"""Redis state helpers for MILP agent and scheduler extender.

This module stores the latest MILP solution and confirmed pod placement in Redis.

HA additions: get_client() now prefers Sentinel discovery when
REDIS_SENTINEL_HOSTS is set; write_placement() accepts an optional
LeaderElector so placement writes are epoch-fenced (split-brain safe);
write_heartbeat() lets the solve loop publish liveness.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Optional

import redis

import k8s_client

log = logging.getLogger("redis-state")

MILP_ID_TO_DEPLOY = {
	"m0": "api-gateway",
	"m1": "ingest",
	"m2": "preprocess",
	"m3": "detection",
	"m4": "gen-ai",
	"m5": "postprocess",
}


def get_client(host: str = "redis.default.svc.cluster.local", port: int = 6379) -> redis.Redis:
	"""Return a Sentinel-aware Redis client when REDIS_SENTINEL_HOSTS is set,
	otherwise fall back to a plain single-host client.

	This function is the single point of Redis connection creation for the
	MILP agent.  Import order: try the ha package first (in-cluster image),
	then fall back to plain redis.Redis for local dev/unit tests.
	"""
	try:
		from ha.redis_client import make_redis_client  # type: ignore[import]
		return make_redis_client()
	except Exception:
		return redis.Redis(host=host, port=port, decode_responses=True, socket_timeout=2)


def write_placement(
	rdb: redis.Redis,
	result,
	e_cpu_unit: list[float] | None = None,
	elector=None,
) -> None:
	"""Serialize PlacementResult to Redis with 35s TTL and emit variant events.

	Args:
		rdb:         Redis client.
		result:      PlacementResult from the MILP solver.
		e_cpu_unit:  Live Kepler W/core per node [n0..n3].
		elector:     Optional LeaderElector.  When provided, writes are
		             epoch-fenced via placement_write_guarded() to prevent
		             split-brain writes from a stale pod.
	"""
	node_map = _node_id_to_hostname_map(result)
	now_iso = datetime.utcnow().isoformat()
	payload = {
		"placement": {
			MILP_ID_TO_DEPLOY.get(svc_id, svc_id): {
				"node": node_map.get(node, node),
				"variant": variant,
			}
			for svc_id, (variant, node) in result.placement.items()
		},
		"migration_types": {
			MILP_ID_TO_DEPLOY.get(svc_id, svc_id): mig
			for svc_id, mig in result.migration_types.items()
		},
		"timestamp": now_iso,
		"objective": float(result.objective_value),
		"solve_time": float(result.solve_time),
		"status": result.status,
		"cost_energy": float(result.cost_energy),
		"cost_disruption": float(result.cost_disruption),
		"gain_accuracy": float(result.gain_accuracy),
		"norm_cost_energy": float(result.norm_cost_energy),
		"norm_cost_disruption": float(result.norm_cost_disruption),
		"norm_gain_accuracy": float(result.norm_gain_accuracy),
		"node_scores": _build_node_scores(result),
		"e_cpu_unit": [round(float(v), 4) for v in e_cpu_unit] if e_cpu_unit else [],
		"resource_usage": {
			node: round(float(usage), 4)
			for node, usage in getattr(result, "resource_usage", {}).items()
		},
		"mem_usage": {
			node: round(float(usage), 4)
			for node, usage in getattr(result, "mem_usage", {}).items()
		},
		"background_load": {
			node: {
				"cpu_cores": round(float(getattr(result, "background_cpu", {}).get(node, 0.0)), 4),
				"mem_gb": round(float(getattr(result, "background_mem", {}).get(node, 0.0)), 4),
			}
			for node in sorted(set(getattr(result, "background_cpu", {})) | set(getattr(result, "background_mem", {})))
		},
	}

	# ── Epoch-fenced write (split-brain safe) ────────────────────────────────
	written = False
	if elector is not None:
		try:
			from ha.leader_election import placement_write_guarded  # type: ignore[import]
			written = placement_write_guarded(rdb, "milp:placement", payload, elector, ttl=35)
			if not written:
				log.warning("milp:placement write skipped (not leader or stale epoch)")
		except Exception as exc:
			log.warning("Leader election write guard failed: %s — plain write fallback", exc)

	if not written:
		# No elector or guard failed — plain write (single-pod / dev mode)
		rdb.setex("milp:placement", 35, json.dumps(payload))
		written = True

	if written:
		rdb.setex("milp:solve_timestamp", 35, now_iso)

		for svc_id, mig_type in result.migration_types.items():
			if "AI Model" not in mig_type:
				continue
			variant, node = result.placement.get(svc_id, ("standard", ""))
			event = {
				"service": MILP_ID_TO_DEPLOY.get(svc_id, svc_id),
				"variant": variant,
				"node": node_map.get(node, node),
			}
			rdb.publish("milp:events", json.dumps(event))
			log.info("Published variant event: %s", event)


def write_confirmed_placement(
	rdb: redis.Redis,
	deploy_name: str,
	node_hostname: str,
	variant: str,
) -> None:
	"""Update confirmed placement snapshot from live Kubernetes pod state."""
	raw = rdb.get("milp:confirmed_placement")
	state = json.loads(raw) if raw else {}
	state[deploy_name] = {"node": node_hostname, "variant": variant}
	rdb.set("milp:confirmed_placement", json.dumps(state))


def read_placement(rdb: redis.Redis) -> dict | None:
	raw = rdb.get("milp:placement")
	return json.loads(raw) if raw else None


def read_confirmed_placement(rdb: redis.Redis) -> dict:
	raw = rdb.get("milp:confirmed_placement")
	return json.loads(raw) if raw else {}


def write_expert_trajectory(
	rdb: redis.Redis,
	state_vec: list[float],
	action_vec: list[int],
	reward: float,
	next_state_vec: list[float],
) -> None:
	"""Append expert sample from MILP solve loop for DRL imitation training."""
	traj = {
		"state": state_vec,
		"action": action_vec,
		"reward": float(reward),
		"next_state": next_state_vec,
		"timestamp": datetime.utcnow().isoformat(),
	}
	rdb.lpush("milp:expert_trajectories", json.dumps(traj))
	rdb.ltrim("milp:expert_trajectories", 0, 4999)


def get_mode(rdb: redis.Redis) -> str:
	"""Read control-plane mode from Redis with safe default."""
	mode = rdb.get("system:mode")
	return mode if mode else "milp"


def set_mode(rdb: redis.Redis, mode: str) -> None:
	"""Persist control-plane mode for prioritize routing."""
	valid_modes = {"milp", "drl", "hybrid", "shadow"}
	if mode not in valid_modes:
		raise ValueError(f"Invalid mode '{mode}'. Expected one of {sorted(valid_modes)}")
	rdb.set("system:mode", mode)


def write_weights(rdb: redis.Redis, w_c: float, w_d: float, w_a: float) -> None:
	"""Persist objective weights to Redis so live updates propagate without redeploy."""
	payload = {"w_c": float(w_c), "w_d": float(w_d), "w_a": float(w_a)}
	rdb.set("milp:weights", json.dumps(payload))
	log.info("milp:weights updated → w_c=%.4f w_d=%.4f w_a=%.4f", w_c, w_d, w_a)


def read_weights(rdb: redis.Redis) -> dict | None:
	"""Return live weights from Redis, or None if the key is absent."""
	raw = rdb.get("milp:weights")
	return json.loads(raw) if raw else None


def write_drl_placement(rdb: redis.Redis, placement_dict: dict[str, Any]) -> None:
	"""Store latest DRL placement payload with same TTL as MILP placement."""
	rdb.setex("drl:placement", 35, json.dumps(placement_dict))


def write_heartbeat(rdb: redis.Redis, controller: str, ttl: int = 90) -> None:
	"""Write a liveness heartbeat for the given controller.

	The HA_Runbook alert fires when the key age exceeds 3× the write interval.
	Use ttl = CONTROL_INTERVAL * 3 so one missed cycle does not fire the alert.

	Args:
		rdb:        Redis client.
		controller: Short name used in the Redis key, e.g. "milp" or "drl".
		ttl:        Key TTL in seconds (default 90 = 30 s interval × 3).
	"""
	key = f"{controller}:heartbeat"
	rdb.setex(key, ttl, datetime.utcnow().isoformat())
	log.debug("Heartbeat written: %s (ttl=%ds)", key, ttl)


def read_drl_placement(rdb: redis.Redis) -> dict | None:
	raw = rdb.get("drl:placement")
	return json.loads(raw) if raw else None


def _build_node_scores(result) -> dict[str, int]:
	"""Build scheduler-extender scores from the latest MILP result."""
	node_map = _node_id_to_hostname_map(result)
	scores: dict[str, int] = {}

	# Base score from aggregate node usage: lower usage => slightly better score.
	for node_id, usage in result.resource_usage.items():
		hostname = node_map.get(node_id, node_id)
		scores[hostname] = max(0, 50 - int(float(usage) * 5))

	# Give MILP-chosen nodes a strong bonus to dominate default scoring.
	for _, (_, node_id) in result.placement.items():
		hostname = node_map.get(node_id, node_id)
		scores[hostname] = min(100, scores.get(hostname, 50) + 50)

	return scores


def _node_id_to_hostname_map(result) -> dict[str, str]:
	"""Build n{i} -> hostname map from current schedulable workers.

	Uses the same sorted-worker convention as metrics_collector._get_nodes().
	Falls back to identity mapping when discovery is unavailable.
	"""
	try:
		hostnames = sorted(k8s_client.get_all_worker_nodes())
		if not hostnames:
			return {}
		mapping = {f"n{i}": h for i, h in enumerate(hostnames)}
		# Keep any explicit non-n{i} IDs stable by mapping to themselves.
		for _, (_, node_id) in result.placement.items():
			mapping.setdefault(node_id, node_id)
		return mapping
	except Exception as exc:
		log.warning("Could not resolve node hostname map: %s", exc)
		return {}
