"""Unit tests for the fixed Digital Twin Tier-2 logic.

Covers:
  - twin_sync._read_background_load: background = total − service_load
  - placement_verifier._resources_per_twin_node: per-node grouping with correct resources
  - placement_verifier._make_test_pod_spec_for_node: nodeSelector pinned to correct twin-node
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock

from src.drl.twin_sync import _NODE_MAPPING, _read_background_load
from src.drl.placement_verifier import (
    _make_test_pod_spec_for_node,
    _resources_per_twin_node,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_redis(milp_payload: dict | None) -> MagicMock:
    """Return a mock Redis client whose GET always returns milp_payload as JSON."""
    rdb = MagicMock()
    rdb.get.return_value = json.dumps(milp_payload) if milp_payload is not None else None
    return rdb


_ALL_STANDARD_ON_N3 = {
    "api-gateway":  {"node": "edge-nodes-4", "variant": "standard"},
    "ingest":       {"node": "edge-nodes-4", "variant": "standard"},
    "preprocess":   {"node": "edge-nodes-4", "variant": "standard"},
    "detection":    {"node": "edge-nodes-4", "variant": "standard"},
    "gen-ai":       {"node": "edge-nodes-4", "variant": "standard"},
    "postprocess":  {"node": "edge-nodes-4", "variant": "standard"},
}

_ALL_ON_N0 = {
    "api-gateway":  {"node": "edge-nodes-1", "variant": "standard"},
    "ingest":       {"node": "edge-nodes-1", "variant": "standard"},
    "preprocess":   {"node": "edge-nodes-1", "variant": "standard"},
    "detection":    {"node": "edge-nodes-1", "variant": "yolo26-medium"},
    "gen-ai":       {"node": "edge-nodes-1", "variant": "llama-3b-small"},
    "postprocess":  {"node": "edge-nodes-1", "variant": "standard"},
}


# ── Tests: _read_background_load ──────────────────────────────────────────────

class TestReadBackgroundLoad(unittest.TestCase):

    def test_no_redis_data_returns_zeros(self):
        """When milp:placement is absent, background load should be all zeros."""
        rdb = _make_redis(None)
        cpu, mem = _read_background_load(rdb)
        for nid in ("n0", "n1", "n2", "n3"):
            self.assertEqual(cpu[nid], 0.0)
            self.assertEqual(mem[nid], 0.0)

    def test_background_equals_total_minus_services(self):
        """Background = resource_usage − service load on each node."""
        svc_cpu_n3 = 6 * 0.20          # 6 standard services × 0.20 CPU = 1.20
        svc_mem_n3 = 6 * 0.25          # 6 standard services × 0.25 GB  = 1.50
        total_cpu_n3 = 2.50
        total_mem_n3 = 2.00

        rdb = _make_redis({
            "placement": _ALL_STANDARD_ON_N3,
            "resource_usage": {"n0": 0.0, "n1": 0.0, "n2": 0.0, "n3": total_cpu_n3},
            "mem_usage":      {"n0": 0.0, "n1": 0.0, "n2": 0.0, "n3": total_mem_n3},
        })
        cpu, mem = _read_background_load(rdb)

        self.assertAlmostEqual(cpu["n3"], total_cpu_n3 - svc_cpu_n3, places=6)
        self.assertAlmostEqual(mem["n3"], total_mem_n3 - svc_mem_n3, places=6)
        for nid in ("n0", "n1", "n2"):
            self.assertEqual(cpu[nid], 0.0)
            self.assertEqual(mem[nid], 0.0)

    def test_background_never_negative(self):
        """Even if service_load > resource_usage (stale data), result must be ≥ 0."""
        rdb = _make_redis({
            "placement": _ALL_STANDARD_ON_N3,
            "resource_usage": {"n3": 0.10},   # less than service load (1.20)
            "mem_usage":      {"n3": 0.10},
        })
        cpu, mem = _read_background_load(rdb)
        for nid in ("n0", "n1", "n2", "n3"):
            self.assertGreaterEqual(cpu[nid], 0.0)
            self.assertGreaterEqual(mem[nid], 0.0)

    def test_no_placement_key_background_equals_total(self):
        """If milp:placement has no placement dict, background = full resource_usage."""
        rdb = _make_redis({
            "resource_usage": {"n0": 0.5, "n1": 0.0, "n2": 0.0, "n3": 1.5},
            "mem_usage":      {"n0": 0.3, "n1": 0.0, "n2": 0.0, "n3": 1.0},
        })
        cpu, mem = _read_background_load(rdb)
        self.assertAlmostEqual(cpu["n0"], 0.5)
        self.assertAlmostEqual(cpu["n3"], 1.5)


# ── Tests: _resources_per_twin_node ───────────────────────────────────────────

class TestResourcesPerTwinNode(unittest.TestCase):

    def test_all_on_n3_produces_single_entry(self):
        """When all services are on edge-nodes-4, only twin-edge-nodes-4 appears."""
        result = _resources_per_twin_node(_ALL_STANDARD_ON_N3)
        self.assertEqual(list(result.keys()), ["twin-edge-nodes-4"])

    def test_all_on_n3_cpu_mem_correct(self):
        """6 standard services on n3 use the verifier's 85% test-pod budget."""
        result = _resources_per_twin_node(_ALL_STANDARD_ON_N3)
        cpu_mc, mem_mi = result["twin-edge-nodes-4"]
        self.assertEqual(cpu_mc, round(6 * 0.20 * 1000 * 0.85))   # 1020 m
        self.assertEqual(mem_mi, round(6 * 0.25 * 1024 * 0.85))   # 1306 Mi

    def test_split_placement_produces_two_entries(self):
        """Services split across two nodes should produce two test-pod entries."""
        placement = {
            "api-gateway": {"node": "edge-nodes-1", "variant": "standard"},
            "ingest":      {"node": "edge-nodes-1", "variant": "standard"},
            "detection":   {"node": "edge-nodes-4", "variant": "yolo26-nano"},
        }
        result = _resources_per_twin_node(placement)
        self.assertIn("twin-edge-nodes-1", result)
        self.assertIn("twin-edge-nodes-4", result)
        self.assertNotIn("twin-edge-nodes-2", result)
        self.assertNotIn("twin-edge-nodes-3", result)

    def test_split_placement_resources_per_node(self):
        """Each twin-node entry aggregates only its own services."""
        placement = {
            "api-gateway": {"node": "edge-nodes-1", "variant": "standard"},   # 200m / 256Mi
            "ingest":      {"node": "edge-nodes-1", "variant": "standard"},   # 200m / 256Mi
            "detection":   {"node": "edge-nodes-4", "variant": "yolo26-nano"},# 500m / 799Mi
        }
        result = _resources_per_twin_node(placement)
        n1_cpu, n1_mem = result["twin-edge-nodes-1"]
        n4_cpu, n4_mem = result["twin-edge-nodes-4"]
        self.assertEqual(n1_cpu, round(2 * 0.20 * 1000 * 0.85))        # 340 m
        self.assertEqual(n1_mem, round(2 * 0.25 * 1024 * 0.85))        # 435 Mi
        self.assertEqual(n4_cpu, round(0.50 * 1000 * 0.85))            # 425 m
        self.assertEqual(n4_mem, round(0.78 * 1024 * 0.85))            # 679 Mi

    def test_all_on_n0_scenario_single_entry(self):
        """All-zeros DRL action (all services on Node 1) → only twin-edge-nodes-1."""
        result = _resources_per_twin_node(_ALL_ON_N0)
        self.assertEqual(list(result.keys()), ["twin-edge-nodes-1"])

    def test_node_mapping_covers_all_four_twins(self):
        """_NODE_MAPPING should have exactly 4 entries covering all twin nodes."""
        twin_names = [twin for _, _, twin in _NODE_MAPPING]
        self.assertEqual(
            sorted(twin_names),
            ["twin-edge-nodes-1", "twin-edge-nodes-2",
             "twin-edge-nodes-3", "twin-edge-nodes-4"],
        )


# ── Tests: _make_test_pod_spec_for_node ───────────────────────────────────────

class TestMakeTestPodSpecForNode(unittest.TestCase):

    def _spec(self, twin="twin-edge-nodes-1", cpu=500, mem=256):
        return _make_test_pod_spec_for_node("test-pod", cpu, mem, twin)

    def test_node_selector_pinned_to_correct_twin(self):
        """nodeSelector must include twin-of pointing to the real node (twin-of label exists on KWOK nodes)."""
        spec = self._spec(twin="twin-edge-nodes-2")
        selector = spec["spec"]["nodeSelector"]
        self.assertEqual(selector.get("twin-of"), "edge-nodes-2")

    def test_node_selector_includes_kwok_type(self):
        """nodeSelector must also have type=kwok for KWOK toleration to work."""
        spec = self._spec()
        self.assertEqual(spec["spec"]["nodeSelector"].get("type"), "kwok")

    def test_resource_requests_match_inputs(self):
        """Container resource requests must exactly match the given cpu_mc / mem_mi."""
        spec = self._spec(cpu=750, mem=512)
        requests = spec["spec"]["containers"][0]["resources"]["requests"]
        self.assertEqual(requests["cpu"], "750m")
        self.assertEqual(requests["memory"], "512Mi")

    def test_kwok_toleration_present(self):
        """Pod must tolerate kwok.x-k8s.io/node=fake:NoSchedule."""
        spec = self._spec()
        tolerations = spec["spec"]["tolerations"]
        kwok_tol = next(
            (t for t in tolerations if t.get("key") == "kwok.x-k8s.io/node"), None
        )
        self.assertIsNotNone(kwok_tol, "KWOK toleration missing")
        self.assertEqual(kwok_tol["effect"], "NoSchedule")

    def test_different_twin_nodes_get_different_selectors(self):
        """Two pods for different nodes must produce different twin-of values."""
        s1 = self._spec(twin="twin-edge-nodes-1")
        s3 = self._spec(twin="twin-edge-nodes-3")
        h1 = s1["spec"]["nodeSelector"]["twin-of"]
        h3 = s3["spec"]["nodeSelector"]["twin-of"]
        self.assertNotEqual(h1, h3)


if __name__ == "__main__":
    unittest.main()
