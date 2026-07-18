#!/usr/bin/env python3
"""Create Kubernetes-native node load scenarios for live controller testing.

Examples:
    python scripts/node_load_scenario.py start --node edge-nodes-3 \
        --cpu 1500m --memory 1500Mi --duration 300 --name stress-edge-3

    python scripts/node_load_scenario.py status
    python scripts/node_load_scenario.py stop --name stress-edge-3
    python scripts/node_load_scenario.py stop --all
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from kubernetes import client, config  # noqa: E402
from kubernetes.client.rest import ApiException  # noqa: E402

from k8s_client import parse_cpu_quantity, parse_memory_quantity_gb  # noqa: E402


DEFAULT_NAMESPACE = "default"
DEFAULT_IMAGE = "polinux/stress:latest"
LOAD_LABEL = "scenario-load"
LOAD_LABEL_VALUE = "true"
MODELED_CPU_ANNOTATION = "kltn.io/modeled-cpu"
MODELED_MEMORY_ANNOTATION = "kltn.io/modeled-memory"


def _load_kube_client() -> client.CoreV1Api:
    try:
        cfg = client.Configuration()
        config.load_incluster_config(client_configuration=cfg)
        return client.CoreV1Api(api_client=client.ApiClient(configuration=cfg))
    except config.ConfigException:
        config.load_kube_config()
        return client.CoreV1Api()


def _safe_name(raw: str) -> str:
    text = raw.lower()
    text = re.sub(r"[^a-z0-9.-]+", "-", text).strip("-.")
    text = re.sub(r"-+", "-", text)
    return text[:63].strip("-.") or "node-load"


def _default_name(node: str) -> str:
    return _safe_name(f"node-load-{node}")


def _cpu_workers(cpu: str) -> int:
    cores = parse_cpu_quantity(cpu)
    return max(1, int(math.ceil(cores)))


def _memory_arg(memory: str, ratio: float = 0.85) -> str:
    mem_gb = parse_memory_quantity_gb(memory)
    safe_ratio = min(1.0, max(0.1, ratio))
    mem_mib = max(1, int(math.ceil(mem_gb * 1024 * safe_ratio)))
    return f"{mem_mib}M"


def _pod_manifest(args: argparse.Namespace) -> dict[str, Any]:
    name = _safe_name(args.name or _default_name(args.node))
    cpu_workers = _cpu_workers(args.cpu)
    stress_args = ["--cpu", str(cpu_workers), "--timeout", f"{int(args.duration)}s"]
    if parse_memory_quantity_gb(args.memory) > 0:
        stress_args.extend([
            "--vm",
            "1",
            "--vm-bytes",
            _memory_arg(args.memory, args.vm_bytes_ratio),
        ])

    annotations = {}
    if args.modeled_cpu:
        annotations[MODELED_CPU_ANNOTATION] = args.modeled_cpu
    if args.modeled_memory:
        annotations[MODELED_MEMORY_ANNOTATION] = args.modeled_memory

    metadata = {
        "name": name,
        "namespace": args.namespace,
        "labels": {
            LOAD_LABEL: LOAD_LABEL_VALUE,
            "app": "node-load-scenario",
            "scenario-target-node": args.node,
        },
    }
    if annotations:
        metadata["annotations"] = annotations

    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": metadata,
        "spec": {
            "nodeName": args.node,
            "restartPolicy": "Never",
            "terminationGracePeriodSeconds": 0,
            "containers": [
                {
                    "name": "stress",
                    "image": args.image,
                    "imagePullPolicy": args.image_pull_policy,
                    "command": ["stress"],
                    "args": stress_args,
                    "resources": {
                        "requests": {
                            "cpu": args.cpu,
                            "memory": args.memory,
                        },
                        "limits": {
                            "cpu": args.limit_cpu or args.cpu,
                            "memory": args.limit_memory or args.memory,
                        },
                    },
                }
            ],
        },
    }


def _node_ready(v1: client.CoreV1Api, node_name: str) -> bool:
    node = v1.read_node(node_name)
    if node.spec.unschedulable:
        return False
    for condition in node.status.conditions or []:
        if condition.type == "Ready":
            return condition.status == "True"
    return False


def _delete_pod_if_exists(v1: client.CoreV1Api, name: str, namespace: str) -> None:
    try:
        v1.delete_namespaced_pod(name, namespace, grace_period_seconds=0)
    except ApiException as exc:
        if exc.status != 404:
            raise


def cmd_start(args: argparse.Namespace) -> int:
    manifest = _pod_manifest(args)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    v1 = _load_kube_client()
    if not args.skip_node_check and not _node_ready(v1, args.node):
        print(f"ERROR: node {args.node!r} is not Ready/schedulable", file=sys.stderr)
        return 2

    name = manifest["metadata"]["name"]
    if args.replace:
        _delete_pod_if_exists(v1, name, args.namespace)

    try:
        v1.create_namespaced_pod(args.namespace, manifest)
    except ApiException as exc:
        if exc.status == 409:
            print(
                f"ERROR: pod {name!r} already exists. Use --replace or stop it first.",
                file=sys.stderr,
            )
            return 2
        raise

    print(
        f"Started load pod {name} on {args.node}: "
        f"cpu={args.cpu} memory={args.memory} duration={args.duration}s"
    )
    if args.modeled_cpu or args.modeled_memory:
        print(
            "MILP modeled load override: "
            f"cpu={args.modeled_cpu or args.cpu} "
            f"memory={args.modeled_memory or args.memory}"
        )
    return 0


def _age(ts: datetime | None) -> str:
    if ts is None:
        return "-"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    seconds = max(0, int((datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds()))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h"


def _pod_requests(pod) -> tuple[float, float]:
    cpu = 0.0
    mem = 0.0
    for container in pod.spec.containers or []:
        resources = container.resources
        requests = resources.requests if resources and resources.requests else {}
        cpu += parse_cpu_quantity(requests.get("cpu", "0"))
        mem += parse_memory_quantity_gb(requests.get("memory", "0"))
    return cpu, mem


def _pod_modeled_load(pod, req_cpu: float, req_mem: float) -> tuple[float, float]:
    annotations = pod.metadata.annotations or {}
    modeled_cpu_raw = annotations.get(MODELED_CPU_ANNOTATION)
    modeled_mem_raw = annotations.get(MODELED_MEMORY_ANNOTATION)
    modeled_cpu = parse_cpu_quantity(modeled_cpu_raw) if modeled_cpu_raw else req_cpu
    modeled_mem = parse_memory_quantity_gb(modeled_mem_raw) if modeled_mem_raw else req_mem
    return modeled_cpu, modeled_mem


def cmd_status(args: argparse.Namespace) -> int:
    v1 = _load_kube_client()
    pods = v1.list_namespaced_pod(
        args.namespace,
        label_selector=f"{LOAD_LABEL}={LOAD_LABEL_VALUE}",
    ).items
    if not pods:
        print("No scenario load pods found.")
        return 0

    print(
        f"{'NAME':<32} {'PHASE':<10} {'NODE':<18} "
        f"{'REQ_CPU':>7} {'REQ_MEM':>8} {'MODEL_CPU':>9} {'MODEL_MEM':>9} {'AGE':>6}"
    )
    for pod in sorted(pods, key=lambda p: p.metadata.name or ""):
        cpu, mem = _pod_requests(pod)
        modeled_cpu, modeled_mem = _pod_modeled_load(pod, cpu, mem)
        ts = pod.status.start_time or pod.metadata.creation_timestamp
        print(
            f"{(pod.metadata.name or '-'):<32} "
            f"{(pod.status.phase or '-'):<10} "
            f"{(pod.spec.node_name or '-'):<18} "
            f"{cpu:>7.3f} "
            f"{mem:>8.3f} "
            f"{modeled_cpu:>9.3f} "
            f"{modeled_mem:>9.3f} "
            f"{_age(ts):>6}"
        )
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    if not args.name and not args.all:
        print("ERROR: provide --name or --all", file=sys.stderr)
        return 2

    v1 = _load_kube_client()
    if args.all:
        pods = v1.list_namespaced_pod(
            args.namespace,
            label_selector=f"{LOAD_LABEL}={LOAD_LABEL_VALUE}",
        ).items
        for pod in pods:
            if pod.metadata.name:
                _delete_pod_if_exists(v1, pod.metadata.name, args.namespace)
        print(f"Stopped {len(pods)} scenario load pod(s).")
        return 0

    name = _safe_name(args.name)
    _delete_pod_if_exists(v1, name, args.namespace)
    print(f"Stopped scenario load pod {name}.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create live node load scenarios.")
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="Start a pinned stress pod on one node.")
    start.add_argument("--node", required=True, help="Target Kubernetes node name.")
    start.add_argument("--cpu", default="1000m", help="CPU request, e.g. 1500m or 2.")
    start.add_argument("--memory", default="512Mi", help="Memory request, e.g. 1500Mi.")
    start.add_argument("--duration", type=int, default=300, help="Stress duration in seconds.")
    start.add_argument("--name", default=None, help="Pod name. Defaults to node-load-<node>.")
    start.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    start.add_argument("--image", default=DEFAULT_IMAGE)
    start.add_argument("--image-pull-policy", default="IfNotPresent")
    start.add_argument("--limit-cpu", default=None, help="CPU limit. Defaults to --cpu.")
    start.add_argument("--limit-memory", default=None, help="Memory limit. Defaults to --memory.")
    start.add_argument(
        "--modeled-cpu",
        default=None,
        help=(
            "Synthetic CPU load seen by MILP, e.g. 4400m. "
            "The pod still requests --cpu from Kubernetes."
        ),
    )
    start.add_argument(
        "--modeled-memory",
        default=None,
        help=(
            "Synthetic memory load seen by MILP, e.g. 512Mi. "
            "Defaults to --memory when omitted."
        ),
    )
    start.add_argument(
        "--vm-bytes-ratio",
        type=float,
        default=0.85,
        help="Fraction of --memory actually touched by stress. Defaults to 0.85 to avoid OOMKilled.",
    )
    start.add_argument("--replace", action="store_true", help="Delete an existing pod with the same name first.")
    start.add_argument("--skip-node-check", action="store_true", help="Do not verify node Ready/schedulable state.")
    start.add_argument("--dry-run", action="store_true", help="Print the pod manifest without creating it.")
    start.set_defaults(func=cmd_start)

    status = sub.add_parser("status", help="List scenario load pods.")
    status.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    status.set_defaults(func=cmd_status)

    stop = sub.add_parser("stop", help="Stop scenario load pods.")
    stop.add_argument("--name", default=None, help="Pod name to delete.")
    stop.add_argument("--all", action="store_true", help="Delete all scenario load pods.")
    stop.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    stop.set_defaults(func=cmd_stop)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
