"""Tests for live scenario load discovery.

The MILP live path treats pods labelled scenario-load=true as background
resource pressure on their pinned node.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src import k8s_client


def _container(cpu: str, memory: str):
    return SimpleNamespace(
        resources=SimpleNamespace(
            requests={"cpu": cpu, "memory": memory},
        ),
    )


def _pod(
    name: str,
    phase: str,
    node: str | None,
    cpu: str,
    memory: str,
    target_label: str | None = None,
):
    labels = {}
    if target_label:
        labels["scenario-target-node"] = target_label
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name, labels=labels),
        spec=SimpleNamespace(node_name=node, containers=[_container(cpu, memory)]),
        status=SimpleNamespace(phase=phase),
    )


class TestScenarioLoadDiscovery(unittest.TestCase):
    def test_quantity_parsers(self):
        self.assertAlmostEqual(k8s_client.parse_cpu_quantity("1500m"), 1.5)
        self.assertAlmostEqual(k8s_client.parse_cpu_quantity("2"), 2.0)
        self.assertAlmostEqual(k8s_client.parse_memory_quantity_gb("1024Mi"), 1.0)
        self.assertAlmostEqual(k8s_client.parse_memory_quantity_gb("2Gi"), 2.0)

    def test_groups_pending_and_running_pods_by_node(self):
        core = MagicMock()
        core.list_namespaced_pod.return_value.items = [
            _pod("load-a", "Running", "edge-nodes-1", "1000m", "512Mi"),
            _pod("load-b", "Pending", None, "500m", "256Mi", "edge-nodes-1"),
            _pod("done", "Succeeded", "edge-nodes-1", "4", "4Gi"),
            _pod("load-c", "Running", "edge-nodes-2", "2", "1Gi"),
        ]

        with patch.object(k8s_client, "_core", return_value=core):
            totals = k8s_client.get_scenario_load_by_node()

        self.assertAlmostEqual(totals["edge-nodes-1"]["cpu_cores"], 1.5)
        self.assertAlmostEqual(totals["edge-nodes-1"]["mem_gb"], 0.75)
        self.assertEqual(len(totals["edge-nodes-1"]["pods"]), 2)
        self.assertAlmostEqual(totals["edge-nodes-2"]["cpu_cores"], 2.0)
        self.assertAlmostEqual(totals["edge-nodes-2"]["mem_gb"], 1.0)
        self.assertNotIn("done", [p["name"] for p in totals["edge-nodes-1"]["pods"]])


if __name__ == "__main__":
    unittest.main()
