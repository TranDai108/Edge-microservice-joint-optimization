from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.ha import leader_election


def _lease(holder: str, transitions: int, resource_version: str = "7"):
    return SimpleNamespace(
        metadata=SimpleNamespace(resource_version=resource_version),
        spec=SimpleNamespace(
            holder_identity=holder,
            lease_duration_seconds=15,
            acquire_time=datetime.now(timezone.utc) - timedelta(seconds=30),
            renew_time=datetime.now(timezone.utc) - timedelta(seconds=20),
            lease_transitions=transitions,
        ),
    )


class TestLeaderElectionFencing(unittest.TestCase):
    def _elector(self) -> leader_election.LeaderElector:
        with patch.object(leader_election, "_K8S_AVAILABLE", True):
            elector = leader_election.LeaderElector("test-leader")
        elector._identity = "pod-a"
        return elector

    def test_stale_leader_cannot_overwrite_new_holder_on_renew(self) -> None:
        elector = self._elector()
        elector._is_leader = True
        api = Mock()
        api.read_namespaced_lease.return_value = _lease("pod-b", 3)

        elector._renew(api)

        self.assertFalse(elector.is_leader())
        api.replace_namespaced_lease.assert_not_called()

    def test_renew_uses_resource_version_compare_and_swap(self) -> None:
        elector = self._elector()
        elector._is_leader = True
        api = Mock()
        api.read_namespaced_lease.return_value = _lease("pod-a", 2, "rv-12")

        elector._renew(api)

        body = api.replace_namespaced_lease.call_args.args[2]
        self.assertEqual(body.metadata.resource_version, "rv-12")
        self.assertEqual(body.spec.holder_identity, "pod-a")
        self.assertEqual(body.spec.lease_transitions, 2)

    def test_takeover_epoch_is_derived_from_durable_transitions(self) -> None:
        elector = self._elector()
        api = Mock()
        api.read_namespaced_lease.return_value = _lease("pod-b", 4, "rv-20")

        elector._check_expired_and_steal(api)

        self.assertTrue(elector.is_leader())
        self.assertEqual(elector.current_epoch(), 6)
        body = api.replace_namespaced_lease.call_args.args[2]
        self.assertEqual(body.metadata.resource_version, "rv-20")
        self.assertEqual(body.spec.lease_transitions, 5)


if __name__ == "__main__":
    unittest.main()
