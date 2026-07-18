"""Tests for leader-gated DRL Tier-2 proposal publishing."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from src.drl import drl_agent


class TestDrlProposedLeaderGate(unittest.TestCase):
    def setUp(self) -> None:
        self.rdb = MagicMock()
        self.payload = {"placement": {"gen-ai": {"node": "edge-nodes-1"}}}

    def test_local_mode_writes_directly(self):
        with patch.object(drl_agent, "_elector", None):
            written = drl_agent._write_proposed_placement(self.rdb, self.payload)

        self.assertTrue(written)
        self.rdb.setex.assert_called_once_with(
            "drl:placement:proposed",
            60,
            json.dumps(self.payload),
        )

    def test_ha_mode_uses_epoch_fenced_guard(self):
        elector = object()
        with (
            patch.object(drl_agent, "_elector", elector),
            patch.object(drl_agent, "placement_write_guarded", return_value=True) as guarded,
        ):
            written = drl_agent._write_proposed_placement(self.rdb, self.payload)

        self.assertTrue(written)
        guarded.assert_called_once_with(
            self.rdb,
            "drl:placement:proposed",
            self.payload,
            elector,
            ttl=60,
        )
        self.rdb.setex.assert_not_called()

    def test_standby_rejection_never_falls_back_to_direct_write(self):
        with (
            patch.object(drl_agent, "_elector", object()),
            patch.object(drl_agent, "placement_write_guarded", return_value=False),
        ):
            written = drl_agent._write_proposed_placement(self.rdb, self.payload)

        self.assertFalse(written)
        self.rdb.setex.assert_not_called()

    def test_guard_exception_never_falls_back_to_direct_write(self):
        with self.assertLogs("drl-agent", level="WARNING") as logs:
            with (
                patch.object(drl_agent, "_elector", object()),
                patch.object(
                    drl_agent,
                    "placement_write_guarded",
                    side_effect=RuntimeError("guard unavailable"),
                ),
            ):
                written = drl_agent._write_proposed_placement(self.rdb, self.payload)

        self.assertFalse(written)
        self.rdb.setex.assert_not_called()
        self.assertTrue(any("leader guard failed" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
