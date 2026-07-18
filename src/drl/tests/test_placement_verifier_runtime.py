"""Runtime-safety tests for placement verifier loop helpers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.drl.placement_verifier import PlacementVerifier, _make_test_pod_spec_for_node


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def get(self, key: str):
        return self.store.get(key)

    def set(self, key: str, value: str) -> None:
        self.store[key] = value

    def setex(self, key: str, _ttl: int, value: str) -> None:
        self.store[key] = value

    def delete(self, *keys: str) -> None:
        for key in keys:
            self.store.pop(key, None)


class TestPlacementVerifierRuntime(unittest.TestCase):
    def _mk_verifier(self) -> PlacementVerifier:
        v = object.__new__(PlacementVerifier)
        v._rdb = _FakeRedis()  # type: ignore[attr-defined]
        v._last_verify_at = 0.0  # type: ignore[attr-defined]
        return v

    @patch("src.drl.placement_verifier.make_redis_client")
    def test_init_uses_sentinel_aware_redis_factory(self, make_client):
        fake_redis = _FakeRedis()
        make_client.return_value = fake_redis
        with patch.object(PlacementVerifier, "_init_k8s_client", return_value=None):
            verifier = PlacementVerifier()

        self.assertIs(verifier._rdb, fake_redis)  # type: ignore[attr-defined]
        make_client.assert_called_once_with()

    def test_should_reverify_when_no_live_verdict(self):
        v = self._mk_verifier()
        self.assertTrue(v._should_reverify_same_fingerprint())  # type: ignore[attr-defined]

    def test_should_not_reverify_when_live_verdict_exists(self):
        v = self._mk_verifier()
        v._rdb.set("drl:placement:verified", '{"ok":true}')  # type: ignore[attr-defined]
        self.assertFalse(v._should_reverify_same_fingerprint())  # type: ignore[attr-defined]

    def test_publish_writes_compatibility_keys(self):
        v = self._mk_verifier()
        payload = {"ok": True, "proposed_fingerprint": "abc123", "timestamp": "2026-01-01T00:00:00Z"}
        v._publish("drl:placement:verified", payload)  # type: ignore[attr-defined]
        self.assertIsNotNone(v._rdb.get("drl:placement:verified"))  # type: ignore[attr-defined]
        self.assertIsNotNone(v._rdb.get("drl:placement:verified:latest"))  # type: ignore[attr-defined]
        self.assertIsNotNone(v._rdb.get("placement:verifier:last"))  # type: ignore[attr-defined]
        self.assertIsNotNone(v._rdb.get("placement:verifier:latest"))  # type: ignore[attr-defined]
        self.assertIsNotNone(v._rdb.get("tier2:placement:verified"))  # type: ignore[attr-defined]
        self.assertIsNotNone(v._rdb.get("tier2:placement:last_result"))  # type: ignore[attr-defined]
        self.assertIsNotNone(v._rdb.get("drl:tier2_stats"))  # type: ignore[attr-defined]

    def test_publish_clears_opposite_durable_verdict(self):
        v = self._mk_verifier()
        revoked = {"ok": False, "reason": "timeout", "proposed_fingerprint": "abc123", "timestamp": "2026-01-01T00:00:00Z"}
        verified = {"ok": True, "proposed_fingerprint": "abc123", "timestamp": "2026-01-01T00:01:00Z"}
        v._publish("drl:placement:revoked", revoked)  # type: ignore[attr-defined]
        self.assertIsNotNone(v._rdb.get("drl:placement:revoked:latest"))  # type: ignore[attr-defined]
        v._publish("drl:placement:verified", verified)  # type: ignore[attr-defined]
        self.assertIsNone(v._rdb.get("drl:placement:revoked"))  # type: ignore[attr-defined]
        self.assertIsNone(v._rdb.get("drl:placement:revoked:latest"))  # type: ignore[attr-defined]
        self.assertIsNone(v._rdb.get("tier2:placement:revoked"))  # type: ignore[attr-defined]

    def test_publish_heartbeat_marks_current_fingerprint(self):
        v = self._mk_verifier()
        v._publish_heartbeat("abc123")  # type: ignore[attr-defined]
        raw = v._rdb.get("drl:placement:verifier_heartbeat")  # type: ignore[attr-defined]
        self.assertIsNotNone(raw)
        self.assertIn("abc123", raw)

    def test_test_pod_uses_generate_name_to_avoid_scheduler_uid_race(self):
        pod = _make_test_pod_spec_for_node(
            "placement-test-abc123-0-",
            cpu_mc=500,
            mem_mi=256,
            twin_node="twin-edge-nodes-3",
        )

        metadata = pod["metadata"]
        self.assertNotIn("name", metadata)
        self.assertEqual(metadata["generateName"], "placement-test-abc123-0-")
        self.assertEqual(metadata["labels"]["proposed-fingerprint"], "abc123")
        self.assertEqual(metadata["labels"]["target-twin-node"], "twin-edge-nodes-3")


if __name__ == "__main__":
    unittest.main()
