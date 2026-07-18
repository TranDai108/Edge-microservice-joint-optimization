"""Unit tests for D2.2 reward alignment with MILP objective."""

from __future__ import annotations

import unittest

from src.drl.edge_env import build_mock_dataset
from src.drl.reward import evaluate_objective_for_placement
from src.solver.milp_model import solve_placement


class TestRewardAlignment(unittest.TestCase):
    def test_reward_matches_negative_milp_objective(self) -> None:
        ds = build_mock_dataset()
        result = solve_placement(ds, time_limit=30, verbose=False)
        self.assertIsNotNone(result, "MILP solver did not return a feasible result")

        assert result is not None
        breakdown = evaluate_objective_for_placement(
            ds,
            placement=result.placement,
            migrations=result.migrations,
        )

        # D2.2 acceptance target: reward should match -J from milp_model within ±0.01.
        self.assertAlmostEqual(
            breakdown.reward,
            -float(result.objective_value),
            delta=0.01,
        )


if __name__ == "__main__":
    unittest.main()
