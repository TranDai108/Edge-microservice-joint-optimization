import os

# ── SINGLE SOURCE OF TRUTH FOR MILP OBJECTIVE WEIGHTS ──
# Recommended balanced weights from the multi-seed true Pareto-front analysis.
# A_norm (Accuracy) remains the primary objective, C_norm (Energy) is secondary,
# and D_norm (Disruption) is a soft constraint backed by hard storm limits.
#
# These are used across the entire codebase (milp_agent, controller, solvers, etc)
# to ensure the MILP agent writes expert trajectories with the exact same
# objective target as what we evaluate.
# ───────────────────────────────────────────────────────

def _env_float(*names: str, default: str) -> float:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return float(value)
    return float(default)


# Prefer MILP_W_* env vars; accept legacy W_* for backwards compatibility.
MILP_W_C = _env_float("MILP_W_C", "W_C", default="0.20")  # Energy efficiency
MILP_W_D = _env_float("MILP_W_D", "W_D", default="0.10")  # Migration disruption penalty
MILP_W_A = _env_float("MILP_W_A", "W_A", default="0.70")  # Detection quality / accuracy

# Ensure weights sum to 1.0 (or warn if not, though normalized objective tolerates it)
total = MILP_W_C + MILP_W_D + MILP_W_A
if abs(total - 1.0) > 1e-4:
    import logging
    logging.getLogger("config").warning(
        f"MILP weights sum to {total:.3f}, not 1.0 (w_c={MILP_W_C}, w_d={MILP_W_D}, w_a={MILP_W_A})"
    )
