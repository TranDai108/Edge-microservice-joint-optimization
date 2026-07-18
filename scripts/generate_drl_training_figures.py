#!/usr/bin/env python3
"""Generate thesis figures from existing DRL/BC/PPO artifacts.

The script intentionally reads post-training evidence instead of re-running
training:

- BC loss JSONL from ``results/bc_training_loss.jsonl``.
- Expert trajectory JSONL from a generated DRL expert dataset.
- TensorBoard scalar events from ``src/drl/logs``.
- Offline evaluation JSONL from ``results/thesis_main_offline``.

Outputs are written as PNG files that can be included directly from the thesis
``img/charts`` directory.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:  # pragma: no cover - handled at runtime
    EventAccumulator = None  # type: ignore[assignment]


# ── Global aesthetics ────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":         "DejaVu Sans",
    "font.size":           11,
    "axes.titlesize":      13,
    "axes.titleweight":    "bold",
    "axes.titlepad":       12,
    "axes.labelsize":      11,
    "axes.labelcolor":     "#1e293b",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.linewidth":      0.9,
    "axes.edgecolor":      "#94a3b8",
    "xtick.labelsize":     10,
    "ytick.labelsize":     10,
    "xtick.color":         "#475569",
    "ytick.color":         "#475569",
    "xtick.major.size":    4,
    "ytick.major.size":    4,
    "legend.fontsize":     10,
    "legend.frameon":      False,
    "legend.borderpad":    0.6,
    "figure.dpi":          150,
    "figure.facecolor":    "white",
    "savefig.facecolor":   "white",
    "savefig.edgecolor":   "none",
    "lines.linewidth":     2.2,
    "lines.solid_capstyle": "round",
})

# ── Curated colour palette ────────────────────────────────────────────────────
C = {
    "MILP":    "#2563eb",   # Royal blue
    "DRL":     "#059669",   # Emerald green
    "Hybrid":  "#dc2626",   # Red
    "K8s":     "#7c3aed",   # Purple
    "BC":      "#0284c7",   # Sky blue
    "PPO":     "#d97706",   # Amber
    "Muted":   "#64748b",   # Slate
    "Grid":    "#e2e8f0",   # Light slate
    "Bg":      "#f8fafc",   # Near-white tint
    "Text":    "#0f172a",   # Dark ink
    "Accept":  "#10b981",   # Green (DRL accepted)
    "Reject":  "#f97316",   # Orange (fallback)
}

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THESIS_DIR = (
    ROOT
    / "docs/thesis/Joint_Optimization_of_Model_Variants_and_Service_Placement_in_Edge_Computing"
)

SERVICE_HEADS = ["m0", "m1", "m2", "m3", "m4", "m5"]
ACTION_DIMS = [4, 4, 4, 12, 12, 4]


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def save_fig(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.relative_to(ROOT)}")


def _style_ax(
    ax: plt.Axes,
    title: str,
    ylabel: str | None = None,
    xlabel: str | None = None,
) -> None:
    """Apply consistent professional styling to an axes."""
    ax.set_title(title, fontsize=13, fontweight="bold", color=C["Text"], pad=12)
    if ylabel:
        ax.set_ylabel(ylabel, color=C["Text"])
    if xlabel:
        ax.set_xlabel(xlabel, color=C["Text"])
    ax.grid(True, axis="y", color=C["Grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors="#475569")


def _infobox(ax: plt.Axes, text: str, loc: str = "lower right") -> None:
    """Place a light annotation box."""
    ha = "right" if "right" in loc else "left"
    va = "bottom" if "bottom" in loc else "top"
    x = 0.98 if "right" in loc else 0.02
    y = 0.04 if "bottom" in loc else 0.96
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        ha=ha, va=va,
        fontsize=9.5,
        color=C["Text"],
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            edgecolor="#cbd5e1",
            linewidth=0.8,
            alpha=0.92,
        ),
    )


def scalar_series(run_dir: Path, tag: str) -> pd.DataFrame:
    if EventAccumulator is None or not run_dir.exists():
        return pd.DataFrame(columns=["step", "value"])
    try:
        acc = EventAccumulator(str(run_dir))
        acc.Reload()
    except Exception:
        return pd.DataFrame(columns=["step", "value"])
    if tag not in acc.Tags().get("scalars", []):
        return pd.DataFrame(columns=["step", "value"])
    rows = [{"step": item.step, "value": float(item.value)} for item in acc.Scalars(tag)]
    return pd.DataFrame(rows)


def discover_default_ppo_runs(tb_log_dir: Path) -> list[Path]:
    preferred = [
        tb_log_dir / "bc_warmstart/PPO_1",
        tb_log_dir / "PPO_2",
    ]
    runs = [p for p in preferred if p.exists()]
    if runs:
        return runs
    candidates: list[tuple[int, Path]] = []
    for event_file in tb_log_dir.glob("**/events.out.tfevents*"):
        run = event_file.parent
        data = scalar_series(run, "rollout/ep_rew_mean")
        if not data.empty:
            candidates.append((len(data), run))
    candidates.sort(reverse=True, key=lambda item: item[0])
    return [path for _, path in candidates[:2]]


def load_offline_mode(offline_dir: Path, filename: str, label: str) -> pd.DataFrame:
    rows = read_jsonl(offline_dir / filename)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["method"] = label
    return df


# ── Figure 1 – DRL Training Pipeline ─────────────────────────────────────────

def plot_training_pipeline(out_dir: Path) -> None:
    """Clean linear layout — no arrow crossings, mathtext-safe labels."""
    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ── Phase backgrounds ────────────────────────────────────────────────────
    # Phase 1: MILP → BC (left third)
    ax.add_patch(FancyBboxPatch(
        (0.01, 0.08), 0.36, 0.78,
        boxstyle="round,pad=0.012", linewidth=1.0,
        edgecolor="#93c5fd", facecolor="#eff6ff", alpha=0.70, zorder=0,
    ))
    ax.text(0.19, 0.91, "Phase 1 — BC Pre-training",
            ha="center", va="center", fontsize=9.5,
            color="#1d4ed8", style="italic", fontweight="bold")

    # Phase 2: PPO loop (right two-thirds)
    ax.add_patch(FancyBboxPatch(
        (0.38, 0.08), 0.61, 0.78,
        boxstyle="round,pad=0.012", linewidth=1.0,
        edgecolor="#fcd34d", facecolor="#fffbeb", alpha=0.70, zorder=0,
    ))
    ax.text(0.685, 0.91, "Phase 2 — MaskablePPO Fine-tuning",
            ha="center", va="center", fontsize=9.5,
            color="#92400e", style="italic", fontweight="bold")

    # ── Nodes — all on a single horizontal row (y=0.50) ──────────────────────
    # Layout: MILP(0.09) → BC(0.24) → [checkpoint above at 0.24/0.75]
    #         EdgeEnv(0.52) ↔ PPO(0.68) ↔ ActionMasker below(0.52/0.25)
    #         → Checkpoint(0.88)
    BOX_W, BOX_H = 0.115, 0.20
    MID_Y = 0.49

    def node(label, cx, cy, fc, ec, fontsize=9.5):
        ax.add_patch(FancyBboxPatch(
            (cx - BOX_W / 2, cy - BOX_H / 2), BOX_W, BOX_H,
            boxstyle="round,pad=0.014",
            linewidth=1.8, edgecolor=ec, facecolor=fc, zorder=2,
        ))
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold",
                color=C["Text"], zorder=3, linespacing=1.35)

    # Phase 1 nodes
    node("MILP Solver\nExpert data",   0.09,  MID_Y, "#dbeafe", C["MILP"])
    node("Behavioral\nCloning",        0.25,  MID_Y, "#e0f2fe", C["BC"])
    node("BC Warm-start\ncheckpoint",  0.31,  0.78,  "#f1f5f9", C["Muted"], fontsize=9)

    # Phase 2 nodes
    node("EdgeEnv +\nScenarioGen",     0.53,  MID_Y, "#d1fae5", "#059669")
    node("ActionMasker\nvalid mask",   0.53,  0.22,  "#ede9fe", "#7c3aed")
    node("MaskablePPO\nupdate",        0.70,  MID_Y, "#fef3c7", C["PPO"])
    node("PPO Policy\ncheckpoint",     0.88,  MID_Y, "#d1fae5", C["DRL"])

    # ── Arrows — designed with no crossings ──────────────────────────────────
    def arr(x0, y0, x1, y1, lbl="", rad=0.0, lbl_dy=0.025):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>", color="#475569",
                lw=1.5, mutation_scale=13,
                connectionstyle=f"arc3,rad={rad}",
            ), zorder=4)
        if lbl:
            mx = (x0 + x1) / 2
            my = (y0 + y1) / 2
            ax.text(mx, my + lbl_dy, lbl, ha="center", va="bottom",
                    fontsize=8.5, color="#64748b", style="italic")

    # MILP → BC (horizontal)
    arr(0.148, MID_Y, 0.192, MID_Y, "expert data")
    # BC → BC checkpoint (diagonal up-right, label clears node)
    arr(0.30, MID_Y + BOX_H/2, 0.253, 0.78 - BOX_H/2, "save weights", lbl_dy=0.02)
    # BC checkpoint → MaskablePPO (curved bridge across phase boundary)
    arr(0.368, 0.78, 0.642, MID_Y + BOX_H/2 + 0.01, "warm-start", rad=-0.22)
    # EdgeEnv → ActionMasker (straight down)
    arr(0.53, MID_Y - BOX_H/2, 0.53, 0.22 + BOX_H/2, "compute\nmask", lbl_dy=0.02)
    # ActionMasker → MaskablePPO (diagonal, no crossing)
    arr(0.588, 0.22, 0.642, MID_Y - BOX_H/2, "masked at", lbl_dy=0.015)
    # EdgeEnv → MaskablePPO (state + reward, straight right)
    arr(0.588, MID_Y, 0.642, MID_Y, "st+1, rt")
    # MaskablePPO → EdgeEnv (policy feedback, curved back above)
    arr(0.700, MID_Y + BOX_H/2, 0.588, MID_Y + BOX_H/2 + 0.03,
        "policy", rad=-0.35, lbl_dy=0.02)
    # MaskablePPO → Final checkpoint
    arr(0.758, MID_Y, 0.822, MID_Y, "save")

    # ── Edge-label overrides with nicer text where mathtext would break ───────
    # (already plain text above — skip LaTeX to avoid mathtext dependency)

    ax.set_title(
        "DRL Training Pipeline: BC Warm-start  →  MaskablePPO Fine-tuning",
        fontsize=14, fontweight="bold", color=C["Text"], pad=8,
    )
    save_fig(fig, out_dir / "fig_drl_training_pipeline.png")


# ── Figure 2 – BC Loss ────────────────────────────────────────────────────────

def plot_bc_loss(bc_loss_path: Path, out_dir: Path) -> None:
    rows = read_jsonl(bc_loss_path)
    if not rows:
        print(f"skip BC loss: missing {bc_loss_path}")
        return
    df = pd.DataFrame(rows)
    df["train_loss"] = df["train_loss"].astype(float)
    df["val_loss"]   = df["val_loss"].astype(float)

    fig, ax = plt.subplots(figsize=(9, 5.2))

    ax.plot(df["epoch"], df["train_loss"],
            color=C["BC"], linewidth=2.4, label="Train loss", zorder=3)
    ax.plot(df["epoch"], df["val_loss"],
            color=C["DRL"], linewidth=2.4, linestyle="--",
            label="Validation loss", zorder=3)

    # Fill BETWEEN the two curves only (not down to zero)
    ax.fill_between(
        df["epoch"],
        df["train_loss"], df["val_loss"],
        where=(df["train_loss"] >= df["val_loss"]),
        alpha=0.12, color=C["BC"], interpolate=True,
    )
    ax.fill_between(
        df["epoch"],
        df["val_loss"], df["train_loss"],
        where=(df["val_loss"] > df["train_loss"]),
        alpha=0.12, color=C["DRL"], interpolate=True,
    )

    best_idx = int(df["val_loss"].idxmin())
    best = df.loc[best_idx]
    ax.scatter([best["epoch"]], [best["val_loss"]],
               s=100, color=C["PPO"], zorder=5,
               edgecolors="white", linewidths=1.5)
    # Place annotation to the left of the point to avoid right-edge clipping
    ax.annotate(
        f"Best val = {best['val_loss']:.4f}\n(epoch {int(best['epoch'])})",
        xy=(best["epoch"], best["val_loss"]),
        xytext=(best["epoch"] - 4.0, best["val_loss"] + 0.06),
        arrowprops=dict(arrowstyle="->", color=C["Muted"], lw=1.2),
        fontsize=9.5, color=C["Text"],
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#cbd5e1", linewidth=0.8, alpha=0.95),
    )

    _style_ax(ax,
              "Behavioral Cloning Loss — MILP Expert Data",
              ylabel="Cross-entropy loss",
              xlabel="Epoch")
    # Tight y-axis: zoom into the meaningful range, not 0
    y_min = min(df["train_loss"].min(), df["val_loss"].min())
    y_max = max(df["train_loss"].max(), df["val_loss"].max())
    margin = (y_max - y_min) * 0.12
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.legend(loc="upper right")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    save_fig(fig, out_dir / "fig_bc_loss_real.png")


# ── Figure 3 – Expert Dataset Quality ────────────────────────────────────────

def plot_expert_dataset_quality(expert_path: Path, out_dir: Path) -> None:
    rows = read_jsonl(expert_path)
    actions = [row.get("action") for row in rows if isinstance(row.get("action"), list)]
    actions = [a for a in actions if len(a) == len(ACTION_DIMS)]
    if not actions:
        print(f"skip expert quality: missing valid actions in {expert_path}")
        return

    arr = np.asarray(actions, dtype=int)
    per_head_cardinality = [len(set(arr[:, idx].tolist())) for idx in range(arr.shape[1])]
    per_head_ratio = [per_head_cardinality[i] / ACTION_DIMS[i] for i in range(len(ACTION_DIMS))]
    joint_unique = len({tuple(a) for a in arr.tolist()})
    joint_ratio = joint_unique / len(arr)
    top_joint_count = Counter(tuple(a) for a in arr.tolist()).most_common(1)[0][1]
    top_joint_ratio = top_joint_count / len(arr)

    sources = Counter(str(row.get("source", "unknown")) for row in rows)
    source_names = [name for name, _ in sources.most_common()]
    source_counts = [sources[name] for name in source_names]

    # Color per bar: highlight < 0.75 coverage in amber
    bar_colors = [C["BC"] if r >= 0.75 else C["PPO"] for r in per_head_ratio]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2),
                             gridspec_kw={"wspace": 0.35})

    # Left: per-head coverage
    ax = axes[0]
    bars = ax.bar(SERVICE_HEADS, per_head_ratio, color=bar_colors,
                  alpha=0.88, width=0.55, zorder=3,
                  edgecolor="white", linewidth=0.5)
    ax.axhline(1.0, color="#94a3b8", linewidth=0.8, linestyle=":")
    for idx, (value, ratio) in enumerate(zip(per_head_cardinality, per_head_ratio)):
        ax.text(idx, ratio + 0.02,
                f"{value}/{ACTION_DIMS[idx]}",
                ha="center", va="bottom", fontsize=9.5,
                fontweight="bold", color=C["Text"])
    ax.set_ylim(0, 1.22)
    _style_ax(ax, "Per-head Action Coverage",
              ylabel="Observed / possible actions",
              xlabel="Action head (service)")

    # Right: sample sources
    ax = axes[1]
    # Color by source type
    src_colors = [C["PPO"] if n in ("regret", "live_correction") else C["BC"]
                  for n in source_names[:8]]
    ax.barh(source_names[:8][::-1], source_counts[:8][::-1],
            color=src_colors[::-1], alpha=0.88,
            edgecolor="white", linewidth=0.5, height=0.55, zorder=3)
    _style_ax(ax, "Expert Sample Sources",
              xlabel="Sample count")
    ax.grid(True, axis="x", color=C["Grid"], linewidth=0.8, zorder=0)
    ax.grid(False, axis="y")
    # Place infobox BELOW the chart as a figure-level text to avoid overlap
    stats_text = (
        f"Total rows: {len(arr):,}   |   "
        f"Unique joint actions: {joint_unique}   |   "
        f"Unique ratio: {joint_ratio:.3f}   |   "
        f"Top-action share: {top_joint_ratio:.3f}"
    )
    fig.text(0.5, -0.02, stats_text, ha="center", va="top",
             fontsize=9.5, color=C["Muted"],
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor="#cbd5e1", linewidth=0.8))

    fig.suptitle("Behavioral Cloning — Expert Dataset Quality",
                 fontsize=14, fontweight="bold", color=C["Text"], y=1.01)
    save_fig(fig, out_dir / "fig_bc_expert_dataset_quality.png")


# ── Figure 4 – PPO Telemetry ──────────────────────────────────────────────────

def plot_ppo_telemetry(run_dirs: list[Path], out_dir: Path) -> None:
    if not run_dirs:
        print("skip PPO telemetry: no TensorBoard runs found")
        return

    tags = [
        ("rollout/ep_rew_mean", "Episode Reward Mean", "reward", False),
        ("train/value_loss",    "Value Loss",          "loss",   True),
        ("train/entropy_loss",  "Entropy Loss",        "entropy (nats)", False),
        ("train/approx_kl",    "Approximate KL",      "KL divergence",  False),
    ]
    run_colors = [C["DRL"], C["PPO"], C["BC"], C["MILP"]]
    run_labels = [
        "BC warm-start → PPO",
        "PPO (scratch)",
        "Run 3", "Run 4",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.0))
    axes = axes.flatten()

    for ax, (tag, title, ylabel, log_scale) in zip(axes, tags):
        plotted = False
        for idx, run in enumerate(run_dirs):
            data = scalar_series(run, tag)
            if data.empty:
                continue
            label = run_labels[idx % len(run_labels)]
            clr = run_colors[idx % len(run_colors)]
            ax.plot(data["step"], data["value"],
                    linewidth=2.2, label=label, color=clr, alpha=0.92, zorder=3)
            ax.fill_between(data["step"], data["value"],
                            alpha=0.07, color=clr, zorder=2)
            plotted = True
        if log_scale and plotted:
            ax.set_yscale("log")
        _style_ax(ax, title, ylabel=ylabel,
                  xlabel="Timesteps" if ax in axes[2:] else None)
        if plotted:
            ax.legend(loc="best")
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}")
        )

    fig.suptitle("PPO Training Telemetry — TensorBoard Logs",
                 fontsize=15, fontweight="bold", color=C["Text"])
    fig.subplots_adjust(top=0.91, hspace=0.44, wspace=0.26)
    save_fig(fig, out_dir / "fig_ppo_training_telemetry.png")


# ── Figure 5 – Scenario Gap Analysis ─────────────────────────────────────────

def plot_scenario_gap(offline_dir: Path, out_dir: Path) -> None:
    milp   = load_offline_mode(offline_dir, "offline_main_milp.jsonl",   "MILP")
    drl    = load_offline_mode(offline_dir, "offline_main_drl.jsonl",    "DRL")
    hybrid = load_offline_mode(offline_dir, "offline_main_hybrid.jsonl", "Hybrid")
    if milp.empty or drl.empty or hybrid.empty:
        print(f"skip scenario gap: missing offline files in {offline_dir}")
        return

    base = milp[["cycle", "scenario", "objective_J", "infeasible"]].rename(
        columns={"objective_J": "milp_J", "infeasible": "milp_infeasible"}
    )
    rows_list = []
    for df, method in [(drl, "DRL"), (hybrid, "Hybrid")]:
        merged = df[["cycle", "objective_J"]].merge(base, on="cycle", how="inner")
        merged = merged[np.isfinite(merged["milp_J"]) & np.isfinite(merged["objective_J"])]
        merged["gap_pct"] = (
            100 * (merged["objective_J"] - merged["milp_J"]) / merged["milp_J"].abs()
        )
        merged["method"] = method
        rows_list.append(merged)
    gaps = pd.concat(rows_list, ignore_index=True)
    agg = gaps.groupby(["scenario", "method"])["gap_pct"].mean().unstack()
    infeas = (
        base.groupby("scenario")["milp_infeasible"]
        .mean().reindex(agg.index).fillna(0) * 100
    )
    order = agg.get("DRL", pd.Series(index=agg.index, data=0)) \
               .sort_values(ascending=False).index
    agg    = agg.reindex(order)
    infeas = infeas.reindex(order)

    fig, ax1 = plt.subplots(figsize=(14, 5.8))
    x = np.arange(len(agg.index))
    width = 0.34

    drl_vals = agg.get("DRL",    pd.Series(0, index=agg.index)).fillna(0).values
    hyb_vals = agg.get("Hybrid", pd.Series(0, index=agg.index)).fillna(0).values

    ax1.bar(x - width / 2, drl_vals, width,
            color=C["DRL"], alpha=0.88, label="DRL gap",
            zorder=3, edgecolor="white", linewidth=0.4)
    ax1.bar(x + width / 2, hyb_vals, width,
            color=C["PPO"], alpha=0.88, label="Hybrid gap",
            zorder=3, edgecolor="white", linewidth=0.4)
    ax1.axhline(0, color="#0f172a", linewidth=1.0, zorder=4)

    # Annotate where Hybrid gap is ~0 (demonstrating Hybrid ≈ MILP quality)
    for xi, hv, dv in zip(x, hyb_vals, drl_vals):
        if abs(hv) < 0.05 and abs(dv) > 0.1:   # Hybrid negligible, DRL has gap
            ax1.text(xi + width / 2, 0.03, "~0",
                     ha="center", va="bottom", fontsize=7.5,
                     color=C["PPO"], fontweight="bold")

    _style_ax(ax1, "Scenario-level Objective Gap vs. MILP",
              ylabel="Mean J gap (%)",
              xlabel="Scenario")
    ax1.set_xticks(x)
    ax1.set_xticklabels(agg.index, rotation=38, ha="right", fontsize=9.5)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(x, infeas.values,
             color=C["MILP"], marker="o", markersize=5,
             linewidth=2, label="MILP infeasible rate",
             zorder=5)
    ax2.fill_between(x, infeas.values, alpha=0.10, color=C["MILP"])
    ax2.set_ylabel("MILP infeasible rate (%)", color=C["MILP"])
    ax2.tick_params(axis="y", colors=C["MILP"])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#93c5fd")
    ax2.legend(loc="upper right")

    save_fig(fig, out_dir / "fig_drl_scenario_gap_analysis.png")


# ── Figure 6 – Hybrid Gate Outcomes ──────────────────────────────────────────

def plot_hybrid_gate(offline_dir: Path, out_dir: Path) -> None:
    hybrid = load_offline_mode(offline_dir, "offline_main_hybrid.jsonl", "Hybrid")
    if hybrid.empty or "hybrid_fell_back_to_milp" not in hybrid:
        print(f"skip hybrid gate: missing hybrid fallback fields in {offline_dir}")
        return

    grouped = (
        hybrid.groupby("scenario")["hybrid_fell_back_to_milp"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "fallback", "count": "total"})
    )
    grouped["accept_drl"]    = grouped["total"] - grouped["fallback"]
    grouped["fallback_rate"] = grouped["fallback"] / grouped["total"]
    grouped = grouped.sort_values("fallback_rate", ascending=False)

    fig, ax = plt.subplots(figsize=(13, 5.8))
    x = np.arange(len(grouped.index))
    bar_w = 0.6

    ax.bar(x, grouped["accept_drl"],
           color=C["Accept"], alpha=0.88, width=bar_w,
           label="DRL accepted", zorder=3, edgecolor="white", linewidth=0.4)
    ax.bar(x, grouped["fallback"],
           bottom=grouped["accept_drl"],
           color=C["Reject"], alpha=0.88, width=bar_w,
           label="Fallback to MILP", zorder=3, edgecolor="white", linewidth=0.4)

    # Annotate fallback % directly on fallback segment (not above bar)
    for xi, (_, row) in zip(x, grouped.iterrows()):
        if row["fallback"] > 0:
            seg_mid_y = row["accept_drl"] + row["fallback"] / 2
            ax.text(
                xi, seg_mid_y,
                f'{row["fallback_rate"]:.0%}',
                ha="center", va="center", fontsize=8.5,
                color="white", fontweight="bold",
            )

    _style_ax(ax, "Hybrid Gate Outcomes by Scenario",
              ylabel="Decision cycles",
              xlabel="Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index, rotation=38, ha="right", fontsize=9.5)
    ax.legend(loc="upper left")

    total_fallback = int(grouped["fallback"].sum())
    total = int(grouped["total"].sum())
    # Add bottom margin so infobox clears rotated x-tick labels
    fig.subplots_adjust(bottom=0.26)
    stats_text = (
        f"Total cycles: {total}   | "
        f"Fallback → MILP: {total_fallback} ({total_fallback/max(total,1):.1%}) | "
        f"DRL accepted: {total - total_fallback} ({1 - total_fallback/max(total,1):.1%})"
    )
    fig.text(0.5, 0.01, stats_text, ha="center", va="bottom",
             fontsize=9.5, color=C["Text"],
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8fafc",
                       edgecolor="#cbd5e1", linewidth=0.8))
    save_fig(fig, out_dir / "fig_hybrid_gate_outcomes.png")


# ── Summary JSON ──────────────────────────────────────────────────────────────

def write_summary(
    out_dir: Path,
    bc_loss_path: Path,
    expert_path: Path,
    ppo_runs: list[Path],
    offline_dir: Path,
) -> None:
    summary: dict[str, Any] = {
        "bc_loss": str(bc_loss_path.relative_to(ROOT)) if bc_loss_path.exists() else None,
        "expert_dataset": str(expert_path.relative_to(ROOT)) if expert_path.exists() else None,
        "ppo_runs": [str(path.relative_to(ROOT)) for path in ppo_runs],
        "offline_dir": str(offline_dir.relative_to(ROOT)) if offline_dir.exists() else None,
    }
    bc_rows = read_jsonl(bc_loss_path)
    if bc_rows:
        df = pd.DataFrame(bc_rows)
        best = df.loc[int(df["val_loss"].astype(float).idxmin())]
        summary["bc"] = {
            "epochs":           int(df["epoch"].max()),
            "best_epoch":       int(best["epoch"]),
            "best_val_loss":    float(best["val_loss"]),
            "final_train_loss": float(df["train_loss"].iloc[-1]),
            "final_val_loss":   float(df["val_loss"].iloc[-1]),
        }
    expert_rows = read_jsonl(expert_path)
    expert_actions = [row.get("action") for row in expert_rows
                      if isinstance(row.get("action"), list)]
    if expert_actions:
        valid = [a for a in expert_actions if len(a) == len(ACTION_DIMS)]
        summary["expert"] = {
            "rows": len(valid),
            "unique_joint_actions": len({tuple(a) for a in valid}),
            "per_head_cardinality": [
                len(set(np.asarray(valid, dtype=int)[:, idx].tolist()))
                for idx in range(len(ACTION_DIMS))
            ],
        }
    hybrid = load_offline_mode(offline_dir, "offline_main_hybrid.jsonl", "Hybrid")
    if not hybrid.empty and "hybrid_fell_back_to_milp" in hybrid:
        summary["hybrid"] = {
            "cycles":        int(len(hybrid)),
            "fallback_count": int(hybrid["hybrid_fell_back_to_milp"].fillna(False).sum()),
            "fallback_rate":  float(hybrid["hybrid_fell_back_to_milp"].fillna(False).mean()),
        }
    out_path = out_dir / "fig_drl_training_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {out_path.relative_to(ROOT)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=DEFAULT_THESIS_DIR / "img/charts")
    parser.add_argument("--bc-loss", type=Path,
                        default=ROOT / "results/bc_training_loss.jsonl")
    parser.add_argument("--expert-jsonl", type=Path,
                        default=ROOT / "results/drl_mixed_expert_v8.jsonl")
    parser.add_argument("--tb-log-dir", type=Path,
                        default=ROOT / "src/drl/logs")
    parser.add_argument("--ppo-run", type=Path, action="append", default=[])
    parser.add_argument("--offline-dir", type=Path,
                        default=ROOT / "results/thesis_main_offline")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    ppo_runs = ([path.resolve() for path in args.ppo_run]
                or discover_default_ppo_runs(args.tb_log_dir))

    plot_training_pipeline(out_dir)
    plot_bc_loss(args.bc_loss.resolve(), out_dir)
    plot_expert_dataset_quality(args.expert_jsonl.resolve(), out_dir)
    plot_ppo_telemetry(ppo_runs, out_dir)
    plot_scenario_gap(args.offline_dir.resolve(), out_dir)
    plot_hybrid_gate(args.offline_dir.resolve(), out_dir)
    write_summary(
        out_dir,
        args.bc_loss.resolve(),
        args.expert_jsonl.resolve(),
        ppo_runs,
        args.offline_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
