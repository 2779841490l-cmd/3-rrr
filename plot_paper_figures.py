"""
plot_paper_figures.py - IEEE 300 DPI Journal Figure Generator (Aligned with Frozen 45ea2aa)
===========================================================================================
Generates 4 high-resolution (300 DPI) IEEE Transactions figures directly from verified
hardware CSV datasets in `experiments/`:
  1. paper/figures/fig1_deadzone_3regime.png    : Three-regime actuator deadband & stick-slip scans
  2. paper/figures/fig2_short_arm_jacobian.png  : Y-crank shortening gain compression (4.332 -> 2.182 cnt/us)
  3. paper/figures/fig3_ablation_evolution.png  : System evolution milestones & controlled 5-shot ablations
  4. paper/figures/fig4_gate1_gate2a_results.png: Frozen 45ea2aa Gate 1 (1.83 cnt) & Gate 2a (2.39 cnt, 98.2 ms)
"""

import csv
import os
import statistics
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
import numpy as np


DPI = 3200
UM_PER_COUNT = 25400.0 / DPI  # 7.9375 um/count
FIG_DIR = os.path.join("paper", "figures")


def set_ieee_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.9,
        "grid.alpha": 0.32,
        "grid.linestyle": "--",
        "savefig.dpi": 300,
        "figure.dpi": 300,
    })


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Figure 1: Three-Regime Deadband & Stick-Slip Quantization Trap
# ---------------------------------------------------------------------------
def plot_fig1_deadzone_3regime() -> str:
    dz_rows = _read_csv("experiments/deadzone_gain_fit.csv")
    ss_rows = _read_csv("experiments/step_scan.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.35), constrained_layout=True)

    # Panel (a): 3-Regime Deadband (pos & neg)
    for d_sign, color, label in [("pos", "#1f77b4", "Positive (+Δu_1)"), ("neg", "#d62728", "Negative (-Δu_1)")]:
        sub = [r for r in dz_rows if r["dir"] == d_sign]
        amps = sorted(set(float(r["amp_us"]) for r in sub))
        xs_all, ys_all, meds = [], [], []
        sgn = 1.0 if d_sign == "pos" else -1.0
        for a in amps:
            vals = [float(r["dx_cnt"]) for r in sub if float(r["amp_us"]) == a]
            for v in vals:
                xs_all.append(sgn * a)
                ys_all.append(v)
            meds.append((sgn * a, statistics.median(vals)))
        meds.sort(key=lambda t: t[0])
        ax1.scatter(xs_all, ys_all, color=color, alpha=0.35, s=24, edgecolors="none")
        ax1.plot([t[0] for t in meds], [t[1] for t in meds], "-o", color=color, lw=2.0, ms=5.0, label=label)

    # Shaded 3 regimes
    ax1.axvspan(-20, 20, color="#95a5a6", alpha=0.14)
    ax1.axvspan(20, 55, color="#f39c12", alpha=0.12)
    ax1.axvspan(-55, -20, color="#f39c12", alpha=0.12)
    ax1.axvspan(55, 86, color="#2ecc71", alpha=0.10)
    ax1.axvspan(-86, -55, color="#2ecc71", alpha=0.10)
    # Sandwich preload band (+10..15 and -10..-15 us)
    ax1.axvspan(9.5, 15.0, color="#8e44ad", alpha=0.22, label="Sandwich Pre-load Window (10–15 μs)")
    ax1.axvspan(-15.0, -9.5, color="#8e44ad", alpha=0.22)

    ax1.axhline(0, color="#333333", lw=0.7)
    ax1.axvline(0, color="#333333", lw=0.7)
    ax1.text(0, 58, "I. Deadband Plateau\n(0–20 μs)\nComparator DB: 4–10 μs", ha="center", va="center",
             fontsize=7.4, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#7f8c8d", alpha=0.92))
    ax1.text(38, -82, "II. Stribeck\nStick-Slip\n(20–55 μs)", ha="center", va="bottom",
             fontsize=7.4, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#e67e22", alpha=0.92))
    ax1.text(69, -118, "III. Coulomb Linear\n+2.10 cnt/μs (pos)\n+2.05 cnt/μs (neg)", ha="center", va="bottom",
             fontsize=7.3, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#27ae60", alpha=0.92))

    ax1.set_xlim(-86, 86)
    ax1.set_ylim(-130, 152)
    ax1.set_xlabel("Commanded Step Amplitude Δu₁ (μs)")
    ax1.set_ylabel("Measured X Displacement Δx (counts, 7.94 μm/cnt)")
    ax1.set_title("(a) Three-Regime Actuator Deadband & Sandwich Pre-load")
    ax1.grid(True)
    ax1.legend(loc="upper left", framealpha=0.94, fontsize=7.6)

    # Panel (b): Discrete Stick-Slip Plateaus from step_scan.csv (run 1 representative per step size)
    styles = [("1", "#1f77b4", "-", "1 μs step (med plateau 12.0 μs)"),
              ("2", "#ff7f0e", "--", "2 μs step (jump 7.75±5.68 cnt)"),
              ("4", "#2ca02c", "-.", "4 μs step (jump 8.50±6.97 cnt)")]
    for step_str, col, ls, lbl in styles:
        sub = [r for r in ss_rows if r["step_us"] == step_str and r["run_id"] == "1"]
        x_off = [float(r["u_offset_us"]) for r in sub]
        x0 = float(sub[0]["pos_x_cnt"])
        y_disp = [float(r["pos_x_cnt"]) - x0 for r in sub]
        ax2.step(x_off, y_disp, where="post", color=col, linestyle=ls, lw=1.9, label=lbl)

    ax2.annotate("Stiction Plateau\n(~24 μs lock before\n12-cnt slip jump)",
                 xy=(24, 12), xytext=(6, 26),
                 arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2),
                 fontsize=7.8, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#1f77b4", alpha=0.92))
    ax2.set_xlim(0, 61)
    ax2.set_ylim(-2, 52)
    ax2.set_xlabel("Quasi-Static Offset Δu₁ (μs)")
    ax2.set_ylabel("Net Displacement Δx (counts)")
    ax2.set_title("(b) Quasi-Static Staircase Stick-Slip Plateaus")
    ax2.grid(True)
    ax2.legend(loc="upper left", framealpha=0.92)

    out_path = os.path.join(FIG_DIR, "fig1_deadzone_3regime.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 2: Y-Crank Shortening & 2x2 Measured Jacobian Sensitivity Reduction
# ---------------------------------------------------------------------------
def plot_fig2_short_arm_jacobian() -> str:
    long_rows = _read_csv("experiments/real_jacobian_2x2.csv")
    short_rows = _read_csv("experiments/real_jacobian_2x2_short_arm.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.35), constrained_layout=True)

    # Panel (a): 2D displacement scatter for u1 and u2 channels (Long vs Short arm)
    for rows, ax_name, s_label, col, marker, alpha in [
        (long_rows, "u1", "Long Y-Arm: u1", "#aec7e8", "s", 0.55),
        (long_rows, "u2", "Long Y-Arm: u2 (4.332 cnt/μs)", "#ff9896", "^", 0.65),
        (short_rows, "u1", "Short Y-Arm: u1 (2.039 cnt/μs)", "#1f77b4", "o", 0.90),
        (short_rows, "u2", "Short Y-Arm: u2 (2.182 cnt/μs)", "#d62728", "D", 0.90),
    ]:
        sub = [r for r in rows if r["axis"] == ax_name]
        xs = [float(r["dx_cnt"]) for r in sub]
        ys = [float(r["dy_cnt"]) for r in sub]
        ax1.scatter(xs, ys, c=col, marker=marker, s=36, alpha=alpha, label=s_label, edgecolors="k", linewidths=0.4)

    # Plot Jacobian column arrows at Δu = +20 us scale
    ax1.annotate("", xy=(2.022 * 20, -0.260 * 20), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2.2))
    ax1.annotate("", xy=(-1.694 * 20, -1.376 * 20), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=2.2))
    ax1.annotate("", xy=(-1.892 * 20, -3.897 * 20), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1.6, ls="--"))

    ax1.axhline(0, color="#444", lw=0.7)
    ax1.axvline(0, color="#444", lw=0.7)
    ax1.set_xlim(-60, 62)
    ax1.set_ylim(-102, 88)
    ax1.set_xlabel("Measured X Displacement Δx (counts)")
    ax1.set_ylabel("Measured Y Displacement Δy (counts)")
    ax1.set_title("(a) Coupled 2×2 Displacement Manifolds: Long vs. Short Y-Arm")
    ax1.grid(True)
    ax1.legend(loc="upper left", fontsize=7.5, framealpha=0.94)

    # Panel (b): Y-Channel Sensitivity & Deadband Comparison (|du2| vs |disp|)
    def _group_mean(rows, axis_name):
        sub = [r for r in rows if r["axis"] == axis_name]
        dus = sorted(set(float(r["du_us"]) for r in sub))
        pts = []
        for du in dus:
            vals = [float(r["disp_cnt"]) for r in sub if float(r["du_us"]) == du]
            pts.append((du, float(np.mean(vals)), float(np.std(vals))))
        return pts

    l_u2 = _group_mean(long_rows, "u2")
    s_u2 = _group_mean(short_rows, "u2")
    s_u1 = _group_mean(short_rows, "u1")

    ax2.errorbar([p[0] for p in l_u2], [p[1] for p in l_u2], yerr=[p[2] for p in l_u2],
                 fmt="--^", color="#ff7f0e", lw=1.8, capsize=3, label="Long Arm u2 (4.332 cnt/μs, d=5.90 μs)")
    ax2.errorbar([p[0] for p in s_u2], [p[1] for p in s_u2], yerr=[p[2] for p in s_u2],
                 fmt="-D", color="#d62728", lw=2.1, capsize=3, label="Short Arm u2 (2.182 cnt/μs, d=4.00 μs)")
    ax2.errorbar([p[0] for p in s_u1], [p[1] for p in s_u1], yerr=[p[2] for p in s_u1],
                 fmt="-o", color="#1f77b4", lw=1.9, capsize=3, label="Short Arm u1 (2.039 cnt/μs, d=9.22 μs)")

    ax2.annotate("−49.6% Gain Compression\n1.0 μs PWM: 4.33 → 2.18 cnt\n0.1 μs PWM: 0.22 cnt (1.73 μm)",
                 xy=(25, 45.2), xytext=(12.2, 62),
                 arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.3),
                 fontsize=7.8, bbox=dict(boxstyle="round,pad=0.3", fc="#fdfefe", ec="#2c3e50", alpha=0.95))

    ax2.set_xlim(10, 34)
    ax2.set_ylim(-2, 86)
    ax2.set_xlabel("Differential Pulse Amplitude |Δu| (μs)")
    ax2.set_ylabel("Euclidean Step Magnitude ||Δx||₂ (counts)")
    ax2.set_title("(b) Resolution Compression & Deadband Intercepts")
    ax2.grid(True)
    ax2.legend(loc="lower right", fontsize=7.6, framealpha=0.92)

    out_path = os.path.join(FIG_DIR, "fig2_short_arm_jacobian.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 3: Architectural Ablation Evolution & Controlled 5-Shot Component Study
# ---------------------------------------------------------------------------
def plot_fig3_ablation_evolution() -> str:
    evo_rows = _read_csv("experiments/ablation_evolution.csv")
    short_rows = _read_csv("experiments/short_ablation_trials.csv")
    g1_rows = _read_csv("experiments/gate1_30_20_eval.csv")
    g2_rows = _read_csv("experiments/gate2_6dir_eval.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.55), constrained_layout=True)

    # Panel (a): Historical 4-Stage Engineering Evolution (exclude PID outlier from bar height or annotate)
    # Plot stages ①..④ clearly, and annotate PID (101.04 cnt, 1244.9 ms) as upper reference
    stages = [r for r in evo_rows if r["stage_id"] != "S0_PID"]
    x_idx = np.arange(len(stages))
    errs = [float(r["err_cnt"]) for r in stages]
    t_sets = [float(r["t_set_ms"]) for r in stages]
    t_bals = [float(r["t_bal_ms"]) for r in stages]
    labels = [
        "① 1.0 μs Timer\n(Software PWM)",
        "② 0.1 μs TCXO\n(Single-Phase)",
        "③ + Atomic Packet\n& 10–12 μs Preload",
        "④ + In-Flight Cutoff\n& Stage-2 Isolation",
    ]
    colors = ["#95a5a6", "#e67e22", "#3498db", "#27ae60"]

    bars = ax1.bar(x_idx - 0.16, errs, width=0.32, color=colors, edgecolor="#2c3e50", linewidth=0.8,
                   label="Positioning Error (counts)")
    for b, e in zip(bars, errs):
        ax1.text(b.get_x() + b.get_width() / 2, e + 1.0,
                 f"{e:.2f} cnt\n({e*UM_PER_COUNT:.1f} μm)", ha="center", va="bottom", fontsize=7.6, fontweight="bold")

    ax1_r = ax1.twinx()
    l1 = ax1_r.plot(x_idx + 0.20, t_sets, "-s", color="#8e44ad", lw=2.0, ms=6.0, label="Total Settling t_set (ms)")
    l2 = ax1_r.plot(x_idx + 0.20, t_bals, "--o", color="#c0392b", lw=1.6, ms=5.0, label="Ballistic Time t_bal (ms)")
    for idx_i, (xi, ts, tb) in enumerate(zip(x_idx + 0.20, t_sets, t_bals)):
        x_txt = xi + (0.14 if idx_i == 0 else 0.04)
        ax1_r.text(x_txt, ts + 11.5, f"{ts:.1f} ms", color="#8e44ad", fontsize=7.4, fontweight="bold", ha="center")

    ax1_r.axhline(100.0, color="#8e44ad", lw=1.0, ls=":", alpha=0.75)
    ax1_r.text(0.05, 104.5, "100 ms Target", color="#8e44ad", fontsize=7.4, style="italic")

    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(labels, fontsize=8.0)
    ax1.set_ylim(0, 55)
    ax1_r.set_ylim(0, 290)
    ax1.set_ylabel("Mean Positioning Error (counts)")
    ax1_r.set_ylabel("Response Latency (ms)", color="#8e44ad")
    ax1.set_title("(a) 4-Stage Architecture Evolution (Ref: PID = 101.04 cnt, 1245 ms)")
    ax1.grid(True, axis="y")

    h1, lbl1 = ax1.get_legend_handles_labels()
    h2, lbl2 = ax1_r.get_legend_handles_labels()
    ax1.legend(h1 + h2, lbl1 + lbl2, loc="upper right", fontsize=7.4, framealpha=0.92)

    # Panel (b): Same-Rig Controlled Single-Factor Ablation (5 shots/dir on 45ea2aa rig)
    # Groups: Stage-1 Only vs w/o Sandwich Preload vs Full Proposed (45ea2aa)
    def _stats_for(cond_id: str, tx: float) -> Tuple[float, float, float]:
        es = sorted(float(r["err_counts"]) for r in short_rows
                    if r["condition_id"] == cond_id and float(r["target_x"]) == tx)
        return float(np.mean(es)), float(np.mean(es[:3])), float(np.median(es))

    s1_3020_mean, s1_3020_t3, _ = _stats_for("STAGE1_ONLY", 30.0)
    np_3020_mean, np_3020_t3, _ = _stats_for("NO_PRELOAD", 30.0)
    s1_035_mean, s1_035_t3, _ = _stats_for("STAGE1_ONLY", 0.0)
    np_035_mean, np_035_t3, _ = _stats_for("NO_PRELOAD", 0.0)

    # Full 45ea2aa stats
    g1_last5 = sorted(float(r["err_counts"]) for r in g1_rows[-5:])
    full_3020_trim4 = float(np.mean(g1_last5[:4]))
    full_3020_raw5 = float(np.mean(g1_last5))

    g2_035 = sorted(float(r["err_counts"]) for r in g2_rows if float(r["dir_x"]) == 0.0 and float(r["dir_y"]) == 35.0)
    full_035_trim3 = float(np.mean(g2_035[:3]))
    full_035_raw5 = float(np.mean(g2_035))

    cats = [
        "Stage-1 Cutoff Only\n(w/o Stage-2 Trim)",
        "w/o Sandwich Preload\n(Δu_pre = 0)",
        "Proposed Full System\n(Co-Designed)",
    ]
    xc = np.arange(len(cats))
    w = 0.34

    vals_3020_trim = [s1_3020_t3, np_3020_t3, full_3020_trim4]
    vals_3020_raw = [s1_3020_mean, np_3020_mean, full_3020_raw5]
    vals_035_trim = [s1_035_t3, np_035_t3, full_035_trim3]
    vals_035_raw = [s1_035_mean, np_035_mean, full_035_raw5]

    b1 = ax2.bar(xc - w / 2, vals_3020_trim, width=w, color="#2980b9", edgecolor="#1b4f72",
                 label="(+30, +20) Trimmed Mean (Solid) / 5-Shot Mean (○)")
    b2 = ax2.bar(xc + w / 2, vals_035_trim, width=w, color="#16a085", edgecolor="#0e6251",
                 label="(+0, +35) Trimmed Mean (Solid) / 5-Shot Mean (◇)")

    ax2.scatter(xc - w / 2, vals_3020_raw, color="white", edgecolors="#1b4f72", marker="o", s=38, zorder=5, linewidths=1.3)
    ax2.scatter(xc + w / 2, vals_035_raw, color="white", edgecolors="#0e6251", marker="D", s=34, zorder=5, linewidths=1.3)

    for bar, v_t, v_r in zip(b1, vals_3020_trim, vals_3020_raw):
        y_lbl = max(v_t, v_r) + (1.20 if max(v_t, v_r) < 3.0 else 0.35)
        ax2.text(bar.get_x() + bar.get_width() / 2, y_lbl,
                 f"{v_t:.2f}\n({v_r:.2f})", ha="center", va="bottom", fontsize=7.5, color="#1b4f72", fontweight="bold")
    for bar, v_t, v_r in zip(b2, vals_035_trim, vals_035_raw):
        y_lbl = max(v_t, v_r) + (1.20 if max(v_t, v_r) < 3.0 else 0.35)
        ax2.text(bar.get_x() + bar.get_width() / 2, y_lbl,
                 f"{v_t:.2f}\n({v_r:.2f})", ha="center", va="bottom", fontsize=7.5, color="#0e6251", fontweight="bold")

    ax2.axhline(3.2, color="#e74c3c", ls="--", lw=1.1, label="Tolerance Threshold (3.2–3.5 cnt)")
    ax2.set_xticks(xc)
    ax2.set_xticklabels(cats, fontsize=8.1)
    ax2.set_ylim(0, 14.2)
    ax2.set_ylabel("Residual Positioning Error (counts)")
    ax2.set_title("(b) Controlled Single-Factor Ablation Study")
    ax2.grid(True, axis="y")
    ax2.legend(loc="upper right", fontsize=7.4, framealpha=0.92)

    out_path = os.path.join(FIG_DIR, "fig3_ablation_evolution.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 4: Frozen 45ea2aa Gate 1 Convergence & Gate 2a 6-Direction Scatter
# ---------------------------------------------------------------------------
def plot_fig4_gate1_gate2a_results() -> str:
    g1_rows = _read_csv("experiments/gate1_30_20_eval.csv")
    g2_rows = _read_csv("experiments/gate2_6dir_eval.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.55), constrained_layout=True)

    # Panel (a): Gate 1 (30, 20) 12-Shot Trajectory (Error + Ballistic/Settled Latency)
    shots = [int(r["shot"]) for r in g1_rows]
    errs = [float(r["err_counts"]) for r in g1_rows]
    bals = [float(r["ballistic_ms"]) for r in g1_rows]
    sets = [float(r["settled_ms"]) for r in g1_rows]

    ax1.axvspan(7.5, 12.5, color="#2ecc71", alpha=0.12, label="Protocol I Eval Window (Shots 8–12)")
    ax1.plot(shots, errs, "-o", color="#1f77b4", lw=2.1, ms=5.5, label="Final Error (counts)")
    ax1.axhline(3.2, color="#e74c3c", ls="--", lw=1.2, label="Protocol I Limit (3.2 cnt / 25.4 μm)")
    ax1.axhline(1.83, color="#27ae60", ls="-.", lw=1.2, label="Last-5 Trim-4 Mean = 1.83 cnt (14.5 μm)")

    ax1_r = ax1.twinx()
    ax1_r.plot(shots, bals, ":^", color="#e67e22", lw=1.5, ms=4.6, alpha=0.85, label="t_bal (mean 84.4 ms)")
    ax1_r.plot(shots, sets, "--s", color="#8e44ad", lw=1.5, ms=4.5, alpha=0.85, label="t_set (mean 116.1 ms)")
    ax1_r.set_ylim(45, 165)
    ax1_r.set_ylabel("Ballistic / Settling Time (ms)", color="#8e44ad")

    ax1.set_xlim(0.5, 12.5)
    ax1.set_xticks(shots)
    ax1.set_ylim(0, 7.6)
    ax1.set_xlabel("Cold-Start Shot Index at Target (30, 20)")
    ax1.set_ylabel("Final Radial Error (counts)")
    ax1.set_title("(a) Protocol I Cold-Start Convergence at (30, 20) [PASS]")
    ax1.grid(True)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7.0, framealpha=0.94, ncol=1)

    # Panel (b): Gate 2a 6-Direction Endpoint Scatter Relative to Target (0, 0) + Per-Direction Box/Mean
    by_dir: Dict[Tuple[float, float], List[Dict[str, str]]] = {}
    for r in g2_rows:
        by_dir.setdefault((float(r["dir_x"]), float(r["dir_y"])), []).append(r)

    palette = {
        (35.0, 0.0): ("#1f77b4", "(+35, 0): 3.22 cnt [PASS]"),
        (-35.0, 0.0): ("#ff7f0e", "(-35, 0): 2.19 cnt [PASS]"),
        (0.0, 35.0): ("#2ca02c", "(0, +35): 2.02 cnt [PASS]"),
        (0.0, -35.0): ("#9467bd", "(0, -35): 0.80 cnt [PASS]"),
        (30.0, 25.0): ("#d62728", "(+30, +25): 3.61 cnt [FAIL]"),
        (-30.0, -25.0): ("#8c564b", "(-30, -25): 2.48 cnt [PASS]"),
    }

    for key, rs in by_dir.items():
        col, lbl = palette[key]
        ex, ey = [], []
        for r in rs:
            fx, fy = [float(v.strip("() ")) for v in r["final"].split(",")]
            ex.append(fx - key[0])
            ey.append(fy - key[1])
        ax2.scatter(ex, ey, color=col, s=46, alpha=0.82, edgecolors="k", linewidths=0.45, label=lbl, zorder=4)

    # Tolerance and physical clearance rings (centered at relative target error = (0,0))
    c_239 = Circle((0, 0), 2.39, fill=False, edgecolor="#27ae60", lw=1.8, ls="-", label="6-Dir Trim-3 Mean = 2.39 cnt (19.0 μm)")
    c_350 = Circle((0, 0), 3.50, fill=False, edgecolor="#e74c3c", lw=1.3, ls="--", label="Per-Dir Tolerance (3.50 cnt / 27.8 μm)")
    c_mech = Circle((0, 0), 6.30, fill=False, edgecolor="#7f8c8d", lw=1.0, ls=":", label="~50 μm Half-Clearance Envelope")
    ax2.add_patch(c_239)
    ax2.add_patch(c_350)
    ax2.add_patch(c_mech)

    ax2.axhline(0, color="#444", lw=0.7)
    ax2.axvline(0, color="#444", lw=0.7)
    ax2.set_xlim(-8.8, 8.8)
    ax2.set_ylim(-7.4, 12.0)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("Relative Final X Error Δe_x (counts)")
    ax2.set_ylabel("Relative Final Y Error Δe_y (counts)")
    ax2.set_title("(b) Protocol II Residuals (t_bal=83.6 ms, t_set=98.2 ms)")
    ax2.grid(True)
    ax2.legend(loc="upper center", fontsize=6.4, framealpha=0.94, ncol=2)

    out_path = os.path.join(FIG_DIR, "fig4_gate1_gate2a_results.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    set_ieee_style()
    outs = [
        plot_fig1_deadzone_3regime(),
        plot_fig2_short_arm_jacobian(),
        plot_fig3_ablation_evolution(),
        plot_fig4_gate1_gate2a_results(),
    ]
    for p in outs:
        print(f"[GENERATED 300DPI] {p}")


if __name__ == "__main__":
    main()
