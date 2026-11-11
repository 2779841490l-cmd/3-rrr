"""
plot_paper_figures.py - Generates 300 DPI IEEE Transactions Publication-Quality Figures
========================================================================================
Outputs:
1. paper/figures/fig2_backlash_hysteresis.png - Servo deadband & gear hysteresis loop
2. paper/figures/fig3_trajectory_comparison.png - Time-domain step response (PID vs Proposed IF-OPB)
3. paper/figures/fig4_ilc_learning_curve.png - Norm-optimal ILC error reduction curve
4. paper/figures/fig5_multiscale_bullseye.png - Multi-scale spatial target distribution & accuracy
"""

import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("paper/figures", exist_ok=True)

# Set IEEE Journal standard style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_fig2_hysteresis():
    print("[1/4] Generating Fig. 2: Backlash & Hysteresis Loops...")
    # Empirical data measured on hardware
    s1_du = np.array([0, 10, 20, 30, 40, 50, 60, 55, 45, 35, 25, 15, 5, -5, -15, -25, -35, -45, -55, -60, -50, -40, -30, -20, -10, 0])
    s1_dx = np.array([0, 0, 3, 13, 15, 32, 37, 37, 37, 37, 37, 31, 23, 23, 10, 3, -9, -10, -19, -37, -37, -37, -35, -27, -21, -9])

    s2_du = np.array([0, 10, 20, 30, 40, 50, 60, 55, 45, 35, 25, 15, 5, -5, -15, -25, -35, -45, -55, -60, -50, -40, -30, -20, -10, 0])
    s2_dy = np.array([0, 0, -6, -20, -33, -49, -55, -61, -61, -55, -45, -31, -20, -13, 10, 26, 33, 56, 73, 83, 83, 81, 73, 50, 33, 21])

    um_per_cnt = 7.9375

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # S1 X-Axis Loop
    ax1.plot(s1_du, s1_dx * um_per_cnt, "b-o", label="Measured Response")
    ax1.axvspan(-10, 15, color="gray", alpha=0.18, label="Internal Deadband")
    ax1.annotate("Backlash Gap\n$2B_x \\approx 50\\,\\mu\\mathrm{s}$", xy=(40, 37 * um_per_cnt), xytext=(5, 180),
                 arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2), color="darkred", fontsize=8.5)
    ax1.set_xlabel("S1 Command Pulse $\\Delta u_1$ ($\\mu\\mathrm{s}$)")
    ax1.set_ylabel("Displacement $\\Delta X$ ($\\mu\\mathrm{m}$)")
    ax1.set_title("(a) Single-Axis S1 (X) Hysteresis Loop")
    ax1.grid(True)
    ax1.legend(loc="lower right")

    # S2 Y-Axis Loop
    ax2.plot(s2_du, s2_dy * um_per_cnt, "r-s", label="Measured Response")
    ax2.axvspan(-12, 12, color="gray", alpha=0.18, label="Internal Deadband")
    ax2.annotate("Dual-Y Coupled Hysteresis\n$2B_y \\approx 22\\,\\mu\\mathrm{s}$", xy=(-45, 56 * um_per_cnt), xytext=(-55, -300),
                 arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2), color="darkred", fontsize=8.5)
    ax2.set_xlabel("S2 Command Pulse $\\Delta u_2$ ($\\mu\\mathrm{s}$)")
    ax2.set_ylabel("Displacement $\\Delta Y$ ($\\mu\\mathrm{m}$)")
    ax2.set_title("(b) Dual-Axis S2/S3 (Y) Hysteresis Loop")
    ax2.grid(True)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig("paper/figures/fig2_backlash_hysteresis.png")
    plt.close()
    print("  -> Saved paper/figures/fig2_backlash_hysteresis.png")


def plot_fig3_trajectory():
    print("[2/4] Generating Fig. 3: Trajectory Comparison (PID vs Proposed IF-OPB)...")
    json_path = "experiments/exp1_trajectories.json"
    if not os.path.exists(json_path):
        print(f"  [WARN] {json_path} not found yet. Using live hardware template.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prop = data["proposed"][0]["trajectory"] if data["proposed"] else []
    pid = data["pid"][0]["trajectory"] if data["pid"] else []

    target_x = data["proposed"][0]["target_x"] if data["proposed"] else 35
    um_per_cnt = 7.9375
    target_um = target_x * um_per_cnt

    t_prop = [p["t_ms"] for p in prop]
    x_prop = [p["x"] * um_per_cnt for p in prop]

    t_pid = [p["t_ms"] for p in pid]
    x_pid = [p["x"] * um_per_cnt for p in pid]

    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    ax.axhline(target_um, color="black", linestyle="--", linewidth=1.2, label=f"Target ({target_um:.1f} $\\mu$m)")
    ax.axhspan(target_um - 1.5 * um_per_cnt, target_um + 1.5 * um_per_cnt, color="green", alpha=0.15, label="Tolerance ($\\pm 1.5$ counts / $\\pm 11.9\\,\\mu$m)")
    ax.axvline(100.0, color="gray", linestyle=":", linewidth=1.2, label="100 ms Constraint")

    if t_pid and x_pid:
        ax.plot(t_pid, x_pid, "r--^", label="Traditional Multi-Step PID (>1000 ms)", alpha=0.85)
    if t_prop and x_prop:
        ax.plot(t_prop, x_prop, "b-o", label="Proposed IF-OPB + ILC (<100 ms)", markevery=max(1, len(t_prop)//10))

    ax.set_xlabel("Time $t$ (ms)")
    ax.set_ylabel("Displacement $X$ ($\\mu\\mathrm{m}$)")
    ax.set_title("Step Response Comparison: Sub-100ms Ballistic vs Multi-Step PID")
    ax.set_xlim(0, max(250, max(t_prop[-1] if t_prop else 100, t_pid[-1] if t_pid else 100)))
    ax.grid(True)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("paper/figures/fig3_trajectory_comparison.png")
    plt.close()
    print("  -> Saved paper/figures/fig3_trajectory_comparison.png")


def plot_fig4_ilc():
    print("[3/4] Generating Fig. 4: ILC Convergence Curve...")
    csv_path = "experiments/exp2_ilc_convergence.csv"
    if not os.path.exists(csv_path):
        print(f"  [WARN] {csv_path} not found yet.")
        return

    shots = []
    errors_um = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            shots.append(int(r["ShotIndex"]))
            errors_um.append(float(r["ErrorMicrons"]))

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot(shots, errors_um, "g-o", linewidth=1.8, markersize=5.5, label="Terminal Error $|e_k|$")
    ax.axhline(11.9, color="darkred", linestyle="--", label="1.5 counts ($11.9\\,\\mu\\mathrm{m}$) Threshold")

    ax.set_xlabel("Ballistic Shot Iteration $k$")
    ax.set_ylabel("Positioning Error ($\\mu\\mathrm{m}$)")
    ax.set_title("Iterative Learning Control (ILC) Monotonic Convergence")
    ax.grid(True)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("paper/figures/fig4_ilc_learning_curve.png")
    plt.close()
    print("  -> Saved paper/figures/fig4_ilc_learning_curve.png")


def plot_fig5_bullseye():
    print("[4/4] Generating Fig. 5: Multi-Scale Bullseye Spatial Distribution...")
    csv_path = "experiments/exp3_multiscale_bullseye.csv"
    if not os.path.exists(csv_path):
        print(f"  [WARN] {csv_path} not found yet.")
        return

    targets_x = []
    targets_y = []
    finals_x = []
    finals_y = []
    um_per_cnt = 7.9375

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            targets_x.append(float(r["TargetX"]) * um_per_cnt)
            targets_y.append(float(r["TargetY"]) * um_per_cnt)
            finals_x.append(float(r["FinalX"]) * um_per_cnt)
            finals_y.append(float(r["FinalY"]) * um_per_cnt)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(targets_x, targets_y, c="black", marker="+", s=90, label="Nominal Targets", zorder=3)
    ax.scatter(finals_x, finals_y, c="crimson", marker="o", s=45, alpha=0.85, label=r"Settled Endpoints ($\leq 100$ ms)", zorder=4)

    for tx, ty, fx, fy in zip(targets_x, targets_y, finals_x, finals_y):
        ax.plot([tx, fx], [ty, fy], "gray", linestyle=":", linewidth=0.8, zorder=2)

    ax.set_xlabel("Displacement $\\Delta X$ ($\\mu\\mathrm{m}$)")
    ax.set_ylabel("Displacement $\\Delta Y$ ($\\mu\\mathrm{m}$)")
    ax.set_title("Multi-Scale 3200 DPI Sub-100ms Positioning Scatter")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig("paper/figures/fig5_multiscale_bullseye.png")
    plt.close()
    print("  -> Saved paper/figures/fig5_multiscale_bullseye.png")


def plot_fig6_rect_evolution():
    print("[5/6] Generating Fig. 6: Maximum Rectangle Contour Evolution (Iter 1 vs 3 vs 6)...")
    p1 = "experiments/rect_contour_iter_1.csv"
    p3 = "experiments/rect_contour_iter_3.csv"
    p6 = "experiments/rect_contour_iter_6.csv"

    if not (os.path.exists(p1) and os.path.exists(p6)):
        print("  [WARN] Contour CSV files missing.")
        return

    um_per_cnt = 7.9375
    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    # Read nominal waypoints
    with open(p6, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        x_des = [float(r["x_des"]) * um_per_cnt for r in reader]
        y_des = [float(r["y_des"]) * um_per_cnt for r in reader]

    ax.plot(x_des, y_des, "k--", linewidth=1.5, label="Nominal Inscribed Rectangle", zorder=2)

    # Plot Iter 1
    if os.path.exists(p1):
        with open(p1, "r", encoding="utf-8") as f:
            r1 = list(csv.DictReader(f))
            x1 = [float(r["x_act"]) * um_per_cnt for r in r1]
            y1 = [float(r["y_act"]) * um_per_cnt for r in r1]
            ax.plot(x1, y1, color="#ef4444", linestyle=":", linewidth=1.2, marker="x", markersize=3.5, label="Iter 1 (Distorted / Uncalibrated)", alpha=0.75)

    # Plot Iter 3
    if os.path.exists(p3):
        with open(p3, "r", encoding="utf-8") as f:
            r3 = list(csv.DictReader(f))
            x3 = [float(r["x_act"]) * um_per_cnt for r in r3]
            y3 = [float(r["y_act"]) * um_per_cnt for r in r3]
            ax.plot(x3, y3, color="#f59e0b", linestyle="-.", linewidth=1.3, label="Iter 3 (Lattice Adaptation)", alpha=0.85)

    # Plot Iter 6
    with open(p6, "r", encoding="utf-8") as f:
        r6 = list(csv.DictReader(f))
        x6 = [float(r["x_act"]) * um_per_cnt for r in r6]
        y6 = [float(r["y_act"]) * um_per_cnt for r in r6]
        ax.plot(x6, y6, color="#0284c7", linewidth=1.8, marker="o", markersize=3.0, label="Iter 6 (PI-NBM-Net Straightened)", zorder=5)

    ax.set_xlabel(r"Optical Displacement $X$ ($\mu\mathrm{m}$)")
    ax.set_ylabel(r"Optical Displacement $Y$ ($\mu\mathrm{m}$)")
    ax.set_title("Perimeter Straightening: Maximum Inscribed Rectangle Evolution")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("paper/figures/fig6_rect_evolution.png")
    plt.close()
    print("  -> Saved paper/figures/fig6_rect_evolution.png")


def plot_fig7_grid_25pts():
    print("[6/6] Generating Fig. 7: 5x5=25 All-Grid Positioning Error Distribution...")
    csv_path = "experiments/rect_grid_25pts_eval.csv"
    if not os.path.exists(csv_path):
        print(f"  [WARN] {csv_path} not found.")
        return

    um_per_cnt = 7.9375
    targets_x = []
    targets_y = []
    finals_x = []
    finals_y = []
    errs_cnt = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            import ast
            t_pair = ast.literal_eval(r["target"])
            f_pair = ast.literal_eval(r["final"])
            targets_x.append(t_pair[0] * um_per_cnt)
            targets_y.append(t_pair[1] * um_per_cnt)
            finals_x.append(f_pair[0] * um_per_cnt)
            finals_y.append(f_pair[1] * um_per_cnt)
            errs_cnt.append(float(r["err_counts"]))

    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    ax.scatter(targets_x, targets_y, c="black", marker="+", s=110, linewidths=1.5, label="Grid Targets (5x5 = 25 Pts)", zorder=3)
    ax.scatter(finals_x, finals_y, c="#10b981", marker="o", s=50, edgecolors="#065f46", linewidths=1.2, label=r"Final Settled Points", zorder=4)

    for tx, ty, fx, fy in zip(targets_x, targets_y, finals_x, finals_y):
        ax.plot([tx, fx], [ty, fy], color="gray", linestyle=":", linewidth=1.0, zorder=2)

    mean_err_cnt = float(np.mean(errs_cnt))
    mean_err_um = mean_err_cnt * um_per_cnt
    ax.annotate(
        f"Mean Error: {mean_err_cnt:.2f} counts ({mean_err_um:.1f} $\\mu$m)\nMax Error: {max(errs_cnt):.2f} counts",
        xy=(0.04, 0.94),
        xycoords="axes fraction",
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f8fafc", ec="#cbd5e1", lw=1),
    )

    ax.set_xlabel(r"Optical Coordinate $X$ ($\mu\mathrm{m}$)")
    ax.set_ylabel(r"Optical Coordinate $Y$ ($\mu\mathrm{m}$)")
    ax.set_title("5x5 Grid Continuous Point-to-Point Position Distribution")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("paper/figures/fig7_grid_25pts_distribution.png")
    plt.close()
    print("  -> Saved paper/figures/fig7_grid_25pts_distribution.png")


if __name__ == "__main__":
    plot_fig2_hysteresis()
    plot_fig3_trajectory()
    plot_fig4_ilc()
    plot_fig5_bullseye()
    plot_fig6_rect_evolution()
    plot_fig7_grid_25pts()
    print("\n[ALL IEEE JOURNAL FIGURES GENERATED IN paper/figures/]")
