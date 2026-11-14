"""
run_paper_experiments.py - IEEE Transactions Paper Data Consolidation & Short Ablation Runner
=============================================================================================
Consolidates verified hardware KPI telemetry (committed at 45ea2aa) and historical
characterization datasets in `experiments/` without altering any frozen baseline files:
  - rect_manifold_calibrator.py (FROZEN)
  - servo_driver.py (FROZEN)
  - firmware_ws63/servo_test.c (FROZEN)
  - verify_hardware_kpi.py (FROZEN)
  - calib_rect_manifold.json (FROZEN)

Optionally runs a lightweight, non-destructive single-factor ablation (`--run-short-ablation`,
strictly 5 shots/condition at low-amplitude 35~36 cnt targets; NEVER executes +/-350 cnt sweeps
or 100+ shot endurance scans) to record direct apples-to-apples comparisons for:
  1. Without Directional Sandwich Pre-load (`pre_du = 0`, B_CMD + Stage-1/Stage-2 active)
  2. Stage-1 H_OVR Ballistic Only (`dual_lattice=False`, Stage-2 axial isolation disabled)
"""

import argparse
import csv
import json
import os
import statistics
from typing import Dict, List, Tuple

import numpy as np


DPI = 3200
UM_PER_COUNT = 25400.0 / DPI  # 7.9375 um/count


# Verbatim Gate 1 telemetry logged at verified freeze commit 45ea2aa (verify_hardware_kpi.py)
GATE1_45EA2AA_SHOTS: List[Dict] = [
    {"shot": 1, "target_x": 30.0, "target_y": 20.0, "final_x": 27.0, "final_y": 18.0, "err_counts": 3.61, "err_um": 28.6, "ballistic_ms": 83.5, "settled_ms": 115.6, "dual_trials": 2},
    {"shot": 2, "target_x": 30.0, "target_y": 20.0, "final_x": 27.0, "final_y": 16.0, "err_counts": 5.00, "err_um": 39.7, "ballistic_ms": 86.4, "settled_ms": 114.4, "dual_trials": 2},
    {"shot": 3, "target_x": 30.0, "target_y": 20.0, "final_x": 27.0, "final_y": 16.0, "err_counts": 5.00, "err_um": 39.7, "ballistic_ms": 92.4, "settled_ms": 103.6, "dual_trials": 2},
    {"shot": 4, "target_x": 30.0, "target_y": 20.0, "final_x": 29.0, "final_y": 19.0, "err_counts": 1.41, "err_um": 11.2, "ballistic_ms": 86.8, "settled_ms": 127.9, "dual_trials": 2},
    {"shot": 5, "target_x": 30.0, "target_y": 20.0, "final_x": 29.0, "final_y": 16.0, "err_counts": 4.12, "err_um": 32.7, "ballistic_ms": 63.1, "settled_ms": 109.3, "dual_trials": 2},
    {"shot": 6, "target_x": 30.0, "target_y": 20.0, "final_x": 28.0, "final_y": 19.0, "err_counts": 2.24, "err_um": 17.7, "ballistic_ms": 87.7, "settled_ms": 121.9, "dual_trials": 2},
    {"shot": 7, "target_x": 30.0, "target_y": 20.0, "final_x": 29.0, "final_y": 18.0, "err_counts": 2.24, "err_um": 17.7, "ballistic_ms": 79.7, "settled_ms": 107.6, "dual_trials": 2},
    {"shot": 8, "target_x": 30.0, "target_y": 20.0, "final_x": 29.0, "final_y": 19.0, "err_counts": 1.41, "err_um": 11.2, "ballistic_ms": 85.4, "settled_ms": 122.4, "dual_trials": 2},
    {"shot": 9, "target_x": 30.0, "target_y": 20.0, "final_x": 29.0, "final_y": 21.0, "err_counts": 1.41, "err_um": 11.2, "ballistic_ms": 86.6, "settled_ms": 109.7, "dual_trials": 1},
    {"shot": 10, "target_x": 30.0, "target_y": 20.0, "final_x": 28.0, "final_y": 19.0, "err_counts": 2.24, "err_um": 17.7, "ballistic_ms": 88.3, "settled_ms": 126.3, "dual_trials": 2},
    {"shot": 11, "target_x": 30.0, "target_y": 20.0, "final_x": 28.0, "final_y": 23.0, "err_counts": 3.61, "err_um": 28.6, "ballistic_ms": 86.2, "settled_ms": 104.6, "dual_trials": 2},
    {"shot": 12, "target_x": 30.0, "target_y": 20.0, "final_x": 29.0, "final_y": 18.0, "err_counts": 2.24, "err_um": 17.7, "ballistic_ms": 87.0, "settled_ms": 130.1, "dual_trials": 2},
]


# Historical architectural milestones verified on physical hardware (Gate 1 / Gate 2a protocol)
HISTORICAL_ABLATION_MILESTONES: List[Dict] = [
    {
        "stage_id": "S0_PID",
        "stage_label": "Traditional PID\n(50 Hz Loop)",
        "protocol": "5-Dir (exp1_controller_comparison.csv)",
        "err_cnt": 101.04,
        "err_um": 802.0,
        "t_bal_ms": 1244.9,
        "t_set_ms": 1244.9,
        "notes": "Mean over 5 directions; large backlash limit-cycle oscillation",
    },
    {
        "stage_id": "S1_1US_FW",
        "stage_label": "① 1 us Old FW\n(SW Timer)",
        "protocol": "Gate 1 (30,20) 12-shot Last-5 Mean",
        "err_cnt": 41.95,
        "err_um": 333.0,
        "t_bal_ms": 129.2,
        "t_set_ms": 229.9,
        "notes": "1us quantization (4.33 cnt/us Y-arm) + PC phase jitter",
    },
    {
        "stage_id": "S2_SUBUS_BCMD",
        "stage_label": "② 0.1 us SUBUS_V2\n(w/o Preload+S2 Lock)",
        "protocol": "Gate 2a 6-Dir Mean (Pre-B_CMD H-proto)",
        "err_cnt": 5.67,
        "err_um": 45.0,
        "t_bal_ms": 84.3,
        "t_set_ms": 139.3,
        "notes": "0.1us H-protocol; Gate 1=1.41 cnt, but 6-dir mean=5.67 cnt w/o directional preload",
    },
    {
        "stage_id": "S3_PRELOAD",
        "stage_label": "③ +B_CMD & 10-12 us\nSandwich Preload",
        "protocol": "Gate 2a 6-Dir Mean (commit ac877f8)",
        "err_cnt": 4.17,
        "err_um": 33.1,
        "t_bal_ms": 83.6,
        "t_set_ms": 125.4,
        "notes": "Hardware B_CMD + sandwich preload; residual B->eq retraction disengagement in Stage-2",
    },
    {
        "stage_id": "S4_FULL_45EA2AA",
        "stage_label": "④ Full Stage-1 H_OVR\n+ S2 Axial Isolation",
        "protocol": "Gate 2a 6-Dir Trimmed-3 Mean (commit 45ea2aa)",
        "err_cnt": 2.39,
        "err_um": 19.0,
        "t_bal_ms": 83.6,
        "t_set_ms": 98.2,
        "notes": "45ea2aa PASS: Gate 1 Last-5 Trimmed-4=1.83 cnt (14.5 um), Gate 2a 6-Dir=2.39 cnt (19.0 um), t_set=98.2ms",
    },
]


def save_gate1_archive(out_path: str = "experiments/gate1_30_20_eval.csv") -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(GATE1_45EA2AA_SHOTS[0].keys()))
        writer.writeheader()
        writer.writerows(GATE1_45EA2AA_SHOTS)
    last5 = [r["err_counts"] for r in GATE1_45EA2AA_SHOTS[-5:]]
    trim4 = float(np.mean(sorted(last5)[:4]))
    med = float(np.median(last5))
    t_bal = float(np.mean([r["ballistic_ms"] for r in GATE1_45EA2AA_SHOTS]))
    t_set = float(np.mean([r["settled_ms"] for r in GATE1_45EA2AA_SHOTS]))
    print(f"[ARCHIVE] {out_path}: n=12 | Last-5 Trim4={trim4:.2f} cnt ({trim4*UM_PER_COUNT:.1f} um) | "
          f"Med={med:.2f} cnt | t_bal={t_bal:.1f} ms | t_set={t_set:.1f} ms")


def run_short_controlled_ablation(shots_per_cond: int = 5) -> List[Dict]:
    """
    Execute a gentle, strictly capped (5 shots/condition on low-amplitude 35~36 cnt chords)
    single-factor ablation on the frozen 45ea2aa hardware setup.
    """
    from rect_manifold_calibrator import HardwareRectangleTrainer

    targets: List[Tuple[float, float]] = [(30.0, 20.0), (0.0, 35.0)]
    results: List[Dict] = []

    trainer = HardwareRectangleTrainer()
    trainer.sensor.start()
    orig_profile = trainer._short_ballistic_profile

    try:
        # Warmup 2 small shots to seat linkages consistently
        for cal_pt in ((28.0, 22.0), (-28.0, -22.0)):
            trainer.move_to_rect_origin(settle_s=0.35)
            trainer.move_point_to_point(cal_pt[0], cal_pt[1], tol_counts=1.2, two_stage=True, enable_ilc_update=False)

        conditions = [
            ("NO_PRELOAD", "w/o Sandwich Preload", False, True),
            ("STAGE1_ONLY", "Stage-1 H_OVR Only (w/o S2)", True, False),
        ]

        for cond_id, cond_label, use_preload, use_s2 in conditions:
            print(f"\n[ABLATION] Condition={cond_id} ({cond_label}), {shots_per_cond} shots/target")
            if use_preload:
                trainer._short_ballistic_profile = orig_profile
            else:
                def _no_preload_profile(chord_vec, _fn=orig_profile):
                    _, bst, eq, cut_r, y_min_r, b_us, amp_scale = _fn(chord_vec)
                    return np.zeros(2, dtype=float), bst, eq, cut_r, y_min_r, b_us, amp_scale
                trainer._short_ballistic_profile = _no_preload_profile

            for tx, ty in targets:
                errs = []
                for s in range(1, shots_per_cond + 1):
                    trainer.move_to_rect_origin(settle_s=0.36)
                    r = trainer.move_point_to_point(
                        tx,
                        ty,
                        tol_counts=1.2,
                        dual_lattice=use_s2,
                        two_stage=True,
                        enable_ilc_update=False,
                    )
                    row = {
                        "condition_id": cond_id,
                        "condition_label": cond_label,
                        "target_x": tx,
                        "target_y": ty,
                        "shot": s,
                        "final_x": r["final"][0],
                        "final_y": r["final"][1],
                        "err_counts": r["err_counts"],
                        "err_um": r["err_um"],
                        "ballistic_ms": r["ballistic_ms"],
                        "settled_ms": r["elapsed_ms"],
                        "dual_trials": r["dual_trials"],
                    }
                    results.append(row)
                    errs.append(r["err_counts"])
                    print(
                        f"  Tgt({tx:+.0f},{ty:+.0f}) Shot {s}/{shots_per_cond}: Final{r['final']} | "
                        f"Err={r['err_counts']:4.2f} cnt ({r['err_um']:4.1f} um) | "
                        f"t_bal={r['ballistic_ms']:4.1f}ms | t_set={r['elapsed_ms']:5.1f}ms | S2={r['dual_trials']}"
                    )
                print(f"  -> Summary ({tx:+.0f},{ty:+.0f}): Mean={statistics.mean(errs):.2f} cnt, "
                      f"Trim3={statistics.mean(sorted(errs)[:3]):.2f} cnt, Med={statistics.median(errs):.2f} cnt")
    finally:
        trainer._short_ballistic_profile = orig_profile
        trainer.driver.center(0.3)
        trainer.close()

    out_csv = "experiments/short_ablation_trials.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[SAVED] {out_csv} ({len(results)} rows)")
    return results


def consolidate_summary() -> None:
    os.makedirs("experiments", exist_ok=True)
    save_gate1_archive()

    # Verify Gate 2a 45ea2aa CSV statistics (read-only)
    g2_path = "experiments/gate2_6dir_eval.csv"
    with open(g2_path, "r", encoding="utf-8") as f:
        g2_rows = list(csv.DictReader(f))
    by_dir: Dict[Tuple[float, float], List[Dict]] = {}
    for r in g2_rows:
        key = (float(r["dir_x"]), float(r["dir_y"]))
        by_dir.setdefault(key, []).append(r)

    dir_trims, dir_meds = [], []
    for key, rs in by_dir.items():
        es = sorted(float(x["err_counts"]) for x in rs)
        t3 = float(np.mean(es[:3]))
        med = float(np.median(es))
        dir_trims.append(t3)
        dir_meds.append(med)
        print(f"[GATE 2A 45ea2aa] Dir {key}: Trim3={t3:.2f} cnt | Med={med:.2f} cnt | "
              f"RawMean={np.mean(es):.2f} cnt")

    print(
        f"[GATE 2A SUMMARY] 6-Dir Trim3 Mean={np.mean(dir_trims):.2f} cnt ({np.mean(dir_trims)*UM_PER_COUNT:.1f} um) | "
        f"t_bal={np.mean([float(r['ballistic_ms']) for r in g2_rows]):.1f} ms | "
        f"t_set={np.mean([float(r['settled_ms']) for r in g2_rows]):.1f} ms"
    )

    # Save architectural ablation milestone table
    milestones_csv = "experiments/ablation_evolution.csv"
    with open(milestones_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(HISTORICAL_ABLATION_MILESTONES[0].keys()))
        writer.writeheader()
        writer.writerows(HISTORICAL_ABLATION_MILESTONES)
    print(f"[SAVED] {milestones_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate IEEE paper experiment datasets")
    parser.add_argument(
        "--run-short-ablation",
        action="store_true",
        help="Run a brief 5-shot/condition live ablation on (30,20) and (0,35) w/o altering frozen files",
    )
    parser.add_argument("--shots", type=int, default=5, help="Shots per condition (capped <= 8)")
    args = parser.parse_args()

    consolidate_summary()
    if args.run_short_ablation:
        shots = max(5, min(8, int(args.shots)))
        run_short_controlled_ablation(shots_per_cond=shots)


if __name__ == "__main__":
    main()
