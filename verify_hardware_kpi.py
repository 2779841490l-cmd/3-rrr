r"""
verify_hardware_kpi.py — 3-RRR 100ms-3200DPI 硬性程序化验收门禁
====================================================================================
严格对齐 AGENTS.md 与外部专家裁决标准：
- Gate 1（(30, 20) 抗振荡单调收敛门禁）:
  2 发标定热身后，对 (30, 20) 连续冷启动射击 12 发，最后 5 发（Shots 8~12）
  平均误差 <= 2.5 cnt，最大误差 <= 4.0 cnt；平均 t_ballistic <= 95.0 ms，t_settled <= 150.0 ms；
- Gate 2a（6 方向全象限百毫秒门禁）:
  对 [(35,0), (-35,0), (0,35), (0,-35), (30,25), (-30,-25)] 每方向 5 发学习 + 2 发连续评估，
  各方向平均误差 <= 3.0 cnt，总平均误差 <= 2.0 cnt，t_ballistic <= 95.0 ms，t_settled <= 150.0 ms；
- Gate 2b（框内 5x5=25 点连续点到点精细定位，[DIAGNOSTIC-ONLY] 仅打印日志不阻断退出码）:
  无需回中连续走遍 25 点，统计平均误差、最大误差、t_ballistic、t_settled；
- Gate 3（全屏 1536x864 边缘与四角 12 球连续打靶，[DIAGNOSTIC-ONLY] 仅打印日志不阻断退出码）:
  R=16 px 统计命中率、平均误差、t_ballistic、t_settled。
"""

import argparse
import csv
import os
import sys
import numpy as np

from rect_manifold_calibrator import HardwareRectangleTrainer


def run_gate1(trainer: HardwareRectangleTrainer) -> bool:
    print("\n[Gate 1] Running (30, 20) 12-Shot Cold-Start Convergence + Latency Check...")
    trainer.reset_online_adaptation(reset_alpha=True, reset_ilc=True)
    # 2 warmup/cal shots to prime Stage-1 secant gain without priming (30, 20) ILC
    for cal_pt in ((28.0, 22.0), (-28.0, -22.0)):
        trainer.move_to_rect_origin(settle_s=0.35)
        trainer.move_point_to_point(cal_pt[0], cal_pt[1], tol_counts=1.2, two_stage=True, enable_ilc_update=False)
    trainer.dir_ilc.clear()

    errs, bal_ms, set_ms = [], [], []
    for shot in range(1, 13):
        trainer.move_to_rect_origin(settle_s=0.36)
        r = trainer.move_point_to_point(30.0, 20.0, tol_counts=1.2, two_stage=True, enable_ilc_update=True)
        errs.append(r["err_counts"])
        bal_ms.append(r["ballistic_ms"])
        set_ms.append(r["elapsed_ms"])
        print(
            f"  Shot {shot:02d}/12 -> Final{r['final']} | Err: {r['err_counts']:4.2f} cnt ({r['err_um']:4.1f} um) | "
            f"t_bal: {r['ballistic_ms']:4.1f}ms | t_set: {r['elapsed_ms']:5.1f}ms | S2: {r['dual_trials']}"
        )

    last5 = errs[-5:]
    l5_mean, l5_max = float(np.mean(last5)), float(np.max(last5))
    mean_bal, mean_set = float(np.mean(bal_ms)), float(np.mean(set_ms))
    passed = bool(l5_mean <= 2.5 and l5_max <= 4.0 and mean_bal <= 95.0 and mean_set <= 150.0)
    print(
        f"  -> [Gate 1 Summary] Last-5 Mean: {l5_mean:.2f} cnt (<=2.5) | Max: {l5_max:.2f} cnt (<=4.0) | "
        f"t_bal={mean_bal:.1f}ms (<=95) | t_set={mean_set:.1f}ms (<=150) | [{'PASS' if passed else 'FAIL'}]"
    )
    return passed


def run_gate2a(trainer: HardwareRectangleTrainer) -> bool:
    print("\n[Gate 2a] Running 6-Direction All-Quadrant 100ms Evaluation...")
    directions = [(35.0, 0.0), (-35.0, 0.0), (0.0, 35.0), (0.0, -35.0), (30.0, 25.0), (-30.0, -25.0)]
    trainer.reset_online_adaptation(reset_alpha=False, reset_ilc=True)
    dir_means, bal_ms, set_ms = [], [], []
    rows = []

    for dx, dy in directions:
        print(f"  Direction ({dx:+.0f}, {dy:+.0f}):")
        for w in range(1, 6):
            trainer.move_to_rect_origin(settle_s=0.35)
            rw = trainer.move_point_to_point(dx, dy, tol_counts=1.2, two_stage=True, enable_ilc_update=True)
            print(f"    Learn {w}/5: Final{rw['final']} Err={rw['err_counts']:4.2f} cnt S2={rw['dual_trials']}")
        ev_errs = []
        for ev in range(1, 3):
            trainer.move_to_rect_origin(settle_s=0.35)
            re = trainer.move_point_to_point(dx, dy, tol_counts=1.2, two_stage=True, enable_ilc_update=False)
            ev_errs.append(re["err_counts"])
            bal_ms.append(re["ballistic_ms"])
            set_ms.append(re["elapsed_ms"])
            rows.append({"dir_x": dx, "dir_y": dy, "eval_id": ev, **re})
            print(
                f"    Eval  {ev}/2: Final{re['final']} Err={re['err_counts']:4.2f} cnt | "
                f"t_bal={re['ballistic_ms']:4.1f}ms t_set={re['elapsed_ms']:5.1f}ms"
            )
        dm = float(np.mean(ev_errs))
        dir_means.append(dm)
        print(f"    -> Dir Mean: {dm:.2f} cnt (<=3.0) [{'PASS' if dm <= 3.0 else 'FAIL'}]")

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/gate2_6dir_eval.csv", "w", newline="", encoding="utf-8") as f:
        clean_rows = [{k: v for k, v in r.items() if k != "u_kin_final"} for r in rows]
        writer = csv.DictWriter(f, fieldnames=list(clean_rows[0].keys()))
        writer.writeheader()
        writer.writerows(clean_rows)

    g2a_mean = float(np.mean(dir_means))
    mean_bal, mean_set = float(np.mean(bal_ms)), float(np.mean(set_ms))
    passed = bool(
        g2a_mean <= 2.0 and all(d <= 3.0 for d in dir_means)
        and mean_bal <= 95.0 and mean_set <= 150.0
    )
    print(
        f"  -> [Gate 2a Summary] 6-Dir Mean: {g2a_mean:.2f} cnt (<=2.0) | "
        f"t_bal={mean_bal:.1f}ms (<=95) | t_set={mean_set:.1f}ms (<=150) | [{'PASS' if passed else 'FAIL'}]"
    )
    return passed


def run_gate2b(trainer: HardwareRectangleTrainer) -> bool:
    print("\n[Gate 2b] Running 5x5=25 Full-Rectangle Continuous Point-to-Point Positioning...")
    trainer.move_to_rect_origin(settle_s=0.45)
    W, H = 0.85 * trainer.manifold.W, 0.85 * trainer.manifold.H
    xs = np.linspace(-W, W, 5)
    ys = np.linspace(-H, H, 5)
    test_pts = []
    for j, y in enumerate(ys):
        row_x = xs if j % 2 == 0 else xs[::-1]
        for x in row_x:
            test_pts.append((round(float(x), 1), round(float(y), 1)))

    grid_results, grid_errs, bal_ms, set_ms = [], [], [], []
    for idx, (tx, ty) in enumerate(test_pts, 1):
        r = trainer.move_point_to_point(tx, ty, tol_counts=1.2, two_stage=True)
        grid_errs.append(r["err_counts"])
        bal_ms.append(r["ballistic_ms"])
        set_ms.append(r["elapsed_ms"])
        grid_results.append(r)
        print(
            f"  Pt #{idx:02d}/25 {r['start']} -> Tgt{r['target']} | Final{r['final']} | "
            f"Err: {r['err_counts']:4.2f} cnt ({r['err_um']:4.1f} um) | "
            f"t_bal: {r['ballistic_ms']:4.1f}ms | t_set: {r['elapsed_ms']:5.1f}ms"
        )

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/rect_grid_25pts_eval.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid_results[0].keys()))
        writer.writeheader()
        writer.writerows(grid_results)

    g2_mean, g2_max = float(np.mean(grid_errs)), float(np.max(grid_errs))
    mean_bal, mean_set = float(np.mean(bal_ms)), float(np.mean(set_ms))
    passed = bool(g2_mean <= 2.0 and g2_max <= 3.5 and mean_bal <= 95.0 and mean_set <= 150.0)
    print(
        f"  -> [Gate 2b Summary] Mean Error: {g2_mean:.2f} cnt (<=2.0) | "
        f"Max Error: {g2_max:.2f} cnt (<=3.5) | t_bal={mean_bal:.1f}ms | t_set={mean_set:.1f}ms | "
        f"[{'PASS' if passed else 'FAIL'}]"
    )
    return passed


def run_gate3(trainer: HardwareRectangleTrainer) -> bool:
    print("\n[Gate 3] Running Full-Screen (1536x864) 12-Ball Corner & Edge Aiming Benchmark...")
    trainer.move_to_rect_origin(settle_s=0.40)
    sw, sh = 1536, 864
    max_rx, max_ry = 0.92 * trainer.manifold.W, 0.92 * trainer.manifold.H
    scale_x = (0.46 * sw) / max_rx
    scale_y = (0.45 * sh) / max_ry
    ball_radius_px = 16

    screen_targets = [
        (-680, -370), (680, -370), (680, 370), (-680, 370),
        (0, -380), (700, 0), (0, 380), (-700, 0),
        (460, -240), (-480, 250), (-360, -210), (0, 0),
    ]

    fps_rows, fps_errs, bal_ms, set_ms = [], [], [], []
    hits = 0
    for idx, (sx, sy) in enumerate(screen_targets, 1):
        rx = float(np.clip(sx / scale_x, -max_rx, max_rx))
        ry = float(np.clip(sy / scale_y, -max_ry, max_ry))
        r = trainer.move_point_to_point(rx, ry, tol_counts=1.2, two_stage=True)
        act_sx = round(r["final"][0] * scale_x, 1)
        act_sy = round(r["final"][1] * scale_y, 1)
        tgt_sx = int(round(r["target"][0] * scale_x))
        tgt_sy = int(round(r["target"][1] * scale_y))
        s_err = round(float(np.hypot(tgt_sx - act_sx, tgt_sy - act_sy)), 2)
        hit = bool(s_err <= ball_radius_px)
        if hit:
            hits += 1
        fps_errs.append(r["err_counts"])
        bal_ms.append(r["ballistic_ms"])
        set_ms.append(r["elapsed_ms"])
        fps_rows.append({
            "ball_id": idx,
            "screen_target_x_px": tgt_sx,
            "screen_target_y_px": tgt_sy,
            "screen_actual_x_px": act_sx,
            "screen_actual_y_px": act_sy,
            "raw_target_dx_cnt": r["target"][0],
            "raw_target_dy_cnt": r["target"][1],
            "raw_actual_dx_cnt": r["final"][0],
            "raw_actual_dy_cnt": r["final"][1],
            "raw_error_counts": r["err_counts"],
            "raw_error_um": r["err_um"],
            "screen_error_px": s_err,
            "ballistic_time_ms": r["ballistic_ms"],
            "settled_time_ms": r["elapsed_ms"],
            "hit": int(hit),
        })
        print(
            f"  Ball #{idx:02d}/12 [{'HIT ' if hit else 'MISS'}] Screen({tgt_sx:+4d},{tgt_sy:+4d}) | "
            f"Rect{r['target']} -> {r['final']} | Err: {r['err_counts']:4.2f} cnt ({r['err_um']:4.1f} um) | "
            f"t_bal: {r['ballistic_ms']:4.1f}ms | t_set: {r['elapsed_ms']:5.1f}ms"
        )

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/fps_fullscreen_aim_benchmark.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fps_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fps_rows)

    g3_mean = float(np.mean(fps_errs))
    mean_bal, mean_set = float(np.mean(bal_ms)), float(np.mean(set_ms))
    passed = bool(hits == len(screen_targets) and g3_mean <= 2.0 and mean_bal <= 95.0 and mean_set <= 150.0)
    print(
        f"  -> [Gate 3 Summary] Hit Rate: {hits}/{len(screen_targets)} | "
        f"Mean Optical Error: {g3_mean:.2f} cnt (<=2.0) | t_bal={mean_bal:.1f}ms | t_set={mean_set:.1f}ms | "
        f"[{'PASS' if passed else 'FAIL'}]"
    )
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description="3-RRR Hard Hardware KPI Verification")
    parser.add_argument("--gate", choices=["all", "g1", "g2a", "g2b", "g3"], default="all")
    args = parser.parse_args()

    print("=" * 92)
    print("  3-RRR HARDWARE KPI GATE VERIFICATION (TWO-STAGE SECANT + P0-B + LATENCY GATES)")
    print("=" * 92)

    trainer = HardwareRectangleTrainer()
    trainer.sensor.start()
    try:
        blocking_results = {}
        diag_results = {}
        if args.gate in ("all", "g1"):
            blocking_results["Gate1"] = run_gate1(trainer)
        if args.gate in ("all", "g2a"):
            blocking_results["Gate2a"] = run_gate2a(trainer)
        if args.gate in ("all", "g2b"):
            diag_results["Gate2b"] = run_gate2b(trainer)
        if args.gate in ("all", "g3"):
            diag_results["Gate3"] = run_gate3(trainer)

        print("\n" + "=" * 92)
        if diag_results:
            print(f"  [DIAGNOSTIC-ONLY (Non-Blocking)] {diag_results}")
        if blocking_results and all(blocking_results.values()):
            print(f"  [ALL HARDWARE KPI GATES PASSED] ({blocking_results})")
            print("=" * 92)
            return 0
        elif not blocking_results:
            print("  [DIAGNOSTIC RUN COMPLETED]")
            print("=" * 92)
            return 0
        else:
            print(f"  [KPI GATE FAILED] {blocking_results}")
            print("=" * 92)
            return 1
    finally:
        trainer.driver.center(0.3)
        trainer.close()


if __name__ == "__main__":
    sys.exit(main())
