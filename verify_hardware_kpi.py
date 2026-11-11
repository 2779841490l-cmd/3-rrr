r"""
verify_hardware_kpi.py — 3-RRR 全屏矩形流形解耦与全点位精细定位硬性程序化验收门禁
====================================================================================
严格执行 PROJECT_PLAN.md 与 AGENTS.md 硬性工程验收标准（三道铁门）：
- Gate 1（最大内接矩形框四边直线度与伴随跳动门禁）：
  连续走完最大内接矩形框 4 条边与中心十字线，水平走线纵向伴随跳动均值 <= 3.0 counts，
  纵向走线水平伴随跳动均值 <= 3.0 counts；
- Gate 2（框内 5x5=25 个均匀网格点全覆盖连续点到点精细定位门禁）：
  无需回中连续走遍 25 个全范围网格点，平均稳态误差 <= 2.0 counts，最大误差 <= 3.5 counts；
- Gate 3（全屏幕 1536x864 边缘与四角 12 球连续打靶门禁）：
  全屏四角与四边随机目标点连续打靶命中率 100%，平均光学定位误差 <= 2.0 counts。
"""

import csv
import os
import sys
import numpy as np

from rect_manifold_calibrator import HardwareRectangleTrainer


def main() -> int:
    print("=" * 92)
    print("  3-RRR HARDWARE KPI GATE VERIFICATION (PI-NBM-NET RECTANGULAR MANIFOLD & FULL-SCREEN AIM)")
    print("=" * 92)

    trainer = HardwareRectangleTrainer()
    trainer.sensor.start()
    try:
        # ------------------------------------------------------------------
        # Gate 1: Rectangle 4-Edge Contour Straightness & Jump Elimination
        # ------------------------------------------------------------------
        print("\n[Gate 1] Running Full Rectangle Perimeter & Crosshair Straightness Check...")
        res_g1 = trainer.run_rectangle_contour_iteration(iter_idx=6, update_weights=False)
        h_jump = res_g1["horiz_parasitic_jump_y_counts"]
        v_jump = res_g1["vert_parasitic_jump_x_counts"]
        g1_pass = bool(h_jump <= 3.0 and v_jump <= 3.0)
        print(
            f"  -> Contour RMSE: {res_g1['rmse_counts']:.2f} cnt | "
            f"Horiz->Vert Jump: {h_jump:.2f} cnt (<=3.0) | "
            f"Vert->Horiz Jump: {v_jump:.2f} cnt (<=3.0) | "
            f"[{'PASS' if g1_pass else 'FAIL'}]"
        )

        # ------------------------------------------------------------------
        # Gate 2: 5x5 = 25 All-Grid Continuous Point-to-Point Fine Control
        # ------------------------------------------------------------------
        print("\n[Gate 2] Running 5x5=25 Full-Rectangle Continuous Point-to-Point Positioning...")
        trainer.move_to_rect_origin(settle_s=0.45)
        W, H = 0.85 * trainer.manifold.W, 0.85 * trainer.manifold.H
        xs = np.linspace(-W, W, 5)
        ys = np.linspace(-H, H, 5)
        test_pts = []
        for j, y in enumerate(ys):
            row_x = xs if j % 2 == 0 else xs[::-1]
            for x in row_x:
                test_pts.append((round(float(x), 1), round(float(y), 1)))

        grid_results = []
        grid_errs = []
        for idx, (tx, ty) in enumerate(test_pts, 1):
            r = trainer.move_point_to_point(tx, ty, tol_counts=1.2)
            grid_errs.append(r["err_counts"])
            grid_results.append(r)
            print(
                f"  Pt #{idx:02d}/25 {r['start']} -> Tgt{r['target']} | Final{r['final']} | "
                f"Err: {r['err_counts']:4.2f} cnt ({r['err_um']:4.1f} um) | "
                f"Ballistic: {r['ballistic_ms']:4.1f}ms"
            )

        os.makedirs("experiments", exist_ok=True)
        with open("experiments/rect_grid_25pts_eval.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(grid_results[0].keys()))
            writer.writeheader()
            writer.writerows(grid_results)

        g2_mean = float(np.mean(grid_errs))
        g2_max = float(np.max(grid_errs))
        g2_pass = bool(g2_mean <= 2.0 and g2_max <= 3.5)
        print(
            f"  -> [Gate 2 Summary] Mean Error: {g2_mean:.2f} cnt (<=2.0) | "
            f"Max Error: {g2_max:.2f} cnt (<=3.5) | [{'PASS' if g2_pass else 'FAIL'}]"
        )

        # ------------------------------------------------------------------
        # Gate 3: Full-Screen 1536x864 12-Ball Random/Edge Aiming Benchmark
        # ------------------------------------------------------------------
        print("\n[Gate 3] Running Full-Screen (1536x864) 12-Ball Corner & Edge Aiming Benchmark...")
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

        fps_rows = []
        fps_errs = []
        hits = 0
        for idx, (sx, sy) in enumerate(screen_targets, 1):
            rx = float(np.clip(sx / scale_x, -max_rx, max_rx))
            ry = float(np.clip(sy / scale_y, -max_ry, max_ry))
            r = trainer.move_point_to_point(rx, ry, tol_counts=1.2)
            act_sx = round(r["final"][0] * scale_x, 1)
            act_sy = round(r["final"][1] * scale_y, 1)
            tgt_sx = int(round(r["target"][0] * scale_x))
            tgt_sy = int(round(r["target"][1] * scale_y))
            s_err = round(float(np.hypot(tgt_sx - act_sx, tgt_sy - act_sy)), 2)
            hit = bool(s_err <= ball_radius_px)
            if hit:
                hits += 1
            fps_errs.append(r["err_counts"])
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
                f"Ballistic: {r['ballistic_ms']:4.1f}ms"
            )

        with open("experiments/fps_fullscreen_aim_benchmark.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(fps_rows[0].keys()))
            writer.writeheader()
            writer.writerows(fps_rows)

        g3_mean = float(np.mean(fps_errs))
        g3_pass = bool(hits == len(screen_targets) and g3_mean <= 2.0)
        print(
            f"  -> [Gate 3 Summary] Hit Rate: {hits}/{len(screen_targets)} | "
            f"Mean Optical Error: {g3_mean:.2f} cnt (<=2.0) | [{'PASS' if g3_pass else 'FAIL'}]"
        )

        print("\n" + "=" * 92)
        if g1_pass and g2_pass and g3_pass:
            print("  [ALL HARDWARE KPI GATES PASSED] (Gate 1, Gate 2 & Gate 3 100% Verified!)")
            print("=" * 92)
            return 0
        else:
            print(f"  [KPI GATE FAILED] Gate1={g1_pass}, Gate2={g2_pass}, Gate3={g3_pass}")
            print("=" * 92)
            return 1
    finally:
        trainer.driver.center(0.3)
        trainer.close()


if __name__ == "__main__":
    sys.exit(main())
