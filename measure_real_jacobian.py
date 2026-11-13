r"""
measure_real_jacobian.py — 定案4：全 2×2 实测雅可比与通道死区标定（修复版）
==========================================================================

修复说明（原 line 53 bug）：
  旧脚本存在三处致命缺陷：
    1) line 53 `dx_duy = (p_x_plus[0] - p_y_minus[0]) / 50.0` 误将 `p_x_plus[0]` 当成 `p_y_plus[0]`；
    2) 未做确定性预载/回原点，跨步滞回混叠，且无重复次数；
    3) 无多档步长区分“死区 d”与“净斜率 J”，未落盘 CSV，未分符号报告。

本脚本协议（与 P0-A 定案3 `PAIR` 模式严格对齐）：
  - 全程保持双 Y 协同锁 `s3 = 3000 - s2`（即真实生产控制基）。
  - 对 u1（S1 单缸，du ∈ {12, 18, 25, 32} us）与 u2（PAIR 协同，du ∈ {12, 16, 20, 25} us）
    分别在 sign ∈ {+1, -1} 下各做 `--repeats` 次独立测量（默认 4 次，合计 64 发，~55 s）。
  - 每次试探流程：
      1) 沿目标运动方向做同向确定性预载：`u0 - 35*sign -> u0` 静置 0.28 s；
      2) `sensor.recenter_cursor()` 防止指针碰边界，读取静基线 `p0`；
      3) 发送 `u0 + sign * du`（保持锁和），静置 0.40 s 读取 `p1`；
      4) 记录净位移 `(dx, dy) = p1 - p0`。
  - 数据分析：
      - 死区分离线性回归（对每个分支拟合 `disp = s * (du - d)`，同时报告 `25 us` 弦雅可比）；
      - 汇总实测 2×2 前向雅可比 `J_meas` (cnt/us)、逆雅可比 `K_meas = inv(J_meas)` (us/cnt)、
        通道死区 `d_hat` (us)，并与 `calib_rect_manifold.json` 中的 `K_trim` 直接对比。

产出：
  - experiments/real_jacobian_2x2.csv
  - experiments/analysis_real_jacobian_2x2.txt
"""

import argparse
import csv
import math
import os
import statistics
import time
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from rect_manifold_calibrator import HardwareRectangleTrainer

PRELOAD_US = 35.0
PRELOAD_HOLD_S = 0.16
ORIGIN_SETTLE_S = 0.28
STEP_SETTLE_S = 0.40

U1_DUS = (12.0, 18.0, 25.0, 32.0)
U2_DUS = (12.0, 16.0, 20.0, 25.0)
DEFAULT_REPEATS = 4

CSV_PATH = os.path.join("experiments", "real_jacobian_2x2.csv")
TXT_PATH = os.path.join("experiments", "analysis_real_jacobian_2x2.txt")


def _single_trial(
    trainer: HardwareRectangleTrainer,
    u0x: float,
    u0y: float,
    axis: str,
    sign: int,
    du: float,
) -> Dict[str, object]:
    """执行单次确定性预载 + 净阶跃测量。"""
    if axis == "u1":
        # 预载 u1 轴，u2/u3 保持 u0y 锁和态
        u1_pre = u0x - PRELOAD_US * sign
        trainer.driver.send_raw_pulses(u1_pre, u0y)
        time.sleep(PRELOAD_HOLD_S)
        trainer.driver.send_raw_pulses(u0x, u0y)
        time.sleep(ORIGIN_SETTLE_S)

        trainer.sensor.recenter_cursor()
        p0 = trainer.get_rect_position()

        u1_cmd, u2_cmd, u3_cmd, _ = trainer.driver.send_raw_pulses(u0x + sign * du, u0y)
    elif axis == "u2":
        # 预载 u2/u3 协同 PAIR 轴，u1 保持 u0x
        u2_pre = u0y - PRELOAD_US * sign
        trainer.driver.send_raw_pulses(u0x, u2_pre)
        time.sleep(PRELOAD_HOLD_S)
        trainer.driver.send_raw_pulses(u0x, u0y)
        time.sleep(ORIGIN_SETTLE_S)

        trainer.sensor.recenter_cursor()
        p0 = trainer.get_rect_position()

        u1_cmd, u2_cmd, u3_cmd, _ = trainer.driver.send_raw_pulses(u0x, u0y + sign * du)
    else:
        raise ValueError(f"Unknown axis: {axis}")

    time.sleep(STEP_SETTLE_S)
    p1 = trainer.get_rect_position()

    dx = float(p1[0]) - float(p0[0])
    dy = float(p1[1]) - float(p0[1])
    return {
        "axis": axis,
        "sign": int(sign),
        "du_us": float(du),
        "u1_cmd": int(u1_cmd),
        "u2_cmd": int(u2_cmd),
        "u3_cmd": int(u3_cmd),
        "dx_cnt": round(dx, 2),
        "dy_cnt": round(dy, 2),
        "disp_cnt": round(float(math.hypot(dx, dy)), 2),
    }


def _fit_signed_branch(rows: List[Dict[str, object]]) -> Tuple[float, float, float, float, float]:
    """对同一 (axis, sign) 下全部样本做线性回归：
       dx = sx * du + bx,  dy = sy * du + by
    注意：因为输入是 du > 0，对于 sign=+1 或 -1：
      物理响应 p = s_dir * (du - d), 故回归给出斜率 wrt (sign*du) 为 J_col，
      且主轴死区截距 d = -b_main / s_main (us)。
    返回 (j_x_cnt_per_us, j_y_cnt_per_us, d_main_us, sd_main_residual, r2_main)。
    """
    u_signed = np.array([r["sign"] * r["du_us"] for r in rows], dtype=float)
    du_abs = np.array([r["du_us"] for r in rows], dtype=float)
    dx = np.array([r["dx_cnt"] for r in rows], dtype=float)
    dy = np.array([r["dy_cnt"] for r in rows], dtype=float)

    # wrt signed u = sign * du: dx = jx * (sign*du) + c_x => jx 直接是 ∂x/∂u (可正可负)
    jx, cx = np.polyfit(u_signed, dx, 1)
    jy, cy = np.polyfit(u_signed, dy, 1)

    # 主分量死区（选 |斜率| 较大的那一轴来定死区阈值，以避免次耦合轴噪声夸大截距）
    if abs(jx) >= abs(jy):
        # dx = sign * jx * du + cx -> when dx == 0, du_deadband = -cx / (sign * jx)
        slope_mag, intercept = np.polyfit(du_abs, dx, 1)
        pred = slope_mag * du_abs + intercept
        res_vals = dx - pred
        ss_tot = float(np.sum((dx - np.mean(dx)) ** 2))
    else:
        slope_mag, intercept = np.polyfit(du_abs, dy, 1)
        pred = slope_mag * du_abs + intercept
        res_vals = dy - pred
        ss_tot = float(np.sum((dy - np.mean(dy)) ** 2))

    d_dead = float(-intercept / slope_mag) if abs(slope_mag) > 1e-6 else float("nan")
    sd_res = float(np.std(res_vals, ddof=1)) if len(res_vals) > 1 else 0.0
    r2 = 1.0 - float(np.sum(res_vals ** 2)) / ss_tot if ss_tot > 1e-9 else 0.0
    return float(jx), float(jy), d_dead, sd_res, r2


def analyze_rows(rows: List[Dict[str, object]], k_trim_ref: np.ndarray) -> str:
    lines: List[str] = []
    lines.append("")
    lines.append("=" * 94)
    lines.append("  定案4 实测全 2x2 雅可比与通道死区报告（PAIR 协同基 + S1 单缸）")
    lines.append("=" * 94)

    grouped: Dict[Tuple[str, int, float], List[Dict[str, object]]] = defaultdict(list)
    by_branch: Dict[Tuple[str, int], List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        grouped[(str(r["axis"]), int(r["sign"]), float(r["du_us"]))].append(r)
        by_branch[(str(r["axis"]), int(r["sign"]))].append(r)

    for axis in ("u1", "u2"):
        lines.append("")
        lines.append(f"[{axis} 通道逐档统计]")
        for sign in (+1, -1):
            dus = sorted({k[2] for k in grouped.keys() if k[0] == axis and k[1] == sign})
            for du in dus:
                cell = grouped[(axis, sign, du)]
                dxs = [float(c["dx_cnt"]) for c in cell]
                dys = [float(c["dy_cnt"]) for c in cell]
                sdx = statistics.stdev(dxs) if len(dxs) > 1 else 0.0
                sdy = statistics.stdev(dys) if len(dys) > 1 else 0.0
                lines.append(
                    f"  sign={sign:+d} du={du:4.1f}us (n={len(cell)}): "
                    f"dx={statistics.fmean(dxs):+6.2f}+-{sdx:4.2f}, "
                    f"dy={statistics.fmean(dys):+6.2f}+-{sdy:4.2f} cnt"
                )

    lines.append("")
    lines.append("[死区分离回归: disp = J_col * (sign * du) + c, 主轴死区 d = -c / slope]")
    branches: Dict[Tuple[str, int], Tuple[float, float, float, float, float]] = {}
    for axis in ("u1", "u2"):
        for sign in (+1, -1):
            brows = by_branch[(axis, sign)]
            res = _fit_signed_branch(brows)
            branches[(axis, sign)] = res
            jx, jy, d_dead, sd_res, r2 = res
            lines.append(
                f"  {axis} (sign={sign:+d}): J_col=({jx:+6.3f}, {jy:+6.3f}) cnt/us | "
                f"主轴死区 d={d_dead:4.2f} us | R^2={r2:.3f}, res_sd={sd_res:.2f} cnt"
            )

    # 合并正向/负向平均斜率构造 2x2 J_meas 与各向死区估计
    j_u1_plus = branches[("u1", +1)][:2]
    j_u1_minus = branches[("u1", -1)][:2]
    j_u2_plus = branches[("u2", +1)][:2]
    j_u2_minus = branches[("u2", -1)][:2]

    j_col1 = 0.5 * (np.array(j_u1_plus) + np.array(j_u1_minus))
    j_col2 = 0.5 * (np.array(j_u2_plus) + np.array(j_u2_minus))
    J_meas = np.column_stack([j_col1, j_col2])
    K_meas = np.linalg.inv(J_meas)

    # 单独也构造正向/负向雅可比，检查正负不对称度
    J_plus = np.column_stack([j_u1_plus, j_u2_plus])
    J_minus = np.column_stack([j_u1_minus, j_u2_minus])
    K_plus = np.linalg.inv(J_plus)
    K_minus = np.linalg.inv(J_minus)

    d_u1_mean = 0.5 * (branches[("u1", +1)][2] + branches[("u1", -1)][2])
    d_u2_mean = 0.5 * (branches[("u2", +1)][2] + branches[("u2", -1)][2])

    # 修复版 25us 弦雅可比（直接对照旧 measure_real_jacobian.py）
    p_x_p = grouped[("u1", +1, 25.0)]
    p_x_m = grouped[("u1", -1, 25.0)]
    p_y_p = grouped[("u2", +1, 25.0)]
    p_y_m = grouped[("u2", -1, 25.0)]
    J_chord25 = np.array([
        [
            (statistics.fmean([float(r["dx_cnt"]) for r in p_x_p]) - statistics.fmean([float(r["dx_cnt"]) for r in p_x_m])) / 50.0,
            (statistics.fmean([float(r["dx_cnt"]) for r in p_y_p]) - statistics.fmean([float(r["dx_cnt"]) for r in p_y_m])) / 50.0,
        ],
        [
            (statistics.fmean([float(r["dy_cnt"]) for r in p_x_p]) - statistics.fmean([float(r["dy_cnt"]) for r in p_x_m])) / 50.0,
            (statistics.fmean([float(r["dy_cnt"]) for r in p_y_p]) - statistics.fmean([float(r["dy_cnt"]) for r in p_y_m])) / 50.0,
        ],
    ], dtype=float)
    K_chord25 = np.linalg.inv(J_chord25)

    lines.append("")
    lines.append("[死区修正后全 2x2 实测雅可比与逆矩阵]")
    lines.append(f"  d_hat_us (通道死区): d_u1 = {d_u1_mean:.2f} us (+:{branches[('u1',+1)][2]:.2f}, -:{branches[('u1',-1)][2]:.2f}), "
                 f"d_u2 = {d_u2_mean:.2f} us (+:{branches[('u2',+1)][2]:.2f}, -:{branches[('u2',-1)][2]:.2f})")
    lines.append(f"  J_meas (cnt/us): [[{J_meas[0,0]:+.4f}, {J_meas[0,1]:+.4f}], [{J_meas[1,0]:+.4f}, {J_meas[1,1]:+.4f}]]")
    lines.append(f"  K_meas = J_meas^-1 (us/cnt): [[{K_meas[0,0]:+.5f}, {K_meas[0,1]:+.5f}], [{K_meas[1,0]:+.5f}, {K_meas[1,1]:+.5f}]]")
    lines.append(f"  K_plus  (sign=+1): [[{K_plus[0,0]:+.5f}, {K_plus[0,1]:+.5f}], [{K_plus[1,0]:+.5f}, {K_plus[1,1]:+.5f}]]")
    lines.append(f"  K_minus (sign=-1): [[{K_minus[0,0]:+.5f}, {K_minus[0,1]:+.5f}], [{K_minus[1,0]:+.5f}, {K_minus[1,1]:+.5f}]]")
    lines.append(f"  (对照) 修复版 25us 弦 J_chord25: [[{J_chord25[0,0]:+.4f}, {J_chord25[0,1]:+.4f}], [{J_chord25[1,0]:+.4f}, {J_chord25[1,1]:+.4f}]]")
    lines.append(f"  (对照) 修复版 25us 弦 K_chord25: [[{K_chord25[0,0]:+.5f}, {K_chord25[0,1]:+.5f}], [{K_chord25[1,0]:+.5f}, {K_chord25[1,1]:+.5f}]]")

    # 与现用 K_trim 对比
    J_trim_inv = np.linalg.inv(k_trim_ref)
    lines.append("")
    lines.append("[与现用 calib_rect_manifold.json K_trim 对比]")
    lines.append(f"  现用 K_trim       : [[{k_trim_ref[0,0]:+.5f}, {k_trim_ref[0,1]:+.5f}], [{k_trim_ref[1,0]:+.5f}, {k_trim_ref[1,1]:+.5f}]]")
    lines.append(f"  等价 J = K_trim^-1: [[{J_trim_inv[0,0]:+.4f}, {J_trim_inv[0,1]:+.4f}], [{J_trim_inv[1,0]:+.4f}, {J_trim_inv[1,1]:+.4f}]]")
    lines.append(f"  K_meas vs K_trim  : 行1比=({K_meas[0,0]/k_trim_ref[0,0]:.2f}x, {K_meas[0,1]/k_trim_ref[0,1]:.2f}x), "
                 f"行2比=({K_meas[1,0]/k_trim_ref[1,0]:.2f}x, {K_meas[1,1]/k_trim_ref[1,1]:.2f}x)")

    # 量化分辨率核验：1 us 单步在 X / Y 通道产生的理论最小位移
    q_u1 = float(math.hypot(J_meas[0, 0], J_meas[1, 0]))
    q_u2 = float(math.hypot(J_meas[0, 1], J_meas[1, 1]))
    lines.append(f"  1 us 量化分辨率 : |du1=1us| = {q_u1:.3f} cnt, |du2=1us| = {q_u2:.3f} cnt")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="定案4：实测全 2x2 雅可比与通道死区")
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="每个 (axis, sign, du) 重复次数")
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--txt", default=TXT_PATH)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)

    trainer = HardwareRectangleTrainer()
    trainer.sensor.start()
    rows: List[Dict[str, object]] = []
    t_start = time.perf_counter()

    try:
        print("[定案4] 归零中立位 ...")
        trainer.move_to_rect_origin(settle_s=0.6)
        u0x = float(trainer.manifold.u0_star_x)
        u0y = float(trainer.manifold.u0_star_y)
        k_trim_ref = np.array(trainer.manifold.K_trim, dtype=float)
        print(f"[定案4] u0 = ({u0x:.1f}, {u0y:.1f}) us | repeats={args.repeats}")

        # 交错执行：先按轴顺序（换轴前重归原点一次）
        schedule: List[Tuple[str, float]] = [
            ("u1", d) for d in U1_DUS
        ] + [
            ("u2", d) for d in U2_DUS
        ]

        prev_axis = None
        for axis, du in schedule:
            if prev_axis is not None and axis != prev_axis:
                trainer.move_to_rect_origin(settle_s=0.45)
            prev_axis = axis

            for sign in (+1, -1):
                dxs: List[float] = []
                dys: List[float] = []
                for rep in range(1, args.repeats + 1):
                    row = _single_trial(trainer, u0x, u0y, axis, sign, du)
                    row["repeat"] = rep
                    rows.append(row)
                    dxs.append(float(row["dx_cnt"]))
                    dys.append(float(row["dy_cnt"]))
                sdx = statistics.stdev(dxs) if len(dxs) > 1 else 0.0
                sdy = statistics.stdev(dys) if len(dys) > 1 else 0.0
                print(
                    f"  {axis} sign={sign:+d} du={du:4.1f}us -> "
                    f"dx={statistics.fmean(dxs):+6.2f}+-{sdx:4.2f}, "
                    f"dy={statistics.fmean(dys):+6.2f}+-{sdy:4.2f}"
                )

        fieldnames = ["axis", "sign", "du_us", "repeat", "u1_cmd", "u2_cmd", "u3_cmd", "dx_cnt", "dy_cnt", "disp_cnt"]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        report = analyze_rows(rows, k_trim_ref)
        print(report)
        with open(args.txt, "w", encoding="utf-8") as f:
            f.write(report + "\n")
            f.write(f"[SAVED] {args.csv}\n")
            f.write(f"[SAVED] {args.txt}\n")
            f.write(f"[ELAPSED] {time.perf_counter() - t_start:.1f} s\n")

        print(f"[SAVED] {args.csv}")
        print(f"[SAVED] {args.txt}")
        print(f"[ELAPSED] {time.perf_counter() - t_start:.1f} s")
        return 0
    finally:
        try:
            trainer.move_to_rect_origin(0.25)
        finally:
            trainer.close()


if __name__ == "__main__":
    raise SystemExit(main())
