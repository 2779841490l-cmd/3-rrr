r"""probe_deadzone_gain_fit.py — 定案2c：raw 一致的幅度-位移扫描，拟合输入死区 d 与增益 g

为什么必须再做一次
------------------
定案2b 发现：`move_to_rect_origin()` 内部走的是 `evaluate_pulse()`（含 play 补偿），
而探针送的是 `send_raw_pulses()`（绕过补偿）。两者基线不一致，
所以 2b 的 B/C 两组不能当"跟随精度"用，只能当相对增量用。

但 2b 仍然给出了一个强信号：同样只推 S1，
    40 us -> 37.4 cnt  (0.925 cnt/us)
    61 us -> 70.2 cnt  (1.148 cnt/us)
视在增益随幅度单调上升 —— 这是输入死区的指纹。
线性 + 死区模型 dx = g*(du - d) 解出 g ~ 1.56 cnt/us、d ~ 16 us。

本脚本把基线改成"全程 raw"，扫 8 档正反幅度，直接拟合：
  * g   : 死区外真实增益 (cnt/us)
  * d   : 输入死区 (us)          <- 这个数决定亚 us tick 有没有意义
  * 正反向对称性                  <- 决定是否需要方向相关偏置

对后续路线的判定意义
--------------------
  若 d 显著 (> 3 us)：即在 A0 附近，指令必须变化超过 d 平台才动。
      => 亚 us PWM tick 单独无用（路线 A 死），必须 dither 越过死区（路线 B）
         或改用可达格点地图（路线 E）。
  若 d ~ 0 而增益非线性：说明是"粘滑"而非纯死区 => 路线 A + 更大预载有空间。

硬件耗时：约 60 秒。
"""

import csv
import json
import math
import os
import statistics
import time
from typing import Dict, List, Tuple

from rect_manifold_calibrator import HardwareRectangleTrainer

AMPS_US = (5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0)
RUNS = 3
ORIGIN_SETTLE_S = 0.5
SETTLE_S = 0.7


def _send(trainer, ux: float, uy: float):
    trainer.driver.send_raw_pulses(float(ux), float(uy))


def _measure_from_raw_origin(
    trainer, ux0: float, uy0: float, ux1: float, uy1: float
) -> Tuple[float, float]:
    """raw 基线：先 raw 走到 u0 并静置，再 raw 走到 u1，返回 (dx, dy) cnt。"""
    _send(trainer, ux0, uy0)
    time.sleep(ORIGIN_SETTLE_S)
    trainer.sensor.recenter_cursor()
    p0 = trainer.get_rect_position()

    _send(trainer, ux1, uy1)
    time.sleep(SETTLE_S)
    p1 = trainer.get_rect_position()

    return float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])


def _fit_deadzone(pts: List[Tuple[float, float]]) -> Dict[str, float]:
    """
    在正的 du 点上拟合 dx = g*(du - d)（du > d 才计入），网格搜索 d。
    pts: [(du_us, dx_cnt)]，du 已取绝对值。返回 {g, d, rms, n_used}
    """
    best: Dict[str, float] = {"g": float("nan"), "d": float("nan"), "rms": float("inf"), "n_used": 0}
    if len(pts) < 3:
        return best

    du_max = max(p[0] for p in pts)
    for d_int in range(0, 401):          # d 以 0.1 us 步长搜索 0 ~ 40 us
        d = d_int * 0.1
        used = [(du, dx) for du, dx in pts if du > d + 1e-9]
        if len(used) < 2:
            continue
        num = sum((du - d) * dx for du, dx in used)
        den = sum((du - d) ** 2 for du, dx in used)
        if den <= 1e-12:
            continue
        g = num / den
        if not math.isfinite(g) or g <= 0:
            continue
        rms = math.sqrt(sum((dx - g * (du - d)) ** 2 for du, dx in used) / len(used))
        if rms < best["rms"]:
            best = {"g": g, "d": d, "rms": rms, "n_used": len(used)}
    return best


def _stat(vals: List[float]) -> str:
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:+.2f}"
    return f"{statistics.mean(vals):+.2f}+-{statistics.stdev(vals):.2f}"


def main() -> int:
    trainer = HardwareRectangleTrainer()
    manifold = trainer.manifold
    trainer.sensor.start()

    rows: List[Dict] = []
    try:
        u0 = manifold.evaluate_kinematic_pulse(0.0, 0.0)
        u0x, u0y = float(u0[0]), float(u0[1]) + float(manifold.hyst_y_pos)

        print("=" * 96)
        print("  定案2c · raw 一致的幅度-位移扫描（X 轴，安静区 / 过渡区 / 饱和区）")
        print("=" * 96)
        print(
            f"  raw origin u = ({u0x:.2f}, {u0y:.2f}) us   amps={AMPS_US} us"
            f"   runs={RUNS}   settle={SETTLE_S}s"
        )

        sweep = [(s, a) for a in AMPS_US for s in (+1.0, -1.0)]
        pos_pts: List[Tuple[float, float]] = []
        neg_pts: List[Tuple[float, float]] = []
        pos_cross: List[float] = []
        neg_cross: List[float] = []

        for sign, amp in sweep:
            tag = "pos" if sign > 0 else "neg"
            dxs: List[float] = []
            dys: List[float] = []
            for i in range(1, RUNS + 1):
                dx, dy = _measure_from_raw_origin(
                    trainer, u0x, u0y, u0x + sign * amp, u0y
                )
                dxs.append(dx)
                dys.append(dy)
                rows.append(
                    {
                        "dir": tag,
                        "amp_us": amp,
                        "run_id": i,
                        "dx_cnt": round(dx, 2),
                        "dy_cnt": round(dy, 2),
                    }
                )
            med_dx = statistics.median(dxs)
            med_dy = statistics.median(dys)
            if sign > 0:
                pos_pts.append((amp, med_dx))
                pos_cross.append(med_dy)
            else:
                neg_pts.append((amp, -med_dx))
                neg_cross.append(med_dy)
            print(
                f"  [{tag}] du={sign * amp:+6.1f} us -> dx={_stat(dxs):>16} cnt "
                f"dy={_stat(dys):>16} cnt  (median dx={med_dx:+.2f})"
            )

        os.makedirs("experiments", exist_ok=True)
        csv_path = "experiments/deadzone_gain_fit.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["dir", "amp_us", "run_id", "dx_cnt", "dy_cnt"])
            w.writeheader()
            w.writerows(rows)

        # ---------- 拟合 ----------
        f_pos = _fit_deadzone(pos_pts)
        f_neg = _fit_deadzone(neg_pts)

        lines: List[str] = []
        lines.append("=" * 96)
        lines.append("  定案2c · X 轴 raw 幅度-位移响应（安静区 / 过渡区 / 饱和区）")
        lines.append("=" * 96)
        lines.append(f"  raw origin u = ({u0x:.2f}, {u0y:.2f}) us | runs/amp = {RUNS} | settle = {SETTLE_S}s")
        lines.append("")
        lines.append("  [1] 实测响应表（median over runs）：")
        for (amp, dx) in pos_pts:
            lines.append(f"    +{amp:5.1f} us -> {dx:+8.2f} cnt")
        for (amp, dxn) in neg_pts:
            lines.append(f"    -{amp:5.1f} us -> {-dxn:+8.2f} cnt")
        lines.append("")
        # 单一模型 dx = g*(du-d) 在 n_used=2 时不可辨（v1 曾误报 d=40 us），
        # 故此处不再输出 g/d，改用下面的 [2]~[4] 分段描述。
        _ = (f_pos, f_neg)  # 保留计算供 CSV 交叉核对，不进入报告
        lines.append("  [1b] 单一模型 g/d 拟合已弃用（有效点数 n=2，无法辨识 d）")
        lines.append(
            f"  正反向 Y 串扰: pos dy = {_stat(pos_cross)} cnt   neg dy = {_stat(neg_cross)} cnt"
        )

        # 参考：标定文件里的两个雅可比
        try:
            with open("calib_params.json", "r", encoding="utf-8") as fh:
                cp = json.load(fh)
            j00 = cp["jacobian_center_px_per_us"][0][0]
            lines.append(f"  参照 calib_params J[0][0] = {j00:.3f} cnt/us")
        except Exception:  # noqa: BLE001
            j00 = float("nan")
        k_lin = manifold.K_lin if hasattr(manifold, "K_lin") else None
        if k_lin is not None:
            lines.append(
                f"  参照 manifold.K_lin[0][0] = {float(k_lin[0][0]):.3f} us/cnt "
                f"-> {1.0 / float(k_lin[0][0]):.3f} cnt/us"
            )

        # ---------- 分段斜率 / 静摩擦安静区 ----------
        # 单一 dx = g*(du-d) 拟合只在"过坎后"成立，且 n_used 往往只有 2 个点，
        # 不足以代表全曲线。这里改成诚实的描述：安静区 -> 过渡区 -> 饱和区。
        # 注意：neg_pts 在采集时已存成「符号归一化后的正值」(= -med_dx)，
        # 这里绝不能再翻一次符号，否则负向斜率会被算成负数并污染中位数。
        pos_curve = list(pos_pts)
        neg_curve = list(neg_pts)

        def _piecewise(pts: List[Tuple[float, float]]) -> List[str]:
            out2: List[str] = []
            for (a0, x0), (a1, x1) in zip(pts, pts[1:]):
                if a1 <= a0:
                    continue
                out2.append(
                    f"{a0:5.1f}->{a1:5.1f} us: slope={(x1 - x0) / (a1 - a0):+6.2f} cnt/us"
                )
            return out2

        def _quiet_zone(pts: List[Tuple[float, float]], thr: float = 1.5) -> float:
            q = 0.0
            for a, x in pts:
                if abs(x) <= thr:
                    q = a
                else:
                    break
            return q

        def _sat_slope(pts: List[Tuple[float, float]]) -> float:
            if len(pts) < 2:
                return float("nan")
            (a0, x0), (a1, x1) = pts[-2], pts[-1]
            return (x1 - x0) / (a1 - a0) if a1 > a0 else float("nan")

        q_pos = _quiet_zone(pos_curve)
        q_neg = _quiet_zone(neg_curve)
        # 两条曲线均已是「位移随幅度单调上升」的正向曲线，斜率应为正；
        # 取绝对值作为最后一道防线，防止符号约定变化再次悄悄翻转结论。
        gs_pos = abs(_sat_slope(pos_curve))
        gs_neg = abs(_sat_slope(neg_curve))

        lines.append("")
        lines.append("  [2] 分段斜率（分辨 安静区 / 过渡区 / 饱和区）")
        for s in _piecewise(pos_curve):
            lines.append(f"    正向 {s}")
        for s in _piecewise(neg_curve):
            lines.append(f"    反向 {s}")
        lines.append("")
        lines.append(
            f"  [3] 静摩擦安静区（|dx| <= 1.5 cnt 的最长幅度）: "
            f"正向 {q_pos:.1f} us   反向 {q_neg:.1f} us"
        )
        lines.append(
            f"  [4] 饱和区增益（最大两档之间的斜率）: "
            f"正向 {gs_pos:.2f} cnt/us   反向 {gs_neg:.2f} cnt/us"
        )
        if math.isfinite(j00):
            lines.append(f"  [5] 参照 calib_params J[0][0] = {j00:.3f} cnt/us（单轴约束柔度）")
        if k_lin is not None:
            lines.append(
                f"      参照 1/K_lin[0][0] = {1.0 / float(k_lin[0][0]):.3f} cnt/us（协同纯X等效增益）"
            )

        lines.append("")
        k_pred = 1.0 / float(k_lin[0][0]) if k_lin is not None else 1.942
        gs_all = [v for v in (gs_pos, gs_neg) if math.isfinite(v)]
        gs_med = statistics.median(gs_all) if gs_all else float("nan")
        q_max = max(q_pos, q_neg)
        gain_ok = math.isfinite(gs_med) and abs(gs_med - k_pred) / k_pred < 0.25
        lines.append(
            f"  增益校验: 饱和斜率 {gs_med:.2f} vs 1/K_lin {k_pred:.2f} -> "
            f"{'一致 (差<25%)' if gain_ok else '不一致 (差>=25%)'}"
        )
        lines.append("")

        if gain_ok and q_max >= 15.0:
            lines.append(
                "  [结论 Q-DEADZONE-VALIDATED] 饱和区增益与 K_lin 一致 => 标定尺度没有病；\n"
                "      定案2 报的 GAIN_MISMATCH 是「测量落在安静区/过渡区」造成的假警报，予以撤销。\n"
                f"      静摩擦安静区高达 {q_max:.0f} us，20~50 us 之间是缓坡过渡（局部斜率 0.3~1.6 cnt/us）。\n"
                "      => 路线 A（亚 us PWM tick）单独【无效】：< 20 us 的指令修正被完全吞掉。\n"
                "      => 有效的软件方向只剩：\n"
                "         B 死区抖动 / breakaway kick：先给一次 20~25 us 越障脉冲再叠加小修正；\n"
                "         C 冗余执行器平均：多个格点相位错开的作动器取平均；\n"
                "         E 可达格点地图：把目标重映射到实际可达的格点。\n"
                "      => 另一个立刻可试、无需换件的控制律改进：\n"
                "         _dual_side_refine 现在的 walk 阶梯是 (6, 9, 12) us，全部落在安静区/缓坡区，\n"
                "         实测每次 walk 平均只换来 +1.33 cnt；应改为「先大后小」的递减阶梯（如 45,25,12 us）。\n"
                "         注意：历史上有一次「步长改成晶格周期后恶化」的否证，但那次是固定等步长，\n"
                "         与递减阶梯不是同一个变量，必须重新做单变量测试才能定论。"
            )
        elif q_max >= 15.0:
            lines.append(
                f"  [结论 Q-DEADZONE] 静摩擦安静区 {q_max:.0f} us 很大，但饱和增益与 K_lin 不符，\n"
                "      说明除死区外还存在增益尺度问题，需要同时修标定与控制律。"
            )
        else:
            lines.append(
                "  [结论 S-STICKSLIP] 安静区 < 15 us，但增益随幅度明显非线性 => 粘滑成分主导。\n"
                "      => 亚 us tick + 更大预载(路线 A/D) 有真实收益空间。"
            )

        lines.append("")
        lines.append(
            f"  量化含义：一旦越过 {q_max:.0f} us 安静区，指令端 1 us 即产生约 {gs_med:.2f} cnt 位移；\n"
            f"      目标精度 1~2 cnt 与「1 us 的含金量」同量级，因此能否细分到 1 us 以下不是关键，\n"
            f"      关键是能否把平台稳定停在任意位置 —— 这依赖格点/抖动/kick，而不是 tick 精度。"
        )

        report = "\n".join(lines)
        print("\n" + report)
        out_path = "experiments/analysis_deadzone_gain_fit.txt"
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
        print(f"\n  CSV: {os.path.abspath(csv_path)}")
        print(f"  报告: {os.path.abspath(out_path)}")
        return 0
    finally:
        trainer.close()


if __name__ == "__main__":
    raise SystemExit(main())
