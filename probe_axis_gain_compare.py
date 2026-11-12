r"""probe_axis_gain_compare.py — 定案2b：单轴激励 vs 协同双轴激励 的增益对照

为什么需要这一支
----------------
定案2 (probe_step_size_scan.py) 的增益对照块测出：只推 S1 +40 us，平台只走
dx=+34.6 cnt（gain=0.875 cnt/us），而 K_lin 隐含的"纯 X 移动"增益是
1/0.515 = 1.942 cnt/us，比值只有 0.45x，脚本据此报了 GAIN_MISMATCH。

但那个参照量很可能选错了。3-RRR 是闭环机构：
  - 只推一个作动器时，平台被其余支链的刚性约束拉住，量到的是"约束柔度"，
    在雅可比里对应 J[0][0]，calib_params.json 实测为 1.190 cnt/us
    => 0.875 / 1.190 = 0.735，其实只差 26%，在 ±35% 门限内。
  - 按 K_lin 做"纯 X 移动"必须两轴同时动（Δu2/Δu1 ≈ -0.70），
    此时才有 1.942 cnt/us。

同时定案2 里 Y 在"完全没被命令"的情况下飞了 -11/+6/-11/+10/+5 cnt（符号随机），
而 J[1][0]*40 = -10.2 恰好对上其中两次。必须判断这是"C 没做协同"的伪影，
还是真实的跨轴寄生跳变（若是后者，它比死区量化更严重，直接解释 4 cnt 残差）。

三组激励，各 RUNS 次（每次都从矩形原点重新出发、游标居中、settle 0.6 s）
  [A] 单轴+   u = u0 + [+40, 0] us              (定案2 用的激励)
  [B] 协同+   u = u_kin(+40 cnt, 0) + [0, hyst_y_pos]
  [C] 协同-   u = u_kin(-40 cnt, 0) + [0, hyst_y_pos]

判据（先定死，不许事后解释）
  B 满足 |dx-40| <= 8 且 |dy| <= 3
      => H1 成立：标定链路自洽，"单轴激励测增益"是实验设计伪影，
         撤销 GAIN_MISMATCH，Y 的 ±11 cnt 也判为伪影，可回到 Q/S 判定。
  否则
      => H2 成立：K_lin 的尺度/符号存在真实偏差，会影响所有目标定位，
         必须先把标定修好，暂停一切"精度上限"结论。

硬件耗时：约 25 秒。
"""

import csv
import json
import os
import statistics
import time
from typing import Dict, List, Tuple

from rect_manifold_calibrator import HardwareRectangleTrainer

TARGET_CNT = 40.0
RUNS = 5
SETTLE_S = 0.6
DX_TOL = 8.0
DY_TOL = 3.0


def _send(trainer, ux: float, uy: float):
    trainer.driver.send_raw_pulses(float(ux), float(uy))


def _measure(trainer, ux: float, uy: float) -> Tuple[float, float]:
    """从矩形原点重新出发，走一步，返回 (dx, dy) [cnt]。"""
    trainer.move_to_rect_origin()
    trainer.sensor.recenter_cursor()
    p0 = trainer.get_rect_position()
    _send(trainer, ux, uy)
    time.sleep(SETTLE_S)
    p1 = trainer.get_rect_position()
    return float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])


def _stat(vals: List[float]) -> str:
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:+.2f}"
    return f"{statistics.mean(vals):+.2f}+-{statistics.stdev(vals):.2f}"


def _load_calib_jacobian() -> Dict:
    try:
        with open("calib_params.json", "r", encoding="utf-8") as f:
            d = json.load(f)
        return {
            "jacobian_cnt_per_us": d.get("jacobian_center_px_per_us"),
            "inverse_us_per_cnt": d.get("inverse_jacobian_center_us_per_px"),
        }
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] 读 calib_params.json 失败: {exc}")
        return {}


def main() -> int:
    trainer = HardwareRectangleTrainer()
    manifold = trainer.manifold
    trainer.sensor.start()

    rows: List[Dict] = []
    try:
        u0 = manifold.evaluate_kinematic_pulse(0.0, 0.0)
        hyst = float(manifold.hyst_y_pos)
        u0x, u0y = float(u0[0]), float(u0[1])

        u_plus = manifold.evaluate_kinematic_pulse(+TARGET_CNT, 0.0)
        u_minus = manifold.evaluate_kinematic_pulse(-TARGET_CNT, 0.0)

        cmd_pos = (float(u_plus[0]), float(u_plus[1]) + hyst)
        cmd_neg = (float(u_minus[0]), float(u_minus[1]) + hyst)
        cmd_axis = (u0x + TARGET_CNT, u0y + hyst)

        print("=" * 96)
        print("  定案2b · 单轴激励 vs 协同双轴激励（X 轴，目标 +-%.0f cnt）" % TARGET_CNT)
        print("=" * 96)
        print(f"  u0_kin = [{u0x:.2f}, {u0y:.2f}] us   hyst_y_pos = {hyst} us   settle = {SETTLE_S}s")

        d_syn_pos = (cmd_pos[0] - u0x, cmd_pos[1] - u0y)
        d_syn_neg = (cmd_neg[0] - u0x, cmd_neg[1] - u0y)
        print(
            f"  [A] 单轴  指令增量 = ({TARGET_CNT:+.2f}, {0.0:+.2f}) us  "
            f"=> 每 cnt 用 {abs(TARGET_CNT) / TARGET_CNT:+.3f}? 见下"
        )
        print(
            f"  [B] 协同+ 指令增量 = ({d_syn_pos[0]:+.3f}, {d_syn_pos[1]:+.3f}) us "
            f"=> 实测雅可比第1列 = ({TARGET_CNT / d_syn_pos[0]:+.3f}, "
            f"{0.0 / d_syn_pos[1] if d_syn_pos[1] else float('nan'):+.3f}) cnt/us"
        )
        print(
            f"  [C] 协同- 指令增量 = ({d_syn_neg[0]:+.3f}, {d_syn_neg[1]:+.3f}) us "
            f"=> 每 cnt 用 {abs(d_syn_neg[0]) / TARGET_CNT:.3f} us/cnt"
        )
        print(
            f"      协同/单轴 指令幅度比 = {abs(d_syn_pos[0]) / TARGET_CNT:.3f} us/cnt"
            f"  (单轴为 1.000 us/cnt)"
        )

        calib = _load_calib_jacobian()
        jac = calib.get("jacobian_cnt_per_us")
        inv = calib.get("inverse_us_per_cnt")
        k_lin = manifold.K_lin if hasattr(manifold, "K_lin") else None
        if jac:
            print(f"  calib_params J[0][0] = {jac[0][0]:.3f} cnt/us  J[1][0] = {jac[1][0]:.3f} cnt/us")
        if inv:
            print(
                f"  calib_params K[0][0] = {inv[0][0]:.3f} us/cnt  K[0][1] = {inv[0][1]:.3f} us/cnt"
                f"  => 纯X移动的 u2/u1 比 = {inv[0][1] / inv[0][0]:+.3f}"
            )
        if k_lin is not None:
            print(
                f"  manifold.K_lin[0] = [{float(k_lin[0][0]):.3f}, {float(k_lin[0][1]):.3f}] us/cnt"
                f"  => 纯X移动的 u2/u1 比 = {float(k_lin[0][1]) / float(k_lin[0][0]):+.3f}"
            )

        groups = [
            ("A_axis", cmd_axis),
            ("B_syn_pos", cmd_pos),
            ("C_syn_neg", cmd_neg),
        ]

        res: Dict[str, Dict[str, List[float]]] = {g: {"dx": [], "dy": []} for g, _ in groups}
        for name, (ux, uy) in groups:
            print(f"\n  [{name}]  u = ({ux:.2f}, {uy:.2f}) us")
            for i in range(1, RUNS + 1):
                dx, dy = _measure(trainer, ux, uy)
                res[name]["dx"].append(dx)
                res[name]["dy"].append(dy)
                print(f"    run {i}/{RUNS}: dx={dx:+7.2f} cnt  dy={dy:+7.2f} cnt")
                rows.append(
                    {
                        "group": name,
                        "run_id": i,
                        "ux_us": round(ux, 3),
                        "uy_us": round(uy, 3),
                        "dx_cnt": round(dx, 2),
                        "dy_cnt": round(dy, 2),
                    }
                )

        os.makedirs("experiments", exist_ok=True)
        csv_path = "experiments/axis_gain_compare.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["group", "run_id", "ux_us", "uy_us", "dx_cnt", "dy_cnt"]
            )
            w.writeheader()
            w.writerows(rows)

        # ---------- 汇总 + 判定 ----------
        ax_dx, ax_dy = res["A_axis"]["dx"], res["A_axis"]["dy"]
        bp_dx, bp_dy = res["B_syn_pos"]["dx"], res["B_syn_pos"]["dy"]
        cn_dx, cn_dy = res["C_syn_neg"]["dx"], res["C_syn_neg"]["dy"]

        bp_dx_med = statistics.median(bp_dx)
        bp_dy_med = statistics.median(bp_dy)
        cn_dx_med = statistics.median(cn_dx)

        lines: List[str] = []
        lines.append("=" * 96)
        lines.append("  定案2b · 单轴 vs 协同 增益对照汇总")
        lines.append("=" * 96)
        lines.append(f"  目标幅度 = {TARGET_CNT:.0f} cnt   runs/group = {RUNS}   settle = {SETTLE_S}s")
        lines.append(f"  u0_kin = [{u0x:.2f}, {u0y:.2f}] us")
        lines.append("")
        lines.append(f"  [A] 单轴+(40us)  dx = {_stat(ax_dx)} cnt   dy = {_stat(ax_dy)} cnt "
                     f"| 单轴增益 = {statistics.median(ax_dx) / TARGET_CNT:.3f} cnt/us")
        lines.append(f"  [B] 协同+        dx = {_stat(bp_dx)} cnt   dy = {_stat(bp_dy)} cnt "
                     f"| 跟随误差 = {bp_dx_med - TARGET_CNT:+.2f} cnt")
        lines.append(f"  [C] 协同-        dx = {_stat(cn_dx)} cnt   dy = {_stat(cn_dy)} cnt "
                     f"| 跟随误差 = {cn_dx_med - (-TARGET_CNT):+.2f} cnt")
        lines.append("")
        lines.append(f"  X 正反向对称性: |B| - |C| = {abs(bp_dx_med) - abs(cn_dx_med):+.2f} cnt")
        if jac and abs(jac[0][0]) > 1e-9:
            lines.append(
                f"  单轴增益 / J[0][0] = {statistics.median(ax_dx) / TARGET_CNT / jac[0][0]:.3f}"
                f"   (接近 1 说明单轴激励量到的就是约束柔度，无额外损耗)"
            )
        lines.append("")

        h1 = (abs(bp_dx_med - TARGET_CNT) <= DX_TOL) and (abs(bp_dy_med) <= DY_TOL)
        lines.append(f"  H1 判据: |dx-40| = {abs(bp_dx_med - TARGET_CNT):.2f} cnt (<= {DX_TOL})"
                     f"  |dy| = {abs(bp_dy_med):.2f} cnt (<= {DY_TOL})")
        lines.append("")
        if h1:
            lines.append("  [结论 H1] 协同激励下位移与指令自洽 => 标定链路正常。")
            lines.append("      => 定案2 的 GAIN_MISMATCH 是「单轴激励」这一实验设计造成的伪影，予以撤销；")
            lines.append("         Y 的 +-11 cnt 串扰同样判为伪影（闭环被单侧拉扯）。")
            lines.append("      => 可以回到 Q/S 判定（用定案2 已落盘的 step_scan.csv）。")
        else:
            lines.append("  [结论 H2] 协同激励下位移仍与指令不符 => K_lin 尺度/符号存在真实偏差。")
            lines.append("      => 这不是精度极限问题，而是标定问题，会影响所有目标定位；")
            lines.append("         必须先重标 K_lin（或在 evaluate_kinematic_pulse 上加一次尺度校正），")
            lines.append("         在此之前暂停一切「精度上限 / 死区量化」结论。")

        report = "\n".join(lines)
        print("\n" + report)
        out_path = "experiments/analysis_axis_gain_compare.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\n  CSV: {os.path.abspath(csv_path)}")
        print(f"  报告: {os.path.abspath(out_path)}")
        return 0
    finally:
        trainer.close()


if __name__ == "__main__":
    raise SystemExit(main())
