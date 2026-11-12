r"""probe_step_size_scan.py — 定案2：A0 步长扫描（区分「纯量化 + 固定死区」vs「粘滑」）

背景
----
已实测到两点：
  1) 短脉冲小步长扫描下，平台位置呈「平台段 + 跳变」的阶梯形态；
  2) 同入口重复逼近时落点聚成双簇，X 簇间距 14.25 cnt、Y 15.75 cnt。
推断：MG996R 的死区把连续位置量化成离散格点。

但「台阶宽度」这个数字有两种可能来源，二者对应的后续路线完全相反：

  (Q) 纯量化 + 固定死区
      => 台阶宽度与「采样步长」无关。
         此时任何更细的指令分辨率（亚 us PWM tick）都无效，
         只能靠 死区抖动 / 冗余执行器平均 / 可达格点地图。

  (S) 粘滑（stick-slip）
      => 大步长下单次脉冲能量更大，更容易突破静摩擦，
         台阶宽度会随采样步长收缩。
         此时亚 us tick + 更大预载 有真实收益空间。

本版相对首版的三处修正（首版结果不可用于判定）
--------------------------------------------
  1) 新增【增益对照块】：从原点单次 +40us、settle 0.6s，重复 5 次。
     首版发现 40us 增量扫描只走 12~26 cnt，而 K_lin=0.515us/cnt 预测 78 cnt，
     差 3~6 倍。必须先确认「大步长真实增益」，否则台阶宽度无意义。
  2) 扫描范围 40 -> 60 us、每档 3 -> 5 轮，提高台阶样本量。
  3) 判定改为统计式：只有组间差异 > 2x 组内合并散布 才下 Q/S 结论，
     否则明确输出 INCONCLUSIVE，绝不用 n=3 + 高散布硬拗结论。

判据
----
  gain_ctrl 中 displacement/40us ≈ 1.94 cnt/us  => 增益与 K_lin 自洽，可用台阶宽度判别
  gain_ctrl 显著低于 1.94                          => 先把增益问题查清（settle/映射/姿态），暂停判 Q/S

硬件耗时：约 3 分钟。
"""

import csv
import os
import statistics
import time
from typing import Dict, List, Tuple

from rect_manifold_calibrator import HardwareRectangleTrainer

SPAN_US = 60
STEPS = (1, 2, 4)
RUNS_PER_STEP = 5
SETTLE_S = 0.25
JUMP_THR_CNT = 0.5
GAIN_STEP_US = 40
GAIN_SETTLE_S = 0.6
GAIN_RUNS = 5


def _send_x(trainer, u0_kin, off_us: float):
    trainer.driver.send_raw_pulses(
        float(u0_kin[0] + off_us),
        float(u0_kin[1] + trainer.manifold.hyst_y_pos),
    )


def gain_control(trainer: HardwareRectangleTrainer, u0_kin) -> List[Dict]:
    """单次大幅步进的真实增益对照（避开增量扫描的累积效应）。"""
    out: List[Dict] = []
    for i in range(1, GAIN_RUNS + 1):
        trainer.move_to_rect_origin()
        trainer.sensor.recenter_cursor()
        p0 = trainer.get_rect_position()
        _send_x(trainer, u0_kin, GAIN_STEP_US)
        time.sleep(GAIN_SETTLE_S)
        p1 = trainer.get_rect_position()
        dx = float(p1[0]) - float(p0[0])
        dy = float(p1[1]) - float(p0[1])
        gain = dx / GAIN_STEP_US
        print(
            f"  [gain_ctrl {i}/{GAIN_RUNS}] +{GAIN_STEP_US}us -> "
            f"dx={dx:+.2f} cnt  dy={dy:+.2f} cnt  gain={gain:.3f} cnt/us "
            f"(K_lin 预测 {1.0/0.515:.3f})"
        )
        out.append(
            {
                "run_id": i,
                "step_us": GAIN_STEP_US,
                "dx_cnt": round(dx, 2),
                "dy_cnt": round(dy, 2),
                "gain_cnt_per_us": round(gain, 4),
            }
        )
    return out


def scan_once(
    trainer: HardwareRectangleTrainer,
    u0_kin,
    step_us: int,
) -> List[Tuple[int, float, float]]:
    """单向递增扫描 0 -> +SPAN_US，返回 [(offset_us, pos_x_cnt, pos_y_cnt)]。"""
    offsets = list(range(0, SPAN_US + 1, step_us))
    recs: List[Tuple[int, float, float]] = []

    trainer.move_to_rect_origin()
    trainer.sensor.recenter_cursor()

    for off in offsets:
        _send_x(trainer, u0_kin, off)
        time.sleep(SETTLE_S)
        pos = trainer.get_rect_position()
        recs.append((off, float(pos[0]), float(pos[1])))
    return recs


def extract_plateaus(
    recs: List[Tuple[int, float, float]],
    thr: float = JUMP_THR_CNT,
) -> Dict:
    """提取跳变点、平台宽度(us) 与跳变幅值(cnt)。首段平台受齿隙影响，统计时剔除。"""
    jumps: List[int] = []
    amps: List[float] = []
    ref = recs[0][1]
    for off, px, _py in recs[1:]:
        if abs(px - ref) > thr:
            jumps.append(off)
            amps.append(px - ref)
            ref = px

    widths = [jumps[i + 1] - jumps[i] for i in range(len(jumps) - 1)]
    return {
        "jump_n": len(jumps),
        "jumps_us": jumps,
        "widths_us": widths,
        "amps_cnt": [round(a, 2) for a in amps],
        "span_cnt": recs[-1][1] - recs[0][1],
        "crosstalk_y_cnt": recs[-1][2] - recs[0][2],
        "pts": len(recs),
    }


def _stat(vals: List[float]) -> str:
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{statistics.mean(vals):.2f}±{statistics.stdev(vals):.2f}"


def _mad(vals: List[float]) -> float:
    """中位绝对偏差（对离群更稳健）。"""
    if len(vals) < 2:
        return 0.0
    med = statistics.median(vals)
    return statistics.median([abs(v - med) for v in vals])


def main() -> int:
    trainer = HardwareRectangleTrainer()
    manifold = trainer.manifold
    trainer.sensor.start()

    rows: List[Dict] = []
    agg: Dict[int, Dict[str, List[float]]] = {
        s: {"widths": [], "jump_n": [], "span": [], "crosstalk": [], "amps": []}
        for s in STEPS
    }

    try:
        u0_kin = manifold.evaluate_kinematic_pulse(0.0, 0.0)
        print("=" * 96)
        print("  定案2 · A0 步长扫描（X 轴，单向 0 -> +%d us，无换向）" % SPAN_US)
        print("=" * 96)
        print(
            f"  u0_kin={[round(float(v), 2) for v in u0_kin]} us  "
            f"hyst_y_pos={manifold.hyst_y_pos} us  settle={SETTLE_S}s  "
            f"jump_thr={JUMP_THR_CNT} cnt  runs/step={RUNS_PER_STEP}"
        )

        # ---------- [0] 增益对照 ----------
        print("\n  [0] 增益对照块（单次大幅步进，验证 K_lin 是否可信）")
        gain_rows = gain_control(trainer, u0_kin)
        gains = [r["gain_cnt_per_us"] for r in gain_rows]
        gain_med = statistics.median(gains)
        # 两个参照必须分开：本块只推 X 一个作动器，
        # 正确参照是单轴约束柔度 J[0][0]=1.190 cnt/us；
        # 1/K_lin[0][0]=1.942 是「两轴协同纯平移」的等效增益，不能直接比。
        gain_pred_axis = 1.190
        gain_pred_syn = 1.0 / 0.515
        print(
            f"  => gain median={gain_med:.3f} cnt/us  (mean={_stat(gains)})  "
            f"| 单轴预测 {gain_pred_axis:.3f} (比值 {gain_med / gain_pred_axis:.2f}x)  "
            f"| 协同预测 {gain_pred_syn:.3f} (比值 {gain_med / gain_pred_syn:.2f}x)"
        )

        # ---------- [1] 步长扫描 ----------
        print("\n  [1] 步长扫描")
        for step in STEPS:
            for run_id in range(1, RUNS_PER_STEP + 1):
                recs = scan_once(trainer, u0_kin, step)
                info = extract_plateaus(recs)
                agg[step]["widths"].extend(info["widths_us"])
                agg[step]["jump_n"].append(info["jump_n"])
                agg[step]["span"].append(info["span_cnt"])
                agg[step]["crosstalk"].append(info["crosstalk_y_cnt"])
                agg[step]["amps"].extend(info["amps_cnt"])

                print(
                    f"  [step={step}us run={run_id}] pts={info['pts']:3d} "
                    f"jumps={info['jump_n']:2d} widths={info['widths_us']} "
                    f"amps={info['amps_cnt']} span={info['span_cnt']:+.2f} cnt "
                    f"y_ct={info['crosstalk_y_cnt']:+.2f}"
                )
                for off, px, py in recs:
                    rows.append(
                        {
                            "step_us": step,
                            "run_id": run_id,
                            "u_offset_us": off,
                            "pos_x_cnt": round(px, 2),
                            "pos_y_cnt": round(py, 2),
                        }
                    )

        os.makedirs("experiments", exist_ok=True)
        csv_path = "experiments/step_scan.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["step_us", "run_id", "u_offset_us", "pos_x_cnt", "pos_y_cnt"],
            )
            writer.writeheader()
            writer.writerows(rows)

        gain_csv = "experiments/step_scan_gain_control.csv"
        with open(gain_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["run_id", "step_us", "dx_cnt", "dy_cnt", "gain_cnt_per_us"]
            )
            writer.writeheader()
            writer.writerows(gain_rows)

        # ---------- [2] 汇总 + 判定 ----------
        lines: List[str] = []
        lines.append("=" * 96)
        lines.append("  定案2 · A0 步长扫描汇总")
        lines.append("=" * 96)
        lines.append(f"  u0_kin={[round(float(v), 2) for v in u0_kin]} us | span={SPAN_US}us "
                     f"| settle={SETTLE_S}s | runs/step={RUNS_PER_STEP}")
        lines.append("")
        lines.append(f"  [0] 增益对照  gain median={gain_med:.3f} cnt/us "
                     f"({_stat(gains)})  "
                     f"单轴预测={gain_pred_axis:.3f} (比值 {gain_med / gain_pred_axis:.2f}x)  "
                     f"协同预测={gain_pred_syn:.3f} (比值 {gain_med / gain_pred_syn:.2f}x)")

        med_by_step: Dict[int, float] = {}
        spread_by_step: Dict[int, float] = {}
        for step in STEPS:
            w = agg[step]["widths"]
            med = statistics.median(w) if w else 0.0
            med_by_step[step] = med
            spread_by_step[step] = _mad(w)
            lines.append(
                f"  [1] step={step}us | 平台宽度 median={med:.2f} us  MAD={_mad(w):.2f}  "
                f"mean={_stat(w)} (n={len(w)}) | 跳变次数={_stat(agg[step]['jump_n'])} "
                f"| 跳变幅值 mean={_stat(agg[step]['amps'])} cnt "
                f"| 总位移={_stat(agg[step]['span'])} cnt "
                f"| Y 串扰={_stat(agg[step]['crosstalk'])} cnt"
            )

        lines.append("")
        # 判定前提 1：增益可信（与单轴 / 协同两个参照的较小偏差比，二者都算自洽）
        r_axis = abs(gain_med - gain_pred_axis) / gain_pred_axis
        r_syn = abs(gain_med - gain_pred_syn) / gain_pred_syn
        gain_ok = min(r_axis, r_syn) < 0.35
        gain_ref = "单轴 J[0][0]=1.190" if r_axis <= r_syn else "协同 1/K_lin=1.942"
        # 判定前提 2：样本量与可分性
        ns = [len(agg[s]["widths"]) for s in STEPS]
        meds = [med_by_step[s] for s in STEPS]
        pooled_spread = max(spread_by_step.values()) if spread_by_step else 0.0
        between = (max(meds) - min(meds)) if meds else 0.0
        separable = (min(ns) >= 4) and (between > 2.0 * max(pooled_spread, 0.5))

        lines.append(f"  增益自洽: {'YES' if gain_ok else 'NO'}  "
                     f"(gain_med={gain_med:.3f} 最接近 {gain_ref}，"
                     f"偏差 {min(r_axis, r_syn):.1%}，门槛 35%)")
        lines.append(f"  样本量 n={ns}  组间极差={between:.2f} us  "
                     f"组内最大 MAD={pooled_spread:.2f} us  可分性门槛=2xMAD")
        lines.append("")

        if not gain_ok:
            verdict = (
                "  [结论 GAIN_MISMATCH] 大步长真实增益与 K_lin 不符（差 >35%）。\n"
                "      => 台阶宽度暂时不能用来判 Q/S。先查三件事：\n"
                "         a) move_to_rect_origin 后的真实起点是否稳定（用单步大幅步进重复性判断）；\n"
                "         b) K_lin 是否对应 raw pulse 空间（可能已含 play/Jacobian 变换）；\n"
                "         c) settle 0.6s 是否足够（看 dy/dx 是否仍在变）。"
            )
        elif not separable:
            verdict = (
                "  [结论 INCONCLUSIVE] 平台宽度组间差异未超过组内散布，无法区分 Q 与 S。\n"
                "      => 不要据此选路线。需要：加大 SPAN_US / 增加轮数 / 降低传感器量化影响，\n"
                "         或直接改用「同一 offset 重复下发 N 次看是否累积」的累积吸收实验。"
            )
        elif med_by_step[1] < med_by_step[4] - 1.5:
            verdict = (
                "  [结论 Q] 采样步长越小台阶越宽、步长越大台阶越窄 => 纯量化 + 固定死区。\n"
                "      亚 us PWM tick / 更细指令步长【不会】有收益；\n"
                "      应转向：死区抖动(dither) / 冗余执行器平均 / 可达格点地图。"
            )
        else:
            verdict = (
                "  [结论 S] 台阶宽度随机指令步长收缩（大步长更容易突破静摩擦）=> 粘滑成分存在。\n"
                "      亚 us tick + 增大预载【可能】有收益，值得做方向 A/D。"
            )
        lines.append(verdict)
        lines.append("")
        lines.append(f"  CSV: {os.path.abspath(csv_path)}")
        lines.append(f"  CSV: {os.path.abspath(gain_csv)}")

        txt_path = "experiments/analysis_step_scan.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("\n" + "\n".join(lines))
        print(f"  报告: {os.path.abspath(txt_path)}")
        return 0
    finally:
        trainer.close()


if __name__ == "__main__":
    raise SystemExit(main())
