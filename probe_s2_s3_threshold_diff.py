r"""
probe_s2_s3_threshold_diff.py — P0-A 定案3：S2 / S3 双缸启停阈值与滑移量子差分探测
===================================================================================

目的（HANDOVER.md P0-A / 假设 H）：
  X 轴（S1 单缸）微步可重复，Y 轴（S2/S3 差动双缸，u2 + u3 = 3000）两方向均崩塌并出现
  ~17-18 cnt 滑移量子与重尾。
  假设 H：S2 与 S3 两条腿的预滑动/启停阈值不一致（Δth != 0），导致先动腿把平台拖出
  偏摆并在闭合链中弹性储能，后动腿崩塌时释放出“一个滑移量子”的跳变。

安全红线（严格遵守 AGENTS.md）：
  - 差模偏置硬限幅 |Δu_yaw| = |(s2 + s3) - 3000| <= 25.0 us
  - 脉宽硬限幅 [1150, 1850] us
  - 不修改 servo_driver.py 的 MAX_DUAL_Y_MISMATCH_US 保护逻辑

做法：
  从确定性预载后的统一中性起点 (u0x, u0y, u3_0=3000-u0y) 出发，在 Y+ 和 Y- 两个方向上，
  分别对三种驱动模式做阶梯扫描（du in [4, 8, 12, 14, 16, 18, 20, 22, 24, 25] us，每档 3 次）：
    1) S2_ONLY : 只推 S2 (s2 = u0y + sign*du, s3 = u3_0 固定)   -> |Δu_yaw| = du <= 25 us
    2) S3_ONLY : 只推 S3 (s2 = u0y 固定, s3 = u3_0 - sign*du)   -> |Δu_yaw| = du <= 25 us
    3) PAIR    : 双缸协同 (s2 = u0y + sign*du, s3 = u3_0 - sign*du) -> Δu_yaw = 0 us

产出：
  - experiments/s2_s3_threshold_diff.csv
  - experiments/analysis_s2_s3_threshold_diff.txt
"""

import argparse
import csv
import math
import os
import statistics
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from rect_manifold_calibrator import HardwareRectangleTrainer

MAX_YAW_DIFF_US = 25.0
DU_LEVELS_US = (4.0, 8.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 25.0)
REPEATS = 3

PRELOAD_US = 35.0
PRELOAD_HOLD_S = 0.16
ORIGIN_SETTLE_S = 0.28
STEP_SETTLE_S = 0.40

CSV_PATH = os.path.join("experiments", "s2_s3_threshold_diff.csv")
TXT_PATH = os.path.join("experiments", "analysis_s2_s3_threshold_diff.txt")


def _trial(
    trainer: HardwareRectangleTrainer,
    u0x: float,
    u0y: float,
    leg: str,
    sign: float,
    du: float,
) -> Dict[str, object]:
    if du > MAX_YAW_DIFF_US:
        raise ValueError(f"du={du} exceeds MAX_YAW_DIFF_US={MAX_YAW_DIFF_US}")

    u3_0 = 3000.0 - u0y

    # 1) 确定性预载（双缸协同走 -35*sign 再回 u0，确保 Δu_yaw=0 且齿面贴向一致）
    trainer.driver.send_raw_pulses(u0x, u0y - PRELOAD_US * sign, s3=3000.0 - (u0y - PRELOAD_US * sign))
    time.sleep(PRELOAD_HOLD_S)
    trainer.driver.send_raw_pulses(u0x, u0y, s3=u3_0)
    time.sleep(ORIGIN_SETTLE_S)

    trainer.sensor.recenter_cursor()
    p0 = trainer.get_rect_position()

    # 2) 按 leg 施加单腿或协同阶跃
    if leg == "S2_ONLY":
        s2_cmd = u0y + sign * du
        s3_cmd = u3_0
    elif leg == "S3_ONLY":
        s2_cmd = u0y
        s3_cmd = u3_0 - sign * du
    elif leg == "PAIR":
        s2_cmd = u0y + sign * du
        s3_cmd = u3_0 - sign * du
    else:
        raise ValueError(f"Unknown leg mode: {leg}")

    yaw_diff = abs((s2_cmd + s3_cmd) - 3000.0)
    if yaw_diff > MAX_YAW_DIFF_US + 1e-6:
        raise RuntimeError(f"Safety violation: yaw_diff={yaw_diff}")

    u1, u2, u3, u4 = trainer.driver.send_raw_pulses(u0x, s2_cmd, s3=s3_cmd)
    time.sleep(STEP_SETTLE_S)
    p1 = trainer.get_rect_position()

    dx = float(p1[0]) - float(p0[0])
    dy = float(p1[1]) - float(p0[1])
    # 按预期 Y 方向投影（calib_rect_manifold 中 s2 增加对应 Y- 还是 Y+? 取实际同号符号或直接记录 dx/dy/r）
    disp = float(math.hypot(dx, dy))

    return {
        "leg": leg,
        "sign": int(sign),
        "du_us": float(du),
        "u1": u1,
        "u2": u2,
        "u3": u3,
        "u_yaw_diff": abs((u2 + u3) - 3000),
        "dx_cnt": round(dx, 2),
        "dy_cnt": round(dy, 2),
        "disp_cnt": round(disp, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="P0-A 定案3：S2/S3 双缸启停阈值差分探测")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--dus", default=",".join(str(int(x)) for x in DU_LEVELS_US), help="逗号分隔 du(us) 档位")
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--txt", default=TXT_PATH)
    args = ap.parse_args()

    du_levels = tuple(float(x.strip()) for x in args.dus.split(",") if x.strip())
    for du in du_levels:
        if du > MAX_YAW_DIFF_US or du < 0:
            raise ValueError(f"du={du} out of safe range [0, {MAX_YAW_DIFF_US}]")

    os.makedirs("experiments", exist_ok=True)
    trainer = HardwareRectangleTrainer()
    trainer.sensor.start()
    time.sleep(0.30)
    trainer.sensor.reset_origin()

    rows: List[Dict[str, object]] = []
    try:
        trainer.move_to_rect_origin(settle_s=0.45)
        u0_kin = trainer.manifold.evaluate_kinematic_pulse(0.0, 0.0)
        u0x = float(round(float(u0_kin[0])))
        u0y = float(round(float(u0_kin[1])))
        u3_0 = 3000.0 - u0y

        print("=" * 96)
        print(f"P0-A 定案3：S2 / S3 启停阈值差分探测 (安全限幅 |Δu_yaw| <= {MAX_YAW_DIFF_US:.1f} us)")
        print(f"中性起点 raw: u0x={u0x:.0f}, u0y(S2)={u0y:.0f}, u3_0(S3)={u3_0:.0f} (sum={u0y+u3_0:.0f})")
        print(f"du 档位: {du_levels} | 每档重复: {args.repeats} 次")
        print("=" * 96)

        for sign in (+1.0, -1.0):
            s_lbl = "Y+" if sign > 0 else "Y-"
            print(f"\n--- 方向 {s_lbl} (sign={int(sign):+d}) ---")
            for leg in ("S2_ONLY", "S3_ONLY", "PAIR"):
                print(f"\n  [{leg}]")
                print(f"    {'du(us)':>6} | {'dy_list(cnt)':<24} | {'dx_mean':>7} | {'dy_mean':>7} | {'dy_sd':>6} | {'disp_med':>8} | {'moved(>=2)':>10}")
                for du in du_levels:
                    cell_rows = []
                    for rep in range(1, args.repeats + 1):
                        r = _trial(trainer, u0x, u0y, leg, sign, du)
                        r["rep"] = rep
                        rows.append(r)
                        cell_rows.append(r)
                    dxs = [float(x["dx_cnt"]) for x in cell_rows]
                    dys = [float(x["dy_cnt"]) for x in cell_rows]
                    disps = [float(x["disp_cnt"]) for x in cell_rows]
                    dy_sd = statistics.stdev(dys) if len(dys) > 1 else 0.0
                    moved = sum(1 for d in disps if d >= 2.0)
                    dy_str = ",".join(f"{v:+.0f}" for v in dys)
                    print(
                        f"    {du:6.1f} | {dy_str:<24} | {statistics.mean(dxs):+7.2f} | "
                        f"{statistics.mean(dys):+7.2f} | {dy_sd:6.2f} | {statistics.median(disps):8.2f} | {moved}/{args.repeats}"
                    )
    finally:
        try:
            trainer.driver.center(settle_s=0.4)
        except Exception:
            pass
        trainer.close()

    # Save CSV
    fieldnames = ["leg", "sign", "du_us", "rep", "u1", "u2", "u3", "u_yaw_diff", "dx_cnt", "dy_cnt", "disp_cnt"]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Summarize thresholds & slip quanta
    summary_lines = []
    def out(s: str = ""):
        print(s)
        summary_lines.append(s)

    out("\n" + "=" * 96)
    out("  P0-A 定案3 汇总分析：S2 vs S3 启停阈值与滑移量子")
    out("=" * 96)

    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["sign"], r["leg"], r["du_us"])].append(r)

    thresholds: Dict[Tuple[int, str], float] = {}
    for sign in (+1, -1):
        s_lbl = "sign=+1" if sign > 0 else "sign=-1"
        out(f"\n[{s_lbl}]")
        for leg in ("S2_ONLY", "S3_ONLY", "PAIR"):
            th_first = float("nan")
            th_rel = float("nan")
            nonzero_disps = []
            for du in du_levels:
                rs = grouped[(sign, leg, du)]
                disps = [float(x["disp_cnt"]) for x in rs]
                if math.isnan(th_first) and any(d >= 2.0 for d in disps):
                    th_first = du
                if math.isnan(th_rel) and sum(1 for d in disps if d >= 2.0) >= max(2, (len(disps) + 1) // 2):
                    th_rel = du
                for d in disps:
                    if d >= 2.0:
                        nonzero_disps.append(d)
            thresholds[(sign, leg)] = th_rel
            q_str = (
                f"median={statistics.median(nonzero_disps):.1f}, min={min(nonzero_disps):.1f}, max={max(nonzero_disps):.1f} cnt"
                if nonzero_disps else "无>=2cnt跳变(<=25us未破阈)"
            )
            out(f"  {leg:<8}: 首次动(>=2cnt)={th_first:>5} us | 稳定动(>=2/3次)={th_rel:>5} us | 有效位移量子: {q_str}")

        t2 = thresholds.get((sign, "S2_ONLY"), float("nan"))
        t3 = thresholds.get((sign, "S3_ONLY"), float("nan"))
        if not math.isnan(t2) and not math.isnan(t3):
            out(f"  => 阈值差 Δth = th(S2) - th(S3) = {t2 - t3:+.1f} us")
        elif math.isnan(t2) and not math.isnan(t3):
            out(f"  => 阈值差 Δth: S3 在 {t3:.1f} us 突破，而 S2 在 {max(du_levels):.1f} us 内全程未动")
        elif not math.isnan(t2) and math.isnan(t3):
            out(f"  => 阈值差 Δth: S2 在 {t2:.1f} us 突破，而 S3 在 {max(du_levels):.1f} us 内全程未动")
        else:
            out(f"  => 阈值差 Δth: {max(du_levels):.1f} us 内 S2 与 S3 均未稳定突破。")

    out(f"\n[SAVED] {args.csv}")
    out(f"[SAVED] {args.txt}")

    with open(args.txt, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

