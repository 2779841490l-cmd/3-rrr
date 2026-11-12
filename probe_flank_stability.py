"""probe_flank_stability.py -- 定案0：台阶(晶格)边界的时间稳定性 + 双向并集晶格间隙。

为什么需要它
------------
probe_flank_state.py 只做"单向"(offset 单调递增)，因此:
  1) 从未测过迟滞/回程，也从未测过"左右两个晶格"能否并集;
  2) 从未测过台阶边界在时间上会不会漂移 -- 而这一点直接决定
     "双向并集晶格"这条路到底能不能用。

本脚本对同一名义位置，间隔 gap_s 秒做两遍:
  上行 pass: offset -40 -> +40 (step_us 递增)
  下行 pass: offset +40 -> -40 (step_us 递减)
X / Y 两轴各做一遍，共 4 个 pass / 每轮，共 2 轮。

预注册判据(先写死，再看数据，不许事后改口径)
--------------------------------------------
  A) 边界时间稳定性:
     对每个 (轴, 方向) 分别估计 "首次起跳 offset" 与 "半程穿越 offset"，
     比较第1轮 vs 第2轮:
        两轮最大 |位移| <= 2.0 us  -> 晶格在 20 s 尺度上稳定 => 【路1 双向并集】成立
        两轮最大 |位移| >  2.0 us  -> 晶格会跑           => 路1 失效，直接换件
  B) 晶格间隙:
     q_single = 单个 pass 内"可达位置集合"的相邻间距中位数
     q_union  = 上行集合 U 下行集合 的并集相邻间距中位数
        若 q_union < 0.7 * q_single -> 双向并集确实把地板压下去(定量证明)

用法
----
  set PYTHONIOENCODING=utf-8 && chcp 65001 >nul
  "E:\\anaconda\\envs\\pytorch\\python.exe" probe_flank_stability.py [gap_s] [step_us] [axes]
    默认 gap_s=20, step_us=2, axes=XY
输出
----
  experiments/flank_stability_<YYYYmmdd_HHMMSS>.csv   (绝不覆盖任何旧文件)
退出码
------
  0 = 判据A通过(晶格稳定)   2 = 判据A失败(晶格会跑)   3 = 数据不足无法判定
"""

import csv
import os
import statistics
import sys
import time

from rect_manifold_calibrator import HardwareRectangleTrainer

TOL_CNT = 1.0          # "首次起跳"判据: 累计位移超过 1 count
DRIFT_LIMIT_US = 2.0   # 判据A阈值
UNION_RATIO = 0.7      # 判据B阈值


# ---------------------------------------------------------------- 分析工具
def _cross_offset(offs, vals, level):
    """在给定遍历顺序下，返回 vals 首次跨过 level 的插值 offset。

    遍历顺序由调用方保持(上行=offset 递增，下行=offset 递减)，
    因此返回值天然带方向信息。
    """
    if len(offs) < 2:
        return None
    for i in range(1, len(vals)):
        a, b = vals[i - 1], vals[i]
        if (a <= level <= b) or (b <= level <= a):
            if abs(b - a) < 1e-9:
                return float(offs[i])
            t = (level - a) / (b - a)
            return float(offs[i - 1]) + t * float(offs[i] - offs[i - 1])
    return None


def _pass_metrics(offs, vals):
    """返回 (首次起跳 offset, 半程穿越 offset, travel)。"""
    if not vals:
        return None, None, 0.0
    v0, v1 = vals[0], vals[-1]
    travel = v1 - v0
    if abs(travel) < 1e-9:
        return None, None, 0.0
    sign = 1.0 if travel > 0 else -1.0
    first = _cross_offset(offs, vals, v0 + sign * TOL_CNT)
    mid = _cross_offset(offs, vals, v0 + 0.5 * travel)
    return first, mid, travel


def _lattice(vals, lo, hi, decimals=1):
    """把一段位置序列收成"可达位置集合"，返回 (唯一值列表, 相邻间距列表)。"""
    uniq = sorted({round(float(v), decimals) for v in vals if lo <= v <= hi})
    diffs = []
    for i in range(len(uniq) - 1):
        d = round(uniq[i + 1] - uniq[i], 2)
        if d > 1e-6:
            diffs.append(d)
    return uniq, diffs


def _stat(vals):
    if not vals:
        return "n=0"
    return (
        f"n={len(vals)}  median={statistics.median(vals):.2f}  "
        f"max={max(vals):.2f}  min={min(vals):.2f}"
    )


def _fmt(v, nd=2):
    return "--" if v is None else f"{v:.{nd}f}"


# ---------------------------------------------------------------- 主流程
def main():
    gap_s = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    step_us = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    axes = (sys.argv[3] if len(sys.argv) > 3 else "XY").upper()

    trainer = HardwareRectangleTrainer()
    manifold = trainer.manifold
    driver = trainer.driver
    sensor = trainer.sensor
    sensor.start()
    time.sleep(0.4)

    offsets_up = list(range(-40, 41, step_us))
    offsets_dn = list(reversed(offsets_up))

    # (repeat_id, axis, approach_dir) -> list of (offset, pos_main, pos_other)
    passes = {}
    rows = []
    t_start = time.time()

    print("=" * 96)
    print("[定案0] 台阶(晶格)边界时间稳定性 + 双向并集间隙")
    print(
        f"  gap_s={gap_s:.1f} s   step={step_us} us   offsets={offsets_up[0]}..{offsets_up[-1]} "
        f"({len(offsets_up)} pts)   axes={axes}"
    )
    print(f"  单点驻留 0.25 s  预计耗时 ~{(len(offsets_up) * len(axes) * 2 * 2) * 0.40:.0f} s + {gap_s:.0f} s 间隔")
    print("=" * 96)

    try:
        for repeat_id in (1, 2):
            if repeat_id == 2:
                print(f"\n[间隔] 保持通电静止 {gap_s:.1f} s (不重新上电，只等时间漂移) ...")
                time.sleep(gap_s)

            trainer.move_to_rect_origin()
            sensor.recenter_cursor()
            time.sleep(0.15)

            u0_kin = manifold.evaluate_kinematic_pulse(0.0, 0.0)
            print(
                f"\n[Repeat {repeat_id}/2] u0_kin={[round(float(v), 2) for v in u0_kin]}  "
                f"play_g={[round(float(v), 3) for v in manifold.pi_net.play_g]}  "
                f"hyst_x_pos={manifold.hyst_x_pos}  hyst_y_pos={manifold.hyst_y_pos}"
            )

            for axis in axes:
                for dname, offs in (("UP", offsets_up), ("DN", offsets_dn)):
                    recs = []
                    t_pass0 = time.time()
                    for offset in offs:
                        if axis == "X":
                            u = [
                                float(u0_kin[0] + offset),
                                float(u0_kin[1] + manifold.hyst_y_pos),
                            ]
                        else:
                            u = [
                                float(u0_kin[0] + manifold.hyst_x_pos),
                                float(u0_kin[1] + offset),
                            ]
                        driver.send_raw_pulses(*u)
                        time.sleep(0.25)
                        pos = trainer.get_rect_position()
                        px, py = float(pos[0]), float(pos[1])
                        recs.append((offset, px, py))
                        rows.append(
                            {
                                "repeat_id": repeat_id,
                                "axis": axis,
                                "approach_dir": dname,
                                "seq": len(recs) - 1,
                                "u_offset_us": offset,
                                "pos_x_cnt": round(px, 2),
                                "pos_y_cnt": round(py, 2),
                                "t_rel_s": round(time.time() - t_start, 2),
                            }
                        )
                    passes[(repeat_id, axis, dname)] = recs
                    print(
                        f"    done {axis}/{dname:<2} {len(recs)} pts in {time.time() - t_pass0:.1f} s"
                    )

        # ------------------------------------------------------------ A. 稳定性
        print("\n" + "=" * 96)
        print(f"=== A. 边界时间稳定性 (Repeat1 vs Repeat2，间隔 {gap_s:.0f} s) ===")
        print(
            f"{'axis':<5}{'dir':<5}{'R1 first':>10}{'R2 first':>10}{'d_first':>9}"
            f"{'R1 mid':>9}{'R2 mid':>9}{'d_mid':>8}{'travel_R1':>11}{'travel_R2':>11}"
        )
        print("-" * 96)

        drifts = []
        for axis in axes:
            for dname in ("UP", "DN"):
                axis_idx = 1 if axis == "X" else 2
                v1 = [r[axis_idx] for r in passes[(1, axis, dname)]]
                v2 = [r[axis_idx] for r in passes[(2, axis, dname)]]
                o1 = [r[0] for r in passes[(1, axis, dname)]]
                o2 = [r[0] for r in passes[(2, axis, dname)]]
                f1, m1, t1 = _pass_metrics(o1, v1)
                f2, m2, t2 = _pass_metrics(o2, v2)
                d_f = None if (f1 is None or f2 is None) else abs(f1 - f2)
                d_m = None if (m1 is None or m2 is None) else abs(m1 - m2)
                for d in (d_f, d_m):
                    if d is not None:
                        drifts.append(d)
                print(
                    f"{axis:<5}{dname:<5}{_fmt(f1):>10}{_fmt(f2):>10}{_fmt(d_f):>9}"
                    f"{_fmt(m1):>9}{_fmt(m2):>9}{_fmt(d_m):>8}{t1:>11.1f}{t2:>11.1f}"
                )

        max_drift = max(drifts) if drifts else None
        print("-" * 96)
        print(f"最大边界位移 = {_fmt(max_drift)} us   (阈值 {DRIFT_LIMIT_US:.1f} us)")
        if max_drift is None:
            print("[A 判定] 数据不足，无法判定 (某次 pass 完全没有位移 => 卡死或量具失效)")
            verdict_a = 3
        elif max_drift <= DRIFT_LIMIT_US:
            print("[A 判定] PASS -> 晶格在时间上稳定，【路1 双向并集晶格】成立，可以先免费做")
            verdict_a = 0
        else:
            print("[A 判定] FAIL -> 晶格会随时间漂移，路1 失效，应直接进入换件，不再投算法工时")
            verdict_a = 2

        # ------------------------------------------------------------ B. 晶格间隙
        print("\n" + "=" * 96)
        print("=== B. 晶格间隙 q (counts，取行程中段 20%~80% 以避开端点伪影) ===")
        q_single_all = []
        q_union_all = []
        for axis in axes:
            axis_idx = 1 if axis == "X" else 2
            for repeat_id in (1, 2):
                up = [r[axis_idx] for r in passes[(repeat_id, axis, "UP")]]
                dn = [r[axis_idx] for r in passes[(repeat_id, axis, "DN")]]
                allp = up + dn
                pmin, pmax = min(allp), max(allp)
                span = pmax - pmin
                if span < 5.0:
                    print(f"  {axis} R{repeat_id}: 行程只有 {span:.1f} cnt，跳过")
                    continue
                lo, hi = pmin + 0.2 * span, pmax - 0.2 * span

                _, d_up = _lattice(up, lo, hi)
                _, d_dn = _lattice(dn, lo, hi)
                _, d_un = _lattice(allp, lo, hi)

                if d_up:
                    q_single_all.append(statistics.median(d_up))
                if d_dn:
                    q_single_all.append(statistics.median(d_dn))
                if d_un:
                    q_union_all.append(statistics.median(d_un))

                print(
                    f"  {axis} R{repeat_id}  range=[{pmin:.1f},{pmax:.1f}] span={span:.1f} cnt"
                    f"  q_single(UP)={_stat(d_up)}"
                )
                print(f"           q_single(DN)={_stat(d_dn)}")
                print(f"           q_union (UP|DN)={_stat(d_un)}")

        print("-" * 96)
        med_single = statistics.median(q_single_all) if q_single_all else None
        med_union = statistics.median(q_union_all) if q_union_all else None
        print(
            f"q_single 全局中位 = {_fmt(med_single)} cnt   "
            f"(单向地板 e_min = q/2 = {_fmt(None if med_single is None else med_single / 2)} cnt)"
        )
        print(
            f"q_union  全局中位 = {_fmt(med_union)} cnt   "
            f"(并集地板 e_min = q/2 = {_fmt(None if med_union is None else med_union / 2)} cnt)"
        )
        if med_single and med_union:
            ratio = med_union / med_single
            print(f"并集/单向 = {ratio:.2f}  (阈值 < {UNION_RATIO:.2f} 视为并集有效)")
            if ratio < UNION_RATIO:
                print("[B 判定] PASS -> 双向并集确实收窄了晶格，值得实现四象限选优")
            else:
                print("[B 判定] FAIL -> 并集没有收窄晶格，双向选优无收益，别做")
        else:
            print("[B 判定] 数据不足")

        # ------------------------------------------------------------ 输出 CSV
        ts = time.strftime("%Y%m%d_%H%M%S")
        csv_path = f"experiments/flank_stability_{ts}.csv"
        os.makedirs("experiments", exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "repeat_id",
                    "axis",
                    "approach_dir",
                    "seq",
                    "u_offset_us",
                    "pos_x_cnt",
                    "pos_y_cnt",
                    "t_rel_s",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        print("\n" + "=" * 96)
        print(f"CSV saved: {os.path.abspath(csv_path)}  ({len(rows)} rows)")
        print(f"总耗时 {(time.time() - t_start) / 60:.1f} min")
        print("=" * 96)
        return verdict_a
    finally:
        trainer.close()


if __name__ == "__main__":
    sys.exit(main())
