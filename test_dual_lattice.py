"""test_dual_lattice.py -- 双向晶格是否确定性？(决定"同次访问内双向选优"能否实现)

背景
----
实测: MG996R + 连杆传动存在 6~14 us 死区，闭环 trim 在 |err|<15 cnt 时
      发出的 2.5 us 微调指令根本推不动机构(L732 max_du 下限 2.5 us)。
实测: 同一 offset 上行/下行落在不同位置(迟滞环宽约 33 cnt)，
      因此"单个方向的晶格间隙 q/2"才是精度地板。

本实验要回答的唯一问题:
  对**同一个名义脉冲** u_t，从下侧接近 与 从上侧接近，
  最终落点是否是**两个各自很窄、但彼此错开**的簇?
    - 是 => 双子格成立，可以在同一次访问内用"双向选优"把地板从 q/2 压到 ~q/4
    - 否 => 落点只是随机散布，双向选优没有可学习结构，路1 彻底死掉，直接换件

判定(预注册)
------------
  对每个轴:
    sep   = |mean(p_below) - mean(p_above)|        两簇间距
    sd_i  = 各簇内的标准差
    sep >= 2 * max(sd_below, sd_above) 且 sep >= 3 cnt  => PASS(双子格确定性)
用法
----
  set PYTHONIOENCODING=utf-8 && chcp 65001 >nul
  "E:\\anaconda\\envs\\pytorch\\python.exe" test_dual_lattice.py [trials] [overshoot_us] [axes]
    默认 trials=8, overshoot=20, axes=XY
输出
----
  experiments/dual_lattice_<YYYYmmdd_HHMMSS>.csv
退出码: 0=PASS  2=FAIL(落点随机)
"""

import csv
import os
import statistics
import sys
import time

from rect_manifold_calibrator import HardwareRectangleTrainer

MIN_SEP_CNT = 3.0      # 两簇最小可辨识间距
SEP_SD_RATIO = 2.0     # sep 至少是簇内标准差的这么多倍


def _axis_pos(rec, axis):
    return float(rec[0]) if axis == "X" else float(rec[1])


def _summ(vals):
    if not vals:
        return None, None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    overshoot = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    axes = (sys.argv[3] if len(sys.argv) > 3 else "XY").upper()

    trainer = HardwareRectangleTrainer()
    manifold = trainer.manifold
    driver = trainer.driver
    sensor = trainer.sensor
    sensor.start()
    time.sleep(0.4)

    rows = []
    results = {}
    t0 = time.time()

    print("=" * 92)
    print("[双向晶格确定性] 同一名义脉冲，从下侧 / 上侧接近的落点分布")
    print(f"  trials={trials}/侧   overshoot={overshoot:.0f} us   axes={axes}")
    print("=" * 92)

    try:
        trainer.move_to_rect_origin()
        sensor.recenter_cursor()
        time.sleep(0.2)

        u0_kin = manifold.evaluate_kinematic_pulse(0.0, 0.0)
        print(
            f"  u0_kin={[round(float(v), 2) for v in u0_kin]}  "
            f"hyst_x_pos={manifold.hyst_x_pos}  hyst_y_pos={manifold.hyst_y_pos}"
        )

        for axis in axes:
            # 名义目标脉冲(与 flank 探针一致: 非扫描轴带上迟滞偏置)
            if axis == "X":
                u_t = [float(u0_kin[0]), float(u0_kin[1] + manifold.hyst_y_pos)]
            else:
                u_t = [float(u0_kin[0] + manifold.hyst_x_pos), float(u0_kin[1])]

            below, above = [], []
            for k in range(trials):
                # 交替做，避免单调温漂偏向某一侧
                # --- 从下侧接近: u_t - overshoot  ->  u_t
                d = -overshoot if axis == "X" else -overshoot
                u_pre = [u_t[0] + d, u_t[1]] if axis == "X" else [u_t[0], u_t[1] + d]
                driver.send_raw_pulses(*u_pre)
                time.sleep(0.25)
                driver.send_raw_pulses(*u_t)
                time.sleep(0.25)
                p = trainer.get_rect_position()
                below.append(_axis_pos(p, axis))
                rows.append(
                    {
                        "axis": axis,
                        "trial": k + 1,
                        "approach_dir": "BELOW",
                        "overshoot_us": overshoot,
                        "pos_axis_cnt": round(_axis_pos(p, axis), 2),
                        "pos_x_cnt": round(float(p[0]), 2),
                        "pos_y_cnt": round(float(p[1]), 2),
                        "t_rel_s": round(time.time() - t0, 2),
                    }
                )

                # --- 从上侧接近: u_t + overshoot  ->  u_t
                u_pre = [u_t[0] + overshoot, u_t[1]] if axis == "X" else [u_t[0], u_t[1] + overshoot]
                driver.send_raw_pulses(*u_pre)
                time.sleep(0.25)
                driver.send_raw_pulses(*u_t)
                time.sleep(0.25)
                p = trainer.get_rect_position()
                above.append(_axis_pos(p, axis))
                rows.append(
                    {
                        "axis": axis,
                        "trial": k + 1,
                        "approach_dir": "ABOVE",
                        "overshoot_us": overshoot,
                        "pos_axis_cnt": round(_axis_pos(p, axis), 2),
                        "pos_x_cnt": round(float(p[0]), 2),
                        "pos_y_cnt": round(float(p[1]), 2),
                        "t_rel_s": round(time.time() - t0, 2),
                    }
                )

            mb, sb = _summ(below)
            ma, sa = _summ(above)
            sep = abs(mb - ma)
            results[axis] = {
                "below": (mb, sb),
                "above": (ma, sa),
                "sep": sep,
                "all": below + above,
            }

        # ------------------------------------------------------------ 报告
        print("\n" + "=" * 92)
        print("=== 落点分布 (counts) ===")
        print(f"{'axis':<6}{'BELOW mean':>12}{'sd':>8}{'ABOVE mean':>12}{'sd':>8}{'sep':>9}{'sep/sd':>9}")
        print("-" * 92)
        verdicts = {}
        for axis in axes:
            r = results[axis]
            mb, sb = r["below"]
            ma, sa = r["above"]
            max_sd = max(sb, sa, 0.001)
            ratio = r["sep"] / max_sd
            ok = (r["sep"] >= MIN_SEP_CNT) and (ratio >= SEP_SD_RATIO)
            verdicts[axis] = ok
            print(
                f"{axis:<6}{mb:>12.2f}{sb:>8.2f}{ma:>12.2f}{sa:>8.2f}"
                f"{r['sep']:>9.2f}{ratio:>9.2f}   {'PASS' if ok else 'FAIL'}"
            )

        print("\n--- 并集后的可达位置与间隙 ---")
        for axis in axes:
            vals = sorted({round(v, 1) for v in results[axis]["all"]})
            gaps = [
                round(vals[i + 1] - vals[i], 2)
                for i in range(len(vals) - 1)
                if vals[i + 1] - vals[i] > 1e-6
            ]
            single = sorted({round(v, 1) for v in results[axis]["below"]})
            sgaps = [
                round(single[i + 1] - single[i], 2)
                for i in range(len(single) - 1)
                if single[i + 1] - single[i] > 1e-6
            ]
            print(
                f"  {axis}: 单侧(下侧)可达点={len(single)} 间隙={sgaps}  |  "
                f"并集可达点={len(vals)} 间隙={gaps}"
            )

        print("\n" + "=" * 92)
        all_ok = all(verdicts.values())
        if all_ok:
            print("[判定] PASS -> 双子格确定性成立，可实现'同次访问内双向选优'，地板由 q/2 降到 ~q/4")
        else:
            bad = [a for a, v in verdicts.items() if not v]
            print(f"[判定] FAIL -> {','.join(bad)} 轴落点只是随机散布，双向选优无可利用结构，路1 应放弃")
        print("=" * 92)

        ts = time.strftime("%Y%m%d_%H%M%S")
        csv_path = f"experiments/dual_lattice_{ts}.csv"
        os.makedirs("experiments", exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "axis",
                    "trial",
                    "approach_dir",
                    "overshoot_us",
                    "pos_axis_cnt",
                    "pos_x_cnt",
                    "pos_y_cnt",
                    "t_rel_s",
                ],
            )
            w.writeheader()
            w.writerows(rows)
        print(f"CSV saved: {os.path.abspath(csv_path)}  ({len(rows)} rows)")
        return 0 if all_ok else 2
    finally:
        trainer.close()


if __name__ == "__main__":
    sys.exit(main())
