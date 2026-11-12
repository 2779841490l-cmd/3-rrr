r"""
test_landing_repeatability.py — 落地重复性判别（决定性：算法极限 vs 执行器极限）
====================================================================================
目的
----
上面所有实验都只能看到"点位误差"，看不到"同一个指令脉冲重复执行时的落点散布"。
本实验把两者拆开：

  每一发都从同一个起点 (-150, -120) 打到同一个目标 (150, 120)，
  记录 (a) 该发最终下发/停留的 kinematic 脉冲 u_kin_final
       (b) 该发最终光学落点 (x, y)

判别逻辑（预登记，先写死再看数据）
----------------------------------
* 若各发的 u_kin_final **逐分量完全相同**（唯一值数 <= 2，即最多差一个步进），
  而落点逐轴散布 sd >= 1.5 counts
      => 落点散布不是算法造成的，是执行器在**同一命令**下的自身重复性极限。
         => 在此执行器上继续调算法无意义 => 进【定案4】换件(总线舵机)。
* 若各发的 u_kin_final 明显不同（唯一值数 >= 5 且逐轴 sd(u) 与 sd(pos) 同量级）
      => 是我们的收敛判据/精修协议在抖，算法层面还有救。

额外输出
--------
* 逐轴 sd(pos)、落点最大两两距离、σ->理论平均 hypot = 1.2533*sd
* 死区换算：1 count = 7.9375 um；平台增益参考 14.3 counts/deg
  => sd 3 counts ≈ 0.21 deg，用于与 MG996R 电位器死区对照

运行
----
    set PYTHONIOENCODING=utf-8 && chcp 65001 >nul
    "E:\anaconda\envs\pytorch\python.exe" test_landing_repeatability.py [trials] [dual]

退出码：0 = 执行器极限（算法无解，进定案4）；2 = 算法仍有抖动空间；1 = 异常
"""

import csv
import os
import sys
import time

import numpy as np

from rect_manifold_calibrator import HardwareRectangleTrainer

# --- 预登记判据 ---
SD_POS_LIMIT = 1.5      # counts / axis，超过就算"落点散布显著"
U_UNIQUE_MAX = 2        # 认为"命令相同"的唯一值上限
U_UNIQUE_MIN = 5        # 认为"命令在抖"的唯一值下限

START_PT = (-150.0, -120.0)
TARGET_PT = (150.0, 120.0)
TOL_COUNTS = 1.2


def _uniq_count(vals, eps=1e-6):
    out = []
    for v in vals:
        if not any(abs(v - o) <= eps for o in out):
            out.append(float(v))
    return len(out)


def main() -> int:
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    dual = (sys.argv[2].lower() not in ("0", "false", "off")) if len(sys.argv) > 2 else True

    print("=" * 92)
    print("test_landing_repeatability.py — 落地重复性判别（同一指令 vs 落点散布）")
    print(f"trials={trials}  dual_lattice={dual}  start={START_PT} -> target={TARGET_PT}  tol={TOL_COUNTS}")
    print(f"预登记: sd_pos >= {SD_POS_LIMIT} cnt 且 u 唯一值 <= {U_UNIQUE_MAX} => 执行器极限")
    print("=" * 92)

    trainer = HardwareRectangleTrainer()
    trainer.sensor.start()
    rows = []
    try:
        trainer.move_to_rect_origin(settle_s=0.45)

        for i in range(1, trials + 1):
            # 每一发都用完全相同的入口条件：从同一起点、关闭双侧精修地打过去
            trainer.move_point_to_point(START_PT[0], START_PT[1], tol_counts=TOL_COUNTS,
                                        dual_lattice=False)
            t0 = time.time()
            r = trainer.move_point_to_point(TARGET_PT[0], TARGET_PT[1], tol_counts=TOL_COUNTS,
                                            dual_lattice=dual)
            dt_ms = (time.time() - t0) * 1000.0
            u = r.get("u_kin_final", [float("nan"), float("nan")])
            rows.append({
                "trial": i,
                "final_x": r["final"][0],
                "final_y": r["final"][1],
                "err_counts": r["err_counts"],
                "err_um": r["err_um"],
                "u_x": u[0],
                "u_y": u[1],
                "dual_trials": r.get("dual_trials", 0),
                "dual_gain_cnt": r.get("dual_gain_cnt", 0.0),
                "ballistic_ms": r["ballistic_ms"],
                "elapsed_ms": r["elapsed_ms"],
                "wall_ms": round(dt_ms, 1),
            })
            print(
                f"  #{i:02d}/{trials} Final({r['final'][0]:7.1f},{r['final'][1]:7.1f}) | "
                f"Err {r['err_counts']:5.2f} cnt ({r['err_um']:6.1f} um) | "
                f"u=({u[0]:8.3f},{u[1]:8.3f}) | dt {r.get('dual_trials', 0)} gain {r.get('dual_gain_cnt', 0.0):5.2f}"
            )

        os.makedirs("experiments", exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join("experiments", f"landing_repeat_{ts}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[CSV] {csv_path}")

        px = np.array([r["final_x"] for r in rows], dtype=float)
        py = np.array([r["final_y"] for r in rows], dtype=float)
        ux = np.array([r["u_x"] for r in rows], dtype=float)
        uy = np.array([r["u_y"] for r in rows], dtype=float)
        err = np.array([r["err_counts"] for r in rows], dtype=float)

        sd_px = float(px.std(ddof=1)) if len(px) > 1 else 0.0
        sd_py = float(py.std(ddof=1)) if len(py) > 1 else 0.0
        sd_ux = float(ux.std(ddof=1)) if len(ux) > 1 else 0.0
        sd_uy = float(uy.std(ddof=1)) if len(uy) > 1 else 0.0
        nu_x = _uniq_count(ux)
        nu_y = _uniq_count(uy)

        print("\n" + "-" * 92)
        print(f"落点   x: mean {px.mean():8.2f}  sd {sd_px:6.3f}  range {px.max()-px.min():6.2f} cnt")
        print(f"落点   y: mean {py.mean():8.2f}  sd {sd_py:6.3f}  range {py.max()-py.min():6.2f} cnt")
        print(f"命令 u_x: mean {ux.mean():8.3f}  sd {sd_ux:6.3f}  range {ux.max()-ux.min():7.3f} us  唯一值 {nu_x}")
        print(f"命令 u_y: mean {uy.mean():8.3f}  sd {sd_uy:6.3f}  range {uy.max()-uy.min():7.3f} us  唯一值 {nu_y}")
        print(f"误差模长: mean {err.mean():6.2f}  max {err.max():6.2f} cnt = {err.max()*7.9375:.1f} um")
        eff_sd = float(np.sqrt(0.5 * (sd_px ** 2 + sd_py ** 2)))
        print(f"等效各项同性 sd = {eff_sd:.3f} cnt -> 理论 mean(hypot) = {1.2533*eff_sd:.2f} cnt "
              f"(实测 {err.mean():.2f})")
        print(f"落点 sd 换算角度 ≈ {eff_sd/14.3:.3f} deg (平台增益 14.3 cnt/deg)")

        cmd_static = (nu_x <= U_UNIQUE_MAX and nu_y <= U_UNIQUE_MAX)
        pos_scatter = (sd_px >= SD_POS_LIMIT or sd_py >= SD_POS_LIMIT)
        cmd_jitter = (nu_x >= U_UNIQUE_MIN or nu_y >= U_UNIQUE_MIN)
        print("-" * 92)
        if cmd_static and pos_scatter:
            print("[VERDICT] 命令完全相同而落点散布显著 => 执行器自身重复性极限（算法无解）")
            print("          => 进【定案4】：换输出轴编码器总线舵机 (STS3215)")
            return 0
        if cmd_jitter:
            print("[VERDICT] 命令在抖（u 唯一值多） => 收敛判据/精修协议仍有改进空间（算法层面可救）")
            return 2
        print("[VERDICT] 数据不足以判定（命令近乎相同且落点散布小）——请加大 trials 或换点复测")
        return 2
    finally:
        try:
            trainer.driver.center(0.3)
        finally:
            trainer.close()


if __name__ == "__main__":
        sys.exit(main())
