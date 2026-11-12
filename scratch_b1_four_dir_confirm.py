r"""scratch_b1_four_dir_confirm.py — 四方向同口径确认 B1 配方（禁止只挑好方向拼）。

【背景】
scratch_b1_dt_sweep.py 已证明：
  - 静态 hold 单独施加 = +0.00 +- 0.00 cnt（废步，专家否决成立）
  - dt 是真实、单调的时域旋钮 ⇒ B1「限时 kick + 反向 hold」是真时域控制

scratch_b1_2d_sweep.py 已找到四方向配方：
  X+ : jac0 / kick=45 / hold=-12 / dt=20 ms
  X- : jac0 / kick=45 / hold=-12 / dt=20 ms
  Y+ : jac1 / kick=21 / hold= -3 / dt=20 ms
  Y- : jac1 / kick=26 / hold= -5 / dt=22 ms
（Y+ 的 hold=-12 会过冲成 -7 cnt，必须用小 kick 21 + hold -3；Y+ 在 kick=45 下
 呈「0 或 ~17」双峰粘滑分岔，绝不可用。
 Y- 的 kick=21 落在触发阈值(~25us)以下呈 0/-8 双峰；kick=27 以上滑移 7~9 cnt。
 加密网格实测 kick=25~26 / dt=21~23 三格连续 PASS（-3.00+-0.00 / -3.25+-1.26
 / -3.00+-0.82），取平台中心 kick=26 / hold=-5 / dt=22 作为代表点。）

【本脚本】
四个方向各跑 N 次（默认 6），统一判据「2~5 cnt 且 sd<=2」，
全部落盘 experiments/b1_four_dir_confirm.csv，全部通过才允许拼接进 _dual_side_refine。
"""

import csv
import json
import math
import os
import statistics
import time
from typing import Dict, List, Tuple

import numpy as np

from probe_timed_kick import _trial
from rect_manifold_calibrator import HardwareRectangleTrainer

PASS_MEAN_LO = 2.0
PASS_MEAN_HI = 5.0
PASS_STD_MAX = 2.0

RECIPES: List[Dict] = [
    {"name": "X+", "mode": "jac0", "sign": +1.0, "kick": 45.0, "hold": -12.0, "dt": 20.0},
    {"name": "X-", "mode": "jac0", "sign": -1.0, "kick": 45.0, "hold": -12.0, "dt": 20.0},
    {"name": "Y+", "mode": "jac1", "sign": +1.0, "kick": 21.0, "hold": -3.0, "dt": 20.0},
    {"name": "Y-", "mode": "jac1", "sign": -1.0, "kick": 26.0, "hold": -5.0, "dt": 22.0},
]


def _load_inv_jac() -> np.ndarray:
    path = "calib_params.json"
    if not os.path.exists(path):
        return np.array([[0.73026, -0.51146], [-0.07728, -0.35898]], dtype=float)
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return np.array(d["inverse_jacobian_center_us_per_px"], dtype=float)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > 1e-12 else np.array([1.0, 0.0])


def _resolve(mode: str, invj: np.ndarray) -> Tuple[str, Tuple[float, float], str]:
    if mode == "jac0":
        u = _unit(invj[:, 0])
        return "x", (float(u[0]), float(u[1])), "dx"
    u = _unit(invj[:, 1])
    return "y", (float(u[0]), float(u[1])), "dy"


def _is_pass(mean: float, sd: float) -> bool:
    return (
        math.isfinite(mean)
        and math.isfinite(sd)
        and PASS_MEAN_LO <= abs(mean) <= PASS_MEAN_HI
        and sd <= PASS_STD_MAX
    )


def main(runs: int = 6) -> int:
    invj = _load_inv_jac()
    trainer = HardwareRectangleTrainer()
    rows: List[Dict] = []
    all_pass = True
    try:
        trainer.sensor.start()   # 关键：HardwareRectangleTrainer.__init__ 不启动传感器线程
        time.sleep(0.30)
        trainer.sensor.reset_origin()
        trainer.move_to_rect_origin(settle_s=0.45)
        u0 = trainer.manifold.evaluate_kinematic_pulse(0.0, 0.0)
        u0x, u0y = float(u0[0]), float(u0[1])

        print("=" * 104)
        print(f"B1 四方向同口径确认  n={runs}/方向  判据 |mean| in [{PASS_MEAN_LO},{PASS_MEAN_HI}] cnt 且 sd<={PASS_STD_MAX}")
        print("=" * 104)
        print(f"{'dir':>4} {'kick':>6} {'hold':>6} {'dt':>5} | {'proj_mean':>10} {'proj_sd':>8} "
              f"{'min':>7} {'max':>7} | {'cross_mean':>11} | verdict")

        for rc in RECIPES:
            axis, vec, proj_ax = _resolve(rc["mode"], invj)
            ps, cs = [], []
            for _ in range(runs):
                r = _trial(
                    trainer, u0x, u0y,
                    kick_us=float(rc["kick"]), dt_ms=float(rc["dt"]), hold_us=float(rc["hold"]),
                    sign=float(rc["sign"]), axis=axis, vec=vec,
                )
                dx, dy = float(r["dx_cnt"]), float(r["dy_cnt"])
                if proj_ax == "dx":
                    ps.append(dx); cs.append(dy)
                else:
                    ps.append(dy); cs.append(dx)
            mp = statistics.mean(ps)
            sp = statistics.stdev(ps) if len(ps) >= 2 else float("nan")
            mc = statistics.mean(cs)
            ok = _is_pass(mp, sp)
            all_pass = all_pass and ok
            print(
                f"{rc['name']:>4} {rc['kick']:+6.0f} {rc['hold']:+6.0f} {rc['dt']:5.0f} | "
                f"{mp:+10.2f} {sp:8.2f} {min(ps):+7.2f} {max(ps):+7.2f} | {mc:+11.2f} | "
                f"{'PASS' if ok else 'FAIL'}"
            )
            for v in ps:
                rows.append({
                    "dir": rc["name"], "mode": rc["mode"], "sign": rc["sign"],
                    "kick_us": rc["kick"], "hold_us": rc["hold"], "dt_ms": rc["dt"],
                    "proj_cnt": v, "proj_ax": proj_ax,
                    "proj_mean": mp, "proj_sd": sp, "cross_mean": mc,
                    "pass": int(ok),
                })

        print("-" * 104)
        os.makedirs("experiments", exist_ok=True)
        csv_path = "experiments/b1_four_dir_confirm.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[OK] CSV -> {csv_path}")

        if all_pass:
            print("[结论 ALL-4-PASS] 四方向配方全部达标，可以拼接进 _dual_side_refine。")
            return 0
        print("[结论 NOT-ALL-PASS] 仍有方向未达标，禁止拼接。")
        return 1
    finally:
        try:
            trainer.driver.center(0.3)
        except Exception:
            pass
        try:
            trainer.sensor.stop()
        except Exception:
            pass
        trainer.close()


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    raise SystemExit(main(n))
