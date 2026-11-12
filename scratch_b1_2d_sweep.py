r"""scratch_b1_2d_sweep.py — B1「限时 kick + 反向 hold」的 dt 旋钮扫描（全方向通用）。

【已被数据确立的两条事实】
1) 静态 hold（无 kick）一律零位移：scratch_b1_dt_sweep.py 实测 hold=-12us 单独施加
   dx = +0.00 +- 0.00 cnt。⇒ 静态微偏置阶梯是废步，专家的否决成立。
2) dt 是真实、单调、低方差的时域旋钮（X+ / hold=-12 / kick=45）：
      dt  0ms -> +0.00 | 5ms -> +1.67 | 10ms -> +1.67 | 20ms -> +4.67
      35ms -> +7.00 | 50ms -> +13.67 | 70ms -> +18.67
   ⇒ 落点由 kick 的暴露时长控制，B1 是时域控制而非静态偏置。

【本脚本】
把同一旋钮扫到任意方向（含 Y+ 这个一直没解的方向），用来找
「2 ~ 5 cnt 且 sd <= 2」的 PASS 格。绝不只测好方向。

用法示例：
  python scratch_b1_2d_sweep.py --mode jac1 --sign +1 --hold -12 --kick 45 --dts "0,3,5,8,12,20" --runs 3
  python scratch_b1_2d_sweep.py --mode x    --sign +1 --hold -12 --kick 45 --dts "15,18,20,22,25" --runs 4

mode: x / y = 纯单轴；jac0 / jac1 = 沿真实逆雅可比列方向（u 空间矢量，归一化）。
"""

import argparse
import csv
import datetime
import json
import math
import os
import statistics
import time
from typing import List, Optional, Tuple

import numpy as np

from probe_timed_kick import _trial
from rect_manifold_calibrator import HardwareRectangleTrainer

PASS_MEAN_LO = 2.0
PASS_MEAN_HI = 5.0
PASS_STD_MAX = 2.0


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


def _resolve_mode(mode: str) -> Tuple[str, Optional[Tuple[float, float]], str]:
    """返回 (axis_label, vec_or_None, 主投影轴说明)。"""
    if mode == "x":
        return "x", None, "dx"
    if mode == "y":
        return "y", None, "dy"
    invj = _load_inv_jac()
    if mode == "jac0":
        u = _unit(invj[:, 0])
        return "x", (float(u[0]), float(u[1])), "dx"
    if mode == "jac1":
        u = _unit(invj[:, 1])
        return "y", (float(u[0]), float(u[1])), "dy"
    raise ValueError(f"unknown mode: {mode}")


def _is_pass(mean: float, sd: float) -> bool:
    return (
        math.isfinite(mean)
        and math.isfinite(sd)
        and PASS_MEAN_LO <= abs(mean) <= PASS_MEAN_HI
        and sd <= PASS_STD_MAX
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="B1 dt 旋钮扫描（全方向通用）")
    ap.add_argument("--mode", default="x", choices=["x", "y", "jac0", "jac1"])
    ap.add_argument("--sign", type=float, default=1.0)
    ap.add_argument("--hold", type=float, default=-12.0)
    ap.add_argument("--kick", type=float, default=45.0)
    ap.add_argument("--kicks", default="", help="逗号分隔的 kick 列表，给了就做 kick x dt 二维网格")
    ap.add_argument("--dts", default="0,5,10,20,35,50,70")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    axis, vec, proj_ax = _resolve_mode(args.mode)
    dts: List[float] = [float(x) for x in args.dts.split(",") if x.strip() != ""]
    kicks: List[float] = (
        [float(x) for x in args.kicks.split(",") if x.strip() != ""]
        if args.kicks.strip() != ""
        else [float(args.kick)]
    )

    trainer = HardwareRectangleTrainer()
    try:
        trainer.sensor.start()   # 关键：HardwareRectangleTrainer.__init__ 不启动传感器线程
        time.sleep(0.30)
        trainer.sensor.reset_origin()
        trainer.move_to_rect_origin(settle_s=0.45)

        u0 = trainer.manifold.evaluate_kinematic_pulse(0.0, 0.0)
        u0x, u0y = float(u0[0]), float(u0[1])

        print("=" * 100)
        print(f"mode={args.mode} axis={axis} sign={args.sign:+.0f} kicks={kicks} "
              f"hold={args.hold:+.1f}us runs={args.runs}")
        print(f"vec={vec}  主投影={proj_ax}  u0=({u0x:.2f}, {u0y:.2f})")
        print("=" * 100)
        print(f"{'kick':>6} {'dt_ms':>7} | {'proj_mean':>10} {'proj_sd':>8} {'proj_min':>9} {'proj_max':>9} | "
              f"{'cross_mean':>11} {'cross_sd':>9} | verdict")

        best = None
        records: List[dict] = []
        grid = [(k, dt) for k in kicks for dt in dts]
        for cell_idx, (kick, dt) in enumerate(grid):
            ps, cs = [], []
            for run_idx in range(args.runs):
                r = _trial(
                    trainer, u0x, u0y,
                    kick_us=float(kick), dt_ms=float(dt), hold_us=float(args.hold),
                    sign=float(args.sign), axis=axis, vec=vec,
                )
                dx, dy = float(r["dx_cnt"]), float(r["dy_cnt"])
                if proj_ax == "dx":
                    ps.append(dx); cs.append(dy)
                    pv, cv = dx, dy
                else:
                    ps.append(dy); cs.append(dx)
                    pv, cv = dy, dx
                records.append({
                    "cell_idx": cell_idx, "run_idx": run_idx,
                    "mode": args.mode, "sign": float(args.sign),
                    "kick_us": float(kick), "hold_us": float(args.hold), "dt_ms": float(dt),
                    "proj_ax": proj_ax, "proj_cnt": pv, "cross_cnt": cv,
                })
            mp = statistics.mean(ps)
            sp = statistics.stdev(ps) if len(ps) >= 2 else float("nan")
            mc = statistics.mean(cs)
            sc = statistics.stdev(cs) if len(cs) >= 2 else float("nan")
            ok = _is_pass(mp, sp)
            print(
                f"{kick:+6.0f} {dt:7.1f} | {mp:+10.2f} {sp:8.2f} {min(ps):+9.2f} {max(ps):+9.2f} | "
                f"{mc:+11.2f} {sc:9.2f} | {'PASS' if ok else 'fail'}"
            )
            if ok:
                score = (abs(mp) - 3.5) ** 2 + sp ** 2
                if best is None or score < best[0]:
                    best = (score, kick, dt, mp, sp)

        # 逐发原始数据落盘（论文级证据：同会话重复性 / 重尾分布分析必需）
        try:
            os.makedirs("experiments", exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            tag = f"{args.mode}_{args.sign:+.0f}".replace("+", "p").replace("-", "m")
            csv_path = f"experiments/b1_2d_sweep_{tag}_{stamp}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                wr = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                wr.writeheader()
                wr.writerows(records)
            print(f"[OK] 逐发明细 CSV -> {csv_path}  ({len(records)} 发)")
        except Exception as exc:
            print(f"[WARN] 写 CSV 失败: {exc}")

        print("-" * 100)
        if best is None:
            print("[结论] 本配置在 kick x dt 网格内无 PASS 格。需要同时调 --hold 再扫。")
            return 1
        _, bk, bdt, bmp, bsp = best
        print(f"[结论 PASS] 最佳 kick={bk:+.0f}us dt={bdt:.1f} ms -> proj={bmp:+.2f} +- {bsp:.2f} cnt"
              f"（判据 {PASS_MEAN_LO}~{PASS_MEAN_HI} cnt, sd<={PASS_STD_MAX}）")
        return 0
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
    raise SystemExit(main())
