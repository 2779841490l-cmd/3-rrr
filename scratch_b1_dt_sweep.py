r"""scratch_b1_dt_sweep.py — 判定 B1 的「限时 kick」到底有没有真正起作用。

【起因】
scratch_brake_trace.py 实测：45us 阶跃下发后 0~25 ms 完全零位移，37 ms 才 +1 cnt，
54 ms 才 +11 cnt ⇒ 执行器起始潜伏期约 35 ms（固件侧已排除：每帧结束 50us 切片
轮询 g_cmd_updated，新命令最迟下一帧 ~5 ms 生效；servo_test.c 主体无斜坡限速）。

而 probe_timed_kick.py 的 PASS 配方是 kick=45us / dt=20 ms / hold=-12us。
20 ms < 35 ms ⇒ kick 相很可能在运动发生前就被撤掉，真正决定落点的是随后一直保持到
稳态的 hold=-12us 静态平衡。

【本脚本】
固定 kick_us / hold_us，只扫 dt_ms ∈ {0,5,10,20,35,50,70}，
外加一列 kick=0（纯静态 -12us，不经 kick）作对照，各 n=REPEAT 次。

判据读法：
  - 若各 dt 的 mean 无显著差异（且与 kick=0 列一致）⇒ B1 的「时序截断」是伪机制，
    落点纯由 hold 的静态平衡决定 ⇒ 必须按「预载 + 静态微偏置」来设计，绝不能声称
    是时域控制。
  - 若 dt 大时 mean 显著变大 ⇒ kick 真的落地了，时序截断有效，可以放心用。

只动 S1（axis='x'），单方向，安全。总时长约 2~3 分钟。
"""

import statistics
import time
from typing import List

from probe_timed_kick import _trial
from rect_manifold_calibrator import HardwareRectangleTrainer

KICK_US = 45.0
HOLD_US = -12.0
REPEAT = 3
DT_LIST: List[float] = [0.0, 5.0, 10.0, 20.0, 35.0, 50.0, 70.0]


def _stat(vals: List[float]):
    if not vals:
        return float("nan"), float("nan")
    m = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) >= 2 else float("nan")
    return m, sd


def main() -> int:
    trainer = HardwareRectangleTrainer()
    try:
        trainer.sensor.start()   # 关键：HardwareRectangleTrainer.__init__ 不启动传感器线程
        time.sleep(0.30)
        trainer.sensor.reset_origin()
        trainer.move_to_rect_origin(settle_s=0.45)

        u0 = trainer.manifold.evaluate_kinematic_pulse(0.0, 0.0)
        u0x, u0y = float(u0[0]), float(u0[1])
        print("=" * 92)
        print(f"u0 = ({u0x:.2f}, {u0y:.2f})   kick={KICK_US:+.1f}us  hold={HOLD_US:+.1f}us  n={REPEAT}")
        print("=" * 92)
        print(f"{'kick':>7} {'dt_ms':>7} | {'dx_mean':>9} {'dx_sd':>7} {'dx_min':>8} {'dx_max':>8} | {'dy_mean':>9}")

        rows = []
        for kick in (KICK_US, 0.0):
            for dt in DT_LIST:
                if kick == 0.0 and dt != 0.0:
                    continue
                xs, ys = [], []
                for _ in range(REPEAT):
                    r = _trial(
                        trainer, u0x, u0y,
                        kick_us=float(kick), dt_ms=float(dt), hold_us=HOLD_US,
                        sign=1.0, axis="x", vec=None,
                    )
                    xs.append(float(r["dx_cnt"]))
                    ys.append(float(r["dy_cnt"]))
                mx, sx = _stat(xs)
                my, _sy = _stat(ys)
                tag = "static" if kick == 0.0 else f"{kick:+.0f}"
                print(
                    f"{tag:>7} {dt:7.1f} | {mx:+9.2f} {sx:7.2f} {min(xs):+8.2f} {max(xs):+8.2f} | {my:+9.2f}"
                )
                rows.append((tag, dt, mx, sx, my))

        print("-" * 92)
        base = next((r for r in rows if r[0] == "static"), None)
        if base is not None:
            print(f"[对照] 纯静态 hold={HOLD_US:+.1f}us（无 kick）: dx={base[2]:+.2f} ± {base[3]:.2f} cnt")
        spread = [r[2] for r in rows if r[0] != "static"]
        if spread:
            print(f"[kick 列] dt 从 {min(DT_LIST):.0f}→70 ms，dx_mean 范围 "
                  f"{min(spread):+.2f} ~ {max(spread):+.2f} cnt，跨度 {max(spread) - min(spread):.2f} cnt")
            if max(spread) - min(spread) <= 2.0:
                print("[结论 DT-INDEPENDENT] dt 几乎不改变落点 ⇒ B1 的『时序截断』不是真机制，")
                print("                      落点由 hold 的静态平衡决定（预载 + 静态微偏置）。")
            else:
                print("[结论 DT-EFFECTIVE] dt 显著改变落点 ⇒ kick 相真实落地，时序截断有效。")
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
