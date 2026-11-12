r"""scratch_brake_trace.py — 单发时间线追踪：45us kick 后逐毫秒打印真实位移。

目的：定位 probe_target_brake.py 中「40ms 内走 19 counts，但轮询声称 proj 始终 < 2」的矛盾。
只发一次脉冲，60 ms 后立刻撤驱动，安全。
"""

import time

import numpy as np

from rect_manifold_calibrator import HardwareRectangleTrainer

KICK_US = 45.0
PRELOAD_US = 35.0
TRACE_MS = 60.0


def main():
    trainer = HardwareRectangleTrainer()
    try:
        trainer.sensor.start()
        time.sleep(0.30)
        trainer.sensor.reset_origin()

        u0 = np.array(trainer.manifold.evaluate_kinematic_pulse(0.0, 0.0), dtype=float)
        print(f"u0 = {u0}")

        trainer.move_to_rect_origin(settle_s=0.45)
        print(f"after origin: rect={trainer.get_rect_position()}  total={trainer.sensor.get_position()}")

        # 预载
        trainer.driver.send_raw_pulses(float(u0[0] - PRELOAD_US), float(u0[1]))
        time.sleep(0.22)
        trainer.driver.send_raw_pulses(float(u0[0]), float(u0[1]))
        time.sleep(0.40)

        p0 = np.array(trainer.get_rect_position(), dtype=float)
        print(f"p0 = {p0}   origin_sensor_xy={trainer.origin_sensor_xy}")

        # kick
        t0 = time.perf_counter()
        trainer.driver.send_raw_pulses(float(u0[0] + KICK_US), float(u0[1]))
        t_send = (time.perf_counter() - t0) * 1000.0
        print(f"-- send_raw_pulses 耗时 {t_send:.2f} ms --")
        print("   t_ms    raw_total        rect          proj_x   d_proj")

        prev = 0.0
        n_reads = 0
        while True:
            el = (time.perf_counter() - t0) * 1000.0
            if el >= TRACE_MS:
                break
            p = np.array(trainer.get_rect_position(), dtype=float)
            tot = trainer.sensor.get_position()
            proj = float(p[0] - p0[0])
            n_reads += 1
            if n_reads <= 8 or (proj - prev) >= 1.0 or n_reads % 8 == 0:
                print(f"  {el:6.2f}  {str(tot):>15s}  {str(p):>18s}  {proj:+7.2f}  {proj - prev:+6.2f}")
                prev = proj
            time.sleep(0.001)

        trainer.driver.send_raw_pulses(float(u0[0]), float(u0[1]))
        time.sleep(0.50)
        p1 = np.array(trainer.get_rect_position(), dtype=float)
        print(f"final rect = {p1}   delta = {p1 - p0}   总轮询次数={n_reads}")
        print(f"event_count={getattr(trainer.sensor, '_event_count', -1)}")
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
