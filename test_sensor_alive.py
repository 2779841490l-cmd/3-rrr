r"""
test_sensor_alive.py — 10 秒存活探针：区分"传感器死了"还是"执行器死了"。

背景：
  2026-11-12 22:21 的 A/B 对照组（--no-metric-w）20 发全部 final=(0.0, 0.0)，
  误差恒等于目标距离本身，ballistic 时长恒为 50.5+48≈98.5 ms —— 这正是
  "get_rect_position() 全程返回 (0,0)" 时 Phase-1 弹道循环超时退出的特征。
  控制器逻辑无异常，是硬件/会话层掉线。本脚本只做三件事：
    1) 读一次位置（应 ≈ (0,0)，因为 init 里 move_to_rect_origin）；
    2) 只发 X 轴 +20 counts 的纯运动学脉冲（动执行器）；
    3) 再读一次位置。
  判读：
    - 第 3 步有明显位移          => 传感器与执行器都好，上一场是偶发掉线；
    - 第 3 步仍为 0 且串口无回显 => 执行器没动（供电/固件/串口）；
    - 第 3 步仍为 0 但肉眼可见平台在动 => 传感器死了（Raw Input 未绑定/鼠标被拔）。
"""

import time

import numpy as np

from rect_manifold_calibrator import HardwareRectangleTrainer


def read_pos(trainer, tag: str):
    p = trainer.get_rect_position()
    print(f"  [{tag}] pos = ({p[0]:+8.2f}, {p[1]:+8.2f}) cnt")
    return np.array(p, dtype=float)


def main():
    print("=" * 78)
    trainer = HardwareRectangleTrainer()
    try:
        # 关键：HardwareRectangleTrainer.__init__ 不启动传感器线程，必须显式 start()
        trainer.sensor.start()
        time.sleep(0.30)
        trainer.sensor.reset_origin()

        print("[1] 回原点并读取基准位置 ...")
        trainer.move_to_rect_origin(settle_s=0.45)
        p0 = read_pos(trainer, "base")

        # 用 60 counts（~31 us 偏移）确保越过 20 us 静默储能区，避免假阴性
        print("[2] 只发一次 X 轴 +60 counts 的纯运动学脉冲 ...")
        u = trainer.manifold.evaluate_kinematic_pulse(60.0, 0.0)
        print(f"      target pulse = (u_x={u[0]:.2f} us, u_y={u[1]:.2f} us)")
        trainer.driver.send_raw_pulses(float(u[0]), float(u[1]))
        time.sleep(0.55)

        p1 = read_pos(trainer, "after +60X")
        d = p1 - p0
        ev = getattr(trainer.sensor, "_event_count", -1)
        pd = dict(getattr(trainer.sensor, "per_device", {}))
        print("-" * 78)
        print(f"  delta = ({d[0]:+8.2f}, {d[1]:+8.2f}) cnt | |delta| = {float(np.hypot(d[0], d[1])):.2f} cnt")
        print(f"  events = {ev}  per_device = {pd}")
        if float(np.hypot(d[0], d[1])) >= 10.0:
            print("  [VERDICT] 传感器 + 执行器都活着（位移 >= 10 cnt）。上一场是偶发掉线，重跑即可。")
        else:
            print("  [VERDICT] 位移 < 10 cnt。请肉眼确认平台是否真的动了：")
            print("            - 平台动了 => 传感器死了（Raw Input 未绑定 / 鼠标被拔 / 换了 USB 口）")
            print("            - 平台没动 => 执行器没动（舵机供电 / WS63 固件 / COM 口）")
        print("=" * 78)
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
    main()
