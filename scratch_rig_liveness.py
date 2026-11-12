r"""scratch_rig_liveness.py — 一次性同时验证「执行器侧」与「传感器侧」是否存活。

设计要点（安全）：
  1. 只用 +200 us / -200 us 的单向大脉冲（相对 u0），脉宽 0.35 s 后**立即归零**，
     绝不长时间满偏置保持 —— 不触发堵转/热保护。
  2. 同时打印 Raw Input 的注册结果、hwnd、event_count、per_device，
     用来区分「传感器没收到事件」与「平台根本没动」。
  3. 不需要任何标定文件写入，不修改任何参数。

用法：
  set PYTHONIOENCODING=utf-8 && chcp 65001 >nul && python scratch_rig_liveness.py
运行期间请：
  - 先**看平台**：+200us 那一步平台应该明显移动约 200~240 counts（1.6~1.9 mm）。
  - 再**用手动一下真实鼠标**：per_device 里应出现设备名。
"""

import time

import numpy as np

from rect_manifold_calibrator import HardwareRectangleTrainer


def _dump(trainer, tag: str):
    s = trainer.sensor
    p = s.get_position()
    ev = getattr(s, "_event_count", -1)
    rej = getattr(s, "rejected_counts", -1)
    pd = dict(getattr(s, "per_device", {}))
    hwnd = getattr(s, "_hwnd", None)
    print(
        f"  [{tag:<10}] pos=({p[0]:+5d},{p[1]:+5d}) cnt | events={ev:<4d} "
        f"rejected={rej:<4d} hwnd={hwnd} per_device={pd}"
    )
    return p


def main():
    print("=" * 92)
    print("  机构存活双向探针：执行器 (+-200 us 大脉冲) + 传感器 (Raw Input 事件计数)")
    print("=" * 92)

    trainer = HardwareRectangleTrainer()
    try:
        # 关键：HardwareRectangleTrainer.__init__ 不启动传感器线程，必须显式 start()
        trainer.sensor.start()
        time.sleep(0.30)
        trainer.sensor.reset_origin()

        drv = trainer.driver
        print(f"  串口端口        = {getattr(drv, 'port', '?')}")
        print(f"  device_filter   = {trainer.sensor.device_filter!r}")
        u0 = np.array(trainer.manifold.evaluate_kinematic_pulse(0.0, 0.0), dtype=float)
        print(f"  u0_kin          = {u0}")

        print("\n[1] 回矩形原点（会先预载 -35us 再回位）...")
        trainer.move_to_rect_origin(settle_s=0.45)
        p_base = _dump(trainer, "base")
        print(f"      get_rect_position = {trainer.get_rect_position()}")

        for sign, name in ((+1.0, "+200us"), (-1.0, "-200us")):
            print(f"\n[2] 下发 u0{'+' if sign > 0 else '-'}200 us（仅保持 0.35 s 后立刻归零）")
            print("    >>> 现在请看平台：它应该明显移动一下 <<<")
            trainer.driver.send_raw_pulses(float(u0[0] + sign * 200.0), float(u0[1]))
            for i in range(5):
                time.sleep(0.07)
                p = trainer.sensor.get_position()
                print(f"      t={(i + 1) * 0.07:.2f}s  pos=({p[0]:+5d},{p[1]:+5d})")
            trainer.driver.send_raw_pulses(float(u0[0]), float(u0[1]))
            time.sleep(0.30)
            p_after = _dump(trainer, name)
            d = np.array(p_after, dtype=float) - np.array(p_base, dtype=float)
            print(f"      delta from base = ({d[0]:+.0f},{d[1]:+.0f})  |d|={np.hypot(*d):.1f} cnt")
            p_base = p_after

        print("\n[3] 请你**现在用手移动一下真实鼠标**（5 秒），检查 Raw Input 是否活着")
        for i in range(5):
            time.sleep(1.0)
            _dump(trainer, f"hover{i + 1}")

        print("\n" + "-" * 92)
        print("  判读表：")
        print("   - 平台明显动了 且 |delta| >= 100 cnt  => 执行器 OK + 传感器 OK")
        print("   - 平台明显动了 但 |delta| ~ 0          => 传感器/Raw Input 掉线")
        print("   - 平台完全不动                          => 执行器侧掉线（供电/WS63/COM）")
        print("   - per_device 在动鼠标后仍为 {}          => Raw Input 会话层失效，需重开终端/重插鼠标")
        print("-" * 92)
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
