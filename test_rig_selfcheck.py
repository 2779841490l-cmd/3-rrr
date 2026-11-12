r"""
test_rig_selfcheck.py — 一条命令跑完的"执行器 + 传感器"端到端自检（读传感器内部计数器）。

## 为什么必须读内部计数器

`get_rect_position()` 返回的是 `sensor.get_position() - origin_sensor_xy`，
而这个差在"累加器冻结"和"平台真的没动"两种情况下都是 (0,0)，无法区分。
本脚本直接拆开传感器的三个内部量：

    _event_count      收到的 HID 位移事件包个数（哪怕 dx=dy=0 也计）
    _total_x/_total_y 通过设备过滤器、真正累加进去的原始计数
    rejected_counts   位移到了但被设备过滤器拒掉的计数

于是四种结果完全可判：

    A) rejected > 0 且 total 未变   => 过滤器绑错了。位移本来就到了，被 mouse_device.txt 拒掉。
    B) total 变了                    => 传感器活着、平台确实在动。上一次是偶发掉线，直接重跑实验。
    C) total 未变 且 events > 0      => 鼠标在线但光学面零相对位移 => 平台真的没动
                                        => 执行器侧问题（舵机独立供电 / PWM 无输出）。
    D) total 未变 且 events == 0     => 一个 HID 事件都没来 => Raw Input 未绑定 / 鼠标掉线。

## 动作

安全范围内做 4 轮 X 与 4 轮 Y 的大幅摆动（±60 us，远大于死区 6~14 us），
每轮 0.6 s 保持 + 采样，全程监控。脉冲硬限幅 [1150,1850] 由驱动层保证。
"""
import time

import numpy as np
import serial.tools.list_ports

from mouse_sensor import HighPrecisionMouseSensor
from servo_driver import WS63ServoDriver

SWING_US = 60.0
HOLD_S = 0.6


def snapshot(sen: HighPrecisionMouseSensor):
    with sen._lock:
        return dict(
            events=sen._event_count,
            tx=sen._total_x,
            ty=sen._total_y,
            rejected=sen.rejected_counts,
        )


def main() -> int:
    print("=" * 88)
    print("  3-RRR RIG SELF-CHECK · 执行器 vs 传感器 端到端判别")
    print("=" * 88)

    # ---------- Step 0: 串口 ----------
    ports = [p for p in serial.tools.list_ports.comports()]
    ch340 = [p for p in ports if "1A86" in (p.hwid or "").upper() or "CH340" in (p.description or "").upper()]
    print(f"\n[0] 串口枚举: {len(ports)} 个; CH340: {[p.device for p in ch340] or '无'}")
    if not ch340:
        print("  [FATAL] 没有 CH340 => USB 线/口问题，后面都不用测了。")
        return 2

    sen = HighPrecisionMouseSensor(dpi=3200)
    dev_filter = sen.device_filter
    print(f"    传感器设备过滤器 = {dev_filter!r}  ({'已绑定' if dev_filter else '空 -> 接受所有鼠标'})")
    sen.start()
    time.sleep(0.35)

    driver = None
    try:
        driver = WS63ServoDriver()
        print(f"    驱动目标端口 = {driver.port}")

        base = snapshot(sen)
        print(f"    基线: events={base['events']} total=({base['tx']},{base['ty']}) rejected={base['rejected']}")

        # ---------- Step 1: X 轴大幅摆动 ----------
        print(f"\n[1] X 轴 (S1) ±{SWING_US:.0f} us 摆动 4 轮，每轮保持 {HOLD_S}s")
        for i in range(4):
            for sign in (+1.0, -1.0):
                driver.send_raw_pulses(driver.NEUTRAL_S1 + sign * SWING_US, driver.NEUTRAL_S2)
                time.sleep(HOLD_S)
                s = snapshot(sen)
                print(f"      X r{i+1}{'+' if sign > 0 else '-'} -> total=({s['tx']:+6d},{s['ty']:+6d}) "
                      f"events={s['events']:5d} rejected={s['rejected']:6d}")

        # ---------- Step 2: Y 轴大幅摆动 ----------
        print(f"\n[2] Y 轴 (S2/S3) ±{SWING_US:.0f} us 摆动 4 轮")
        for i in range(4):
            for sign in (+1.0, -1.0):
                driver.send_raw_pulses(driver.NEUTRAL_S1, driver.NEUTRAL_S2 + sign * SWING_US)
                time.sleep(HOLD_S)
                s = snapshot(sen)
                print(f"      Y r{i+1}{'+' if sign > 0 else '-'} -> total=({s['tx']:+6d},{s['ty']:+6d}) "
                      f"events={s['events']:5d} rejected={s['rejected']:6d}")

        end = snapshot(sen)
        d_total = int(np.hypot(end["tx"] - base["tx"], end["ty"] - base["ty"]))
        d_rej = end["rejected"] - base["rejected"]
        d_evt = end["events"] - base["events"]

        print("\n" + "-" * 88)
        print(f"  全程增量 |Δtotal| = {d_total} cnt | Δrejected = {d_rej} | Δevents = {d_evt}")
        if sen.per_device:
            print("  各设备位移累计（诊断用，含被拒的）:")
            for nm, c in sorted(sen.per_device.items(), key=lambda kv: -kv[1]):
                print(f"      {c:8d}  {nm}")
        print("-" * 88)

        if d_rej > 0 and d_total == 0:
            print("  [VERDICT A] 过滤器绑错：位移到了却被拒掉。")
            print("              => 改写 mouse_device.txt 为当前在线设备的 VID/PID 子串。")
            rc = 1
        elif d_total > 0:
            print("  [VERDICT B] 传感器活着且平台确实在动 —— 硬件完全正常。")
            print("              => 上一次 B 组(1/20 命中、final 全 (0,0)) 是偶发掉线，直接重跑即可。")
            rc = 0
        elif d_evt > 0:
            print("  [VERDICT C] HID 事件在来，但光学面零相对位移 => 平台真的没动。")
            print("              => 执行器侧：舵机独立供电掉电 / PWM 无输出 / 联杆脱开。")
            print("                 请查舵机 5~6V 电源、WS63 是否还在跑本固件。")
            rc = 1
        else:
            print("  [VERDICT D] 一个 HID 事件都没来 => Raw Input 未绑定 / 鼠标掉线。")
            print("              => 重插 Razer DeathAdder（当前应为 HID\\VID_1532&PID_0098&MI_00\\7&49C0815）。")
            rc = 1
        print("=" * 88)
        return rc

    finally:
        if driver is not None:
            try:
                driver.center(0.3)
                driver.close()
            except Exception:
                pass
        try:
            sen.stop()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
