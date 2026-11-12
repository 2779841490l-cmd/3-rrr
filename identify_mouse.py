"""
identify_mouse.py — 找出机构鼠标的设备名，写入 mouse_device.txt
用法:
  python identify_mouse.py          # 被动：10 秒内列出每个鼠标设备的累计位移（你可手推机构或动自己的鼠标）
  python identify_mouse.py --jog    # 主动：让 S1 在中立位 ±25us 来回 6 次，位移最大的设备即机构鼠标
  python identify_mouse.py --save <子串>   # 手动写入过滤子串
"""
import os
import re
import sys
import time

import mouse_sensor
from mouse_sensor import HighPrecisionMouseSensor

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), mouse_sensor.DEVICE_FILTER_FILE)


def vidpid(name: str) -> str:
    m = re.search(r"VID_[0-9A-F]{4}&PID_[0-9A-F]{4}", name.upper())
    return m.group(0) if m else name


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--save":
        open(OUT, "w", encoding="utf-8").write(sys.argv[2])
        print(f"saved filter '{sys.argv[2]}' -> {OUT}")
        return

    if os.path.exists(OUT):
        os.rename(OUT, OUT + ".bak")      # 识别期间需接受全部设备
    s = HighPrecisionMouseSensor(dpi=3200)
    s.start()
    try:
        if "--jog" in sys.argv:
            from servo_driver import WS63ServoDriver
            d = WS63ServoDriver()
            d.set_pwm_frequency(200)
            d.center(settle_s=0.4)
            for _ in range(6):
                d.send_raw_pulses(d.NEUTRAL_S1 + 25, d.NEUTRAL_S2)
                time.sleep(0.3)
                d.send_raw_pulses(d.NEUTRAL_S1 - 25, d.NEUTRAL_S2)
                time.sleep(0.3)
            d.center(settle_s=0.4)
            d.close()
        else:
            print("10 秒采集中 ...")
            time.sleep(10)
    finally:
        s.stop()
        if os.path.exists(OUT + ".bak") and not os.path.exists(OUT):
            os.rename(OUT + ".bak", OUT)

    ranking = sorted(s.per_device.items(), key=lambda kv: -kv[1])
    for name, c in ranking:
        print(f"{c:8d} counts  {vidpid(name)}  ({name})")
    if "--jog" in sys.argv and ranking:
        best = vidpid(ranking[0][0])
        open(OUT, "w", encoding="utf-8").write(best)
        print(f"\n机构鼠标 = {best}，已写入 {OUT}")


if __name__ == "__main__":
    main()
