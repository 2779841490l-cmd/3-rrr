r"""
test_ws63_alive.py — 判定 WS63 固件是否还活着（串口回读 + 复位抓开机横幅）。

为什么必须要它：
    servo_driver.send_raw_pulses() 是"只写不读"（servo_driver.py:133 只 ser.write，
    没有回执读取）。所以"写成功"完全不能证明舵机动了 —— 对已枚举但固件挂死、
    或舵机独立供电掉电的板子，写操作照样返回成功。
    但固件在 servo_test_task 启动时会 osal_printk 打印开机横幅
    （firmware_ws63/servo_test.c:196-204），这一路是 UART0 -> CH340 -> COM4 可读的。
    因此：只要能从 COM4 读回任何字节，就证明 WS63 与固件都活着；
    若复位后长时间一个字节都读不到，则 WS63/固件侧死了（或 USB 只供了 TTL 信号未供板电）。

判读表：
    [VERDICT] 读到开机横幅            => WS63 活着，问题在"传感器"或"舵机独立供电"这一侧
    [VERDICT] 复位后仍 0 字节          => WS63 没跑 / 没供电 / UART 只连了 TX 没连 RX
    [VERDICT] 复位后读到乱码但不含横幅 => 波特率或固件不符（可能被刷成了别的固件）

注意：本脚本会拉 DTR/RTS 触发一次复位。若板子接线支持，复位后固件会重新初始化
      四路 PWM 并回中立位 —— 平台会有一次可见的抽动。请盯着平台看。
"""
import time

import serial

from servo_driver import WS63ServoDriver

PORT = WS63ServoDriver.auto_detect_port()
BAUD = 115200


def drain(ser: serial.Serial, seconds: float, label: str) -> bytes:
    """在 seconds 秒内把所有可读字节攒起来并打印。"""
    t0 = time.time()
    chunks = []
    while time.time() - t0 < seconds:
        n = ser.in_waiting
        if n:
            chunks.append(ser.read(n))
        else:
            time.sleep(0.02)
    data = b"".join(chunks)
    print(f"  [{label:<18}] {len(data):4d} bytes")
    if data:
        txt = data.decode("utf-8", "replace")
        txt = txt.replace("\r\n", "\n").replace("\r", "\n")
        for line in txt.split("\n"):
            if line.strip():
                print(f"      | {line}")
    return data


def main() -> int:
    print("=" * 84)
    print(f"  WS63 liveness probe | PORT = {PORT} | BAUD = {BAUD}")
    print("=" * 84)

    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.2)
    except Exception as e:
        print(f"  [FATAL] 打不开 {PORT}: {e}")
        print("  => CH340 不在了：换 USB 口 / 换线 / 检查设备管理器")
        print("=" * 84)
        return 2

    try:
        # ---- 1) 被动听：板子若刚上电/刚复位，横幅还在缓冲里 ----
        print("\n[1] 被动监听 1.5 s（板子若刚上电，横幅会在这里）")
        d1 = drain(ser, 1.5, "passive-1.5s")

        # ---- 2) 发一条合法 P 指令，看固件是否回吐任何东西 ----
        print("\n[2] 发送 'P 1525 1545 1455 1500' 并监听 1.2 s")
        ser.write(b"P 1525 1545 1455 1500\n")
        ser.flush()
        d2 = drain(ser, 1.2, "after-P")

        # ---- 3) 拉 RTS 触发复位，抓开机横幅 ----
        print("\n[3] 拉 RTS/DTR 触发一次复位，随后监听 4 s")
        print("    >>> 请现在盯住平台：若它抽动一下，说明舵机是有电的 <<<")
        try:
            ser.dtr = False
            ser.rts = True
            time.sleep(0.20)
            ser.rts = False
            ser.dtr = True
        except Exception as e:
            print(f"    (DTR/RTS 不可用: {e})")
        d3 = drain(ser, 4.0, "after-reset-4s")

        # ---- 4) 复位后显式发 'C'（回中立位），再听 ----
        print("\n[4] 发送 'C'（全轴回中立位）并监听 1.5 s")
        ser.write(b"C\n")
        ser.flush()
        d4 = drain(ser, 1.5, "after-C")

        total = d1 + d2 + d3 + d4
        print("-" * 84)
        print(f"  总回读字节数 = {len(total)}")
        has_banner = b"BearPi" in total or b"H3863" in total or b"WS63" in total
        print("-" * 84)
        if has_banner:
            print("  [VERDICT] 读到开机横幅 => WS63 固件活着（UART 双向通）。")
            print('            那么"平台不动"只能是：舵机独立供电掉了，或传感器侧死了。')
        elif len(total) > 0:
            print("  [VERDICT] 有回读但不是已知横幅 => 波特率/固件不符，或板子在刷别的固件。")
            print("            检查是否有别的串口工具占着 COM4，或固件被覆盖。")
        else:
            print("  [VERDICT] 复位前后都是 0 字节 => WS63 侧没在跑（或 UART 只有 TX 没接 RX）。")
            print("            => 优先查：板子 5V/3.3V 供电、USB 线是否只是充电线、")
            print("               WS63 是否被刷坏、COM4 是否被别的程序独占。")
        print("=" * 84)
        return 0 if has_banner else 1
    finally:
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
