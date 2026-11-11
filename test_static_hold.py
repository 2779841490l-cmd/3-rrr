import time
from mouse_sensor import HighPrecisionMouseSensor
from servo_driver import WS63ServoDriver

drv = WS63ServoDriver()
sensor = HighPrecisionMouseSensor()
sensor.start()

print("Testing theoretical Jacobian holding positions for (30, 20)...")
for u1 in [1528.0, 1530.0, 1532.0, 1534.0]:
    for u2 in [1524.0, 1526.0, 1528.0]:
        drv.center(settle_s=0.40, preload=True)
        sensor.read_delta_and_clear()
        x0, y0 = sensor.get_position()
        drv.send_raw_pulses(u1, u2)
        time.sleep(0.12)
        fx, fy = sensor.get_position()
        print(f"u1={u1:.1f}, u2={u2:.1f} -> Held Pos: ({fx-x0:+2d}, {fy-y0:+2d})")

drv.center(settle_s=0.2, preload=True)
sensor.stop()
drv.close()
