import time
from mouse_sensor import HighPrecisionMouseSensor
from servo_driver import WS63ServoDriver

drv = WS63ServoDriver()
sensor = HighPrecisionMouseSensor()
sensor.start()

for s_time in [0.45, 0.55, 0.65, 0.75]:
    print(f"\n--- Testing settle_s = {s_time} ---")
    drv.center(settle_s=s_time, preload=True)
    sensor.read_delta_and_clear()
    t0 = time.time()
    drift = [0, 0]
    while time.time() - t0 < 0.3:
        time.sleep(0.02)
        dx, dy = sensor.read_delta_and_clear()
        drift[0] += dx
        drift[1] += dy
    print(f"Residual drift in 0.3s after center({s_time}): dx={drift[0]}, dy={drift[1]}")

sensor.stop()
drv.close()

