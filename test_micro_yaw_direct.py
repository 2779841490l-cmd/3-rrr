import time
from mouse_sensor import HighPrecisionMouseSensor
from servo_driver import WS63ServoDriver

drv = WS63ServoDriver()
sensor = HighPrecisionMouseSensor()
sensor.start()

print("Testing Micro-Yaw Differential Displacement...")
# S2 (GPIO 5) is Y1, S3 (GPIO 7) is Y2.
# In pure translation: s2=1545, s3=1455 (sum=3000).
# Differential micro-yaw: s2 = 1545 + delta/2, s3 = 1455 - delta/2 (sum=3000 always!)
for dyaw in [-16.0, -8.0, 0.0, 8.0, 16.0]:
    drv.center(settle_s=0.40, preload=True)
    sensor.read_delta_and_clear()
    x0, y0 = sensor.get_position()
    
    s2 = 1545.0 + dyaw / 2.0
    s3 = 1455.0 - dyaw / 2.0
    drv.send_raw_pulses(1510.0, s2, s3, 1500.0)
    time.sleep(0.12)
    fx, fy = sensor.get_position()
    print(f"dyaw={dyaw:+5.1f}us (s2={s2:.1f}, s3={s3:.1f}) -> Shift: dx={fx-x0:+2d}, dy={fy-y0:+2d}")

drv.center(settle_s=0.2, preload=True)
sensor.stop()
drv.close()

