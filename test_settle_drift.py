import time
from mouse_sensor import HighPrecisionMouseSensor
from servo_driver import WS63ServoDriver

drv = WS63ServoDriver()
sensor = HighPrecisionMouseSensor()
sensor.start()

print("Centering with preload...")
drv.center(settle_s=0.45, preload=True)

print("Monitoring mouse for 1.0 second after center...")
t0 = time.time()
while time.time() - t0 < 1.0:
    time.sleep(0.05)
    dx, dy = sensor.read_delta_and_clear()
    if dx != 0 or dy != 0:
        print(f"  t={time.time()-t0:.2f}s residual drift: dx={dx}, dy={dy}")

sensor.stop()
drv.close()

