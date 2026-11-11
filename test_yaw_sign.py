import time
import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.sensor.start()

print("Testing differential yaw kick response from neutral:")
for yk in [-10.0, 0.0, 10.0]:
    ctrl.driver.center(settle_s=0.35, preload=True)
    ctrl.sensor.read_delta_and_clear()
    x0, y0 = ctrl.sensor.get_position()
    
    # Apply differential yaw: S2 = neutral - yk, S3 = neutral + yk
    u2 = ctrl.driver.NEUTRAL_S2 - yk
    u3 = ctrl.driver.NEUTRAL_S3 + yk
    ctrl.driver.send_raw_pulses(ctrl.driver.NEUTRAL_S1, u2, s3=u3)
    time.sleep(0.08)
    
    x1, y1 = ctrl.sensor.get_position()
    print(f"yaw_kick = {yk:+5.1f} us -> Shift: ({x1-x0:+3d}, {y1-y0:+3d})")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

