import time
import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.sensor.start()

print("Testing Micro-Yaw Trim on (+30, +20):")
for shot in range(4):
    ctrl.driver.center(settle_s=0.35, preload=True)
    ctrl.sensor.read_delta_and_clear()
    x0, y0 = ctrl.sensor.get_position()
    
    # Fire shot using controller
    res = ctrl.flick_shot(30, 20, timeout_ms=100.0, enable_ilc_update=False)
    fd = res['final_delta']
    err = res['final_error_counts']
    print(f"Shot {shot+1}: Final {fd} | Err: {err:.2f} cnt ({err*7.9375:.1f} um)")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

