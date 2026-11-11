import time
import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.ilc.params['target_table'] = {}
ctrl.sensor.start()

print("=" * 72)
print("  TESTING PREDICTIVE BRAKING WITH active_boost_limit_ms = 46.0ms")
print("=" * 72)

# Override in flick_shot by monkeypatching or editing
for i in range(1, 8):
    ctrl.driver.center(settle_s=0.35, preload=True)
    res = ctrl.flick_shot(30, 20, timeout_ms=100.0, enable_ilc_update=False)
    fd = res["final_delta"]
    err = res["final_error_counts"]
    ton = res["t_onset_ms"]
    tb = res["t_brake_ms"]
    print(f"Shot {i}: final={fd}, err={err:4.2f} cnt, t_onset={ton:4.1f}ms, t_brake={tb:4.1f}ms, flight={tb-ton:4.1f}ms")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()
