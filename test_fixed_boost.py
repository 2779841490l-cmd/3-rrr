import time
import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.ilc.params['target_table'] = {}
ctrl.sensor.start()

print("=" * 72)
print("  TESTING GENTLE BOOST OVERDRIVE (PEAK VELOCITY REGULATION)")
print("=" * 72)

errs = []
for i in range(1, 13):
    ctrl.driver.center(settle_s=0.35, preload=True)
    res = ctrl.flick_shot(30, 20, timeout_ms=100.0, enable_ilc_update=True)
    fd = res["final_delta"]
    err = res["final_error_counts"]
    tb = res["t_brake_ms"]
    errs.append(err)
    print(f"Shot {i:2d}: final=({fd[0]:+3d}, {fd[1]:+3d}), err={err:4.2f} cnt ({err*7.9375:4.1f} um), brake={tb:4.1f}ms")

l5 = errs[-5:]
print("\n" + "=" * 72)
print(f"Last-5 Mean: {np.mean(l5):.2f} cnt (<=2.50?), Last-5 Max: {np.max(l5):.2f} cnt (<=4.00?)")
print(f"Gate 1 Status: {'PASS' if np.mean(l5) <= 2.5 and np.max(l5) <= 4.0 else 'FAIL'}")
print("=" * 72)

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()
