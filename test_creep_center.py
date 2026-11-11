import time
import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.ilc.params["target_table"] = {}
ctrl.sensor.start()

print("=" * 72)
print("  GATE 1: 12 CONSECUTIVE SHOTS ON (+30, +20) WITH DETERMINISTIC PRELOAD")
print("=" * 72)

errors = []
for i in range(1, 13):
    ctrl.driver.center(settle_s=0.42, preload=True)
    res = ctrl.flick_shot(30, 20, timeout_ms=100.0, enable_ilc_update=True)
    fd = res["final_delta"]
    err = res["final_error_counts"]
    tb = res["t_brake_ms"]
    errors.append(err)
    corr = ctrl.ilc.get_correction(30, 20)
    print(f"  Shot {i:2d}/12 | Final ({fd[0]:+3d}, {fd[1]:+3d}) | Err: {err:4.2f} cnt ({err*7.9375:4.1f} um) | Brake: {tb:4.1f}ms | ILC: [{corr[0]:+5.1f}, {corr[1]:+5.1f}]")

last5 = errors[-5:]
mean_l5 = np.mean(last5)
max_l5 = np.max(last5)
print("\n" + "=" * 72)
print(f"Gate 1 Summary: Last-5 Mean = {mean_l5:.2f} cnt (<=2.50?), Last-5 Max = {max_l5:.2f} cnt (<=4.00?)")
print(f"Status: {'PASS' if mean_l5 <= 2.5 and max_l5 <= 4.0 else 'FAIL'}")
print("=" * 72)

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()
