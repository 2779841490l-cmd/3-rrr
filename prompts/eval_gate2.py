import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.ilc.params["target_table"] = {}
ctrl.sensor.start()

directions = [(35, 0), (-35, 0), (0, 35), (0, -35), (30, 25), (-30, -25)]
errors = []

print("Running Gate 2 Test on 6 directions...")
for dx, dy in directions:
    for w in range(6):
        ctrl.driver.center(settle_s=0.38, preload=True)
        ctrl.flick_shot(dx, dy, timeout_ms=100.0, enable_ilc_update=True)

    ev_errs = []
    ev_finals = []
    for ev in range(2):
        ctrl.driver.center(settle_s=0.38, preload=True)
        res = ctrl.flick_shot(dx, dy, timeout_ms=100.0, enable_ilc_update=False)
        ev_errs.append(res["final_error_counts"])
        ev_finals.append(res["final_delta"])

    mean_err = float(np.mean(ev_errs))
    errors.append(mean_err)
    corr = ctrl.ilc.get_correction(dx, dy)
    status = "PASS" if mean_err <= 3.0 else "FAIL"
    print(f"[{status}] ({dx:+3d}, {dy:+3d}) -> Finals: {ev_finals[0]}, {ev_finals[1]} | Mean: {mean_err:4.2f} cnt | ILC_u: [{corr[0]:+5.1f}, {corr[1]:+5.1f}]")

overall = float(np.mean(errors))
status = "PASS" if overall <= 2.0 and all(e <= 3.0 for e in errors) else "FAIL"
print(f"\n[{status}] Overall Mean Error: {overall:.2f} cnt ({overall*7.9375:.1f} um)")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()
