import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.ilc.params['target_table'] = {}
ctrl.sensor.start()

print("Testing pure X moves with S2=-8.2us (+X) and S2=-3.5us (-X)...")
for target_x in [35, -35]:
    for i in range(2):
        ctrl.driver.center(settle_s=0.40, preload=True)
        res = ctrl.flick_shot(target_x, 0, timeout_ms=100.0, enable_ilc_update=False)
        fd = res["final_delta"]
        err = res["final_error_counts"]
        print(f"Target ({target_x:+3d}, 0) Shot {i+1} -> Final ({fd[0]:+3d}, {fd[1]:+3d}) | Err {err:4.1f} cnt")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()
