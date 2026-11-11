import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
# Override poly_jac coeffs with exact center values
ctrl.poly_jac.coeffs["K11"]["Const"] = 1.03823
ctrl.poly_jac.coeffs["K12"]["Const"] = -1.04691
ctrl.poly_jac.coeffs["K21"]["Const"] = -0.13882
ctrl.poly_jac.coeffs["K22"]["Const"] = -0.73746

ctrl.ilc.params["target_table"] = {}
ctrl.sensor.start()
ctrl.driver.center(settle_s=0.45, preload=True)

print("Tracing (0, 35) with exact center Jacobian...")
for i in range(5):
    ctrl.driver.center(settle_s=0.38, preload=True)
    res = ctrl.flick_shot(0, 35, timeout_ms=100.0, enable_ilc_update=True)
    corr = ctrl.ilc.get_correction(0, 35)
    fd = res["final_delta"]
    err = res["final_error_counts"]
    print(f"Shot {i+1}: Final ({fd[0]:+3d}, {fd[1]:+3d}) | Err {err:4.1f} cnt | ILC_u: [{corr[0]:+5.1f}, {corr[1]:+5.1f}]")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()
