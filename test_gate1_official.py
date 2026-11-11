import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.ilc.params["target_table"] = {}
ctrl.ilc.save()

ctrl.sensor.start()

tx, ty = 30, 20
gate1_errors = []
print("Testing Gate 1 (12 shots) with baseline du_base=[30.0, -17.2]...")
for i in range(1, 13):
    ctrl.driver.center(settle_s=0.50, preload=True)
    res = ctrl.flick_shot(tx, ty, timeout_ms=100.0, enable_ilc_update=True)
    fd = res["final_delta"]
    err = res["final_error_counts"]
    gate1_errors.append(err)
    corr = ctrl.ilc.get_correction(tx, ty)
    print(f"Shot {i:2d}/12 -> Final ({fd[0]:+3d}, {fd[1]:+3d}) | Err: {err:4.2f} cnt | Onset: {res['t_onset_ms']:4.1f}ms | Brake: {res['t_brake_ms']:4.1f}ms | ILC: [{corr[0]:+5.2f}, {corr[1]:+5.2f}]")

last5 = gate1_errors[-5:]
mean_last5 = float(np.mean(last5))
max_last5 = float(np.max(last5))
print(f"\nGate 1 Summary: Last-5 Mean = {mean_last5:.2f} cnt (limit <= 2.50), Last-5 Max = {max_last5:.2f} cnt (limit <= 4.00)")
if mean_last5 <= 2.50 and max_last5 <= 4.00:
    print(">>> GATE 1 PASSED! <<<")
else:
    print(">>> GATE 1 FAILED! <<<")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()
