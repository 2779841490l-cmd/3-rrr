from ballistic_controller import Ballistic100msController
import numpy as np

ctrl = Ballistic100msController()
ctrl.ilc.params["target_table"] = {}
ctrl.ilc.save()
ctrl.sensor.start()

print("Testing du_base[0]=23.2, du_base[1]=-14.2 on (30, 20)...")
errs = []
for i in range(1, 11):
    ctrl.driver.center(settle_s=0.45, preload=True)
    # du_base in ballistic_controller is currently [22.0, -16.2]
    # so we add [23.2 - 22.0, -14.2 - (-16.2)] = [+1.2, +2.0]
    ctrl.ilc.params["target_table"]["30_20"] = [1.2, 2.0]
    ctrl.ilc.save()
    res = ctrl.flick_shot(30, 20, timeout_ms=100.0, enable_ilc_update=False)
    err = res['final_error_counts']
    errs.append(err)
    print(f"Shot {i:2d} -> Final {res['final_delta']} | Err: {err:4.2f} | Onset: {res['t_onset_ms']:4.1f}ms | Brake: {res['t_brake_ms']:4.1f}ms")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()
print(f"\nOverall Mean: {np.mean(errs):.2f} cnt, Last-5 Mean: {np.mean(errs[-5:]):.2f} cnt, Last-5 Max: {np.max(errs[-5:]):.2f} cnt")
