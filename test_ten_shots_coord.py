import time
import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.ilc.params["target_table"] = {}
ctrl.ilc.save()
ctrl.sensor.start()

print("Testing 10 shots on (30, 20) with coordinated kinematic preload...")
# Monkey-patch or use coordinated preload
def coordinated_center(settle_s=0.45):
    ctrl.driver.send_raw_pulses(1510, 1545, 1455, 1500)
    time.sleep(0.10)
    ctrl.driver.send_raw_pulses(1480, 1570, 1430, 1500)
    time.sleep(0.20)
    ctrl.driver.send_raw_pulses(1510, 1545, 1455, 1500)
    time.sleep(settle_s)

errs = []
for i in range(1, 11):
    coordinated_center(0.40)
    # Target (30, 20)
    res = ctrl.flick_shot(30, 20, timeout_ms=100.0, enable_ilc_update=False)
    err = res['final_error_counts']
    errs.append(err)
    print(f"Shot {i:2d} -> Final {res['final_delta']} | Err: {err:4.2f} cnt | Onset: {res['t_onset_ms']}ms | Brake: {res['t_brake_ms']}ms")

coordinated_center(0.2)
ctrl.close()
print(f"\nOverall Mean: {np.mean(errs):.2f} cnt")
print(f"Last-5 Mean : {np.mean(errs[-5:]):.2f} cnt")
print(f"Last-5 Max  : {np.max(errs[-5:]):.2f} cnt")

