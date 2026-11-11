import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.sensor.start()
ctrl.driver.center(settle_s=0.45, preload=True)
res = ctrl.flick_shot(0, 35, timeout_ms=100.0, enable_ilc_update=False)

print("Final Delta:", res["final_delta"])
print("Error:", res["final_error_counts"])
print("Onset ms:", res["t_onset_ms"])
print("Brake ms:", res["t_brake_ms"])
print("Settled ms:", res["settled_time_ms"])

print("\nTrajectory sample:")
for p in res["trajectory"][::4]: # every 4ms
    print(f"t={p['t_ms']:5.1f}ms (act={p['active_ms']:5.1f}ms) [{p['phase']:5s}] | x={p['x']:3d}, y={p['y']:3d} | vx={p['vx']:6.1f}, vy={p['vy']:6.1f} | err={p['err_counts']:5.1f}")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

