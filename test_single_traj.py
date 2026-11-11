import sys
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.sensor.start()
ctrl.driver.center(settle_s=0.38, preload=True)
res = ctrl.flick_shot(30, 20, timeout_ms=100.0, enable_ilc_update=False)
ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

print(f"Target (30, 20) -> Final {res['final_delta']}, Err: {res['final_error_counts']}, Brake: {res['t_brake_ms']}ms, Onset: {res['t_onset_ms']}ms")
print("Trajectory samples around onset and brake:")
for pt in res["trajectory"]:
    if pt["t_ms"] >= 40 and pt["t_ms"] <= 95 and int(pt["t_ms"]) % 4 == 0:
        print(f"  t={pt['t_ms']:4.1f}ms act={pt['active_ms']:4.1f}ms phase={pt['phase']} pos=({pt['x']:2d},{pt['y']:2d}) v=({pt['vx']:5.1f},{pt['vy']:5.1f}) err={pt['err_counts']:4.1f}")

