from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.sensor.start()
ctrl.driver.center(settle_s=0.45, preload=True)
res = ctrl.flick_shot(160, -90, timeout_ms=100.0, enable_ilc_update=False)
ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

print(f"Final: {res['final_delta']}, Err: {res['final_error_counts']}, Brake: {res['t_brake_ms']}ms, Onset: {res['t_onset_ms']}ms")
for pt in res["trajectory"]:
    if pt["t_ms"] >= 30 and pt["t_ms"] <= 90 and int(pt["t_ms"]) % 5 == 0:
        print(f"  t={pt['t_ms']} act={pt['active_ms']} phase={pt['phase']} pos=({pt['x']},{pt['y']}) v=({pt['vx']},{pt['vy']})")

