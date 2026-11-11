from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.ilc.params["target_table"] = {}
ctrl.ilc.save()
ctrl.sensor.start()

print("Checking Onset and Brake timing on (30, 20)...")
for i in range(1, 7):
    ctrl.driver.center(settle_s=0.45, preload=True)
    res = ctrl.flick_shot(30, 20, timeout_ms=100.0, enable_ilc_update=False)
    print(f"Shot {i:2d} -> Final {res['final_delta']} | Err: {res['final_error_counts']:4.2f} | Onset: {res['t_onset_ms']:4.1f}ms | Brake: {res['t_brake_ms']:4.1f}ms")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

