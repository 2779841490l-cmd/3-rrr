from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.sensor.start()

for i in range(5):
    ctrl.driver.center(settle_s=0.45, preload=True)
    # ux = 23.0 -> delta_ux = 23.0 - 24.8 = -1.8
    # uy = -17.0 -> delta_uy = -17.0 - (-15.0) = -2.0
    ctrl.ilc.params["target_table"]["30_20"] = [-1.8, -2.0]
    ctrl.ilc.save()
    res = ctrl.flick_shot(30, 20, timeout_ms=100.0, enable_ilc_update=False)
    print(f"Shot {i+1} -> Final {res['final_delta']}, Err: {res['final_error_counts']:.2f} cnt, Brake: {res['t_brake_ms']}ms")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

