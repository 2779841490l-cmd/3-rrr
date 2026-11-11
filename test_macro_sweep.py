from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.sensor.start()

for u_y in [11.0, 13.0, 15.0]:
    ctrl.driver.center(settle_s=0.45, preload=True)
    # du_base is [218.0, 9.0]
    ctrl.ilc.params["target_table"]["160_-90"] = [194.0 - 218.0, u_y - 9.0]
    ctrl.ilc.save()
    res = ctrl.flick_shot(160, -90, timeout_ms=100.0, enable_ilc_update=False)
    print(f"u_x=194.0, u_y={u_y:.1f} -> Final {res['final_delta']}, Err: {res['final_error_counts']:.2f}")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()
