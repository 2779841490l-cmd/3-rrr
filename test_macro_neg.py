from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.sensor.start()

for u_x in [-171.0, -168.0, -165.0]:
    ctrl.driver.center(settle_s=0.45, preload=True)
    # du_base is [-171.0, -14.5]
    ctrl.ilc.params["target_table"]["-160_90"] = [u_x - (-171.0), 0.0]
    ctrl.ilc.save()
    res = ctrl.flick_shot(-160, 90, timeout_ms=100.0, enable_ilc_update=False)
    print(f"u_x={u_x:.1f} -> Final {res['final_delta']}, Err: {res['final_error_counts']:.2f}")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

