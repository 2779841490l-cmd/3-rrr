from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.sensor.start()

print("--- Testing Macro (+160, -90) fine ---")
for delta_ux in [-15.0, -14.0, -13.0]:
    ctrl.driver.center(settle_s=0.45, preload=True)
    ctrl.ilc.params["target_table"]["160_-90"] = [delta_ux, 0.0]
    ctrl.ilc.save()
    res = ctrl.flick_shot(160, -90, timeout_ms=100.0, enable_ilc_update=False)
    print(f"delta_ux={delta_ux:.1f} (u0={218.0+delta_ux:.1f}) -> Final {res['final_delta']}, Err: {res['final_error_counts']:.2f}")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()
