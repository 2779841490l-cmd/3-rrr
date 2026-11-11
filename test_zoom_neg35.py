from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.sensor.start()

print("--- Zoom in on (-35, 0) ---")
for delta_ux in [-3.0, -1.0, 0.0]:
    for uy_val in [-4.0, -3.0, -2.0, -1.0]:
        ctrl.driver.center(settle_s=0.45, preload=True)
        # Target (-35, 0) has du_base[0] ~ -32.8, du_base[1] = -2.2
        ctrl.ilc.params["target_table"]["-35_0"] = [delta_ux, uy_val - (-2.2)]
        ctrl.ilc.save()
        res = ctrl.flick_shot(-35, 0, timeout_ms=100.0, enable_ilc_update=False)
        print(f"du_x={delta_ux:+4.1f}, uy={uy_val:+4.1f} -> Final {res['final_delta']}, Err: {res['final_error_counts']:.2f}")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

