import time
import numpy as np
from rect_manifold_calibrator import HardwareRectangleTrainer
from servo_driver import WS63ServoDriver

trainer = HardwareRectangleTrainer()
trainer.sensor.start()

try:
    print("Zeroing at origin...")
    trainer.move_to_rect_origin(settle_s=0.6)
    ux0 = 1496.0
    uy0 = 1466.0

    def goto_target(tx, ty):
        t0 = time.perf_counter()
        p0 = trainer.get_rect_position()
        dx = tx - p0[0]
        dy = ty - p0[1]
        
        # Open-loop baseline command
        u_kin_x = 0.62 * tx - 0.36 * ty
        u_kin_y = -0.045 * tx - 0.38 * ty
        ux = float(np.clip(ux0 + u_kin_x, 1150.0, 1850.0))
        uy = float(np.clip(uy0 + u_kin_y, 1150.0, 1850.0))
        trainer.driver.send_raw_pulses(ux, uy)
        
        # Primary settle (190ms)
        time.sleep(0.190)
        p_prim = trainer.get_rect_position()
        e_prim = float(np.hypot(tx - p_prim[0], ty - p_prim[1]))
        
        # Pure deadbeat trims: gain = 0.72, NO artificial kicks
        cur_ux, cur_uy = ux, uy
        for it in range(3):
            p_m = trainer.get_rect_position()
            e = np.array([tx - p_m[0], ty - p_m[1]], dtype=float)
            err_norm = float(np.hypot(e[0], e[1]))
            if err_norm <= 2.0:
                break
                
            du_x = 0.72 * (0.65 * e[0] - 0.36 * e[1])
            du_y = 0.72 * (-0.045 * e[0] - 0.38 * e[1])
            
            # Pure proportional update (clipped to safe limits)
            cur_ux = float(np.clip(cur_ux + du_x, WS63ServoDriver.SAFE_MIN_US, WS63ServoDriver.SAFE_MAX_US))
            cur_uy = float(np.clip(cur_uy + du_y, WS63ServoDriver.SAFE_MIN_US, WS63ServoDriver.SAFE_MAX_US))
            trainer.driver.send_raw_pulses(cur_ux, cur_uy)
            time.sleep(0.095)
            
        p_final = trainer.get_rect_position()
        e_final = float(np.hypot(tx - p_final[0], ty - p_final[1]))
        ms = (time.perf_counter() - t0) * 1000.0
        return e_prim, e_final, ms, p_final

    test_pts = [
        (60.0, 45.0),
        (-60.0, 45.0),
        (-60.0, -45.0),
        (60.0, -45.0),
        (0.0, 0.0),
        (100.0, 0.0),
        (-100.0, 0.0),
        (0.0, 80.0),
        (0.0, -80.0),
        (0.0, 0.0),
    ]

    print("\n--- Testing Pure Proportional Deadbeat Trims (gain=0.72, no kicks) ---")
    errs = []
    for tx, ty in test_pts:
        e_prim, e_fin, ms, pf = goto_target(tx, ty)
        errs.append(e_fin)
        print(f"  Tgt({tx:+6.1f}, {ty:+6.1f}) | Prim: {e_prim:4.1f} cnt -> Final: {e_fin:4.2f} cnt ({e_fin*7.9375:4.1f} um) | FinalPos: ({pf[0]:+5.1f}, {pf[1]:+5.1f}) | {ms:5.1f} ms")

    print(f"\n[Result] Mean Error: {np.mean(errs):.2f} cnt ({np.mean(errs)*7.9375:.1f} um) | Max Error: {np.max(errs):.2f} cnt")

finally:
    trainer.move_to_rect_origin(0.2)
    trainer.close()

