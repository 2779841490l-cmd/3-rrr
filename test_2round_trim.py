import time
import numpy as np
from rect_manifold_calibrator import HardwareRectangleTrainer
from servo_driver import WS63ServoDriver

trainer = HardwareRectangleTrainer()
trainer.sensor.start()

try:
    print("Zeroing at origin...")
    trainer.move_to_rect_origin(settle_s=0.6)
    ux0 = trainer.manifold.u0_star_x - 13.0
    uy0 = trainer.manifold.u0_star_y

    def goto_target(tx, ty):
        t0 = time.perf_counter()
        p0 = trainer.get_rect_position()
        dx = tx - p0[0]
        dy = ty - p0[1]
        
        u_kin_x = 0.65 * tx - 0.36 * ty
        u_kin_y = -0.045 * tx - 0.38 * ty
        hx = 6.0 if dx > 0 else (-6.0 if dx < 0 else 0.0)
        hy = 3.0 if dy > 0 else (-3.0 if dy < 0 else 0.0)
        ux = float(np.clip(ux0 + u_kin_x + hx, 1150.0, 1850.0))
        uy = float(np.clip(uy0 + u_kin_y + hy, 1150.0, 1850.0))
        trainer.driver.send_raw_pulses(ux, uy)
        
        # 1. Primary settle (190ms)
        time.sleep(0.190)
        p_prim = trainer.get_rect_position()
        e_prim = float(np.hypot(tx - p_prim[0], ty - p_prim[1]))
        
        # 2. Deadbeat micro-trim: up to 2 rounds
        cur_ux, cur_uy = ux, uy
        for trim_idx in range(2):
            p_m = trainer.get_rect_position()
            err_vec = np.array([tx - p_m[0], ty - p_m[1]], dtype=float)
            if np.hypot(err_vec[0], err_vec[1]) <= 2.0:
                break
                
            du_x = 0.65 * err_vec[0] - 0.36 * err_vec[1]
            du_y = -0.045 * err_vec[0] - 0.38 * err_vec[1]
            
            # Stiction kick
            kx = np.sign(du_x) * 2.5 if abs(err_vec[0]) > 2.0 and abs(du_x) < 2.5 else 0.0
            ky = np.sign(du_y) * 1.5 if abs(err_vec[1]) > 2.0 and abs(du_y) < 1.5 else 0.0
            
            cur_ux = float(np.clip(cur_ux + du_x + kx, 1150.0, 1850.0))
            cur_uy = float(np.clip(cur_uy + du_y + ky, 1150.0, 1850.0))
            trainer.driver.send_raw_pulses(cur_ux, cur_uy)
            time.sleep(0.110)
            
        p_final = trainer.get_rect_position()
        e_final = float(np.hypot(tx - p_final[0], ty - p_final[1]))
        ms = (time.perf_counter() - t0) * 1000.0
        return e_prim, e_final, ms, p_final

    test_pts = [
        (80.0, 60.0),
        (-80.0, 60.0),
        (-80.0, -60.0),
        (80.0, -60.0),
        (0.0, 0.0),
        (120.0, 0.0),
        (-120.0, 0.0),
        (0.0, 80.0),
        (0.0, -80.0),
        (0.0, 0.0),
    ]

    print("\n--- Testing 2-Round Deadbeat Trim (110ms settle each) ---")
    errs = []
    for tx, ty in test_pts:
        e_prim, e_fin, ms, pf = goto_target(tx, ty)
        errs.append(e_fin)
        print(f"  Tgt({tx:+6.1f}, {ty:+6.1f}) | Prim: {e_prim:4.1f} cnt -> Final: {e_fin:4.2f} cnt ({e_fin*7.9375:4.1f} um) | FinalPos: ({pf[0]:+5.1f}, {pf[1]:+5.1f}) | {ms:5.1f} ms")

    print(f"\n[Result] Mean Error: {np.mean(errs):.2f} cnt ({np.mean(errs)*7.9375:.1f} um) | Max Error: {np.max(errs):.2f} cnt")

finally:
    trainer.move_to_rect_origin(0.2)
    trainer.close()

