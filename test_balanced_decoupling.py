import time
import numpy as np
from rect_manifold_calibrator import HardwareRectangleTrainer
from servo_driver import WS63ServoDriver

trainer = HardwareRectangleTrainer()
trainer.sensor.start()

try:
    print("Zeroing at origin...")
    trainer.move_to_rect_origin(settle_s=0.6)
    # Calibrated physical neutral center
    ux0 = trainer.manifold.u0_star_x - 13.0
    uy0 = trainer.manifold.u0_star_y

    def goto_target(tx, ty):
        t0 = time.perf_counter()
        p0 = trainer.get_rect_position()
        dx = tx - p0[0]
        dy = ty - p0[1]
        
        # Primary command with exact physics decoupling
        u_kin_x = 0.65 * tx - 0.36 * ty
        u_kin_y = -0.045 * tx - 0.38 * ty
        hx = 6.0 if dx > 0 else (-6.0 if dx < 0 else 0.0)
        hy = 3.0 if dy > 0 else (-3.0 if dy < 0 else 0.0)
        ux = float(np.clip(ux0 + u_kin_x + hx, 1150.0, 1850.0))
        uy = float(np.clip(uy0 + u_kin_y + hy, 1150.0, 1850.0))
        trainer.driver.send_raw_pulses(ux, uy)
        
        # 1. Primary settle
        time.sleep(0.180)
        p_last = trainer.get_rect_position()
        for _ in range(6):
            time.sleep(0.015)
            p_now = trainer.get_rect_position()
            if np.hypot(p_now[0] - p_last[0], p_now[1] - p_last[1]) <= 1.2:
                break
            p_last = p_now
            
        p_primary = trainer.get_rect_position()
        e_primary = float(np.hypot(tx - p_primary[0], ty - p_primary[1]))
        
        # 2. Deadbeat micro-trim (up to 2 rounds, 85ms each)
        cur_ux, cur_uy = ux, uy
        for _ in range(2):
            p_m = trainer.get_rect_position()
            err_vec = np.array([tx - p_m[0], ty - p_m[1]], dtype=float)
            err_norm = float(np.hypot(err_vec[0], err_vec[1]))
            if err_norm <= 2.0:
                break
                
            du_x = 0.65 * err_vec[0] - 0.36 * err_vec[1]
            du_y = -0.045 * err_vec[0] - 0.38 * err_vec[1]
            
            # Stiction kick
            kx = np.sign(du_x) * 2.5 if abs(err_vec[0]) > 2.0 and abs(du_x) < 2.5 else 0.0
            ky = np.sign(du_y) * 1.5 if abs(err_vec[1]) > 2.0 and abs(du_y) < 1.5 else 0.0
            
            cur_ux = float(np.clip(cur_ux + du_x + kx, WS63ServoDriver.SAFE_MIN_US, WS63ServoDriver.SAFE_MAX_US))
            cur_uy = float(np.clip(cur_uy + du_y + ky, WS63ServoDriver.SAFE_MIN_US, WS63ServoDriver.SAFE_MAX_US))
            trainer.driver.send_raw_pulses(cur_ux, cur_uy)
            time.sleep(0.085)
            
        p_final = trainer.get_rect_position()
        e_final = float(np.hypot(tx - p_final[0], ty - p_final[1]))
        ms = (time.perf_counter() - t0) * 1000.0
        return e_primary, e_final, ms, p_final

    test_pts = [
        (100.0, 80.0),
        (-100.0, 80.0),
        (-100.0, -80.0),
        (100.0, -80.0),
        (0.0, 0.0),
        (160.0, 0.0),
        (-160.0, 0.0),
        (0.0, 120.0),
        (0.0, -120.0),
        (180.0, 140.0),
        (-180.0, -140.0),
        (0.0, 0.0),
    ]

    print("\n--- Testing Balanced Physics Decoupling Positioning ---")
    errs = []
    times = []
    for tx, ty in test_pts:
        e_prim, e_fin, ms, pf = goto_target(tx, ty)
        errs.append(e_fin)
        times.append(ms)
        print(f"  Tgt({tx:+6.1f}, {ty:+6.1f}) | PrimErr: {e_prim:4.1f} cnt -> Final: {e_fin:4.2f} cnt ({e_fin*7.9375:4.1f} um) | FinalPos: ({pf[0]:+5.1f}, {pf[1]:+5.1f}) | {ms:5.1f} ms")

    print(f"\n[Result] Mean Error: {np.mean(errs):.2f} cnt ({np.mean(errs)*7.9375:.1f} um) | Max Error: {np.max(errs):.2f} cnt ({np.max(errs)*7.9375:.1f} um) | Mean Time: {np.mean(times):.1f} ms")

finally:
    trainer.move_to_rect_origin(0.2)
    trainer.close()

