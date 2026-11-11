import time
import numpy as np
from rect_manifold_calibrator import HardwareRectangleTrainer
from servo_driver import WS63ServoDriver

trainer = HardwareRectangleTrainer()
trainer.sensor.start()

try:
    print("Zeroing at origin...")
    trainer.move_to_rect_origin(settle_s=0.6)
    ux0 = trainer.manifold.u0_star_x
    uy0 = trainer.manifold.u0_star_y

    def eval_pulse(xc, yc, dx_h=0, dy_h=0):
        # Nonlinear gain: 0.62 for moderate travel, smooth 2nd order
        u_kin_x = 0.58 * xc - 0.09 * yc
        u_kin_y = -0.05 * xc - 0.38 * yc
        hx = 12.0 if dx_h > 0 else (-16.0 if dx_h < 0 else 0.0)
        hy = 4.0 if dy_h > 0 else (-4.0 if dy_h < 0 else 0.0)
        ux = float(np.clip(ux0 + u_kin_x + hx, 1150.0, 1850.0))
        uy = float(np.clip(uy0 + u_kin_y + hy, 1150.0, 1850.0))
        return ux, uy

    def move_to(tx, ty):
        t0 = time.perf_counter()
        p_cur = trainer.get_rect_position()
        dx = tx - p_cur[0]
        dy = ty - p_cur[1]
        L = np.hypot(dx, dy)
        
        ux, uy = eval_pulse(tx, ty, dx_h=dx, dy_h=dy)
        trainer.driver.send_raw_pulses(ux, uy)
        
        # Primary settle
        time.sleep(max(0.12, 0.08 + 0.0003 * L))
        p_last = trainer.get_rect_position()
        for _ in range(8):
            time.sleep(0.015)
            p_now = trainer.get_rect_position()
            if np.hypot(p_now[0] - p_last[0], p_now[1] - p_last[1]) <= 1.5:
                break
            p_last = p_now
            
        # Direction tracking for backlash
        cur_dir_x = np.sign(dx) if abs(dx) > 2.0 else 0.0
        cur_dir_y = np.sign(dy) if abs(dy) > 2.0 else 0.0
        
        # Deadbeat trim (up to 3 rounds)
        cur_ux, cur_uy = ux, uy
        for trim_idx in range(3):
            p_m = trainer.get_rect_position()
            e = np.array([tx - p_m[0], ty - p_m[1]], dtype=float)
            err_norm = np.hypot(e[0], e[1])
            if err_norm <= 2.0:
                break
                
            du_x = 0.60 * e[0] - 0.09 * e[1]
            du_y = -0.05 * e[0] - 0.38 * e[1]
            
            # Reversal backlash traversal
            trim_dir_x = np.sign(du_x)
            trim_dir_y = np.sign(du_y)
            bx = 0.0
            if abs(e[0]) > 2.0:
                if cur_dir_x * trim_dir_x < -0.1:  # reversing
                    bx = trim_dir_x * 14.0
                else:
                    bx = trim_dir_x * 2.5
            by = 0.0
            if abs(e[1]) > 2.0:
                if cur_dir_y * trim_dir_y < -0.1:  # reversing
                    by = trim_dir_y * 4.0
                else:
                    by = trim_dir_y * 1.5
                    
            cur_dir_x = trim_dir_x
            cur_dir_y = trim_dir_y
            
            cur_ux = float(np.clip(cur_ux + du_x + bx, 1150.0, 1850.0))
            cur_uy = float(np.clip(cur_uy + du_y + by, 1150.0, 1850.0))
            trainer.driver.send_raw_pulses(cur_ux, cur_uy)
            time.sleep(0.075)
            
        pf = trainer.get_rect_position()
        ef = float(np.hypot(tx - pf[0], ty - pf[1]))
        ms = (time.perf_counter() - t0) * 1000.0
        return ef, ms, pf

    extreme_targets = [
        (-240.0, -180.0),
        (240.0, -180.0),
        (240.0, 180.0),
        (-240.0, 180.0),
        (0.0, 0.0),
    ]

    print("\n--- Testing Extreme Corners with Reversal Backlash Deadbeat ---")
    for tx, ty in extreme_targets:
        ef, ms, pf = move_to(tx, ty)
        print(f"  Tgt({tx:+6.1f}, {ty:+6.1f}) -> Final({pf[0]:+6.1f}, {pf[1]:+6.1f}) | Err: {ef:4.2f} cnt ({ef*7.9375:4.1f} um) | {ms:5.1f} ms")

finally:
    trainer.move_to_rect_origin(0.2)
    trainer.close()

