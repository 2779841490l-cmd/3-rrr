import time
import numpy as np
from rect_manifold_calibrator import HardwareRectangleTrainer
from servo_driver import WS63ServoDriver

trainer = HardwareRectangleTrainer()
trainer.sensor.start()

try:
    print("Zeroing at true calibrated neutral center...")
    ux0 = 1496.0
    uy0 = 1466.0
    
    trainer.driver.send_raw_pulses(ux0, uy0)
    time.sleep(0.65)
    trainer.sensor.recenter_cursor()
    sx, sy = trainer.sensor.get_position()
    p_orig = np.array([float(sx), float(sy)], dtype=float)
    
    def get_pos():
        cur_x, cur_y = trainer.sensor.get_position()
        return np.array([float(cur_x) - p_orig[0], float(cur_y) - p_orig[1]], dtype=float)

    def move_p2p(tx, ty):
        t0 = time.perf_counter()
        p_cur = get_pos()
        dx = tx - p_cur[0]
        dy = ty - p_cur[1]
        
        # Primary decoupled pulses
        u_kin_x = 0.65 * tx - 0.36 * ty
        u_kin_y = -0.045 * tx - 0.38 * ty
        hx = 6.0 if dx > 0 else (-6.0 if dx < 0 else 0.0)
        hy = 3.0 if dy > 0 else (-3.0 if dy < 0 else 0.0)
        
        cmd_ux = float(np.clip(ux0 + u_kin_x + hx, 1150.0, 1850.0))
        cmd_uy = float(np.clip(uy0 + u_kin_y + hy, 1150.0, 1850.0))
        trainer.driver.send_raw_pulses(cmd_ux, cmd_uy)
        
        # 1. Primary mechanical transit settle (180ms)
        time.sleep(0.180)
        p_last = get_pos()
        for _ in range(6):
            time.sleep(0.015)
            p_now = get_pos()
            if np.hypot(p_now[0] - p_last[0], p_now[1] - p_last[1]) <= 1.2:
                break
            p_last = p_now
            
        # 2. Precision Deadbeat micro-trim (up to 3 rounds with full settle)
        cur_ux, cur_uy = cmd_ux, cmd_uy
        for _ in range(3):
            p_m = get_pos()
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
            
        p_final = get_pos()
        e_final = float(np.hypot(tx - p_final[0], ty - p_final[1]))
        ms = (time.perf_counter() - t0) * 1000.0
        return e_final, ms, p_final

    W, H = 220.0, 160.0
    xs = np.linspace(-W, W, 5)
    ys = np.linspace(-H, H, 5)
    test_pts = []
    for j, y in enumerate(ys):
        row_x = xs if j % 2 == 0 else xs[::-1]
        for x in row_x:
            test_pts.append((round(float(x), 1), round(float(y), 1)))

    print(f"\n--- Testing 25 Grid Points with True Calibrated Center (ux0={ux0}, uy0={uy0}) ---")
    errs = []
    times = []
    for idx, (tx, ty) in enumerate(test_pts, 1):
        ef, ms, pf = move_p2p(tx, ty)
        errs.append(ef)
        times.append(ms)
        print(f"  #{idx:02d}/25 Tgt({tx:+6.1f}, {ty:+6.1f}) -> Final({pf[0]:+6.1f}, {pf[1]:+6.1f}) | Err: {ef:4.2f} cnt ({ef*7.9375:4.1f} um) | {ms:5.1f} ms")

    mean_err = float(np.mean(errs))
    max_err = float(np.max(errs))
    print(f"\n[Summary] Mean Error: {mean_err:.2f} cnt ({mean_err*7.9375:.1f} um) | Max Error: {max_err:.2f} cnt ({max_err*7.9375:.1f} um) | Mean Time: {np.mean(times):.1f} ms")

finally:
    trainer.driver.send_raw_pulses(1496.0, 1466.0)
    time.sleep(0.2)
    trainer.close()

