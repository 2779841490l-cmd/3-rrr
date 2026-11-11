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

    def eval_clean_pulse(xc, yc, dx_h=0, dy_h=0):
        # Accurate linear decoupling baseline
        u_kin_x = 0.68 * xc - 0.09 * yc
        u_kin_y = -0.05 * xc - 0.38 * yc
        
        # Hysteresis along moving axis
        hx = 14.0 if dx_h > 0 else (-14.0 if dx_h < 0 else 0.0)
        hy = 5.0 if dy_h > 0 else (-5.0 if dy_h < 0 else 0.0)
        
        ux = float(np.clip(ux0 + u_kin_x + hx, 1150.0, 1850.0))
        uy = float(np.clip(uy0 + u_kin_y + hy, 1150.0, 1850.0))
        return ux, uy

    def move_to(tx, ty):
        t0 = time.perf_counter()
        p_cur = trainer.get_rect_position()
        dx = tx - p_cur[0]
        dy = ty - p_cur[1]
        L = np.hypot(dx, dy)
        
        ux, uy = eval_clean_pulse(tx, ty, dx_h=dx, dy_h=dy)
        trainer.driver.send_raw_pulses(ux, uy)
        
        # Settle
        time.sleep(max(0.12, 0.08 + 0.0003 * L))
        p_last = trainer.get_rect_position()
        for _ in range(8):
            time.sleep(0.015)
            p_now = trainer.get_rect_position()
            if np.hypot(p_now[0] - p_last[0], p_now[1] - p_last[1]) <= 1.5:
                break
            p_last = p_now
            
        # Deadbeat trim (up to 2 rounds with 70ms settle)
        cur_ux, cur_uy = ux, uy
        for _ in range(2):
            p_m = trainer.get_rect_position()
            e = np.array([tx - p_m[0], ty - p_m[1]], dtype=float)
            if np.hypot(e[0], e[1]) <= 2.0:
                break
            du_x = 0.68 * e[0] - 0.09 * e[1]
            du_y = -0.05 * e[0] - 0.38 * e[1]
            kx = np.sign(du_x) * 3.0 if abs(e[0]) > 2.0 and abs(du_x) < 3.0 else 0.0
            ky = np.sign(du_y) * 2.0 if abs(e[1]) > 2.0 and abs(du_y) < 2.0 else 0.0
            cur_ux = float(np.clip(cur_ux + du_x + kx, 1150.0, 1850.0))
            cur_uy = float(np.clip(cur_uy + du_y + ky, 1150.0, 1850.0))
            trainer.driver.send_raw_pulses(cur_ux, cur_uy)
            time.sleep(0.075)
            
        pf = trainer.get_rect_position()
        ef = float(np.hypot(tx - pf[0], ty - pf[1]))
        ms = (time.perf_counter() - t0) * 1000.0
        return ef, ms, pf

    W, H = 240.0, 180.0
    xs = np.linspace(-W, W, 5)
    ys = np.linspace(-H, H, 5)
    test_pts = []
    for j, y in enumerate(ys):
        row_x = xs if j % 2 == 0 else xs[::-1]
        for x in row_x:
            test_pts.append((round(float(x), 1), round(float(y), 1)))

    print(f"\n--- Testing 25 Grid Points with Clean Decoupled Baseline ---")
    errs = []
    times = []
    for idx, (tx, ty) in enumerate(test_pts, 1):
        ef, ms, pf = move_to(tx, ty)
        errs.append(ef)
        times.append(ms)
        print(f"  #{idx:02d}/25 Tgt({tx:+6.1f}, {ty:+6.1f}) -> Final({pf[0]:+6.1f}, {pf[1]:+6.1f}) | Err: {ef:4.2f} cnt ({ef*7.9375:4.1f} um) | {ms:5.1f} ms")

    print(f"\n[Summary] Mean Error: {np.mean(errs):.2f} cnt ({np.mean(errs)*7.9375:.1f} um) | Max Error: {np.max(errs):.2f} cnt ({np.max(errs)*7.9375:.1f} um) | Mean Time: {np.mean(times):.1f} ms")

finally:
    trainer.move_to_rect_origin(0.2)
    trainer.close()

