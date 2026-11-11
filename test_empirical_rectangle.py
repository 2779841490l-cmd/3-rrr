import time
import numpy as np
from rect_manifold_calibrator import HardwareRectangleTrainer

trainer = HardwareRectangleTrainer()
trainer.sensor.start()
try:
    print("Zeroing at origin...")
    trainer.move_to_rect_origin(settle_s=0.6)
    ux0, uy0 = trainer.manifold.u0_star_x, trainer.manifold.u0_star_y
    
    # Decoupling matrix K_real
    # [du_x] = [ 1.38   -0.094 ] [dx]
    # [du_y]   [-0.052  -0.375 ] [dy]
    # Plus backlash for X: +6 us if dx > 0, -18 us if dx < 0
    # Plus backlash for Y: +3 us if dy > 0, -3 us if dy < 0
    
    def eval_decoupled_pulse(x, y, dx_hint=0, dy_hint=0):
        du_x = 1.38 * x - 0.094 * y
        du_y = -0.052 * x - 0.375 * y
        bx = 6.0 if dx_hint > 0 else (-18.0 if dx_hint < 0 else 0.0)
        by = 2.0 if dy_hint > 0 else (-2.0 if dy_hint < 0 else 0.0)
        ux = np.clip(ux0 + du_x + bx, 1150, 1850)
        uy = np.clip(uy0 + du_y + by, 1150, 1850)
        return float(ux), float(uy)

    # Let's test 4 edges of a 200x150 rectangle:
    # Corner 0: (-100, -75)
    # Edge 1 (TOP): (-100, -75) -> (+100, -75)
    # Edge 2 (RIGHT): (+100, -75) -> (+100, +75)
    # Edge 3 (BOTTOM): (+100, +75) -> (-100, +75)
    # Edge 4 (LEFT): (-100, +75) -> (-100, -75)
    
    W, H = 100.0, 75.0
    
    # Move to start corner
    ux, uy = eval_decoupled_pulse(-W, -H, dx_hint=-1, dy_hint=-1)
    trainer.driver.send_raw_pulses(ux, uy)
    time.sleep(0.4)
    
    print("\nTracing Rectangle with Empirical Decoupled Matrix...")
    edges = [
        ("TOP_EDGE", np.linspace(-W, W, 9), np.full(9, -H), 1, 0),
        ("RIGHT_EDGE", np.full(9, W), np.linspace(-H, H, 9), 0, 1),
        ("BOTTOM_EDGE", np.linspace(W, -W, 9), np.full(9, H), -1, 0),
        ("LEFT_EDGE", np.full(9, -W), np.linspace(H, -H, 9), 0, -1),
    ]
    
    for name, xs, ys, dx_h, dy_h in edges:
        devs = []
        for x, y in zip(xs, ys):
            ux, uy = eval_decoupled_pulse(x, y, dx_hint=dx_h, dy_hint=dy_h)
            trainer.driver.send_raw_pulses(ux, uy)
            time.sleep(0.08)
            p = trainer.get_rect_position()
            perp_err = (y - p[1]) if dx_h != 0 else (x - p[0])
            devs.append(abs(perp_err))
        print(f"  {name:12s}: Mean Cross-Axis Deviation = {np.mean(devs):4.2f} cnt, Max = {np.max(devs):4.2f} cnt")
        time.sleep(0.1)

finally:
    trainer.move_to_rect_origin(0.2)
    trainer.close()

