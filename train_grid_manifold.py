import time
import json
import numpy as np
from rect_manifold_calibrator import HardwareRectangleTrainer
from servo_driver import WS63ServoDriver

trainer = HardwareRectangleTrainer()
trainer.sensor.start()

try:
    print("Zeroing at origin...")
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

    # 9x9 lattice
    grid_size = 9
    W, H = 220.0, 160.0
    gx = np.linspace(-W, W, grid_size)
    gy = np.linspace(-H, H, grid_size)
    
    # Load or initialize residual lattice
    res_ux = np.zeros((grid_size, grid_size), dtype=float)
    res_uy = np.zeros((grid_size, grid_size), dtype=float)

    def interp_res(x, y):
        xc = float(np.clip(x, -W, W))
        yc = float(np.clip(y, -H, H))
        tx = (xc + W) / (2.0 * W) * (grid_size - 1)
        ty = (yc + H) / (2.0 * H) * (grid_size - 1)
        ix = int(np.clip(np.floor(tx), 0, grid_size - 2))
        iy = int(np.clip(np.floor(ty), 0, grid_size - 2))
        fx = tx - ix
        fy = ty - iy
        rx = (1-fx)*(1-fy)*res_ux[ix,iy] + fx*(1-fy)*res_ux[ix+1,iy] + (1-fx)*fy*res_ux[ix,iy+1] + fx*fy*res_ux[ix+1,iy+1]
        ry = (1-fx)*(1-fy)*res_uy[ix,iy] + fx*(1-fy)*res_uy[ix+1,iy] + (1-fx)*fy*res_uy[ix,iy+1] + fx*fy*res_uy[ix+1,iy+1]
        return float(rx), float(ry)

    def eval_pulse(x, y, dx=0, dy=0):
        rx, ry = interp_res(x, y)
        u_kin_x = 0.65 * x - 0.36 * y + rx
        u_kin_y = -0.045 * x - 0.38 * y + ry
        hx = 6.0 if dx > 0 else (-6.0 if dx < 0 else 0.0)
        hy = 3.0 if dy > 0 else (-3.0 if dy < 0 else 0.0)
        ux = float(np.clip(ux0 + u_kin_x + hx, 1150.0, 1850.0))
        uy = float(np.clip(uy0 + u_kin_y + hy, 1150.0, 1850.0))
        return ux, uy

    # 5x5 = 25 test points
    xs = np.linspace(-W, W, 5)
    ys = np.linspace(-H, H, 5)
    test_pts = []
    for j, y in enumerate(ys):
        row_x = xs if j % 2 == 0 else xs[::-1]
        for x in row_x:
            test_pts.append((round(float(x), 1), round(float(y), 1)))

    # Run 2 training epochs over the 25 points
    for epoch in range(1, 3):
        print(f"\n=== Training Epoch {epoch}/2 (25 Points) ===")
        epoch_errs = []
        for idx, (tx, ty) in enumerate(test_pts, 1):
            p_cur = get_pos()
            dx, dy = tx - p_cur[0], ty - p_cur[1]
            cmd_ux, cmd_uy = eval_pulse(tx, ty, dx, dy)
            trainer.driver.send_raw_pulses(cmd_ux, cmd_uy)
            time.sleep(0.180)
            
            p_meas = get_pos()
            err_vec = np.array([tx - p_meas[0], ty - p_meas[1]], dtype=float)
            err_norm = float(np.hypot(err_vec[0], err_vec[1]))
            epoch_errs.append(err_norm)
            
            # Compute lattice correction
            du_x = 0.65 * err_vec[0] - 0.36 * err_vec[1]
            du_y = -0.045 * err_vec[0] - 0.38 * err_vec[1]
            du_x = np.clip(du_x, -15.0, 15.0)
            du_y = np.clip(du_y, -15.0, 15.0)
            
            # Update lattice nodes within radius
            rx_rad = 0.35 * W
            ry_rad = 0.35 * H
            for i, nx in enumerate(gx):
                for j, ny in enumerate(gy):
                    d2 = ((tx - nx) / rx_rad)**2 + ((ty - ny) / ry_rad)**2
                    if d2 <= 2.5:
                        w = float(np.exp(-0.5 * d2))
                        res_ux[i, j] += 0.50 * w * du_x
                        res_uy[i, j] += 0.50 * w * du_y

        print(f"Epoch {epoch} Mean Error: {np.mean(epoch_errs):.2f} cnt, Max: {np.max(epoch_errs):.2f} cnt")

    # Now verify with precision deadbeat trims:
    print("\n=== Stage III Verification with Learned Lattice & Deadbeat Trim ===")
    final_errs = []
    results = []
    for idx, (tx, ty) in enumerate(test_pts, 1):
        t0 = time.perf_counter()
        p_cur = get_pos()
        dx, dy = tx - p_cur[0], ty - p_cur[1]
        cmd_ux, cmd_uy = eval_pulse(tx, ty, dx, dy)
        trainer.driver.send_raw_pulses(cmd_ux, cmd_uy)
        time.sleep(0.180)
        
        # Deadbeat trim (up to 3 rounds)
        cur_ux, cur_uy = cmd_ux, cmd_uy
        for _ in range(3):
            p_m = get_pos()
            err_vec = np.array([tx - p_m[0], ty - p_m[1]], dtype=float)
            if np.hypot(err_vec[0], err_vec[1]) <= 2.0:
                break
            du_x = 0.65 * err_vec[0] - 0.36 * err_vec[1]
            du_y = -0.045 * err_vec[0] - 0.38 * err_vec[1]
            kx = np.sign(du_x) * 2.5 if abs(err_vec[0]) > 2.0 and abs(du_x) < 2.5 else 0.0
            ky = np.sign(du_y) * 1.5 if abs(err_vec[1]) > 2.0 and abs(du_y) < 1.5 else 0.0
            cur_ux = float(np.clip(cur_ux + du_x + kx, 1150.0, 1850.0))
            cur_uy = float(np.clip(cur_uy + du_y + ky, 1150.0, 1850.0))
            trainer.driver.send_raw_pulses(cur_ux, cur_uy)
            time.sleep(0.100)

        p_final = get_pos()
        ef = float(np.hypot(tx - p_final[0], ty - p_final[1]))
        ms = (time.perf_counter() - t0) * 1000.0
        final_errs.append(ef)
        results.append({
            "target": (tx, ty),
            "final": (round(float(p_final[0]), 1), round(float(p_final[1]), 1)),
            "err_counts": round(ef, 2),
            "err_um": round(ef * 7.9375, 1),
            "elapsed_ms": round(ms, 1)
        })
        print(f"  #{idx:02d}/25 Tgt({tx:+6.1f}, {ty:+6.1f}) -> Final({p_final[0]:+6.1f}, {p_final[1]:+6.1f}) | Err: {ef:4.2f} cnt ({ef*7.9375:4.1f} um) | {ms:5.1f} ms")

    mean_err = float(np.mean(final_errs))
    max_err = float(np.max(final_errs))
    print(f"\n[FINAL EVALUATION] 25-Point Grid Mean Error: {mean_err:.2f} cnt ({mean_err*7.9375:.1f} um) | Max Error: {max_err:.2f} cnt ({max_err*7.9375:.1f} um)")

    # Save to calib_rect_manifold.json
    with open("calib_rect_manifold.json", "r") as f:
        d = json.load(f)
    d["u0_star_us"] = {"s1_x": ux0, "s2_y1": uy0}
    d["residual_ux"] = res_ux.tolist()
    d["residual_uy"] = res_uy.tolist()
    with open("calib_rect_manifold.json", "w") as f:
        json.dump(d, f, indent=2)
    print("Saved trained residual lattice to calib_rect_manifold.json")

    # Save experiments/rect_grid_25pts_eval.csv
    import csv
    with open("experiments/rect_grid_25pts_eval.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print("Saved evaluation trace to experiments/rect_grid_25pts_eval.csv")

finally:
    trainer.driver.send_raw_pulses(1496.0, 1466.0)
    time.sleep(0.2)
    trainer.close()

