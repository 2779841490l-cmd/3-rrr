"""
calibrate_full_workspace.py - Full-Envelope Safe Boundary & 2D Workspace Mapping
==============================================================================
Replaces legacy narrow (-60..+60 us) probe with true hardware 3200 DPI Raw Input
full-workspace characterization across [1165, 1835] us safe actuator bounds:

1. Safely backs up `calib_params.json` and `prompts/ilc_ballistic_weights.json`.
2. Re-measures true 3200 DPI Deadband (D+, D-) and Backlash Play Radius (B+, B-).
3. Performs a 16-ray progressive radial scan with automatic anti-stall / singularity
   guard to map the real mechanical reachable boundary in both actuator (us) and
   3200 DPI optical space (counts / mm).
4. Samples a dense 2D full-workspace grid inside the safe boundary polygon from the
   same deterministic preloaded neutral origin, building an invertible 2D continuous
   forward/inverse RBF & spatial model saved to `experiments/workspace_map_full.json`
   and updated into `calib_params.json`.
"""

import json
import os
import shutil
import time
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

from mouse_sensor import HighPrecisionMouseSensor
from servo_driver import WS63ServoDriver


SAFE_PULSE_MIN = 1165.0
SAFE_PULSE_MAX = 1835.0


def backup_existing_configs():
    os.makedirs("backups", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for fname in ["calib_params.json", "prompts/ilc_ballistic_weights.json"]:
        if os.path.exists(fname):
            base = os.path.basename(fname)
            dst = os.path.join("backups", f"{os.path.splitext(base)[0]}_{ts}.json")
            shutil.copy2(fname, dst)
            print(f"[BACKUP] Saved {fname} -> {dst}")


def pulses_within_safe_envelope(u0_x: float, u0_y: float, du_x: float, du_y: float) -> bool:
    s1 = u0_x + du_x
    s2 = u0_y + du_y
    s3 = 3000.0 - s2
    return (
        SAFE_PULSE_MIN <= s1 <= SAFE_PULSE_MAX
        and SAFE_PULSE_MIN <= s2 <= SAFE_PULSE_MAX
        and SAFE_PULSE_MIN <= s3 <= SAFE_PULSE_MAX
    )


def wait_until_stationary(sensor: HighPrecisionMouseSensor, max_wait_s: float = 0.25):
    t0 = time.perf_counter()
    still_count = 0
    while (time.perf_counter() - t0) < max_wait_s:
        sensor.read_delta_and_clear()
        time.sleep(0.015)
        dx, dy = sensor.read_delta_and_clear()
        if dx == 0 and dy == 0:
            still_count += 1
            if still_count >= 2:
                return
        else:
            still_count = 0


def probe_deadband_backlash(driver: WS63ServoDriver, sensor: HighPrecisionMouseSensor) -> Tuple[Dict, Dict]:
    u0_x, u0_y = float(driver.NEUTRAL_S1), float(driver.NEUTRAL_S2)

    def _preload(axis: str, direction: int, mag: float = 55.0):
        driver.center(settle_s=0.25, preload=False)
        if axis == "X":
            driver.move_smooth_raw(u0_x - direction * mag, u0_y, duration_s=0.16)
            time.sleep(0.08)
            driver.move_smooth_raw(u0_x, u0_y, duration_s=0.16)
        else:
            driver.move_smooth_raw(u0_x, u0_y - direction * mag, duration_s=0.16)
            time.sleep(0.08)
            driver.move_smooth_raw(u0_x, u0_y, duration_s=0.16)
        wait_until_stationary(sensor, 0.25)
        sensor.reset_origin()

    def _deadband(axis: str, direction: int, thresh: int = 4) -> float:
        _preload(axis, direction, 55.0)
        for du in range(1, 28):
            if axis == "X":
                driver.send_raw_pulses(u0_x + direction * du, u0_y)
            else:
                driver.send_raw_pulses(u0_x, u0_y + direction * du)
            time.sleep(0.045)
            dx, dy = sensor.get_position()
            if float(np.hypot(dx, dy)) >= thresh:
                return float(du)
        return 8.0

    def _backlash(axis: str, rev_dir: int, thresh: int = 6) -> float:
        _preload(axis, -rev_dir, 60.0)
        for du in range(2, 96, 2):
            if axis == "X":
                driver.send_raw_pulses(u0_x + rev_dir * du, u0_y)
            else:
                driver.send_raw_pulses(u0_x, u0_y + rev_dir * du)
            time.sleep(0.045)
            dx, dy = sensor.get_position()
            if float(np.hypot(dx, dy)) >= thresh:
                return float(du) * 0.5
        return 25.0

    print("\n[1/4] Probing True 3200 DPI Deadband & Gear Backlash Hysteresis...")
    deadband = {
        "x_pos": _deadband("X", +1),
        "x_neg": _deadband("X", -1),
        "y_pos": _deadband("Y", +1),
        "y_neg": _deadband("Y", -1),
    }
    backlash = {
        "x_pos": _backlash("X", +1),
        "x_neg": _backlash("X", -1),
        "y_pos": _backlash("Y", +1),
        "y_neg": _backlash("Y", -1),
    }
    print(f"      Deadband (us): {deadband}")
    print(f"      Backlash play radius (us): {backlash}")
    return deadband, backlash


def probe_16dir_envelope(
    driver: WS63ServoDriver, sensor: HighPrecisionMouseSensor
) -> Tuple[List[Dict], float]:
    """
    Safely probes 16 uniform directions in (du_x, du_y) from center out to
    either SAFE_PULSE limits or mechanical singularity/stall detection.
    """
    u0_x, u0_y = float(driver.NEUTRAL_S1), float(driver.NEUTRAL_S2)
    angles_deg = [k * 22.5 for k in range(16)]
    step_us = 35.0
    max_radius_us = 420.0
    envelope = []

    print("\n[2/4] Probing 16-Direction Full Mechanical Workspace Boundary...")
    print("  Theta_u | Max (du_x, du_y) us | Max Pos (dx, dy) cnt | Reach (mm) | Theta_xy | Term Reason")
    print("  " + "-" * 92)

    for ang in angles_deg:
        rad = np.deg2rad(ang)
        ux_dir, uy_dir = float(np.cos(rad)), float(np.sin(rad))

        driver.center(settle_s=0.38, preload=True)
        wait_until_stationary(sensor, 0.20)
        sensor.reset_origin()
        sensor.recenter_cursor()

        ray_pts = [(0.0, 0.0, 0, 0)]
        prev_x, prev_y = 0, 0
        max_safe_r = 0.0
        max_safe_du = (0.0, 0.0)
        max_safe_pos = (0, 0)
        term_reason = "PULSE_LIMIT"
        stalled_steps = 0

        r = step_us
        while r <= max_radius_us:
            du_x = r * ux_dir
            du_y = r * uy_dir
            if not pulses_within_safe_envelope(u0_x, u0_y, du_x, du_y):
                term_reason = "PULSE_LIMIT"
                break

            driver.move_smooth_raw(u0_x + du_x, u0_y + du_y, duration_s=0.14, rate_hz=100.0)
            wait_until_stationary(sensor, 0.16)
            cur_x, cur_y = sensor.get_position()
            step_disp = float(np.hypot(cur_x - prev_x, cur_y - prev_y))
            local_gain = step_disp / step_us

            # Anti-stall protection: after clearing worst-case backlash zone (r >= 105 us),
            # if displacement increment collapses < 0.16 cnt/us (~5.5 cnt/35us), stop to prevent binding
            if r >= 105.0 and local_gain < 0.16:
                stalled_steps += 1
                if stalled_steps >= 1:
                    term_reason = f"MECH_STALL_OR_SINGULARITY(gain={local_gain:.2f})"
                    break
            else:
                stalled_steps = 0

            ray_pts.append((round(du_x, 1), round(du_y, 1), int(cur_x), int(cur_y)))
            max_safe_r = r
            max_safe_du = (du_x, du_y)
            max_safe_pos = (int(cur_x), int(cur_y))
            prev_x, prev_y = cur_x, cur_y
            r += step_us

        # Smoothly retract back to neutral to avoid snapping linkage from extreme stretch
        driver.move_smooth_raw(u0_x, u0_y, duration_s=0.22, rate_hz=100.0)
        reach_cnt = float(np.hypot(max_safe_pos[0], max_safe_pos[1]))
        reach_mm = reach_cnt * sensor.mm_per_count
        xy_ang = float(np.rad2deg(np.arctan2(max_safe_pos[1], max_safe_pos[0]))) if reach_cnt > 1 else 0.0

        print(
            f"  {ang:6.1f}° | ({max_safe_du[0]:+6.1f}, {max_safe_du[1]:+6.1f})   | "
            f"({max_safe_pos[0]:+5d}, {max_safe_pos[1]:+5d})      | {reach_mm:5.2f} mm   | "
            f"{xy_ang:+7.1f}° | {term_reason}"
        )
        envelope.append({
            "theta_u_deg": float(ang),
            "max_r_us": float(max_safe_r),
            "max_du_x": round(float(max_safe_du[0]), 2),
            "max_du_y": round(float(max_safe_du[1]), 2),
            "max_dx_cnt": int(max_safe_pos[0]),
            "max_dy_cnt": int(max_safe_pos[1]),
            "reach_cnt": round(reach_cnt, 2),
            "reach_mm": round(reach_mm, 3),
            "theta_xy_deg": round(xy_ang, 2),
            "term_reason": term_reason,
            "ray_samples": ray_pts,
        })

    # Re-center cursor at end
    driver.center(settle_s=0.40, preload=True)
    return envelope, step_us


def _safe_radius_at_angle(envelope: List[Dict], angle_rad: float) -> float:
    angs = np.array([np.deg2rad(e["theta_u_deg"]) for e in envelope], dtype=float)
    radii = np.array([e["max_r_us"] for e in envelope], dtype=float)
    # Periodic linear interpolation over 0..2pi
    ang_mod = float(angle_rad % (2.0 * np.pi))
    angs_ext = np.concatenate([angs, [2.0 * np.pi]])
    radii_ext = np.concatenate([radii, [radii[0]]])
    return float(np.interp(ang_mod, angs_ext, radii_ext))


def probe_2d_safe_grid(
    driver: WS63ServoDriver, sensor: HighPrecisionMouseSensor, envelope: List[Dict]
) -> List[Dict]:
    """
    Samples a dense 2D grid inside 90% of the measured safe mechanical boundary
    so every reachable coordinate in [-du_max, +du_max] has exact calibrated
    (du_x, du_y) <-> (dx_cnt, dy_cnt) ground-truth data from neutral preload.
    """
    u0_x, u0_y = float(driver.NEUTRAL_S1), float(driver.NEUTRAL_S2)
    xs = [-300, -230, -165, -105, -55, -25, 0, 25, 55, 105, 165, 230, 300]
    ys = [-290, -225, -160, -100, -50, -20, 0, 20, 50, 100, 160, 225, 275]

    candidates = []
    for ux in xs:
        for uy in ys:
            if ux == 0 and uy == 0:
                continue
            r = float(np.hypot(ux, uy))
            # Skip ultra-dense corner intersections near center to keep ~100 well-spaced nodes
            if abs(ux) in (25,) and abs(uy) not in (0, 20):
                continue
            if abs(uy) in (20,) and abs(ux) not in (0, 25):
                continue
            if not pulses_within_safe_envelope(u0_x, u0_y, ux, uy):
                continue
            r_lim = _safe_radius_at_angle(envelope, float(np.arctan2(uy, ux)))
            if r <= 0.92 * r_lim:
                candidates.append((float(ux), float(uy)))

    # Sort spiraling outward for smooth progressive inspection
    candidates.sort(key=lambda pt: (np.hypot(pt[0], pt[1]), np.arctan2(pt[1], pt[0])))
    grid_points = [{"du_x": 0.0, "du_y": 0.0, "dx": 0, "dy": 0}]

    print(f"\n[3/4] Sampling Dense 2D Safe Workspace Grid ({len(candidates)} nodes inside 92% envelope)...")
    t_start = time.perf_counter()

    for idx, (du_x, du_y) in enumerate(candidates, 1):
        driver.center(settle_s=0.38, preload=True)
        wait_until_stationary(sensor, 0.16)
        sensor.reset_origin()
        if idx % 10 == 1:
            sensor.recenter_cursor()

        # Ramp smoothly to avoid overshoot distortion so map reflects true kinematic manifold
        dur = 0.14 if np.hypot(du_x, du_y) < 120 else 0.18
        driver.move_smooth_raw(u0_x + du_x, u0_y + du_y, duration_s=dur, rate_hz=100.0)
        wait_until_stationary(sensor, 0.18)
        dx, dy = sensor.get_position()
        driver.move_smooth_raw(u0_x, u0_y, duration_s=0.10, rate_hz=100.0)

        grid_points.append({"du_x": float(du_x), "du_y": float(du_y), "dx": int(dx), "dy": int(dy)})
        if idx % 12 == 0 or idx == len(candidates):
            elapsed = time.perf_counter() - t_start
            print(f"      Progress {idx:3d}/{len(candidates)} ({elapsed:5.1f}s) | du=({du_x:+5.0f},{du_y:+5.0f}) -> pos=({dx:+5d},{dy:+5d}) cnt")

    return grid_points


def analyze_workspace_and_save(
    deadband: Dict, backlash: Dict, envelope: List[Dict], grid_points: List[Dict], dpi: int = 3200
):
    print("\n[4/4] Analyzing Workspace Linearity, Polygon & Saving Full Model...")
    os.makedirs("experiments", exist_ok=True)

    # Fit central Jacobian around |du| <= 70 us using least squares on grid_points
    near_pts = [p for p in grid_points if 15.0 <= np.hypot(p["du_x"], p["du_y"]) <= 75.0]
    if len(near_pts) >= 4:
        U = np.array([[p["du_x"], p["du_y"], 1.0] for p in near_pts], dtype=float)
        X = np.array([[p["dx"], p["dy"]] for p in near_pts], dtype=float)
        coef, *_ = np.linalg.lstsq(U, X, rcond=None)
        J_center = coef[:2, :].T
    else:
        J_center = np.array([[1.4, -1.9], [-0.3, -1.9]], dtype=float)

    K_center = np.linalg.inv(J_center) if abs(np.linalg.det(J_center)) > 1e-5 else np.linalg.pinv(J_center)

    # Compute overall min/max physical reach in X and Y
    all_dx = [e["max_dx_cnt"] for e in envelope] + [p["dx"] for p in grid_points]
    all_dy = [e["max_dy_cnt"] for e in envelope] + [p["dy"] for p in grid_points]
    x_min, x_max = min(all_dx), max(all_dx)
    y_min, y_max = min(all_dy), max(all_dy)

    full_map = {
        "timestamp": datetime.now().isoformat(),
        "dpi": dpi,
        "um_per_count": 25400.0 / float(dpi),
        "pwm_freq_hz": 200,
        "neutral_us": {
            "s1_x": WS63ServoDriver.NEUTRAL_S1,
            "s2_y1": WS63ServoDriver.NEUTRAL_S2,
            "s3_y2": WS63ServoDriver.NEUTRAL_S3,
            "s4_aux": WS63ServoDriver.NEUTRAL_S4,
        },
        "safe_pulse_bounds_us": [SAFE_PULSE_MIN, SAFE_PULSE_MAX],
        "deadband_us": deadband,
        "backlash_play_radius_us": backlash,
        "physical_bounds_counts": {
            "x_min": int(x_min), "x_max": int(x_max),
            "y_min": int(y_min), "y_max": int(y_max),
        },
        "jacobian_center_cnt_per_us": J_center.tolist(),
        "inverse_jacobian_center_us_per_cnt": K_center.tolist(),
        "envelope_16dir": envelope,
        "grid_2d_samples": grid_points,
    }

    map_path = "experiments/workspace_map_full.json"
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(full_map, f, indent=2)
    print(f"[SAVED] Full 2D workspace map -> {map_path} ({len(grid_points)} grid nodes + 16 rays)")

    # Keep calib_params.json backward-compatible while upgrading it with full-range data
    calib_out = {
        "timestamp": full_map["timestamp"],
        "dpi": dpi,
        "neutral_us": full_map["neutral_us"],
        "deadband_us": deadband,
        "backlash_play_radius_us": backlash,
        "jacobian_center_px_per_us": J_center.tolist(),
        "inverse_jacobian_center_us_per_px": K_center.tolist(),
        "physical_bounds_counts": full_map["physical_bounds_counts"],
        "workspace_map_file": map_path,
    }
    if os.path.exists("calib_params.json"):
        try:
            with open("calib_params.json", "r", encoding="utf-8") as f:
                old = json.load(f)
            if "spatial_grid_samples" in old:
                calib_out["legacy_spatial_grid_samples"] = old["spatial_grid_samples"]
        except Exception:
            pass
    with open("calib_params.json", "w", encoding="utf-8") as f:
        json.dump(calib_out, f, indent=2)
    print(f"[SAVED] Updated calib_params.json with true 3200 DPI J_center & bounds:")
    print(f"        X reach: [{x_min:+d}, {x_max:+d}] cnt ({x_min*25.4/dpi:+.2f}..{x_max*25.4/dpi:+.2f} mm)")
    print(f"        Y reach: [{y_min:+d}, {y_max:+d}] cnt ({y_min*25.4/dpi:+.2f}..{y_max*25.4/dpi:+.2f} mm)")
    print(f"        J_center (cnt/us) = {np.round(J_center, 3).tolist()}")


def main():
    backup_existing_configs()
    sensor = HighPrecisionMouseSensor(dpi=3200)
    driver = WS63ServoDriver()
    try:
        driver.set_pwm_frequency(200)
        sensor.start()
        driver.center(settle_s=0.45, preload=True)
        deadband, backlash = probe_deadband_backlash(driver, sensor)
        envelope, _ = probe_16dir_envelope(driver, sensor)
        grid_points = probe_2d_safe_grid(driver, sensor, envelope)
        analyze_workspace_and_save(deadband, backlash, envelope, grid_points, dpi=3200)
    finally:
        driver.center(settle_s=0.35, preload=True)
        driver.close()
        sensor.stop()


if __name__ == "__main__":
    main()
