"""
ballistic_controller.py - Sub-100ms 3200 DPI In-Flight Optical Predictive Braking Controller
=============================================================================================
Core Concept (Targeting IEEE TIM / IEEE T-MECH):
Achieves sub-100ms micrometer-level positioning on low-cost MG996R servos with severe
backlash (50us) by enforcing the "Monotonic Zero-Reversal Time-Optimality Principle".

Three-Phase Ballistic Profile:
1. Phase 1 (0 ~ 35 ms)  : Boost Overdrive - Overcomes gear inertia & static friction.
2. Phase 2 (35 ~ 75 ms) : In-Flight Optical Predictive Braking (IF-OPB) - Tracks 1000Hz
                          optical velocity v(t), triggers premature brake at S(x, v) = 0.
3. Phase 3 (75 ~ 100 ms): Lock & Micro-Yaw Trim - Dissipates remaining kinetic energy
                          without reversing gear direction.
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from mouse_sensor import HighPrecisionMouseSensor
from servo_driver import WS63ServoDriver


class SpatialInverseJacobian2ndOrder:
    def __init__(self, poly_file: str = "prompts/fitted_jacobian_poly.json"):
        self.coeffs = self._load_coeffs(poly_file)

    @staticmethod
    def _load_coeffs(poly_file: str) -> Dict:
        if os.path.exists(poly_file):
            with open(poly_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "K11": {"Const": 0.8851, "dUx": -0.0012, "dUy": -0.0003, "dUx^2/1k": -0.015, "dUy^2/1k": -0.002, "dUx*dUy/1k": 0.0005},
            "K12": {"Const": -0.5632, "dUx": 0.0001, "dUy": -0.0045, "dUx^2/1k": 0.030, "dUy^2/1k": -0.045, "dUx*dUy/1k": -0.010},
            "K21": {"Const": -0.2414, "dUx": -0.0005, "dUy": -0.0010, "dUx^2/1k": -0.004, "dUy^2/1k": -0.002, "dUx*dUy/1k": -0.002},
            "K22": {"Const": -0.4828, "dUx": -0.0004, "dUy": -0.0025, "dUx^2/1k": 0.015, "dUy^2/1k": -0.055, "dUx*dUy/1k": -0.006},
        }

    def evaluate(self, du_x: float, du_y: float) -> np.ndarray:
        K = np.zeros((2, 2), dtype=float)
        basis = {
            "Const": 1.0,
            "dUx": float(du_x),
            "dUy": float(du_y),
            "dUx^2/1k": float(du_x**2) / 1000.0,
            "dUy^2/1k": float(du_y**2) / 1000.0,
            "dUx*dUy/1k": float(du_x * du_y) / 1000.0,
        }
        for (r, c), key in [((0, 0), "K11"), ((0, 1), "K12"), ((1, 0), "K21"), ((1, 1), "K22")]:
            c_dict = self.coeffs.get(key, {})
            K[r, c] = sum(c_dict.get(name, 0.0) * val for name, val in basis.items())
        return K


class BallisticILCLearner:
    """
    Norm-Optimal Iterative Learning Control (ILC) engine for ballistic flick-shots.
    Adapts boost duration, overdrive ratio, and brake lead time based on past terminal errors.
    """

    def __init__(self, ilc_file: str = "prompts/ilc_ballistic_weights.json"):
        self.ilc_file = ilc_file
        self.params = self._load()

    def _load(self) -> Dict:
        if os.path.exists(self.ilc_file):
            try:
                with open(self.ilc_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "boost_ratio": 1.25,        # Overdrive factor: u_boost = u_lock * ratio
            "tau_brake_ms": 28.0,       # Predicted glide time constant
            "micro_yaw_gain": 0.65,     # Yaw compensation sensitivity
            "iteration": 0,
            "loss_history": [],
        }

    def save(self):
        os.makedirs(os.path.dirname(self.ilc_file), exist_ok=True)
        with open(self.ilc_file, "w", encoding="utf-8") as f:
            json.dump(self.params, f, indent=2)

    def update(self, target_dist: float, final_dist: float, err_counts: float):
        """Monotonic ILC gradient step on glide time constant and boost ratio."""
        if target_dist < 4.0:
            return
        overshoot = final_dist - target_dist
        lr_tau = 0.40
        lr_boost = 0.005

        # If overshot, brake earlier (increase tau_brake) and decrease boost ratio
        if overshoot > 1.5:
            self.params["tau_brake_ms"] = min(50.0, self.params["tau_brake_ms"] + lr_tau)
            self.params["boost_ratio"] = max(1.05, self.params["boost_ratio"] - lr_boost)
        # If undershot, brake later (decrease tau_brake) and increase boost ratio
        elif overshoot < -1.5:
            self.params["tau_brake_ms"] = max(12.0, self.params["tau_brake_ms"] - lr_tau)
            self.params["boost_ratio"] = min(1.90, self.params["boost_ratio"] + lr_boost)

        self.params["iteration"] += 1
        self.params["loss_history"].append(float(round(abs(err_counts), 2)))
        if len(self.params["loss_history"]) > 200:
            self.params["loss_history"].pop(0)
        self.save()


class Ballistic100msController:
    def __init__(self, calib_file: str = "calib_params.json"):
        with open(calib_file, "r", encoding="utf-8") as f:
            self.calib = json.load(f)

        self.sensor = HighPrecisionMouseSensor(dpi=self.calib.get("dpi", 3200))
        self.driver = WS63ServoDriver()
        self.poly_jac = SpatialInverseJacobian2ndOrder()
        self.ilc = BallisticILCLearner()

        self.u0_x = float(self.calib["neutral_us"]["s1_x"])
        self.u0_y = float(self.calib["neutral_us"]["s2_y1"])

        # Backlash engagement radii
        self.bx_pos = float(self.calib["backlash_play_radius_us"]["x_pos"])
        self.bx_neg = float(self.calib["backlash_play_radius_us"]["x_neg"])
        self.by_pos = float(self.calib["backlash_play_radius_us"]["y_pos"])
        self.by_neg = float(self.calib["backlash_play_radius_us"]["y_neg"])

    def flick_shot(
        self,
        delta_x: int,
        delta_y: int,
        timeout_ms: float = 100.0,
        enable_ilc_update: bool = True,
    ) -> Dict:
        """
        Executes a sub-100ms ballistic flick shot to relative target (delta_x, delta_y).
        Returns complete high-resolution telemetry dictionary.
        """
        target_dist = float(np.hypot(delta_x, delta_y))
        if target_dist < 0.1:
            return {"status": "NOOP", "final_error_counts": 0.0, "time_to_settle_ms": 0.0}

        # 1. Evaluate 2nd-order Jacobian at current position
        cur_ux = self.driver.cur_s1 - self.u0_x
        cur_uy = self.driver.cur_s2 - self.u0_y
        K = self.poly_jac.evaluate(cur_ux, cur_uy)

        # 2. Compute nominal pulse offsets and backlash pre-engagement
        e_vec = np.array([float(delta_x), float(delta_y)], dtype=float)
        du_nom = K @ e_vec

        # Directional backlash pre-engagement
        b_offset_x = self.bx_pos if du_nom[0] > 0 else (-self.bx_neg if du_nom[0] < 0 else 0.0)
        b_offset_y = self.by_pos if du_nom[1] > 0 else (-self.by_neg if du_nom[1] < 0 else 0.0)

        # Lock pulse holds target position after backlash is traversed
        u_lock_x = self.driver.cur_s1 + du_nom[0] + 0.65 * b_offset_x
        u_lock_y = self.driver.cur_s2 + du_nom[1] + 0.65 * b_offset_y

        # Clamp lock pulse to physical safety envelope
        u_lock_x = np.clip(u_lock_x, 1200.0, 1800.0)
        u_lock_y = np.clip(u_lock_y, 1200.0, 1800.0)

        # 3. Phase 1: Boost Overdrive Calculation (includes full backlash clearance)
        boost_ratio = max(1.20, min(2.0, self.ilc.params["boost_ratio"]))
        tau_brake = self.ilc.params["tau_brake_ms"] / 1000.0  # seconds

        u_boost_x = self.driver.cur_s1 + du_nom[0] * boost_ratio + b_offset_x
        u_boost_y = self.driver.cur_s2 + du_nom[1] * boost_ratio + b_offset_y
        u_boost_x = np.clip(u_boost_x, 1170.0, 1830.0)
        u_boost_y = np.clip(u_boost_y, 1170.0, 1830.0)

        # Start motion
        self.sensor.read_delta_and_clear()
        x0, y0 = self.sensor.get_position()
        target_abs_x = x0 + delta_x
        target_abs_y = y0 + delta_y

        t_start = time.perf_counter()
        trajectory = []

        # Send Phase 1 Boost Command
        self.driver.send_raw_pulses(u_boost_x, u_boost_y)
        phase = "BOOST"
        brake_triggered = False
        t_brake_ms = 0.0

        last_pos_x, last_pos_y = x0, y0
        last_t = t_start
        settled_t_ms = None

        # 4. Phase 2: In-Flight Optical Predictive Braking (1000Hz Loop)
        while True:
            t_now = time.perf_counter()
            elapsed_ms = (t_now - t_start) * 1000.0

            cur_x, cur_y = self.sensor.get_position()
            dt_step = max(1e-4, t_now - last_t)
            vx = (cur_x - last_pos_x) / dt_step  # counts/sec
            vy = (cur_y - last_pos_y) / dt_step
            last_pos_x, last_pos_y = cur_x, cur_y
            last_t = t_now

            # Current error to target
            err_x = target_abs_x - cur_x
            err_y = target_abs_y - cur_y
            err_counts = float(np.hypot(err_x, err_y))

            # Predicted landing location along travel trajectory
            pred_x = cur_x + vx * tau_brake
            pred_y = cur_y + vy * tau_brake
            pred_err_x = target_abs_x - pred_x
            pred_err_y = target_abs_y - pred_y

            # Check if settled within tolerance (<= 1.5 counts / 12 um)
            if err_counts <= 1.5 and settled_t_ms is None and elapsed_ms >= 40.0:
                settled_t_ms = elapsed_ms

            trajectory.append({
                "t_ms": round(elapsed_ms, 2),
                "phase": phase,
                "x": cur_x,
                "y": cur_y,
                "vx": round(vx, 1),
                "vy": round(vy, 1),
                "err_counts": round(err_counts, 2),
                "err_um": round(err_counts * self.sensor.um_per_count, 2),
            })

            # Check Switching Surface S(x, v) = 0 for In-Flight Early Braking
            if not brake_triggered and elapsed_ms >= 35.0:
                # Monotonic dot product check: if predicted vector has reached target
                progress_pred = (pred_x - x0) * delta_x + (pred_y - y0) * delta_y
                target_mag_sq = delta_x**2 + delta_y**2
                speed = float(np.hypot(vx, vy))

                # Trigger brake if predicted landing reaches 90% of target distance, or at 72ms if already moving
                if (progress_pred >= 0.88 * target_mag_sq and speed > 50.0) or (elapsed_ms >= 72.0):
                    # Switch to Lock Pulse
                    self.driver.send_raw_pulses(u_lock_x, u_lock_y)
                    phase = "LOCK"
                    brake_triggered = True
                    t_brake_ms = elapsed_ms

            # Phase 3: Micro-Yaw Fine Trim around 80ms if small error remains
            if elapsed_ms >= 80.0 and phase == "LOCK":
                if 1.5 < err_counts <= 6.0:
                    # Redundant micro-yaw differential kick
                    yaw_kick = float(np.clip(-err_y * 1.5, -20.0, 20.0))
                    self.driver.send_raw_pulses(u_lock_x, u_lock_y + yaw_kick, s3=3000.0 - u_lock_y - yaw_kick)
                    phase = "MICRO_YAW"

            if elapsed_ms >= timeout_ms:
                break
            time.sleep(0.001)  # 1000Hz sampling

        # 5. Terminal Result & ILC Learning Update
        final_x, final_y = self.sensor.get_position()
        final_err = float(np.hypot(target_abs_x - final_x, target_abs_y - final_y))
        actual_dist = float(np.hypot(final_x - x0, final_y - y0))

        if enable_ilc_update:
            self.ilc.update(target_dist, actual_dist, final_err)

        if settled_t_ms is None and final_err <= 1.5:
            settled_t_ms = timeout_ms

        return {
            "target_delta": (delta_x, delta_y),
            "final_delta": (final_x - x0, final_y - y0),
            "final_error_counts": round(final_err, 2),
            "final_error_um": round(final_err * self.sensor.um_per_count, 2),
            "settled_time_ms": round(settled_t_ms if settled_t_ms is not None else timeout_ms, 1),
            "t_brake_ms": round(t_brake_ms, 1),
            "trajectory": trajectory,
        }

    def close(self):
        try:
            self.sensor.stop()
            self.driver.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sub-100ms 3200 DPI Ballistic Controller")
    parser.add_argument("--dx", type=int, default=30, help="Target Delta X (counts)")
    parser.add_argument("--dy", type=int, default=0, help="Target Delta Y (counts)")
    parser.add_argument("--test-suite", action="store_true", help="Run full 100ms benchmark suite")
    args = parser.parse_args()

    ctrl = Ballistic100msController()
    ctrl.sensor.start()
    ctrl.driver.center(settle_s=0.5)

    if args.test_suite:
        print("=" * 72)
        print("  3-RRR Sub-100ms 3200 DPI Ballistic Flick-Shot Benchmark Suite")
        print("=" * 72)
        waypoints = [
            (30, 0),    # Medium step X (~238 um)
            (-30, 0),   # Reversal X
            (0, 35),    # Medium step Y (~278 um)
            (0, -35),   # Reversal Y
            (100, 50),  # Large step (~900 um)
            (-100, -50),# Large reversal
            (5, -4),    # Fine step (~50 um)
        ]
        results = []
        for i, (tdx, tdy) in enumerate(waypoints):
            res = ctrl.flick_shot(tdx, tdy, timeout_ms=100.0)
            results.append(res)
            print(
                f"Shot {i+1:2d} | Target ({tdx:+4d}, {tdy:+4d}) -> "
                f"Final ({res['final_delta'][0]:+4d}, {res['final_delta'][1]:+4d}) | "
                f"Err: {res['final_error_counts']:4.1f} cnt ({res['final_error_um']:5.1f} um) | "
                f"Brake: {res['t_brake_ms']:4.1f} ms | Settle: {res['settled_time_ms']:5.1f} ms"
            )
            time.sleep(0.30)
    else:
        print(f"Firing single 100ms shot to target ({args.dx}, {args.dy})...")
        res = ctrl.flick_shot(args.dx, args.dy, timeout_ms=100.0)
        print(f"Target: ({args.dx}, {args.dy})")
        print(f"Final : {res['final_delta']} | Error: {res['final_error_counts']} counts ({res['final_error_um']} um)")
        print(f"Settled Time: {res['settled_time_ms']} ms | Brake Trigger: {res['t_brake_ms']} ms")

    ctrl.driver.center(settle_s=0.3)
    ctrl.close()
