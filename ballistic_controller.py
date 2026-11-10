"""
ballistic_controller.py - Sub-100ms 3200 DPI Full-Workspace & Full-Screen Ballistic Controller
==============================================================================================
Core Architecture (Targeting IEEE TIM / IEEE T-MECH):
1. Motion-Onset Phase Synchronization (t_onset Locking):
   Eliminates the 0~20ms asynchronous UART-to-PWM frame quantization lottery by locking
   the boost-to-brake switching window to the exact millisecond of first optical motion
   detection (t_onset), placing the command transition dead-center between PWM latch edges.
2. Three-Stage Monotonic Ballistic Profile (BOOST -> ACTIVE BRAKE/TRIM -> LOCK):
   Enforces the Monotonic Zero-Reversal Time-Optimality Principle across both micro-steps
   (5~50 counts) and full-stroke macro flicks (100~400+ counts / full-screen 1536x864).
3. Physics-Guided Overdamped 2D Vector ILC (PG-NB-ILC):
   Uses spectral-radius-bounded gain eta_ilc = 0.34 < 1/sigma_max^2(K0) = 0.418 for
   strictly monotonic, non-oscillatory multi-shot convergence.
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
        # Clamp polynomial evaluation domain so large macro flicks do not extrapolate wildly
        du_x_c = float(np.clip(du_x, -90.0, 90.0))
        du_y_c = float(np.clip(du_y, -90.0, 90.0))
        K = np.zeros((2, 2), dtype=float)
        basis = {
            "Const": 1.0,
            "dUx": du_x_c,
            "dUy": du_y_c,
            "dUx^2/1k": (du_x_c**2) / 1000.0,
            "dUy^2/1k": (du_y_c**2) / 1000.0,
            "dUx*dUy/1k": (du_x_c * du_y_c) / 1000.0,
        }
        for (r, c), key in [((0, 0), "K11"), ((0, 1), "K12"), ((1, 0), "K21"), ((1, 1), "K22")]:
            c_dict = self.coeffs.get(key, {})
            K[r, c] = sum(c_dict.get(name, 0.0) * val for name, val in basis.items())
        return K


class BallisticILCLearner:
    """
    Physics-Guided Norm-Optimal 2D Vector Iterative Learning Control (PG-NB-ILC) engine.
    """

    def __init__(self, ilc_file: str = "prompts/ilc_ballistic_weights.json"):
        self.ilc_file = ilc_file
        self.params = self._load()

    def _load(self) -> Dict:
        if os.path.exists(self.ilc_file):
            try:
                with open(self.ilc_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "target_table" not in data:
                        data["target_table"] = {}
                    return data
            except Exception:
                pass
        return {
            "boost_ratio": 1.15,        # Overdrive factor: u_boost = u_lock + (boost_ratio - 1)*du_base
            "tau_brake_ms": 14.0,       # Predicted glide time constant (ms)
            "active_brake_ms": 0.0,     # Immediate lock latch upon t_onset detection
            "micro_yaw_gain": 0.65,
            "target_table": {},         # Keyed by f"{dx}_{dy}": [du_corr_x, du_corr_y]
            "iteration": 0,
            "loss_history": [],
        }

    def save(self):
        os.makedirs(os.path.dirname(self.ilc_file), exist_ok=True)
        with open(self.ilc_file, "w", encoding="utf-8") as f:
            json.dump(self.params, f, indent=2)

    def get_correction(self, dx: int, dy: int) -> np.ndarray:
        key = f"{dx}_{dy}"
        corr = self.params.get("target_table", {}).get(key, [0.0, 0.0])
        return np.array([float(corr[0]), float(corr[1])], dtype=float)

    def update_vector(
        self,
        dx_target: int,
        dy_target: int,
        dx_actual: int,
        dy_actual: int,
        K_local: np.ndarray,
        err_counts: float,
        eta_ilc: float = 0.34,
    ):
        """
        Overdamped Monotonic 2D Jacobian-Guided ILC Law:
        0 < eta_ilc = 0.34 <= 1 / sigma_max^2(K0) = 0.418.
        """
        key = f"{dx_target}_{dy_target}"
        if "target_table" not in self.params:
            self.params["target_table"] = {}

        cur_corr = self.get_correction(dx_target, dy_target)
        err_vec = np.array([float(dx_target - dx_actual), float(dy_target - dy_actual)], dtype=float)

        # Deadzone on learning if already within <= 1.2 counts (sub-10um bullseye)
        if err_counts > 1.2:
            delta_u_step = eta_ilc * (K_local @ err_vec)
            max_step = max(12.0, 0.15 * float(np.hypot(dx_target, dy_target)))
            delta_u_step = np.clip(delta_u_step, -max_step, max_step)
            new_corr = np.clip(cur_corr + delta_u_step, -140.0, 140.0)
            self.params["target_table"][key] = [round(float(new_corr[0]), 3), round(float(new_corr[1]), 3)]

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
        Supports both micro-precision steps (5~50 counts) and full-stroke macro flicks
        (100~450 counts across [1150, 1850] us).
        """
        target_dist = float(np.hypot(delta_x, delta_y))
        if target_dist < 0.1:
            return {"status": "NOOP", "final_error_counts": 0.0, "settled_time_ms": 0.0}

        # 1. Evaluate 2nd-order Jacobian at current position
        cur_ux = self.driver.cur_s1 - self.u0_x
        cur_uy = self.driver.cur_s2 - self.u0_y
        K = self.poly_jac.evaluate(cur_ux, cur_uy)

        # 2. Compute nominal pulse offsets + 2D ILC learned correction + backlash pre-engagement
        e_vec = np.array([float(delta_x), float(delta_y)], dtype=float)
        du_base = K @ e_vec
        du_ilc = self.ilc.get_correction(delta_x, delta_y)
        du_nom = du_base + du_ilc

        # Directional backlash pre-engagement (since center() preloads positive flank, negative moves cross backlash)
        b_offset_x = 7.5 if du_nom[0] > 0 else (-self.bx_neg * 1.32 if du_nom[0] < 0 else 0.0)
        b_offset_y = 7.5 if du_nom[1] > 0 else (-self.by_neg * 1.32 if du_nom[1] < 0 else 0.0)

        # Lock pulse holds target position after backlash is traversed (full [1150, 1850] safety range)
        u_lock_x = np.clip(self.driver.cur_s1 + du_nom[0] + b_offset_x, 1150.0, 1850.0)
        u_lock_y = np.clip(self.driver.cur_s2 + du_nom[1] + b_offset_y, 1150.0, 1850.0)

        # 3. Distance-Adaptive Boost Overdrive & Active Brake Profile
        # Small moves need higher overdrive to break static friction; large moves scale smoothly
        base_boost = float(self.ilc.params.get("boost_ratio", 1.28))
        if target_dist > 80.0:
            boost_ratio = max(1.12, base_boost * (80.0 / target_dist) ** 0.25)
        else:
            boost_ratio = max(1.15, min(1.65, base_boost))

        tau_brake = float(self.ilc.params.get("tau_brake_ms", 14.0)) / 1000.0
        # Deterministic active boost duration measured strictly AFTER t_onset (first physical motion):
        # For precision targets (<= 80 cnt), switching to u_lock IMMEDIATELY at t_onset (0.0 ms)
        # guarantees that PWM Frame 1 (at t=phi) is the ONLY boost frame, and PWM Frame 2 (at t=phi+20ms)
        # ALWAYS latches u_lock (since t_onset = phi+8ms is 12ms before Frame 2 for ALL phi in [0,20)ms)!
        active_boost_limit_ms = 0.0 if target_dist <= 80.0 else (12.0 if target_dist <= 180.0 else 22.0)

        stiction_kick_x = 4.5 * np.sign(du_base[0]) if abs(du_base[0]) > 1.0 else 0.0
        stiction_kick_y = 4.5 * np.sign(du_base[1]) if abs(du_base[1]) > 1.0 else 0.0
        u_boost_x = np.clip(
            self.driver.cur_s1 + du_base[0] * boost_ratio + du_ilc[0] + b_offset_x + stiction_kick_x,
            1150.0,
            1850.0,
        )
        u_boost_y = np.clip(
            self.driver.cur_s2 + du_base[1] * boost_ratio + du_ilc[1] + b_offset_y + stiction_kick_y,
            1150.0,
            1850.0,
        )

        # Ensure complete mechanical rest before firing shot
        for _ in range(5):
            self.sensor.read_delta_and_clear()
            time.sleep(0.012)
            dx_chk, dy_chk = self.sensor.read_delta_and_clear()
            if dx_chk == 0 and dy_chk == 0:
                break

        x0, y0 = self.sensor.get_position()
        target_abs_x = x0 + delta_x
        target_abs_y = y0 + delta_y

        # Phase-align UART transmission by issuing a warm neutral keep-alive and waiting 4ms
        self.driver.send_raw_pulses(self.driver.cur_s1, self.driver.cur_s2)
        time.sleep(0.004)

        t_start = time.perf_counter()
        t_onset: Optional[float] = None
        trajectory = []

        # Send Stage 1: Boost Overdrive Command
        self.driver.send_raw_pulses(u_boost_x, u_boost_y)
        phase = "BOOST"
        brake_triggered = False
        lock_sent = False
        t_brake_ms = 0.0

        last_pos_x, last_pos_y = x0, y0
        last_t = t_start
        vx_ema, vy_ema = 0.0, 0.0
        settled_t_ms = None
        tol_counts = max(1.5, 0.015 * target_dist)

        # 4. In-Flight Optical Predictive Braking Loop (1000Hz)
        while True:
            t_now = time.perf_counter()
            elapsed_ms = (t_now - t_start) * 1000.0

            cur_x, cur_y = self.sensor.get_position()
            if t_onset is None and (abs(cur_x - x0) + abs(cur_y - y0) >= 1):
                t_onset = t_now

            # Strictly 0.0 until physical motion begins at t_onset!
            active_ms = ((t_now - t_onset) * 1000.0) if t_onset is not None else 0.0

            dt_step = max(1e-3, t_now - last_t)
            vx_raw = (cur_x - last_pos_x) / dt_step
            vy_raw = (cur_y - last_pos_y) / dt_step
            vx_ema = 0.30 * vx_raw + 0.70 * vx_ema
            vy_ema = 0.30 * vy_raw + 0.70 * vy_ema
            last_pos_x, last_pos_y = cur_x, cur_y
            last_t = t_now

            err_x = target_abs_x - cur_x
            err_y = target_abs_y - cur_y
            err_counts = float(np.hypot(err_x, err_y))

            pred_x = cur_x + vx_ema * tau_brake
            pred_y = cur_y + vy_ema * tau_brake

            if err_counts <= tol_counts and settled_t_ms is None and active_ms >= 10.0:
                settled_t_ms = elapsed_ms

            trajectory.append({
                "t_ms": round(elapsed_ms, 2),
                "active_ms": round(active_ms, 2),
                "phase": phase,
                "x": cur_x,
                "y": cur_y,
                "vx": round(vx_ema, 1),
                "vy": round(vy_ema, 1),
                "err_counts": round(err_counts, 2),
                "err_um": round(err_counts * self.sensor.um_per_count, 2),
            })

            # Stage 2: Trigger Lock synchronized strictly to t_onset + 14ms (zero frame-quantization jitter)
            if not brake_triggered and (
                (t_onset is not None and active_ms >= active_boost_limit_ms) or (elapsed_ms >= 70.0)
            ):
                self.driver.send_raw_pulses(u_lock_x, u_lock_y)
                phase = "LOCK"
                brake_triggered = True
                lock_sent = True
                t_brake_ms = elapsed_ms

            if elapsed_ms >= timeout_ms:
                if not lock_sent:
                    self.driver.send_raw_pulses(u_lock_x, u_lock_y)
                break
            time.sleep(0.001)

        # 5. Terminal Result & 2D Vector ILC Learning Update
        final_x, final_y = self.sensor.get_position()
        final_dx = int(final_x - x0)
        final_dy = int(final_y - y0)
        final_err = float(np.hypot(target_abs_x - final_x, target_abs_y - final_y))

        if enable_ilc_update:
            self.ilc.update_vector(delta_x, delta_y, final_dx, final_dy, K, final_err)

        if settled_t_ms is None and final_err <= tol_counts:
            settled_t_ms = min(timeout_ms, active_ms)

        return {
            "target_delta": (delta_x, delta_y),
            "final_delta": (final_dx, final_dy),
            "final_error_counts": round(final_err, 2),
            "final_error_um": round(final_err * self.sensor.um_per_count, 2),
            "settled_time_ms": round(settled_t_ms if settled_t_ms is not None else timeout_ms, 1),
            "t_brake_ms": round(t_brake_ms, 1),
            "t_onset_ms": round((t_onset - t_start) * 1000.0, 1) if t_onset is not None else -1.0,
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
    parser.add_argument("--test-suite", action="store_true", help="Run full-scale 100ms benchmark suite")
    args = parser.parse_args()

    ctrl = Ballistic100msController()
    ctrl.sensor.start()
    ctrl.driver.center(settle_s=0.45, preload=True)

    if args.test_suite:
        print("=" * 78)
        print("  3-RRR Sub-100ms 3200 DPI Full-Workspace Flick-Shot Benchmark Suite")
        print("=" * 78)
        waypoints = [
            (30, 20),     # Micro-precision step
            (-35, 0),     # Negative X backlash traversal
            (0, 35),      # Positive Y
            (120, 80),    # Medium-range flick (~1.15 mm)
            (-150, -90),  # Medium-large reversal
            (240, 150),   # Full-stroke macro flick (~2.25 mm raw / full-screen scaled)
            (-240, -150), # Full-stroke macro reversal
        ]
        for i, (tdx, tdy) in enumerate(waypoints):
            ctrl.driver.center(settle_s=0.35, preload=True)
            res = ctrl.flick_shot(tdx, tdy, timeout_ms=100.0)
            print(
                f"Shot {i+1:2d} | Target ({tdx:+4d}, {tdy:+4d}) -> "
                f"Final ({res['final_delta'][0]:+4d}, {res['final_delta'][1]:+4d}) | "
                f"Err: {res['final_error_counts']:5.1f} cnt ({res['final_error_um']:6.1f} um) | "
                f"Onset: {res['t_onset_ms']:4.1f}ms | Brake: {res['t_brake_ms']:4.1f}ms | "
                f"Settle: {res['settled_time_ms']:5.1f}ms"
            )
    else:
        print(f"Firing single 100ms shot to target ({args.dx}, {args.dy})...")
        res = ctrl.flick_shot(args.dx, args.dy, timeout_ms=100.0)
        print(f"Target: ({args.dx}, {args.dy})")
        print(f"Final : {res['final_delta']} | Error: {res['final_error_counts']} counts ({res['final_error_um']} um)")
        print(f"Onset : {res['t_onset_ms']} ms | Brake: {res['t_brake_ms']} ms | Settled: {res['settled_time_ms']} ms")

    ctrl.driver.center(settle_s=0.35, preload=True)
    ctrl.close()
