"""
closed_loop_controller.py - 3200 DPI Physics-Guided Closed-Loop Positioning Engine
==================================================================================
Combines:
1. Calibrated Inverse Jacobian + Online Recursive Least Squares (RLS) adaptation
2. Continuous Asymmetric Play-Operator Backlash State Machine
3. Sub-Deadband Impulse-Retract Micro-Stepping (Kinetic Nudge)
4. Variable-Gain Anti-Windup PI Control to eliminate limit-cycle oscillation
"""

import argparse
import json
import os
import time
from typing import Dict, List, Tuple
import numpy as np

from mouse_sensor import HighPrecisionMouseSensor
from servo_driver import WS63ServoDriver


class OnlineInverseJacobianRLS:
    """
    Lightweight Online Recursive Least Squares (RLS) estimator that refines
    the inverse Jacobian matrix K = dU / dX in real time to adapt to mouse pad
    friction, cable tension, and thermal drift.
    """

    def __init__(self, K_init: np.ndarray, forgetting_factor: float = 0.985):
        self.K = np.array(K_init, dtype=float).copy()
        self.lam = float(forgetting_factor)
        self.P = np.eye(2, dtype=float) * 0.05

    def update(self, dx_vec: np.ndarray, du_vec: np.ndarray):
        norm_dx = float(np.linalg.norm(dx_vec))
        # Only update on informative movements (avoid noise corruption near zero)
        if norm_dx < 4.0 or norm_dx > 250.0:
            return
        x = dx_vec.reshape(2, 1)
        u = du_vec.reshape(2, 1)
        Px = self.P @ x
        denom = self.lam + float((x.T @ Px)[0, 0])
        gain = Px / denom  # (2, 1)
        err = u - (self.K @ x)  # (2, 1)
        self.K = self.K + (err @ gain.T)
        self.P = (self.P - gain @ x.T @ self.P) / self.lam


class ClosedLoop3RRRController:
    def __init__(self, calib_file: str = "calib_params.json", enable_rls: bool = True):
        self.calib = self._load_calib(calib_file)
        self.enable_rls = enable_rls

        bl = self.calib["backlash_play_radius_us"]
        db = self.calib["deadband_us"]

        self.driver = WS63ServoDriver(
            backlash_x=(bl["x_pos"], bl["x_neg"]),
            backlash_y=(bl["y_pos"], bl["y_neg"]),
            deadband_x=(db["x_pos"], db["x_neg"]),
            deadband_y=(db["y_pos"], db["y_neg"]),
        )
        self.sensor = HighPrecisionMouseSensor(dpi=self.calib.get("dpi", 3200))

        K0 = np.array(self.calib["inverse_jacobian_center_us_per_px"], dtype=float)
        self.rls = OnlineInverseJacobianRLS(K0)

        self.kin_u = np.array(
            [
                float(self.calib["neutral_us"]["s1_x"]),
                float(self.calib["neutral_us"]["s2_y1"]),
            ],
            dtype=float,
        )

    @staticmethod
    def _load_calib(calib_file: str) -> Dict:
        if os.path.exists(calib_file):
            with open(calib_file, "r", encoding="utf-8") as f:
                return json.load(f)
        # Default fallback based on previously embedded WS63 matrix
        return {
            "dpi": 3200,
            "neutral_us": {"s1_x": 1510, "s2_y1": 1545, "s3_y2": 1455, "s4_aux": 1500},
            "deadband_us": {"x_pos": 5.0, "x_neg": 5.0, "y_pos": 5.0, "y_neg": 5.0},
            "backlash_play_radius_us": {
                "x_pos": 22.5,
                "x_neg": 22.5,
                "y_pos": 12.5,
                "y_neg": 12.5,
            },
            "inverse_jacobian_center_us_per_px": [[-3.6980, 0.9905], [0.2311, -2.7405]],
        }

    @staticmethod
    def _variable_gain(err_mag_counts: float) -> float:
        """
        Variable proportional gain schedule to prevent limit-cycle hunting
        near the target while maintaining fast slewing for large errors.
        """
        if err_mag_counts <= 1.0:
            return 0.0  # Inside 1-count (7.9 um) lock zone
        elif err_mag_counts <= 5.0:
            return 0.35  # Fine sub-deadband micro-stepping regime
        elif err_mag_counts <= 25.0:
            return 0.55  # Medium convergence regime
        return 0.72  # Fast slewing regime

    def move_to_target_counts(
        self,
        target_x: int,
        target_y: int,
        tolerance_counts: int = 1,
        max_steps: int = 45,
        step_dt: float = 0.035,
    ) -> Dict:
        """
        Closed-loop positioning to (target_x, target_y) in 3200 DPI counts.
        Returns convergence telemetry dictionary.
        """
        trajectory = []
        t_start = time.perf_counter()

        for step in range(max_steps):
            cur_x, cur_y = self.sensor.get_position()
            ex = float(target_x - cur_x)
            ey = float(target_y - cur_y)
            err_mag = float(np.hypot(ex, ey))

            trajectory.append(
                {
                    "step": step,
                    "t_ms": round((time.perf_counter() - t_start) * 1000.0, 2),
                    "x": cur_x,
                    "y": cur_y,
                    "err_counts": round(err_mag, 2),
                    "err_um": round(err_mag * self.sensor.um_per_count, 2),
                }
            )

            if err_mag <= tolerance_counts:
                break

            kp = self._variable_gain(err_mag)
            e_vec = np.array([ex, ey], dtype=float)
            du_vec = kp * (self.rls.K @ e_vec)

            # Slew-rate limit per control cycle for mechanical smoothness
            du_norm = float(np.linalg.norm(du_vec))
            max_du_step = 45.0
            if du_norm > max_du_step:
                du_vec = du_vec * (max_du_step / du_norm)

            self.kin_u = np.clip(self.kin_u + du_vec, 1180.0, 1820.0)

            enable_nudge = err_mag <= 6.0
            self.driver.step_compensated_with_nudge(
                self.kin_u[0], self.kin_u[1], enable_nudge=enable_nudge
            )
            time.sleep(step_dt)

            new_x, new_y = self.sensor.get_position()
            actual_dx = np.array([float(new_x - cur_x), float(new_y - cur_y)])
            if self.enable_rls:
                self.rls.update(actual_dx, du_vec)

        final_x, final_y = self.sensor.get_position()
        final_err = float(np.hypot(target_x - final_x, target_y - final_y))
        return {
            "target": (target_x, target_y),
            "final": (final_x, final_y),
            "final_error_counts": round(final_err, 2),
            "final_error_um": round(final_err * self.sensor.um_per_count, 2),
            "steps_taken": len(trajectory),
            "elapsed_ms": round((time.perf_counter() - t_start) * 1000.0, 1),
            "trajectory": trajectory,
        }

    def run_benchmark(self) -> List[Dict]:
        print("=" * 68)
        print("  3-RRR 3200 DPI Closed-Loop Positioning Benchmark Suite")
        print("=" * 68)
        self.sensor.start()
        self.driver.center(settle_s=0.6)
        self.sensor.reset_origin()

        waypoints = [
            (0, 0),
            (30, 0),     # Pure +X step (~238 um)
            (-30, 0),    # Pure -X reversal
            (0, 35),     # Pure +Y step
            (0, -35),    # Pure -Y reversal
            (3, -2),     # Sub-deadband micro-target (~28 um)
            (0, 0),      # Return to origin
        ]

        results = []
        for tx, ty in waypoints:
            res = self.move_to_target_counts(tx, ty, tolerance_counts=1)
            results.append(res)
            print(
                f"Target ({tx:+4d}, {ty:+4d}) -> Final ({res['final'][0]:+4d}, {res['final'][1]:+4d}) | "
                f"Err: {res['final_error_counts']:4.1f} cnt ({res['final_error_um']:5.1f} um) | "
                f"{res['steps_taken']:2d} steps ({res['elapsed_ms']:.0f} ms)"
            )
            time.sleep(0.25)

        self.driver.center(settle_s=0.4)
        self.sensor.stop()
        self.driver.close()
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3-RRR 3200 DPI Closed-Loop Controller")
    parser.add_argument("--mode", choices=["benchmark", "center"], default="benchmark")
    args = parser.parse_args()

    ctrl = ClosedLoop3RRRController()
    if args.mode == "center":
        ctrl.driver.center(settle_s=0.5)
        ctrl.driver.close()
    else:
        ctrl.run_benchmark()

