"""
workspace_model.py - Full-Workspace 2D Inverse Kinematics & Spatial ILC Field
=============================================================================
Replaces narrow [-60, +60] us 2nd-order polynomials and hardcoded 11-anchor
dictionaries with:
1. WorkspaceField2D:
   - Loads true 3200 DPI full-stroke 2D map (`experiments/workspace_map_full.json`)
   - Exact ConvexHull reachable boundary check & radial safety projection
   - Backlash-stripped Thin-Plate-Spline RBF continuous inverse manifold
     Phi_inv(x, y) -> (du_x, du_y) across [-947..+1080, -365..+1184] counts
   - Analytic finite-difference local inverse Jacobian K(x, y) anywhere in space
2. SpatialResidualILC:
   - Direction-gated Gaussian-kernel spatial residual field over (dx, dy)
   - Zero-shot generalization to unseen random targets across all 360 degrees
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import ConvexHull


class WorkspaceField2D:
    """
    Continuous full-envelope 2D forward/inverse kinematic model built from
    empirical 3200 DPI Raw Input measurements over [1165, 1835] us.
    """

    def __init__(
        self,
        map_file: str = "experiments/workspace_map_full.json",
        calib_file: str = "calib_params.json",
    ):
        self.map_file = map_file
        self.calib_file = calib_file
        self.bx_neg = 18.0
        self.bx_pos = 6.0
        self.by_neg = 4.0
        self.by_pos = 0.0

        self.hull: Optional[ConvexHull] = None
        self.rbf_inv: Optional[RBFInterpolator] = None
        self.K0 = np.array([[0.80, -0.56], [-0.08, -0.40]], dtype=float)
        self.u0_x = 1510.0
        self.u0_y = 1545.0
        self.x_bounds = (-900, 1000)
        self.y_bounds = (-340, 1100)

        self._load_and_fit()

    def _strip_u(self, u: np.ndarray) -> np.ndarray:
        ux, uy = float(u[0]), float(u[1])
        if ux > 0.0:
            kx = max(0.0, ux - self.bx_pos)
        elif ux < 0.0:
            kx = min(0.0, ux + self.bx_neg)
        else:
            kx = 0.0

        if uy > 0.0:
            ky = max(0.0, uy - self.by_pos)
        elif uy < 0.0:
            ky = min(0.0, uy + self.by_neg)
        else:
            ky = 0.0
        return np.array([kx, ky], dtype=float)

    def _unstrip_u(self, u_kin: np.ndarray) -> np.ndarray:
        kx, ky = float(u_kin[0]), float(u_kin[1])
        # Smooth ramp in |k| < 1.5 us avoids discontinuous chatter near zero coupling seams
        if kx > 1.5:
            ux = kx + self.bx_pos
        elif kx < -1.5:
            ux = kx - self.bx_neg
        elif kx >= 0.0:
            ux = kx * (1.0 + self.bx_pos / 1.5)
        else:
            ux = kx * (1.0 + self.bx_neg / 1.5)

        if ky > 1.0:
            uy = ky + self.by_pos
        elif ky < -1.0:
            uy = ky - self.by_neg
        elif ky >= 0.0:
            uy = ky * (1.0 + self.by_pos)
        else:
            uy = ky * (1.0 + self.by_neg)
        return np.array([ux, uy], dtype=float)

    def _load_and_fit(self):
        if os.path.exists(self.calib_file):
            with open(self.calib_file, "r", encoding="utf-8") as f:
                cal = json.load(f)
            self.u0_x = float(cal.get("neutral_us", {}).get("s1_x", 1510.0))
            self.u0_y = float(cal.get("neutral_us", {}).get("s2_y1", 1545.0))
            if "inverse_jacobian_center_us_per_px" in cal:
                self.K0 = np.array(cal["inverse_jacobian_center_us_per_px"], dtype=float)

        if not os.path.exists(self.map_file):
            return

        with open(self.map_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        pts = data.get("grid_2d_samples", [])
        if len(pts) < 10:
            return

        U_raw = np.array([[p["du_x"], p["du_y"]] for p in pts], dtype=float)
        XY = np.array([[p["dx"], p["dy"]] for p in pts], dtype=float)

        # Keep origin plus all non-stalled nodes (exclude |du|>5 & |xy|<4 backlash dead-zone traps)
        keep = (np.linalg.norm(U_raw, axis=1) < 1.0) | (np.linalg.norm(XY, axis=1) >= 4.0)
        U_c = U_raw[keep]
        XY_c = XY[keep]

        # Add outer 16-ray polygon vertices for ConvexHull boundary safety
        hull_pts = list(XY_c)
        for ray in data.get("envelope_16dir", []):
            hull_pts.append([float(ray["max_dx_cnt"]), float(ray["max_dy_cnt"])])
        self.hull = ConvexHull(np.array(hull_pts, dtype=float))

        bounds = data.get("physical_bounds_counts", {})
        if bounds:
            self.x_bounds = (int(bounds["x_min"]), int(bounds["x_max"]))
            self.y_bounds = (int(bounds["y_min"]), int(bounds["y_max"]))

        U_kin = np.array([self._strip_u(u) for u in U_c], dtype=float)
        self.rbf_inv = RBFInterpolator(XY_c, U_kin, kernel="thin_plate_spline", smoothing=0.5)

    def max_reach_at_angle(self, angle_rad: float) -> float:
        """Exact ray intersection with empirical 2D mechanical reachable hull (counts)."""
        if self.hull is None:
            return 350.0
        d = np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=float)
        t_vals = []
        for eq in self.hull.equations:
            a, b = eq[:2], eq[2]
            denom = float(np.dot(a, d))
            if denom > 1e-9:
                t_vals.append(-float(b) / denom)
        return float(min(t_vals)) if t_vals else 350.0

    def is_reachable(self, dx: float, dy: float, safety_ratio: float = 0.96) -> bool:
        r = float(np.hypot(dx, dy))
        if r < 1e-6:
            return True
        ang = float(np.arctan2(dy, dx))
        return r <= safety_ratio * self.max_reach_at_angle(ang)

    def clamp_to_workspace(self, dx: float, dy: float, safety_ratio: float = 0.94) -> Tuple[float, float]:
        r = float(np.hypot(dx, dy))
        if r < 1e-6:
            return 0.0, 0.0
        ang = float(np.arctan2(dy, dx))
        r_max = safety_ratio * self.max_reach_at_angle(ang)
        if r <= r_max:
            return float(dx), float(dy)
        scale = r_max / r
        return float(dx * scale), float(dy * scale)

    def predict_kinematic_u(self, x: float, y: float) -> np.ndarray:
        """Smooth kinematic servo delta (backlash-free manifold coordinates)."""
        if self.rbf_inv is None:
            return self.K0 @ np.array([float(x), float(y)], dtype=float)
        xc, yc = self.clamp_to_workspace(x, y, safety_ratio=0.98)
        return self.rbf_inv(np.array([[xc, yc]], dtype=float))[0]

    def target_to_equilibrium_du(self, dx: float, dy: float) -> np.ndarray:
        """Full backlash-compensated equilibrium pulse delta (du_x, du_y) from center preload."""
        if abs(dx) < 1e-3 and abs(dy) < 1e-3:
            return np.zeros(2, dtype=float)
        u_kin = self.predict_kinematic_u(dx, dy)
        du = self._unstrip_u(u_kin)
        # Enforce hardware pulse limits [1155, 1845] relative to neutral
        du[0] = float(np.clip(du[0], 1155.0 - self.u0_x, 1845.0 - self.u0_x))
        du[1] = float(np.clip(du[1], 1155.0 - self.u0_y, 1845.0 - self.u0_y))
        return du

    def local_inverse_jacobian(self, x: float, y: float, eps: float = 15.0) -> np.ndarray:
        """
        True position-dependent 2x2 inverse Jacobian K(x, y) = d(u_kin)/d(x, y)
        computed directly on the full-envelope 2D nonlinear RBF manifold.
        """
        if self.rbf_inv is None:
            return self.K0.copy()
        xc, yc = self.clamp_to_workspace(x, y, safety_ratio=0.92)
        p_xp = self.rbf_inv(np.array([[xc + eps, yc]], dtype=float))[0]
        p_xm = self.rbf_inv(np.array([[xc - eps, yc]], dtype=float))[0]
        p_yp = self.rbf_inv(np.array([[xc, yc + eps]], dtype=float))[0]
        p_ym = self.rbf_inv(np.array([[xc, yc - eps]], dtype=float))[0]
        col_x = (p_xp - p_xm) / (2.0 * eps)
        col_y = (p_yp - p_ym) / (2.0 * eps)
        K = np.column_stack([col_x, col_y])
        # Guard spectral norm against extreme edge gradients
        norm = float(np.linalg.norm(K, ord=2))
        if norm > 3.0:
            K = K * (3.0 / norm)
        return K


class SpatialResidualILC:
    """
    Continuous 2D Spatial Residual Iterative Learning Field.
    Every shot updates a direction-aware spatial neighborhood so unseen coordinates
    inherit calibrated actuator trim immediately across all 360 degrees.
    """

    def __init__(self, ilc_file: str = "prompts/ilc_ballistic_weights.json"):
        self.ilc_file = ilc_file
        self.params = self._load()

    def _load(self) -> Dict:
        if os.path.exists(self.ilc_file):
            try:
                with open(self.ilc_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "spatial_nodes" not in data:
                        data["spatial_nodes"] = []
                    if "target_table" not in data:
                        data["target_table"] = {}
                    return data
            except Exception:
                pass
        return {
            "boost_ratio": 1.20,
            "tau_brake_ms": 22.0,
            "active_brake_ms": 0.0,
            "micro_yaw_gain": 0.65,
            "spatial_nodes": [],
            "target_table": {},
            "iteration": 0,
            "loss_history": [],
        }

    def save(self):
        dname = os.path.dirname(self.ilc_file)
        if dname:
            os.makedirs(dname, exist_ok=True)
        with open(self.ilc_file, "w", encoding="utf-8") as f:
            json.dump(self.params, f, indent=2)

    def reset_spatial_field(self):
        self.params["spatial_nodes"] = []
        self.params["target_table"] = {}
        self.params["iteration"] = 0
        self.params["loss_history"] = []
        self.save()

    def get_correction(self, dx: float, dy: float, sigma_cnt: float = 95.0) -> np.ndarray:
        """Smooth direction-gated Gaussian interpolation over spatial support nodes."""
        nodes = self.params.get("spatial_nodes", [])
        if not nodes:
            return np.zeros(2, dtype=float)
        q = np.array([float(dx), float(dy)], dtype=float)
        q_norm = float(np.linalg.norm(q))
        if q_norm < 1e-3:
            return np.zeros(2, dtype=float)
        q_dir = q / q_norm

        w_sum = 0.0
        du_sum = np.zeros(2, dtype=float)
        inv_2s2 = 1.0 / (2.0 * (sigma_cnt ** 2))
        for n in nodes:
            p = np.array([float(n["dx"]), float(n["dy"])], dtype=float)
            p_norm = float(np.linalg.norm(p))
            if p_norm < 1e-3:
                continue
            cos_sim = float(np.dot(q_dir, p / p_norm))
            if cos_sim <= 0.15:
                continue
            dist_sq = float(np.sum((q - p) ** 2))
            # Scale radial kernel slightly wider for ultra-far (>300 cnt) targets
            rad_factor = 1.0 if q_norm < 250.0 else (250.0 / q_norm) ** 2
            w = (cos_sim ** 2) * float(np.exp(-dist_sq * inv_2s2 * rad_factor))
            if w > 1e-4:
                w_sum += w
                du_sum += w * np.array([float(n["du_x"]), float(n["du_y"])], dtype=float)
        if w_sum < 1e-4:
            return np.zeros(2, dtype=float)
        # Attenuate when max support weight is weak (far away from all seen targets)
        return du_sum / max(w_sum, 0.35)

    def update_vector(
        self,
        dx_target: float,
        dy_target: float,
        dx_actual: float,
        dy_actual: float,
        K_local: np.ndarray,
        err_counts: float,
        eta_ilc: float = 0.25,
        tol_counts: float = 3.2,
    ):
        """
        Updates the continuous 2D spatial residual field cleanly without isolated lookup overfit.
        """
        if err_counts <= tol_counts:
            return

        err_vec = np.array([float(dx_target - dx_actual), float(dy_target - dy_actual)], dtype=float)
        dist = float(np.hypot(dx_target, dy_target))

        # Spectral-radius safe damped step (< 0.837 limit), larger cap at macro distances
        max_step = 1.2 if dist <= 80.0 else (2.5 if dist <= 250.0 else 4.0)
        du_step = np.clip(eta_ilc * (K_local @ err_vec), -max_step, max_step)

        nodes: List[Dict] = self.params.setdefault("spatial_nodes", [])
        nearest_idx = -1
        nearest_d = 1e9
        for i, n in enumerate(nodes):
            d = float(np.hypot(n["dx"] - dx_target, n["dy"] - dy_target))
            if d < nearest_d:
                nearest_d = d
                nearest_idx = i

        cur_field = self.get_correction(dx_target, dy_target)
        if nearest_idx >= 0 and nearest_d <= 22.0:
            n = nodes[nearest_idx]
            new_du = np.clip(np.array([n["du_x"], n["du_y"]]) + du_step, -65.0, 65.0)
            n["du_x"] = round(float(new_du[0]), 3)
            n["du_y"] = round(float(new_du[1]), 3)
            n["hits"] = int(n.get("hits", 1)) + 1
        else:
            new_du = np.clip(cur_field + du_step, -65.0, 65.0)
            nodes.append({
                "dx": round(float(dx_target), 1),
                "dy": round(float(dy_target), 1),
                "du_x": round(float(new_du[0]), 3),
                "du_y": round(float(new_du[1]), 3),
                "hits": 1,
            })
            if len(nodes) > 120:
                nodes.pop(0)

        self.params["iteration"] = int(self.params.get("iteration", 0)) + 1
        lh = self.params.setdefault("loss_history", [])
        lh.append(float(round(abs(err_counts), 2)))
        if len(lh) > 200:
            lh.pop(0)
        self.save()


if __name__ == "__main__":
    wf = WorkspaceField2D()
    print("Loaded 2D Workspace Model:")
    for deg in [0, 45, 90, 135, 180, -135, -90, -45]:
        rmax = wf.max_reach_at_angle(np.deg2rad(deg))
        tx, ty = 0.65 * rmax * np.cos(np.deg2rad(deg)), 0.65 * rmax * np.sin(np.deg2rad(deg))
        du = wf.target_to_equilibrium_du(tx, ty)
        K = wf.local_inverse_jacobian(tx, ty)
        print(
            f"  Dir {deg:+4d} deg | 65% target=({tx:+6.1f},{ty:+6.1f}) cnt -> "
            f"du=({du[0]:+6.1f},{du[1]:+6.1f}) us | det(K)={np.linalg.det(K):+.3f}"
        )

