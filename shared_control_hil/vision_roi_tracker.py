"""
vision_roi_tracker.py - Module 2: Low-Latency 256x256 Center ROI Capture, Dynamic Recoil Crosshair & Target CV-KF Tracker
========================================================================================================================
Features:
1. Center 256x256 ROI capture via dxcam -> mss -> native Windows GDI BitBlt fallback (< 1.5 ms),
   plus zero-overhead frame/detection injection for HIL benchmark replay.
2. Vectorized RGB threshold + separable cross-kernel convolution (< 0.2 ms) to track dynamic
   recoil crosshair c(t) under cl_crosshair_recoil 1, with Kalman/recoil prediction fallback.
3. Enemy head/chest keypoint extraction (optional ONNX/YOLO hook + fast contour/synthetic fallback)
   with per-target 4-state Constant-Velocity Kalman Filter (CV-KF) velocity & lead prediction:
       h_hat_i(t + t_lead) = h_i(t) + k_lead * v_target_i(t)
"""

import ctypes
import ctypes.wintypes as wintypes
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


DEFAULT_YOLO_WEIGHTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "weights",
    "cs2_head_yolov8n.pt",
)


@dataclass
class TargetDetection:
    target_id: int
    head_px: np.ndarray        # (u, v) in ROI or screen pixels
    head_radius_px: float
    chest_px: np.ndarray       # (u, v) in ROI or screen pixels
    chest_radius_px: float
    confidence: float = 1.0
    vel_px_s: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    head_pred_px: Optional[np.ndarray] = None
    chest_pred_px: Optional[np.ndarray] = None
    head_bbox_px: Optional[Tuple[float, float, float, float]] = None
    chest_bbox_px: Optional[Tuple[float, float, float, float]] = None
    seen_frames: int = 1


class ConstantVelocityKF:
    """4-state [x, y, vx, vy] Constant-Velocity Kalman Filter in 2D image or count space."""

    def __init__(self, x0: float = 0.0, y0: float = 0.0, q_acc: float = 4500.0, r_pos: float = 0.8):
        self.x = np.array([x0, y0, 0.0, 0.0], dtype=float)
        self.P = np.diag([4.0, 4.0, 2500.0, 2500.0])
        self.q_acc = float(q_acc)
        self.R = np.eye(2, dtype=float) * float(r_pos)
        self.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
        self.I4 = np.eye(4, dtype=float)
        self.last_t: Optional[float] = None

    def step(self, z_xy: Optional[np.ndarray], dt_s: float) -> Tuple[np.ndarray, np.ndarray]:
        dt = float(max(1e-4, min(0.25, dt_s)))
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        dt2 = 0.5 * dt * dt
        G = np.array([[dt2, 0.0], [0.0, dt2], [dt, 0.0], [0.0, dt]], dtype=float)
        Q = (G @ G.T) * self.q_acc

        # Predict
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        # Update if measurement present
        if z_xy is not None:
            z = np.asarray(z_xy, dtype=float).reshape(2)
            y = z - self.x[:2]
            S = self.P[:2, :2] + self.R
            # Closed-form 2x2 inverse for sub-0.02ms execution
            det = float(S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0])
            if abs(det) > 1e-12:
                inv_S = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]], dtype=float) / det
            else:
                inv_S = np.linalg.pinv(S)
            K = self.P[:, :2] @ inv_S
            self.x = self.x + K @ y
            self.P = (self.I4 - K @ self.H) @ self.P

        return self.x[:2].copy(), self.x[2:].copy()

    def predict_lead(self, k_lead_s: float) -> np.ndarray:
        return self.x[:2] + float(k_lead_s) * self.x[2:]


class DynamicCrosshairTracker:
    """
    Fast (< 0.2 ms) crosshair tracker for cl_crosshair_recoil 1 in a 256x256 ROI.
    Combines RGB threshold segmentation + separable 1D cross-response accumulation,
    with CV-KF + recoil fallback when occluded by muzzle flash/smoke.
    """

    def __init__(
        self,
        roi_size: int = 256,
        target_rgb: Tuple[int, int, int] = (0, 255, 80),
        rgb_tol: int = 70,
    ):
        self.roi_size = int(roi_size)
        self.target_rgb = np.array(target_rgb, dtype=np.int16)
        self.rgb_tol = int(rgb_tol)
        center = 0.5 * self.roi_size
        self.kf = ConstantVelocityKF(x0=center, y0=center, q_acc=20000.0, r_pos=0.4)
        self.last_pos = np.array([center, center], dtype=float)

    def detect_in_frame(
        self,
        roi_rgb: np.ndarray,
        dt_s: float = 0.004,
        recoil_prior_px: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool]:
        """Locate recoil crosshair c(t) inside roi_rgb (HxWx3 uint8). Returns (pos_xy, detected_flag)."""
        rgb_i16 = roi_rgb.astype(np.int16)
        diff = np.abs(rgb_i16 - self.target_rgb)
        mask_target = (diff[:, :, 0] <= self.rgb_tol) & (diff[:, :, 1] <= self.rgb_tol) & (diff[:, :, 2] <= self.rgb_tol)
        # Also match crisp classic green / cyan CS2 crosshairs under varying background lighting
        r_ch, g_ch, b_ch = rgb_i16[:, :, 0], rgb_i16[:, :, 1], rgb_i16[:, :, 2]
        mask_cs2 = (g_ch > 185) & (r_ch < 130) & ((g_ch - r_ch) > 50)
        mask = mask_target | mask_cs2
        pixel_count = int(mask.sum())

        z_meas: Optional[np.ndarray] = None
        detected = False
        if 4 <= pixel_count <= 450:
            # Separable row & col projections localize the cross center in ~0.08 ms
            row_sum = mask.sum(axis=1).astype(float)
            col_sum = mask.sum(axis=0).astype(float)
            # Emphasize dense crosshair arms over isolated noise
            v_idx = int(np.argmax(row_sum))
            u_idx = int(np.argmax(col_sum))
            v_win = slice(max(0, v_idx - 10), min(mask.shape[0], v_idx + 11))
            u_win = slice(max(0, u_idx - 10), min(mask.shape[1], u_idx + 11))
            ys, xs = np.nonzero(mask[v_win, u_win])
            if len(xs) >= 3:
                u_c = float(xs.mean() + u_win.start)
                v_c = float(ys.mean() + v_win.start)
                z_meas = np.array([u_c, v_c], dtype=float)
                detected = True

        if not detected and recoil_prior_px is not None:
            z_meas = np.asarray(recoil_prior_px, dtype=float)

        pos, _ = self.kf.step(z_meas, dt_s=dt_s)
        self.last_pos = pos
        return pos, detected


class _WindowsGDICapture:
    """Zero-dependency Windows BitBlt capture for a fixed center screen ROI (< 1.5 ms)."""

    def __init__(self, width: int, height: int, roi_size: int):
        self.user32 = ctypes.windll.user32
        self.gdi32 = ctypes.windll.gdi32
        self.roi_size = int(roi_size)
        self.left = max(0, (int(width) - self.roi_size) // 2)
        self.top = max(0, (int(height) - self.roi_size) // 2)

        self.hwnd = self.user32.GetDesktopWindow()
        self.hdc_screen = self.user32.GetDC(self.hwnd)
        self.hdc_mem = self.gdi32.CreateCompatibleDC(self.hdc_screen)
        self.hbmp = self.gdi32.CreateCompatibleBitmap(self.hdc_screen, self.roi_size, self.roi_size)
        self.gdi32.SelectObject(self.hdc_mem, self.hbmp)

        class _BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", wintypes.DWORD),
                ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD),
            ]

        self._bmi = _BITMAPINFOHEADER()
        self._bmi.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
        self._bmi.biWidth = self.roi_size
        self._bmi.biHeight = -self.roi_size  # top-down DIB
        self._bmi.biPlanes = 1
        self._bmi.biBitCount = 24
        self._bmi.biCompression = 0
        self._buf = ctypes.create_string_buffer(self.roi_size * self.roi_size * 3)

    def grab_rgb(self) -> np.ndarray:
        SRCCOPY = 0x00CC0020
        DIB_RGB_COLORS = 0
        self.gdi32.BitBlt(
            self.hdc_mem, 0, 0, self.roi_size, self.roi_size,
            self.hdc_screen, self.left, self.top, SRCCOPY,
        )
        self.gdi32.GetDIBits(
            self.hdc_mem, self.hbmp, 0, self.roi_size,
            self._buf, ctypes.byref(self._bmi), DIB_RGB_COLORS,
        )
        arr = np.frombuffer(self._buf, dtype=np.uint8).reshape((self.roi_size, self.roi_size, 3))
        return arr[:, :, ::-1]  # BGR -> RGB

    def close(self) -> None:
        try:
            self.gdi32.DeleteObject(self.hbmp)
            self.gdi32.DeleteDC(self.hdc_mem)
            self.user32.ReleaseDC(self.hwnd, self.hdc_screen)
        except Exception:
            pass


class VisionROITracker:
    """
    256x256 Center ROI Perception Engine:
      - Capture backend priority: dxcam -> mss -> native Windows GDI BitBlt -> synthetic frames
      - Crosshair tracking + enemy head/chest detection + per-target CV-KF velocity lead prediction
    """

    def __init__(
        self,
        screen_w: int = 1920,
        screen_h: int = 1080,
        roi_size: int = 256,
        onnx_model_path: Optional[str] = None,
        yolo_model_path: Optional[str] = None,
        yolo_conf: float = 0.48,
        yolo_imgsz: int = 256,
        max_box_w: float = 90.0,
        max_box_h: float = 140.0,
        synthetic_only: bool = False,
    ):
        self.screen_w = int(screen_w)
        self.screen_h = int(screen_h)
        self.roi_size = int(roi_size)
        self.synthetic_only = bool(synthetic_only)
        self.yolo_conf = float(yolo_conf)
        self.yolo_imgsz = int(yolo_imgsz)
        self.max_box_w = float(max_box_w)
        self.max_box_h = float(max_box_h)
        self.backend_name = "synthetic"
        self.detector_backend = "synthetic_blobs" if self.synthetic_only else "none"
        self._dxcam = None
        self._mss = None
        self._gdi = None
        self._ort_session = None
        self._yolo_model = None
        self._yolo_device = "cpu"
        self._next_track_id = 0

        self.crosshair_tracker = DynamicCrosshairTracker(roi_size=roi_size)
        self.target_kfs: Dict[int, ConstantVelocityKF] = {}
        self.track_hits: Dict[int, int] = {}
        self.last_capture_ms = 0.0
        self.last_infer_ms = 0.0

        if not self.synthetic_only:
            self._init_capture_backend()
            resolved_yolo = yolo_model_path if yolo_model_path is not None else DEFAULT_YOLO_WEIGHTS
            if resolved_yolo and os.path.exists(resolved_yolo):
                self._init_yolo(resolved_yolo)
        if onnx_model_path and os.path.exists(onnx_model_path):
            self._init_onnx(onnx_model_path)

    def _init_capture_backend(self) -> None:
        left = max(0, (self.screen_w - self.roi_size) // 2)
        top = max(0, (self.screen_h - self.roi_size) // 2)
        region = (left, top, left + self.roi_size, top + self.roi_size)
        try:
            import dxcam
            cam = dxcam.create(output_color="RGB")
            cam.start(region=region, target_fps=240)
            self._dxcam = cam
            self.backend_name = "dxcam"
            return
        except Exception:
            pass

        try:
            import mss
            self._mss = mss.mss()
            self._mss_mon = {"left": left, "top": top, "width": self.roi_size, "height": self.roi_size}
            self.backend_name = "mss"
            return
        except Exception:
            pass

        try:
            self._gdi = _WindowsGDICapture(self.screen_w, self.screen_h, self.roi_size)
            self.backend_name = "gdi_bitblt"
        except Exception:
            self.backend_name = "synthetic"

    def _init_onnx(self, model_path: str) -> None:
        try:
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._ort_session = ort.InferenceSession(model_path, providers=providers)
        except Exception:
            self._ort_session = None

    def _init_yolo(self, model_path: str) -> None:
        """Load trained YOLOv8n CS2 head/chest detector on CUDA GPU (RTX 4060) if available."""
        try:
            import torch
            from ultralytics import YOLO

            self._yolo_device = 0 if torch.cuda.is_available() else "cpu"
            self._yolo_model = YOLO(model_path)
            # Warmup once with dummy ROI so first live frame has no CUDA init stall
            dummy = np.zeros((self.roi_size, self.roi_size, 3), dtype=np.uint8)
            self._yolo_model.predict(
                dummy,
                imgsz=self.yolo_imgsz,
                conf=self.yolo_conf,
                device=self._yolo_device,
                verbose=False,
            )
            self.detector_backend = f"yolov8n_cuda" if self._yolo_device != "cpu" else "yolov8n_cpu"
        except Exception:
            self._yolo_model = None
            self.detector_backend = "none"

    def _assign_track_id(self, pos_xy: np.ndarray, used_ids: set) -> int:
        best_id = -1
        best_d = 38.0
        for tid, kf in self.target_kfs.items():
            if tid in used_ids:
                continue
            d = float(np.linalg.norm(kf.x[:2] - pos_xy))
            if d < best_d:
                best_d = d
                best_id = tid
        if best_id >= 0:
            return best_id
        tid = self._next_track_id
        self._next_track_id += 1
        return tid

    def _detect_yolo(self, roi_rgb: np.ndarray) -> List[TargetDetection]:
        """Run trained YOLOv8n (0: head, 1: chest) on a 256x256 RGB ROI frame."""
        if self._yolo_model is None:
            return []
        # Ultralytics expects BGR ndarray when passing numpy arrays
        roi_bgr = roi_rgb[:, :, ::-1]
        results = self._yolo_model.predict(
            roi_bgr,
            imgsz=self.yolo_imgsz,
            conf=self.yolo_conf,
            device=self._yolo_device,
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        boxes = results[0].boxes
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        clss = boxes.cls.detach().cpu().numpy().astype(int)

        heads = []
        chests = []
        for b, conf, cls_id in zip(xyxy, confs, clss):
            c_val = float(conf)
            if c_val < self.yolo_conf:
                continue
            x1, y1, x2, y2 = [float(v) for v in b]
            w = max(2.0, x2 - x1)
            h = max(2.0, y2 - y1)
            # Filter out weapon viewmodels, walls, and malformed aspect ratios
            if w > self.max_box_w or h > self.max_box_h:
                continue
            if w < 5.0 or h < 5.0:
                continue
            ar = w / h
            if cls_id == 0:
                if ar > 2.6 or (1.0 / ar) > 2.6:
                    continue
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                heads.append((cx, cy, w, h, c_val, (x1, y1, x2, y2)))
            elif cls_id == 1:
                if w < 8.0 or h < 10.0 or ar > 2.8 or (1.0 / ar) > 3.2:
                    continue
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                chests.append((cx, cy, w, h, c_val, (x1, y1, x2, y2)))

        detections: List[TargetDetection] = []
        used_ids = set()
        used_chests = set()

        # Pair each detected head with its best matching chest box underneath
        heads.sort(key=lambda item: item[4], reverse=True)
        for hx, hy, hw, hh, hconf, hbox in heads:
            h_radius = max(4.0, 0.5 * min(hw, hh))
            head_xy = np.array([hx, hy], dtype=float)

            best_ci = -1
            best_score = 1e9
            for ci, (cx, cy, cw, ch, _, _) in enumerate(chests):
                if ci in used_chests:
                    continue
                if abs(cx - hx) <= max(24.0, 1.2 * cw) and -4.0 <= (cy - hy) <= max(70.0, 2.5 * ch):
                    score = abs(cx - hx) + 0.5 * abs(cy - (hy + 2.0 * h_radius))
                    if score < best_score:
                        best_score = score
                        best_ci = ci

            if best_ci >= 0:
                used_chests.add(best_ci)
                cx, cy, cw, ch, _, cbox = chests[best_ci]
                chest_xy = np.array([cx, cy], dtype=float)
                c_radius = max(8.0, 0.5 * min(cw, ch))
            else:
                chest_xy = head_xy + np.array([0.0, max(12.0, 2.2 * h_radius)], dtype=float)
                c_radius = max(10.0, 1.8 * h_radius)
                cbox = None

            tid = self._assign_track_id(head_xy, used_ids)
            used_ids.add(tid)
            detections.append(
                TargetDetection(
                    target_id=tid,
                    head_px=head_xy,
                    head_radius_px=h_radius,
                    chest_px=chest_xy,
                    chest_radius_px=c_radius,
                    confidence=hconf,
                    head_bbox_px=hbox,
                    chest_bbox_px=cbox,
                )
            )

        # Fallback if only high-confidence torso/chest is visible (infer head position above chest)
        if not detections and chests:
            chests.sort(key=lambda item: item[4], reverse=True)
            for cx, cy, cw, ch, cconf, cbox in chests:
                if cconf < max(0.55, self.yolo_conf):
                    continue
                c_radius = max(8.0, 0.5 * min(cw, ch))
                chest_xy = np.array([cx, cy], dtype=float)
                head_xy = np.array([cx, max(2.0, cy - 0.55 * ch)], dtype=float)
                h_radius = max(4.0, 0.35 * min(cw, ch))
                tid = self._assign_track_id(head_xy, used_ids)
                used_ids.add(tid)
                detections.append(
                    TargetDetection(
                        target_id=tid,
                        head_px=head_xy,
                        head_radius_px=h_radius,
                        chest_px=chest_xy,
                        chest_radius_px=c_radius,
                        confidence=0.8 * cconf,
                        chest_bbox_px=cbox,
                    )
                )

        return detections

    def grab_roi(self) -> np.ndarray:
        """Capture 256x256 center ROI RGB image (H, W, 3) uint8."""
        t0 = time.perf_counter()
        if self._dxcam is not None:
            frame = self._dxcam.get_latest_frame()
            if frame is None:
                frame = np.zeros((self.roi_size, self.roi_size, 3), dtype=np.uint8)
        elif self._mss is not None:
            raw = np.asarray(self._mss.grab(self._mss_mon))
            frame = raw[:, :, 2::-1]
        elif self._gdi is not None:
            frame = self._gdi.grab_rgb()
        else:
            frame = np.zeros((self.roi_size, self.roi_size, 3), dtype=np.uint8)
        self.last_capture_ms = (time.perf_counter() - t0) * 1000.0
        return frame

    def _detect_color_blobs_fallback(self, roi_rgb: np.ndarray) -> List[TargetDetection]:
        """Lightweight vector color detector for synthetic benchmark RGB frames (red heads, orange chests)."""
        r = roi_rgb[:, :, 0].astype(np.int16)
        g = roi_rgb[:, :, 1].astype(np.int16)
        b = roi_rgb[:, :, 2].astype(np.int16)
        head_mask = (r > 190) & (g < 80) & (b < 80)
        if int(head_mask.sum()) < 4:
            return []
        ys, xs = np.nonzero(head_mask)
        head_xy = np.array([float(xs.mean()), float(ys.mean())], dtype=float)
        chest_xy = head_xy + np.array([0.0, 14.0], dtype=float)
        return [
            TargetDetection(
                target_id=0,
                head_px=head_xy,
                head_radius_px=6.0,
                chest_px=chest_xy,
                chest_radius_px=12.0,
            )
        ]

    def update_target_filters(
        self,
        detections: List[TargetDetection],
        dt_s: float,
        k_lead_s: float = 0.085,
    ) -> List[TargetDetection]:
        """Update per-target Constant-Velocity Kalman Filters and attach lead predictions h_hat."""
        t0 = time.perf_counter()
        active_ids = set()
        for det in detections:
            tid = int(det.target_id)
            active_ids.add(tid)
            if tid not in self.target_kfs:
                self.target_kfs[tid] = ConstantVelocityKF(float(det.head_px[0]), float(det.head_px[1]))
            pos, vel = self.target_kfs[tid].step(det.head_px, dt_s=dt_s)
            det.vel_px_s = vel
            det.head_pred_px = pos + float(k_lead_s) * vel
            det.chest_pred_px = det.chest_px + float(k_lead_s) * vel
            self.track_hits[tid] = self.track_hits.get(tid, 0) + 1
            det.seen_frames = self.track_hits[tid]

        # Decay missing tracks immediately so consecutive-frame gate strictly enforces continuous visibility
        missing = [tid for tid in list(self.target_kfs.keys()) if tid not in active_ids]
        for tid in missing:
            self.track_hits.pop(tid, None)
            if len(active_ids) > 0:
                self.target_kfs[tid].step(None, dt_s=dt_s)
            else:
                self.target_kfs.pop(tid, None)

        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0
        return detections

    def process_frame(
        self,
        roi_rgb: Optional[np.ndarray] = None,
        dt_s: float = 0.004,
        k_lead_s: float = 0.085,
        injected_detections: Optional[List[TargetDetection]] = None,
        recoil_prior_px: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[TargetDetection]]:
        """Run one perception cycle: ROI crosshair tracking + target KF lead estimation."""
        t0 = time.perf_counter()
        if roi_rgb is not None:
            c_px, _ = self.crosshair_tracker.detect_in_frame(roi_rgb, dt_s=dt_s, recoil_prior_px=recoil_prior_px)
        else:
            c_px, _ = self.crosshair_tracker.kf.step(recoil_prior_px, dt_s=dt_s)

        if injected_detections is not None:
            dets = injected_detections
        elif roi_rgb is not None:
            if self._yolo_model is not None and not self.synthetic_only:
                dets = self._detect_yolo(roi_rgb)
            elif self.synthetic_only:
                dets = self._detect_color_blobs_fallback(roi_rgb)
            else:
                dets = []
        else:
            dets = []

        dets = self.update_target_filters(dets, dt_s=dt_s, k_lead_s=k_lead_s)
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0
        return c_px, dets

    def reset_filters(self) -> None:
        self.target_kfs.clear()
        self.track_hits.clear()
        c0 = 0.5 * self.roi_size
        self.crosshair_tracker.kf = ConstantVelocityKF(x0=c0, y0=c0, q_acc=20000.0, r_pos=0.4)

    def close(self) -> None:
        if self._dxcam is not None:
            try:
                self._dxcam.stop()
            except Exception:
                pass
        if self._mss is not None:
            try:
                self._mss.close()
            except Exception:
                pass
        if self._gdi is not None:
            self._gdi.close()


def render_synthetic_roi_frame(
    crosshair_xy: np.ndarray,
    targets: List[TargetDetection],
    roi_size: int = 256,
) -> np.ndarray:
    """Render a clean 256x256 synthetic RGB frame for benchmarking crosshair & blob speed."""
    frame = np.full((roi_size, roi_size, 3), 24, dtype=np.uint8)
    for tgt in targets:
        hx, hy = int(round(float(tgt.head_px[0]))), int(round(float(tgt.head_px[1])))
        if 3 <= hx < roi_size - 3 and 3 <= hy < roi_size - 3:
            frame[hy - 2 : hy + 3, hx - 2 : hx + 3] = (230, 35, 35)
    cx, cy = int(round(float(crosshair_xy[0]))), int(round(float(crosshair_xy[1])))
    if 4 <= cx < roi_size - 4 and 4 <= cy < roi_size - 4:
        frame[cy, cx - 3 : cx + 4] = (0, 255, 80)
        frame[cy - 3 : cy + 4, cx] = (0, 255, 80)
    return frame
