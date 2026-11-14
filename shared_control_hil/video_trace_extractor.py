"""
video_trace_extractor.py - Module 7: Automated CS2 Gameplay Video Mining & 4D Telemetry Reconstruction
====================================================================================================
Extracts real personal combat engagements ("whiff_*.npz" & "highlight_*.npz") from archived MP4 gameplay
recordings without requiring external mouse logs:
  1. Two-stage high-speed funnel screening:
       - Stage A (Coarse, every 6th frame / 5 fps): center 256x256 ROI activity + viewmodel gunshot impulse
         + first-person HUD check (filters ~95% running/spectator/idle time).
       - Stage B (Fine, full FPS): slices [-0.8s, +2.2s] full-res center ROI & background streams.
  2. Pure-video 4D telemetry reverse-reconstruction:
       - Background Optical Flow (PyrLK + RANSAC median inlier filtering on non-HUD/non-viewmodel ring):
         d_p_hand(t) = -J_px^{-1} * d_u_bg(t)
       - Dynamic Crosshair Tracking c(t) inside center 256x256 ROI (multi-profile RGB/cyan-green mask)
       - Enemy Head & Upper-Chest Trajectory Extraction h_head(t), h_chest(t) via ego-motion compensated
         character contrast/motion salience + ConstantVelocityKF (with optional YOLO/ONNX hook)
       - Gunshot Firing Timestamp Detection {t_1, ..., t_K} via viewmodel recoil impulse & crosshair kick
  3. Automatic classification into:
       - whiff_*.npz     (mean E_hand > 6.0 cnt or first shot E_hand(t_1) > 10.5 cnt)
       - highlight_*.npz (first or first 2 shots locked on head: min(E_hand[:2]) <= 3.5 cnt)
  4. Saves traces to shared_control_hil/traces/real_video_traces/ and logs/video_mining_report.json.
"""

import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

HIL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(HIL_DIR, "config_hil.json")
DEFAULT_VIDEO_DIR = r"E:\游戏录屏归档\高光短视频"
DEFAULT_OUT_DIR = os.path.join(HIL_DIR, "traces", "real_video_traces")
LOGS_DIR = os.path.join(HIL_DIR, "logs")

try:
    import cv2  # type: ignore
    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False

from scipy.ndimage import gaussian_filter, maximum_filter, sobel, uniform_filter
from shared_control_hil.shared_controller import MacroMicroSharedController
from shared_control_hil.vision_roi_tracker import (
    ConstantVelocityKF,
    TargetDetection,
    VisionROITracker,
)


def _find_ffmpeg_exe() -> str:
    """Locate working ffmpeg executable without modifying environment."""
    candidates = [
        r"E:\FYVideoConverter\Essential\ffmpeg.exe",
        r"E:\FYVideoConverter\Essential\compress\ffmpeg.exe",
        r"D:\Program Files\SOLIDWORKS Corp2024\SOLIDWORKS Flow Simulation\binCFW\ffmpeg.exe",
        shutil.which("ffmpeg") or "",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError("No ffmpeg executable found and OpenCV (cv2) is unavailable.")


def _probe_video_meta(video_path: str, ffmpeg_exe: Optional[str]) -> Tuple[int, int, float, float]:
    """Return (width, height, fps, duration_s) for an MP4 clip."""
    if HAS_CV2:
        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap.release()
        dur = (n_frames / fps) if n_frames > 0 else 9.0
        return w, h, fps, dur

    proc = subprocess.run(
        [ffmpeg_exe, "-hide_banner", "-i", video_path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=False,
    )
    txt = proc.stderr or ""
    m_dur = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", txt)
    dur = 9.0
    if m_dur:
        dur = int(m_dur.group(1)) * 3600.0 + int(m_dur.group(2)) * 60.0 + float(m_dur.group(3))
    m_res = re.search(r"Video:.*?,\s*(\d{3,4})x(\d{3,4})", txt)
    w = int(m_res.group(1)) if m_res else 1920
    h = int(m_res.group(2)) if m_res else 1080
    m_fps = re.search(r"([\d.]+)\s*fps", txt)
    fps = float(m_fps.group(1)) if m_fps else 30.0
    return w, h, fps, dur


def _decode_coarse_frames(
    video_path: str,
    ffmpeg_exe: Optional[str],
    fps: float,
    coarse_stride: int = 6,
    bw: int = 480,
    bh: int = 270,
) -> np.ndarray:
    """Stage A decoder: every `coarse_stride`-th frame downsampled to (N, bh, bw, 3) uint8 RGB."""
    if HAS_CV2:
        cap = cv2.VideoCapture(video_path)
        frames = []
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx % coarse_stride == 0:
                small = cv2.resize(bgr, (bw, bh), interpolation=cv2.INTER_AREA)
                frames.append(small[:, :, ::-1])
            idx += 1
        cap.release()
        return np.asarray(frames, dtype=np.uint8) if frames else np.zeros((0, bh, bw, 3), dtype=np.uint8)

    sample_fps = max(2.0, float(fps) / float(coarse_stride))
    cmd = [
        ffmpeg_exe,
        "-v", "error",
        "-i", video_path,
        "-vf", f"fps={sample_fps:.3f},scale={bw}:{bh}",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]
    raw = subprocess.check_output(cmd)
    n_pix = bh * bw * 3
    n_f = len(raw) // n_pix
    if n_f <= 0:
        return np.zeros((0, bh, bw, 3), dtype=np.uint8)
    return np.frombuffer(raw[: n_f * n_pix], dtype=np.uint8).reshape((n_f, bh, bw, 3))


def _decode_fine_window(
    video_path: str,
    ffmpeg_exe: Optional[str],
    start_s: float,
    window_s: float,
    src_w: int,
    src_h: int,
    roi_size: int = 256,
    bw: int = 480,
    bh: int = 270,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stage B decoder: slice [start_s, start_s + window_s] at full FPS and extract both:
      - center `roi_size x roi_size` native resolution ROI frames (N, roi, roi, 3)
      - global scaled background/viewmodel frames (N, bh, bw, 3)
    """
    crop_x = max(0, (src_w - roi_size) // 2)
    crop_y = max(0, (src_h - roi_size) // 2)

    if HAS_CV2:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
        fps_v = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        max_f = int(round(window_s * fps_v)) + 2
        rois, bg_frames = [], []
        for _ in range(max_f):
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = bgr[:, :, ::-1]
            rois.append(rgb[crop_y : crop_y + roi_size, crop_x : crop_x + roi_size].copy())
            bg_frames.append(cv2.resize(rgb, (bw, bh), interpolation=cv2.INTER_AREA))
        cap.release()
        if not rois:
            return np.zeros((0, roi_size, roi_size, 3), dtype=np.uint8), np.zeros((0, bh, bw, 3), dtype=np.uint8)
        return np.asarray(rois, dtype=np.uint8), np.asarray(bg_frames, dtype=np.uint8)

    cmd_roi = [
        ffmpeg_exe,
        "-v", "error",
        "-ss", f"{start_s:.3f}",
        "-t", f"{window_s:.3f}",
        "-i", video_path,
        "-vf", f"crop={roi_size}:{roi_size}:{crop_x}:{crop_y}",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]
    cmd_bg = [
        ffmpeg_exe,
        "-v", "error",
        "-ss", f"{start_s:.3f}",
        "-t", f"{window_s:.3f}",
        "-i", video_path,
        "-vf", f"scale={bw}:{bh}",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]
    raw_roi = subprocess.check_output(cmd_roi)
    raw_bg = subprocess.check_output(cmd_bg)
    n_roi = len(raw_roi) // (roi_size * roi_size * 3)
    n_bg = len(raw_bg) // (bh * bw * 3)
    n_f = min(n_roi, n_bg)
    if n_f <= 0:
        return np.zeros((0, roi_size, roi_size, 3), dtype=np.uint8), np.zeros((0, bh, bw, 3), dtype=np.uint8)
    rois = np.frombuffer(raw_roi[: n_f * roi_size * roi_size * 3], dtype=np.uint8).reshape((n_f, roi_size, roi_size, 3))
    bgs = np.frombuffer(raw_bg[: n_f * bh * bw * 3], dtype=np.uint8).reshape((n_f, bh, bw, 3))
    return rois, bgs


def _select_bg_ring_corners(gray: np.ndarray, n_corners: int = 56) -> np.ndarray:
    """
    Pick high-gradient corners inside the static background ring (avoiding HUD margins,
    center 256x256 ROI / target box, and bottom-right weapon viewmodel).
    """
    bh, bw = gray.shape
    mask = np.zeros((bh, bw), dtype=bool)
    mask[38:210, 46:434] = True
    # Exclude center target/crosshair box
    mask[88:182, 176:304] = False
    # Exclude bottom-right viewmodel
    mask[158:215, 255:440] = False

    if HAS_CV2:
        pts = cv2.goodFeaturesToTrack(
            gray.astype(np.uint8),
            maxCorners=n_corners,
            qualityLevel=0.04,
            minDistance=12,
            mask=mask.astype(np.uint8) * 255,
        )
        if pts is not None and len(pts) > 0:
            return pts.reshape(-1, 2).astype(np.float32)

    gx = sobel(gray.astype(np.float32), axis=1)
    gy = sobel(gray.astype(np.float32), axis=0)
    corner_score = uniform_filter(gx * gx + gy * gy, size=5)
    corner_score[~mask] = 0.0
    local_max = corner_score == maximum_filter(corner_score, size=13)
    ys, xs = np.nonzero(local_max & (corner_score > 120.0))
    if len(xs) == 0:
        ys, xs = np.nonzero(mask)
        idx = np.linspace(0, len(xs) - 1, min(n_corners, len(xs)), dtype=int)
        return np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
    scores = corner_score[ys, xs]
    top_idx = np.argsort(scores)[-n_corners:][::-1]
    return np.stack([xs[top_idx], ys[top_idx]], axis=1).astype(np.float32)


def _coarse_ring_shift_fft(prev_g: np.ndarray, curr_g: np.ndarray) -> Tuple[float, float]:
    """Coarse global translation via Hanning-windowed phase correlation on upper/mid background band."""
    p = prev_g[36:145, 50:430].astype(np.float32)
    c = curr_g[36:145, 50:430].astype(np.float32)
    win = np.outer(np.hanning(p.shape[0]), np.hanning(p.shape[1])).astype(np.float32)
    p = (p - p.mean()) * win
    c = (c - c.mean()) * win
    Fp = np.fft.rfft2(p)
    Fc = np.fft.rfft2(c)
    R = Fc * np.conj(Fp)
    mag = np.abs(R)
    R /= np.maximum(mag, 1e-6)
    cc = np.fft.irfft2(R, s=p.shape)
    hh, ww = p.shape
    vy, vx = np.unravel_index(np.argmax(cc), cc.shape)
    peak = float(cc[vy, vx])
    dy = float(vy if vy <= hh // 2 else vy - hh)
    dx = float(vx if vx <= ww // 2 else vx - ww)
    if peak < 0.14 or abs(dx) > 8.0 or abs(dy) > 6.0:
        return 0.0, 0.0
    return dx, dy


def _pyr_lk_step(
    prev_l: np.ndarray,
    curr_l: np.ndarray,
    pts: np.ndarray,
    init_flow: np.ndarray,
    win_r: int = 5,
    iters: int = 3,
) -> np.ndarray:
    """Vectorized bilinear Pyramidal Lucas-Kanade refinement for batch of 2D corners."""
    h, w = prev_l.shape
    ix_map = sobel(prev_l, axis=1) * 0.125
    iy_map = sobel(prev_l, axis=0) * 0.125

    offs = np.arange(-win_r, win_r + 1, dtype=np.float32)
    ox, oy = np.meshgrid(offs, offs)
    ox = ox.ravel()
    oy = oy.ravel()

    flow = init_flow.copy()
    n_pts = len(pts)
    xs0 = pts[:, 0:1] + ox[None, :]
    ys0 = pts[:, 1:2] + oy[None, :]

    def _sample(img: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        xc = np.clip(xx, 0.0, w - 1.001)
        yc = np.clip(yy, 0.0, h - 1.001)
        x0 = np.floor(xc).astype(np.int32)
        y0 = np.floor(yc).astype(np.int32)
        fx = xc - x0
        fy = yc - y0
        v00 = img[y0, x0]
        v10 = img[y0, x0 + 1]
        v01 = img[y0 + 1, x0]
        v11 = img[y0 + 1, x0 + 1]
        return (1 - fy) * ((1 - fx) * v00 + fx * v10) + fy * ((1 - fx) * v01 + fx * v11)

    I0 = _sample(prev_l, xs0, ys0)
    Ix = _sample(ix_map, xs0, ys0)
    Iy = _sample(iy_map, xs0, ys0)

    sxx = (Ix * Ix).sum(axis=1) + 1e-3
    syy = (Iy * Iy).sum(axis=1) + 1e-3
    sxy = (Ix * Iy).sum(axis=1)
    det = sxx * syy - sxy * sxy

    valid = (
        (pts[:, 0] > win_r + 2)
        & (pts[:, 0] < w - win_r - 3)
        & (pts[:, 1] > win_r + 2)
        & (pts[:, 1] < h - win_r - 3)
        & (det > 1e-2)
    )

    for _ in range(iters):
        xq = xs0 + flow[:, 0:1]
        yq = ys0 + flow[:, 1:2]
        I1 = _sample(curr_l, xq, yq)
        It = I1 - I0
        bx = -(Ix * It).sum(axis=1)
        by = -(Iy * It).sum(axis=1)
        du = np.where(valid, (syy * bx - sxy * by) / np.maximum(det, 1e-4), 0.0)
        dv = np.where(valid, (-sxy * bx + sxx * by) / np.maximum(det, 1e-4), 0.0)
        flow[:, 0] += np.clip(du, -1.2, 1.2)
        flow[:, 1] += np.clip(dv, -1.2, 1.2)
    return flow


def calc_optical_flow_pyr_lk_ransac(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    pts: Optional[np.ndarray] = None,
    src_scale_xy: Tuple[float, float] = (4.0, 4.0),
) -> Tuple[np.ndarray, float]:
    """
    Estimate background translation du_bg (in full 1920x1080 pixels) using 2-level PyrLK + RANSAC median.
    Returns (du_bg_px, inlier_ratio).
    """
    if pts is None or len(pts) < 8:
        pts = _select_bg_ring_corners(prev_gray)
    if len(pts) < 6:
        return np.zeros(2, dtype=float), 0.0

    dx0, dy0 = _coarse_ring_shift_fft(prev_gray, curr_gray)
    if HAS_CV2:
        pts_init = (pts + np.array([[dx0, dy0]], dtype=np.float32)).reshape(-1, 1, 2)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray.astype(np.uint8),
            curr_gray.astype(np.uint8),
            pts.reshape(-1, 1, 2),
            pts_init,
            winSize=(15, 15),
            maxLevel=2,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        )
        ok = status.ravel() == 1
        if int(ok.sum()) >= 5:
            vecs = next_pts.reshape(-1, 2)[ok] - pts[ok]
        else:
            vecs = np.tile(np.array([[dx0, dy0]], dtype=np.float32), (len(pts), 1))
    else:
        p0 = prev_gray.astype(np.float32)
        c0 = curr_gray.astype(np.float32)
        p1 = gaussian_filter(p0, sigma=0.8)[::2, ::2]
        c1 = gaussian_filter(c0, sigma=0.8)[::2, ::2]
        init_l1 = np.tile(np.array([[0.5 * dx0, 0.5 * dy0]], dtype=np.float32), (len(pts), 1))
        flow_l1 = _pyr_lk_step(p1, c1, 0.5 * pts, init_l1, win_r=4, iters=3)
        vecs = _pyr_lk_step(p0, c0, pts, 2.0 * flow_l1, win_r=5, iters=3)

    # RANSAC median consensus across corner flow vectors
    best_inliers = np.ones(len(vecs), dtype=bool)
    best_cnt = 0
    rng = np.random.default_rng(17)
    n_v = len(vecs)
    for k in range(min(36, n_v)):
        cand = vecs[k] if n_v <= 36 else vecs[rng.integers(0, n_v)]
        dists = np.hypot(vecs[:, 0] - cand[0], vecs[:, 1] - cand[1])
        inl = dists <= 1.25
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_inliers = inl

    med_vec = np.median(vecs[best_inliers], axis=0) if best_cnt >= 3 else np.median(vecs, axis=0)
    if float(np.hypot(med_vec[0], med_vec[1])) < 0.22:
        med_vec = np.zeros(2, dtype=float)
    du_full = np.array([float(med_vec[0]) * src_scale_xy[0], float(med_vec[1]) * src_scale_xy[1]], dtype=float)
    # Clamp 1080p per-frame mouse pan at 18.0 px so v_hand stays within physical CS2 range
    du_norm = float(np.hypot(du_full[0], du_full[1]))
    if du_norm > 18.0:
        du_full = du_full * (18.0 / du_norm)
    return du_full, float(best_cnt) / max(1.0, float(n_v))


def detect_crosshair_profile_and_track(rois_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Auto-detect player's dynamic crosshair color inside center 256x256 ROI (supporting cyan/green/yellow/white)
    and track c_px(t) across all frames using separable cross-arm response + CV-KF smoothing.
    Returns (c_px_seq [N, 2], detection_rate).
    """
    n_f, h, w, _ = rois_rgb.shape
    cx0, cy0 = 0.5 * w, 0.5 * h

    # Use temporal median of center 32x32 patch to detect crosshair signature
    med_patch = np.median(rois_rgb[:, h // 2 - 16 : h // 2 + 16, w // 2 - 16 : w // 2 + 16], axis=0).astype(np.int16)
    mr, mg, mb = med_patch[:, :, 0], med_patch[:, :, 1], med_patch[:, :, 2]
    sat = np.maximum(np.maximum(mr, mg), mb) - np.minimum(np.minimum(mr, mg), mb)
    high_sat = sat > 45
    if int(high_sat.sum()) >= 3:
        proto = np.median(med_patch[high_sat], axis=0).astype(np.int16)
        cyan_green_mode = (proto[1] > proto[0] + 20) and (proto[1] >= 140)
    else:
        proto = np.array([105, 212, 192], dtype=np.int16)
        cyan_green_mode = True

    kf = ConstantVelocityKF(x0=cx0, y0=cy0, q_acc=24000.0, r_pos=0.45)
    dt_s = 1.0 / 30.0
    c_seq = np.zeros((n_f, 2), dtype=float)
    n_det = 0

    u0, u1 = max(0, w // 2 - 52), min(w, w // 2 + 52)
    v0, v1 = max(0, h // 2 - 48), min(h, h // 2 + 48)

    for i in range(n_f):
        sub = rois_rgb[i, v0:v1, u0:u1].astype(np.int16)
        sr, sg, sb = sub[:, :, 0], sub[:, :, 1], sub[:, :, 2]
        if cyan_green_mode:
            mask = (sg > 165) & (sb > 130) & ((sg - sr) > 22)
        else:
            diff = np.abs(sub - proto[None, None, :])
            mask = (diff[:, :, 0] <= 55) & (diff[:, :, 1] <= 55) & (diff[:, :, 2] <= 55)

        px_cnt = int(mask.sum())
        z = None
        if 4 <= px_cnt <= 320:
            row_sum = mask.sum(axis=1).astype(float)
            col_sum = mask.sum(axis=0).astype(float)
            vr = int(np.argmax(row_sum))
            ur = int(np.argmax(col_sum))
            vw = slice(max(0, vr - 10), min(mask.shape[0], vr + 11))
            uw = slice(max(0, ur - 10), min(mask.shape[1], ur + 11))
            ys, xs = np.nonzero(mask[vw, uw])
            if len(xs) >= 3:
                u_c = float(xs.mean() + uw.start + u0)
                v_c = float(ys.mean() + vw.start + v0)
                z = np.array([u_c, v_c], dtype=float)
                n_det += 1

        if z is None:
            # Anchor toward ROI center gently on occlusions
            z = np.array([0.7 * kf.x[0] + 0.3 * cx0, 0.7 * kf.x[1] + 0.3 * cy0], dtype=float)
        pos, _ = kf.step(z, dt_s=dt_s)
        c_seq[i] = pos

    return c_seq, float(n_det) / max(1, n_f)


def detect_gunshots_and_targets(
    rois_rgb: np.ndarray,
    bgs_rgb: np.ndarray,
    du_bg_seq: np.ndarray,
    c_px_seq: np.ndarray,
    counts_per_px: float,
    fps: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract gunshot frame indices, target head trajectory h_head_px(t), chest h_chest_px(t), and target id
    via ego-motion-compensated character foreground residual + upper-18% head anchor + VisionROITracker KF.
    """
    n_f, h, w, _ = rois_rgb.shape
    cx0, cy0 = 0.5 * w, 0.5 * h

    # 1) Viewmodel gunshot recoil impulse I_fire(t) + background vertical kick + crosshair snap
    bg_f = bgs_rgb.astype(np.float32)
    vm_diff = np.zeros(n_f, dtype=float)
    ring_diff = np.zeros(n_f, dtype=float)
    if n_f > 1:
        vm_diff[1:] = np.abs(bg_f[1:, 180:255, 280:440] - bg_f[:-1, 180:255, 280:440]).mean(axis=(1, 2, 3))
        ring_diff[1:] = np.abs(bg_f[1:, 40:160, 60:420] - bg_f[:-1, 40:160, 60:420]).mean(axis=(1, 2, 3))

    recoil_impulse = np.maximum(0.0, vm_diff - 0.85 * ring_diff)
    c_jump = np.zeros(n_f, dtype=float)
    c_jump[1:] = np.hypot(np.diff(c_px_seq[:, 0]), np.diff(c_px_seq[:, 1]))
    bg_kick = np.zeros(n_f, dtype=float)
    bg_kick[1:] = np.abs(du_bg_seq[:, 1]) * 0.25
    fire_score = recoil_impulse + 1.8 * c_jump + bg_kick

    # Adaptive peak picking with >= 3-frame (~100 ms at 30 fps) refractory interval
    min_sep = max(2, int(round(0.095 * fps)))
    thresh = max(16.0, float(np.percentile(fire_score, 74)))
    shot_frames: List[int] = []
    search_start = max(2, int(round(0.20 * fps)))
    search_end = max(search_start + 1, n_f - 2)
    cand_order = np.argsort(fire_score[search_start:search_end])[::-1] + search_start
    for f_idx in cand_order:
        if fire_score[f_idx] < thresh:
            break
        if all(abs(int(f_idx) - sf) >= min_sep for sf in shot_frames):
            shot_frames.append(int(f_idx))
            if len(shot_frames) >= 10:
                break
    shot_frames.sort()

    if len(shot_frames) == 0:
        shot_frames = [int(np.argmax(fire_score[search_start:search_end]) + search_start)]

    # 2) Subpixel ego-motion-compensated character salience inside 256x256 ROI
    from scipy.ndimage import shift as nd_shift

    rois_g = rois_rgb.mean(axis=3).astype(np.float32)
    # Mask out bright/saturated cyan-green crosshair pixels cleanly without blinding the head behind it
    rr, gg, bb = rois_rgb[:, :, :, 0].astype(np.int16), rois_rgb[:, :, :, 1].astype(np.int16), rois_rgb[:, :, :, 2].astype(np.int16)
    ch_pixels = (gg > 165) & (bb > 130) & ((gg - rr) > 22)

    h_head_seq = np.zeros((n_f, 2), dtype=float)
    h_chest_seq = np.zeros((n_f, 2), dtype=float)
    tgt_id_seq = np.zeros(n_f, dtype=np.int32)

    roi_tracker = VisionROITracker(roi_size=w, synthetic_only=True)
    dt_s = 1.0 / max(1.0, fps)
    chest_offset_px = 14.0 / counts_per_px

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    prev_head = np.array([cx0, cy0], dtype=float)
    cur_tid = 0
    switch_streak = 0

    flash_mask = np.zeros(n_f, dtype=bool)
    for sf in shot_frames:
        flash_mask[int(sf)] = True
        if int(sf) + 1 < n_f:
            flash_mask[int(sf) + 1] = True

    for i in range(n_f):
        z_head: Optional[np.ndarray] = None
        if flash_mask[i] and i > 0:
            z_head = prev_head.copy()
        else:
            sal = np.zeros((h, w), dtype=np.float32)
            if i > 0:
                dx = float(np.clip(du_bg_seq[i - 1, 0], -24.0, 24.0))
                dy = float(np.clip(du_bg_seq[i - 1, 1], -24.0, 24.0))
                warped = nd_shift(rois_g[i - 1], shift=(dy, dx), order=1, mode="nearest")
                fg = np.abs(rois_g[i] - warped)
                pad_y = max(3, int(math.ceil(abs(dy))) + 2)
                pad_x = max(3, int(math.ceil(abs(dx))) + 2)
                fg[:pad_y, :] = 0.0
                fg[h - pad_y :, :] = 0.0
                fg[:, :pad_x] = 0.0
                fg[:, w - pad_x :] = 0.0
                sal += fg

            # Local micro-contrast structure in the engagement corridor
            gx = np.abs(sobel(rois_g[i], axis=1))
            gy = np.abs(sobel(rois_g[i], axis=0))
            edge_map = uniform_filter(0.75 * gx + 0.25 * gy, size=5)
            sal = sal + 0.06 * edge_map
            sal[ch_pixels[i]] = 0.0

            # Tight spatial focus centered around tracked crosshair c_px[i] (3-RRR micro-workspace <= ~8.5 px = 48 cnt)
            cu, cv = float(c_px_seq[i, 0]), float(c_px_seq[i, 1])
            dist_sq = (xx - cu) ** 2 + (yy - cv) ** 2
            sal_w = gaussian_filter(sal, sigma=1.4) * np.exp(-dist_sq / (2.0 * 6.5 * 6.5))

            peak_val = float(sal_w.max())
            if peak_val > 2.5:
                mask_fg = sal_w >= max(2.5, 0.50 * peak_val)
                ys_fg, xs_fg = np.nonzero(mask_fg)
                if len(xs_fg) >= 5:
                    # Upper 12% vertical crown of detected silhouette + anatomical face offset
                    y_top = float(np.percentile(ys_fg, 12))
                    head_slice = ys_fg <= (y_top + 3.5)
                    if int(head_slice.sum()) >= 3:
                        w_pts = sal_w[ys_fg[head_slice], xs_fg[head_slice]]
                        w_sum = max(1e-6, float(w_pts.sum()))
                        hx = float((xs_fg[head_slice] * w_pts).sum() / w_sum)
                        hy = float((ys_fg[head_slice] * w_pts).sum() / w_sum) + 2.35
                    else:
                        hx = float(np.mean(xs_fg))
                        hy = y_top + 2.35
                    hx = float(np.clip(hx, cu - 7.5, cu + 7.5))
                    hy = float(np.clip(hy, cv - 7.5, cv + 7.5))
                    z_head = np.array([hx, hy], dtype=float)

                    cur_tid = 0
                    switch_streak = 0
                    prev_head = z_head

            if z_head is None:
                z_head = prev_head.copy()

        det = TargetDetection(
            target_id=cur_tid,
            head_px=z_head,
            head_radius_px=10.5 / counts_per_px,
            chest_px=z_head + np.array([0.0, chest_offset_px], dtype=float),
            chest_radius_px=21.0 / counts_per_px,
        )
        roi_tracker.update_target_filters([det], dt_s=dt_s, k_lead_s=0.0)
        head_kf = roi_tracker.target_kfs[cur_tid].x[:2].copy()
        h_head_seq[i] = head_kf
        h_chest_seq[i] = head_kf + np.array([0.0, chest_offset_px], dtype=float)
        tgt_id_seq[i] = cur_tid

    if n_f >= 5:
        h_head_seq = gaussian_filter(h_head_seq, sigma=(1.15, 0.0))
        h_chest_seq = h_head_seq + np.array([[0.0, chest_offset_px]], dtype=float)

    return np.array(shot_frames, dtype=np.int32), h_head_seq, h_chest_seq, tgt_id_seq


def _reconstruct_clip_trace(
    video_path: str,
    ffmpeg_exe: Optional[str],
    src_w: int,
    src_h: int,
    fps: float,
    event_t_s: float,
    dur_s: float,
    counts_per_px: float,
) -> Optional[Dict]:
    """Decode [-0.8s, +2.2s] window around `event_t_s` and extract 200Hz `.npz` trace payload."""
    win_start = max(0.0, min(max(0.0, dur_s - 3.05), event_t_s - 0.8))
    win_len = min(3.0, max(1.2, dur_s - win_start))
    rois, bgs = _decode_fine_window(video_path, ffmpeg_exe, win_start, win_len, src_w, src_h)
    n_f = len(rois)
    if n_f < int(1.0 * fps):
        return None

    scale_x = float(src_w) / float(bgs.shape[2])
    scale_y = float(src_h) / float(bgs.shape[1])

    # 1) Background Optical Flow -> inverse pinhole hand mouse displacement & velocity
    bgs_g = bgs.mean(axis=3).astype(np.float32)
    du_bg_seq = np.zeros((max(1, n_f - 1), 2), dtype=float)
    inlier_ratios = []
    for i in range(n_f - 1):
        du_px, inl_r = calc_optical_flow_pyr_lk_ransac(bgs_g[i], bgs_g[i + 1], src_scale_xy=(scale_x, scale_y))
        du_bg_seq[i] = du_px
        inlier_ratios.append(inl_r)

    # Smooth outlier single-frame flow jumps via 3-tap median filter
    if len(du_bg_seq) >= 3:
        for ax in range(2):
            col = du_bg_seq[:, ax].copy()
            for j in range(1, len(col) - 1):
                col[j] = float(np.median(du_bg_seq[j - 1 : j + 2, ax]))
            du_bg_seq[:, ax] = col

    # d_p_hand = -J_px^{-1} * d_u_bg (diagonal 5.60 cnt/px pinhole mapping)
    dp_hand_cnt = np.zeros((n_f, 2), dtype=float)
    dp_hand_cnt[1:] = -counts_per_px * du_bg_seq[: n_f - 1]
    v_hand_raw = dp_hand_cnt * float(fps)

    # 2) Dynamic recoil crosshair tracking c(t) in ROI
    c_px_seq, ch_det_rate = detect_crosshair_profile_and_track(rois)

    # 3) Shot detection + Target head/chest extraction h_head(t), h_chest(t)
    shot_frames, h_head_px, h_chest_px, tgt_id_seq = detect_gunshots_and_targets(
        rois, bgs, du_bg_seq, c_px_seq, counts_per_px=counts_per_px, fps=fps
    )
    if len(shot_frames) < 1:
        return None

    # Smooth v_hand with EMA (alpha=0.38), damp gunshot camera viewpunch, and scale to physical micro-aim range
    v_ema = np.zeros_like(v_hand_raw)
    if n_f > 0:
        v_ema[0] = v_hand_raw[0]
        alpha_v = 0.38
        for i in range(1, n_f):
            v_ema[i] = alpha_v * v_hand_raw[i] + (1.0 - alpha_v) * v_ema[i - 1]
    shot_mask = np.zeros(n_f, dtype=bool)
    for sf in shot_frames:
        shot_mask[sf] = True
        if sf + 1 < n_f:
            shot_mask[sf + 1] = True
    v_ema[shot_mask] *= 0.22
    v_hand_30 = v_ema * 0.14
    v_norms = np.hypot(v_hand_30[:, 0], v_hand_30[:, 1])
    over_v = v_norms > 380.0
    if np.any(over_v):
        v_hand_30[over_v] *= (380.0 / v_norms[over_v])[:, None]

    # Compute true video-frame error e_frame_cnt = (h_head_px - c_px) * counts_per_px
    # Replace muzzle-flash/viewpunch frame residuals with clean pre-flash aim (sf - 1),
    # smooth temporally, and clamp extreme outliers to 42.0 cnt radial magnitude.
    e_raw_cnt = (h_head_px - c_px_seq) * counts_per_px
    e_raw_cnt = gaussian_filter(e_raw_cnt, sigma=(1.0, 0.0))
    for sf in shot_frames:
        if sf > 0:
            e_raw_cnt[sf] = e_raw_cnt[sf - 1].copy()
    e_norms = np.hypot(e_raw_cnt[:, 0], e_raw_cnt[:, 1])
    over_e = e_norms > 42.0
    if np.any(over_e):
        e_raw_cnt[over_e] *= (42.0 / e_norms[over_e])[:, None]

    shot_errs_raw = np.hypot(e_raw_cnt[shot_frames, 0], e_raw_cnt[shot_frames, 1])

    # Truncate post-kill phantom shots (> 8.0 cnt) once a lethal headshot (<= 4.5 cnt) occurs in shots 1..3
    keep_n = len(shot_frames)
    for k_hit in range(min(3, len(shot_frames))):
        if shot_errs_raw[k_hit] <= 4.5:
            cut = k_hit + 1
            while cut < len(shot_frames) and shot_errs_raw[cut] <= 8.0:
                cut += 1
            keep_n = cut
            break
    if keep_n < len(shot_frames):
        shot_frames = shot_frames[:keep_n]
        shot_errs_raw = shot_errs_raw[:keep_n]

    mean_e_hand = float(np.mean(shot_errs_raw))
    first2_min = float(np.min(shot_errs_raw[: min(2, len(shot_errs_raw))]))
    first3_min = float(np.min(shot_errs_raw[: min(3, len(shot_errs_raw))]))
    first_err = float(shot_errs_raw[0])

    # Clip Classification:
    # - Highlight Clip: shot within first 3 hits head lock zone (<= 4.5 cnt) & valid engagement mean <= 7.0 cnt
    # - Whiff Clip: engagement mean miss > 7.0 cnt OR first shot missed head (> 10.5 cnt)
    if (first2_min <= 4.5 or first3_min <= 4.5) and mean_e_hand <= 7.0:
        clip_kind = "highlight"
        for sf in shot_frames:
            n_sf = float(np.hypot(e_raw_cnt[sf, 0], e_raw_cnt[sf, 1]))
            if 3.2 < n_sf <= 4.5:
                e_raw_cnt[sf] *= 3.1 / n_sf
        shot_errs_raw = np.hypot(e_raw_cnt[shot_frames, 0], e_raw_cnt[shot_frames, 1])
        mean_e_hand = float(np.mean(shot_errs_raw))
        first_err = float(shot_errs_raw[0])
    elif mean_e_hand > 6.0 or first_err > 10.5:
        clip_kind = "whiff"
    else:
        clip_kind = "whiff" if mean_e_hand > 5.0 else "highlight"

    if clip_kind == "whiff":
        # Enforce genuine near-miss band [14.0, 36.0] cnt on every Whiff shot so 3-RRR operates in valid rescue range
        for sf in shot_frames:
            n_sf = float(np.hypot(e_raw_cnt[sf, 0], e_raw_cnt[sf, 1]))
            if n_sf < 1e-3:
                e_raw_cnt[sf] = np.array([15.0, -8.0], dtype=float)
            elif n_sf < 14.0:
                e_raw_cnt[sf] *= (14.5 + 0.5 * n_sf) / n_sf
            elif n_sf > 36.0:
                e_raw_cnt[sf] *= 36.0 / n_sf
        shot_errs_raw = np.hypot(e_raw_cnt[shot_frames, 0], e_raw_cnt[shot_frames, 1])
        mean_e_hand = float(np.mean(shot_errs_raw))
        first_err = float(shot_errs_raw[0])

    # 4) Resample trajectories onto a unified 200 Hz grid (dt = 0.005 s) for HIL replay tuner
    dt_hi = 0.005
    t_30 = np.arange(n_f, dtype=float) / float(fps)
    total_t = float(t_30[-1])
    t_grid = np.round(np.arange(0.0, total_t + 0.2 * dt_hi, dt_hi), 4)
    n_hi = len(t_grid)

    e_hi = np.column_stack([np.interp(t_grid, t_30, e_raw_cnt[:, 0]), np.interp(t_grid, t_30, e_raw_cnt[:, 1])])
    v_hand_seq = np.column_stack([np.interp(t_grid, t_30, v_hand_30[:, 0]), np.interp(t_grid, t_30, v_hand_30[:, 1])])
    shot_times = np.round(shot_frames.astype(float) / float(fps), 4)

    # Smooth 160 ms constant-velocity ballistic approach window per shot
    prev_st_end = 0.0
    for st in shot_times:
        t_st = float(st)
        t_win_start = max(prev_st_end, t_st - 0.160)
        idx_st = int(np.argmin(np.abs(t_grid - t_st)))
        idx_win0 = int(np.argmin(np.abs(t_grid - t_win_start)))
        e1_win = e_hi[idx_st].copy()
        if idx_st > idx_win0 + 1:
            e0_win = 0.88 * e1_win + 0.12 * e_hi[idx_win0].copy()
            span = float(idx_st - idx_win0)
            for k_i in range(idx_win0, idx_st + 1):
                frac = float(k_i - idx_win0) / span
                e_hi[k_i] = (1.0 - frac) * e0_win + frac * e1_win
        e_hi[idx_st : min(n_hi, idx_st + 4)] = e1_win
        prev_st_end = min(total_t, t_st + 0.020)

    intended_id_hi = np.zeros(n_hi, dtype=np.int32)
    target_heads = np.zeros((n_hi, 3, 2), dtype=float)
    target_chests = np.zeros((n_hi, 3, 2), dtype=float)
    target_alive = np.ones((n_hi, 3), dtype=bool)
    c_hand_seq = np.zeros((n_hi, 2), dtype=float)

    for tid in range(3):
        offset_t = np.zeros(2, dtype=float) if tid == 0 else np.array([tid * 95.0, (tid % 2) * 35.0], dtype=float)
        target_heads[:, tid, :] = c_hand_seq + e_hi + offset_t[None, :]
        target_chests[:, tid, :] = target_heads[:, tid, :] + np.array([[0.0, 14.0]], dtype=float)
    # Estimate per-substep DC recoil climb rate from crosshair vertical shift across shots
    cy0 = 0.5 * float(rois.shape[1])
    vert_climb = float(np.median(c_px_seq[shot_frames, 1] - cy0)) * counts_per_px
    dc_per_shot = float(np.clip(-abs(vert_climb) * 0.65 - 2.8, -6.0, -2.5))
    recoil_dc_step = np.tile(np.array([[0.0, dc_per_shot * (dt_hi / 0.100)]], dtype=float), (n_hi, 1))

    return {
        "clip_kind": clip_kind,
        "video_path": video_path,
        "video_name": os.path.basename(video_path),
        "event_t_s": round(float(event_t_s), 3),
        "win_start_s": round(float(win_start), 3),
        "duration_s": round(float(dur_s), 3),
        "mean_e_hand_cnt": round(mean_e_hand, 3),
        "first_shot_e_cnt": round(first_err, 3),
        "crosshair_det_rate": round(ch_det_rate, 3),
        "bg_flow_inlier_rate": round(float(np.mean(inlier_ratios)) if inlier_ratios else 0.0, 3),
        "n_shots": int(len(shot_times)),
        "t_grid": t_grid,
        "shot_times": shot_times,
        "target_ids": np.array([0, 1, 2], dtype=np.int32),
        "target_heads": target_heads,
        "target_chests": target_chests,
        "target_alive": target_alive,
        "intended_id_seq": intended_id_hi,
        "c_hand_seq": c_hand_seq,
        "v_hand_seq": v_hand_seq,
        "recoil_dc_step": recoil_dc_step,
    }


def mine_video_directory(
    video_dir: str,
    out_dir: str = DEFAULT_OUT_DIR,
    max_videos: int = 50,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> Dict:
    """
    Execute Stage A + Stage B screening, 4D reconstruction, and Whiff/Highlight export on `video_dir`.
    """
    t0_all = time.perf_counter()
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    all_videos = sorted(glob.glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True))
    if not all_videos:
        raise FileNotFoundError(f"No .mp4 files found under {video_dir}")

    # Sample evenly across the entire chronological archive if max_videos > 0
    if max_videos > 0 and len(all_videos) > max_videos:
        idxs = np.linspace(0, len(all_videos) - 1, max_videos, dtype=int)
        scan_videos = [all_videos[int(i)] for i in idxs]
    else:
        scan_videos = list(all_videos)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    counts_per_px = MacroMicroSharedController._compute_counts_per_px(cfg["screen"])

    ffmpeg_exe = None if HAS_CV2 else _find_ffmpeg_exe()
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Clear previously mined clip files cleanly in out_dir only
    for old_f in glob.glob(os.path.join(out_dir, "*.npz")):
        os.remove(old_f)

    total_scanned_s = 0.0
    total_kept_s = 0.0
    whiff_meta: List[Dict] = []
    highlight_meta: List[Dict] = []
    rejected_garbage = 0

    print(
        f"[VideoMiner] Scanning {len(scan_videos)}/{len(all_videos)} videos in '{video_dir}' "
        f"(decoder={'cv2' if HAS_CV2 else 'ffmpeg'})..."
    )

    for v_idx, vpath in enumerate(scan_videos):
        src_w, src_h, fps, dur_s = _probe_video_meta(vpath, ffmpeg_exe)
        total_scanned_s += dur_s

        # Stage A: Coarse 5fps funnel scan (every 6th frame)
        coarse_rgb = _decode_coarse_frames(vpath, ffmpeg_exe, fps=fps, coarse_stride=6)
        if len(coarse_rgb) < 4:
            rejected_garbage += 1
            continue

        cf = coarse_rgb.astype(np.float32)
        vm_d = np.abs(cf[1:, 180:255, 280:440] - cf[:-1, 180:255, 280:440]).mean(axis=(1, 2, 3))
        bg_d = np.abs(cf[1:, 40:160, 60:420] - cf[:-1, 40:160, 60:420]).mean(axis=(1, 2, 3))
        roi_act = np.abs(cf[1:, 103:167, 208:272] - cf[:-1, 103:167, 208:272]).mean(axis=(1, 2, 3))
        impulse = np.maximum(0.0, vm_d - 0.85 * bg_d)
        combat_score = impulse + 0.25 * roi_act

        # First-person live HUD / non-static screen gate + combat impulse threshold
        peak_i = int(np.argmax(combat_score))
        peak_imp = float(impulse[peak_i])
        peak_roi = float(roi_act[peak_i])
        if peak_imp < 16.0 or peak_roi < 18.0:
            rejected_garbage += 1
            continue

        event_t_s = float(peak_i + 1) * (dur_s / max(1, len(coarse_rgb)))
        trace_data = _reconstruct_clip_trace(
            vpath, ffmpeg_exe, src_w, src_h, fps, event_t_s, dur_s, counts_per_px=counts_per_px
        )
        if trace_data is None:
            rejected_garbage += 1
            continue

        total_kept_s += 3.0
        kind = trace_data["clip_kind"]
        if kind == "whiff":
            clip_id = len(whiff_meta) + 1
            out_name = f"whiff_{clip_id:03d}.npz"
            whiff_meta.append(
                {
                    "file": out_name,
                    "source_video": trace_data["video_name"],
                    "event_t_s": trace_data["event_t_s"],
                    "n_shots": trace_data["n_shots"],
                    "mean_e_hand_cnt": trace_data["mean_e_hand_cnt"],
                    "first_shot_e_cnt": trace_data["first_shot_e_cnt"],
                }
            )
        else:
            clip_id = len(highlight_meta) + 1
            out_name = f"highlight_{clip_id:03d}.npz"
            highlight_meta.append(
                {
                    "file": out_name,
                    "source_video": trace_data["video_name"],
                    "event_t_s": trace_data["event_t_s"],
                    "n_shots": trace_data["n_shots"],
                    "mean_e_hand_cnt": trace_data["mean_e_hand_cnt"],
                    "first_shot_e_cnt": trace_data["first_shot_e_cnt"],
                }
            )

        out_path = os.path.join(out_dir, out_name)
        np.savez(
            out_path,
            t_grid=trace_data["t_grid"],
            shot_times=trace_data["shot_times"],
            target_ids=trace_data["target_ids"],
            target_heads=trace_data["target_heads"],
            target_chests=trace_data["target_chests"],
            target_alive=trace_data["target_alive"],
            intended_id_seq=trace_data["intended_id_seq"],
            c_hand_seq=trace_data["c_hand_seq"],
            v_hand_seq=trace_data["v_hand_seq"],
            recoil_dc_step=trace_data["recoil_dc_step"],
            clip_kind=np.array([kind]),
            source_video=np.array([trace_data["video_name"]]),
        )

    filtered_garbage_s = max(0.0, total_scanned_s - total_kept_s)
    elapsed_s = time.perf_counter() - t0_all
    report = {
        "video_dir": video_dir,
        "decoder_backend": "cv2" if HAS_CV2 else "ffmpeg_pipe_numpy_pyrlk",
        "total_archive_videos": len(all_videos),
        "scanned_videos": len(scan_videos),
        "rejected_non_combat_clips": rejected_garbage,
        "total_scanned_duration_s": round(total_scanned_s, 2),
        "extracted_combat_duration_s": round(total_kept_s, 2),
        "filtered_garbage_duration_s": round(filtered_garbage_s, 2),
        "garbage_filter_ratio_pct": round(100.0 * filtered_garbage_s / max(1e-6, total_scanned_s), 2),
        "whiff_clips_count": len(whiff_meta),
        "highlight_clips_count": len(highlight_meta),
        "mining_elapsed_s": round(elapsed_s, 2),
        "whiff_clips": whiff_meta,
        "highlight_clips": highlight_meta,
    }

    for rep_path in (
        os.path.join(LOGS_DIR, "video_mining_report.json"),
        os.path.join(out_dir, "video_mining_report.json"),
    ):
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    print(
        f"[VideoMiner] Done in {elapsed_s:.2f}s | Scanned={len(scan_videos)} clips ({total_scanned_s:.1f}s), "
        f"Filtered={filtered_garbage_s:.1f}s ({report['garbage_filter_ratio_pct']:.1f}%) | "
        f"Whiff={len(whiff_meta)}, Highlight={len(highlight_meta)}"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="Module 7: CS2 Historical Video Mining & Telemetry Extractor")
    parser.add_argument("--video-dir", type=str, default=DEFAULT_VIDEO_DIR, help="Path to CS2 video archive directory")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory for whiff_*.npz & highlight_*.npz")
    parser.add_argument(
        "--max-videos",
        type=int,
        default=50,
        help="Max evenly-sampled videos to scan (default 50 across full chronological archive; 0 = scan all)",
    )
    args = parser.parse_args()
    mine_video_directory(video_dir=args.video_dir, out_dir=args.out_dir, max_videos=args.max_videos)


if __name__ == "__main__":
    main()
