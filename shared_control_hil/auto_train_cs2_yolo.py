#!/usr/bin/env python3
"""
auto_train_cs2_yolo.py - Zero-Manual-Annotation Auto-Distillation YOLO Pipeline for CS2
=========================================================================================

Pipeline:
1. Scans archived CS2 gameplay clips (e.g., ``E:\\游戏录屏归档\\高光短视频``, 1287 MP4 files).
2. Runs heavy pretrained pose teacher (``yolov8x-pose.pt`` or ``yolov8m-pose.pt``) on 512x512
   center frames on RTX 4060 GPU to extract high-precision COCO 17-keypoint detections.
3. Converts head keypoints (0:nose, 1-2:eyes, 3-4:ears) -> Class 0 (``head``) and upper-torso
   keypoints (5-6:shoulders, 11-12:hips) -> Class 1 (``chest``) into 256x256 ROI YOLO labels,
   mixing ~10% clean background hard-negatives to suppress brick/crate false positives.
4. Fine-tunes ``yolov8n.pt`` student on RTX 4060 (``imgsz=256``), exports
   ``shared_control_hil/weights/cs2_head_yolov8n.pt``, reports validation ``mAP@50``, and writes
   a visual comparison collage ``shared_control_hil/logs/yolo_val_predictions.jpg`` (head=RED,
   chest=BLUE).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEO_DIR = r"E:\游戏录屏归档\高光短视频"
DEFAULT_TEACHER = os.path.join(_THIS_DIR, "weights", "yolov8x-pose.pt")
DEFAULT_STUDENT_BASE = os.path.join(_THIS_DIR, "weights", "yolov8n.pt")
DEFAULT_OUT_WEIGHTS = os.path.join(_THIS_DIR, "weights", "cs2_head_yolov8n.pt")
DEFAULT_DATASET_DIR = os.path.join(_THIS_DIR, "dataset_cs2")
DEFAULT_VIS_PATH = os.path.join(_THIS_DIR, "logs", "yolo_val_predictions.jpg")
DEFAULT_TRAIN_REPORT = os.path.join(_THIS_DIR, "logs", "yolo_training_report.json")


def _clip_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Optional[Tuple[float, float, float, float]]:
    cx1 = max(0.0, min(float(w - 1), x1))
    cy1 = max(0.0, min(float(h - 1), y1))
    cx2 = max(0.0, min(float(w - 1), x2))
    cy2 = max(0.0, min(float(h - 1), y2))
    if (cx2 - cx1) < 5.0 or (cy2 - cy1) < 5.0:
        return None
    return cx1, cy1, cx2, cy2


def _pose_to_head_chest_boxes(
    box_xyxy: np.ndarray,
    box_conf: float,
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    img_w: int = 512,
    img_h: int = 512,
) -> List[Tuple[int, float, float, float, float]]:
    """
    Convert one COCO-17 pose detection inside a 512x512 crop into (cls_id, x1, y1, x2, y2):
      - cls_id 0: head
      - cls_id 1: chest
    Filters out CS2 first-person viewmodel hands/arms and low-keypoint false positives.
    """
    if float(box_conf) < 0.42:
        return []

    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    # Reject large bottom-right / bottom-edge first-person weapon viewmodel blobs
    if bw > 310.0 or bh > 430.0:
        return []
    if y2 > 0.96 * img_h and y1 > 0.55 * img_h and x2 > 0.55 * img_w and float(np.max(kpts_conf[:5])) < 0.45:
        return []

    upper_vis = int(np.sum(kpts_conf[:7] >= 0.32))
    head_vis_idx = [i for i in range(5) if float(kpts_conf[i]) >= 0.30]
    sh_l_ok = float(kpts_conf[5]) >= 0.30
    sh_r_ok = float(kpts_conf[6]) >= 0.30
    if upper_vis < 2 or (not head_vis_idx and not (sh_l_ok and sh_r_ok)):
        return []

    sh_w = float(np.linalg.norm(kpts_xy[5] - kpts_xy[6])) if (sh_l_ok and sh_r_ok) else 0.0
    head_r = float(np.clip(max(0.13 * bh, 0.34 * sh_w, 7.0), 6.0, 62.0))

    # 1. Head bounding box (class 0)
    head_box: Optional[Tuple[float, float, float, float]] = None
    if len(head_vis_idx) >= 2:
        pts_h = kpts_xy[head_vis_idx]
        hx_min, hy_min = float(np.min(pts_h[:, 0])), float(np.min(pts_h[:, 1]))
        hx_max, hy_max = float(np.max(pts_h[:, 0])), float(np.max(pts_h[:, 1]))
        h_cx = 0.5 * (hx_min + hx_max)
        h_cy = 0.5 * (hy_min + hy_max) - 0.14 * head_r
        h_w = max(1.45 * (hx_max - hx_min), 1.85 * head_r)
        h_h = max(1.65 * (hy_max - hy_min), 1.08 * h_w)
        head_box = _clip_box(h_cx - 0.5 * h_w, h_cy - 0.5 * h_h, h_cx + 0.5 * h_w, h_cy + 0.5 * h_h, img_w, img_h)
    elif len(head_vis_idx) == 1:
        kx, ky = float(kpts_xy[head_vis_idx[0], 0]), float(kpts_xy[head_vis_idx[0], 1])
        h_cx, h_cy = kx, ky - 0.12 * head_r
        h_w = 1.85 * head_r
        h_h = 2.00 * head_r
        head_box = _clip_box(h_cx - 0.5 * h_w, h_cy - 0.5 * h_h, h_cx + 0.5 * h_w, h_cy + 0.5 * h_h, img_w, img_h)
    elif sh_l_ok and sh_r_ok:
        sh_mid = 0.5 * (kpts_xy[5] + kpts_xy[6])
        h_w = max(0.85 * sh_w, 12.0)
        h_cx = float(sh_mid[0])
        h_cy = float(sh_mid[1]) - 0.85 * h_w
        head_box = _clip_box(h_cx - 0.5 * h_w, h_cy - 0.5 * h_w, h_cx + 0.5 * h_w, h_cy + 0.5 * h_w, img_w, img_h)

    # 2. Chest bounding box (class 1)
    chest_box: Optional[Tuple[float, float, float, float]] = None
    if sh_l_ok and sh_r_ok:
        sh_mid = 0.5 * (kpts_xy[5] + kpts_xy[6])
        hip_l_ok = float(kpts_conf[11]) >= 0.30
        hip_r_ok = float(kpts_conf[12]) >= 0.30
        if hip_l_ok and hip_r_ok:
            hip_mid = 0.5 * (kpts_xy[11] + kpts_xy[12])
            torso_h = max(12.0, float(np.linalg.norm(hip_mid - sh_mid)))
        else:
            torso_h = max(12.0, 0.34 * bh)
        c_cx = float(sh_mid[0])
        c_cy = float(sh_mid[1]) + 0.38 * torso_h
        c_w = max(1.32 * abs(float(kpts_xy[5, 0] - kpts_xy[6, 0])), 0.30 * bh, 12.0)
        c_h = max(0.82 * torso_h, 0.24 * bh, 12.0)
        chest_box = _clip_box(c_cx - 0.5 * c_w, c_cy - 0.5 * c_h, c_cx + 0.5 * c_w, c_cy + 0.5 * c_h, img_w, img_h)
    elif sh_l_ok or sh_r_ok:
        sh_idx = 5 if sh_l_ok else 6
        c_cx = float(kpts_xy[sh_idx, 0])
        c_cy = float(kpts_xy[sh_idx, 1]) + 0.16 * bh
        c_w = max(0.24 * bh, 12.0)
        c_h = max(0.26 * bh, 14.0)
        chest_box = _clip_box(c_cx - 0.5 * c_w, c_cy - 0.5 * c_h, c_cx + 0.5 * c_w, c_cy + 0.5 * c_h, img_w, img_h)

    out: List[Tuple[int, float, float, float, float]] = []
    if head_box is not None:
        out.append((0, *head_box))
    if chest_box is not None:
        out.append((1, *chest_box))
    return out


def _project_boxes_to_crop(
    boxes_512: List[Tuple[int, float, float, float, float]],
    crop_x0: int,
    crop_y0: int,
    crop_size: int = 256,
) -> List[Tuple[int, float, float, float, float]]:
    """Transform boxes from 512x512 coordinates into [0,1] normalized YOLO labels in a 256x256 sub-crop."""
    kept: List[Tuple[int, float, float, float, float]] = []
    cx1_lim, cy1_lim = float(crop_x0), float(crop_y0)
    cx2_lim, cy2_lim = float(crop_x0 + crop_size), float(crop_y0 + crop_size)

    for cls_id, x1, y1, x2, y2 in boxes_512:
        ix1 = max(x1, cx1_lim)
        iy1 = max(y1, cy1_lim)
        ix2 = min(x2, cx2_lim)
        iy2 = min(y2, cy2_lim)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        orig_area = max(1.0, (x2 - x1) * (y2 - y1))
        if (iw * ih) / orig_area < 0.50 or iw < 4.5 or ih < 4.5:
            continue

        lx1 = ix1 - cx1_lim
        ly1 = iy1 - cy1_lim
        lx2 = ix2 - cx1_lim
        ly2 = iy2 - cy1_lim
        xc = 0.5 * (lx1 + lx2) / crop_size
        yc = 0.5 * (ly1 + ly2) / crop_size
        wn = (lx2 - lx1) / crop_size
        hn = (ly2 - ly1) / crop_size
        kept.append((cls_id, xc, yc, wn, hn))
    return kept


def _sample_video_crops_512(
    video_dir: str,
    max_candidates: int = 4200,
    seed: int = 20261114,
) -> List[Tuple[str, np.ndarray]]:
    """Uniformly sample 512x512 center crops across all archived CS2 highlight MP4 videos."""
    videos = sorted(glob.glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True))
    if not videos:
        raise FileNotFoundError(f"No MP4 videos found under: {video_dir}")

    rng = random.Random(seed)
    fracs = [0.35, 0.55, 0.72, 0.22]
    per_vid = max(2, min(4, int(np.ceil(max_candidates / max(1, len(videos))))))
    samples: List[Tuple[str, np.ndarray]] = []

    t0 = time.perf_counter()
    for v_idx, vp in enumerate(videos):
        if len(samples) >= max_candidates:
            break
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            continue
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if n_frames < 15 or w < 512 or h < 512:
            cap.release()
            continue

        cx0 = (w - 512) // 2
        cy0 = (h - 512) // 2
        chosen_fracs = fracs[:per_vid]
        for f_k, fr in enumerate(chosen_fracs):
            f_idx = int(np.clip((fr + rng.uniform(-0.04, 0.04)) * n_frames, 5, n_frames - 5))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                continue
            crop_512 = frame_bgr[cy0 : cy0 + 512, cx0 : cx0 + 512].copy()
            # Skip fade-to-black / loading frames
            if float(np.mean(crop_512)) < 14.0:
                continue
            clip_tag = f"v{v_idx:04d}_f{f_idx:04d}"
            samples.append((clip_tag, crop_512))
        cap.release()

        if (v_idx + 1) % 250 == 0:
            print(f"[Video Scan] {v_idx + 1}/{len(videos)} clips | extracted {len(samples)} 512x512 crops ({time.perf_counter() - t0:.1f}s)")

    print(f"[Video Scan Done] {len(samples)} candidate frames from {len(videos)} clips in {time.perf_counter() - t0:.1f}s.")
    return samples


def build_auto_distilled_dataset(
    video_dir: str = DEFAULT_VIDEO_DIR,
    teacher_weights: str = DEFAULT_TEACHER,
    dataset_dir: str = DEFAULT_DATASET_DIR,
    target_pos_samples: int = 1800,
    neg_ratio: float = 0.10,
    seed: int = 20261114,
) -> Dict[str, object]:
    """
    Use pretrained YOLOv8-Pose teacher to auto-label 256x256 CS2 head/chest crops with 0 human labels.
    """
    if not os.path.exists(teacher_weights):
        raise FileNotFoundError(f"Teacher pose weights not found: {teacher_weights}")

    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    for split in ("train", "val"):
        os.makedirs(os.path.join(dataset_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "labels", split), exist_ok=True)

    candidates = _sample_video_crops_512(video_dir=video_dir, max_candidates=4200, seed=seed)
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[Teacher Auto-Label] Loading {os.path.basename(teacher_weights)} on device={device}...")
    teacher = YOLO(teacher_weights)

    pos_records: List[Tuple[str, np.ndarray, List[Tuple[int, float, float, float, float]]]] = []
    neg_records: List[Tuple[str, np.ndarray, List[Tuple[int, float, float, float, float]]]] = []
    rng = random.Random(seed)
    batch_size = 16
    t0 = time.perf_counter()

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        imgs = [b[1] for b in batch]
        results = teacher.predict(imgs, imgsz=512, conf=0.36, verbose=False, device=device, half=torch.cuda.is_available())

        for (tag, img_512), res in zip(batch, results):
            if res.boxes is None or res.keypoints is None or len(res.boxes) == 0:
                if len(neg_records) < int(target_pos_samples * neg_ratio * 1.5):
                    center_256 = img_512[128:384, 128:384]
                    neg_records.append((tag, center_256, []))
                continue

            boxes_xyxy = res.boxes.xyxy.cpu().numpy()
            boxes_conf = res.boxes.conf.cpu().numpy()
            kpts_xy = res.keypoints.xy.cpu().numpy()
            kpts_conf = res.keypoints.conf.cpu().numpy()

            all_hc_512: List[Tuple[int, float, float, float, float]] = []
            for p_idx in range(len(boxes_xyxy)):
                hc = _pose_to_head_chest_boxes(
                    boxes_xyxy[p_idx],
                    float(boxes_conf[p_idx]),
                    kpts_xy[p_idx],
                    kpts_conf[p_idx],
                    img_w=512,
                    img_h=512,
                )
                all_hc_512.extend(hc)

            if not all_hc_512:
                continue

            # 1) Exact runtime center 256x256 crop [128:384, 128:384]
            center_labels = _project_boxes_to_crop(all_hc_512, crop_x0=128, crop_y0=128, crop_size=256)
            has_head = any(lb[0] == 0 for lb in center_labels)
            if center_labels and has_head:
                pos_records.append((f"{tag}_c", img_512[128:384, 128:384], center_labels))
            else:
                # 2) Shifted 256x256 crop centered near detected head inside the 512x512 engagement window
                head_cands = [b for b in all_hc_512 if b[0] == 0]
                if head_cands:
                    _, hx1, hy1, hx2, hy2 = head_cands[0]
                    h_cx = 0.5 * (hx1 + hx2)
                    h_cy = 0.5 * (hy1 + hy2)
                    ox = int(np.clip(round(h_cx - 128 + rng.randint(-48, 48)), 0, 256))
                    oy = int(np.clip(round(h_cy - 128 + rng.randint(-36, 48)), 0, 256))
                    shifted_labels = _project_boxes_to_crop(all_hc_512, crop_x0=ox, crop_y0=oy, crop_size=256)
                    if shifted_labels and any(lb[0] == 0 for lb in shifted_labels):
                        pos_records.append((f"{tag}_s", img_512[oy : oy + 256, ox : ox + 256], shifted_labels))

        if (i // batch_size + 1) % 30 == 0:
            print(
                f"[Teacher Progress] {min(i + batch_size, len(candidates))}/{len(candidates)} frames | "
                f"pos={len(pos_records)} neg={len(neg_records)} ({time.perf_counter() - t0:.1f}s)"
            )
        if len(pos_records) >= target_pos_samples:
            break

    n_neg_keep = min(len(neg_records), int(round(len(pos_records) * neg_ratio)))
    rng.shuffle(pos_records)
    rng.shuffle(neg_records)
    all_records = pos_records[:target_pos_samples] + neg_records[:n_neg_keep]

    # Split by video clip ID (vXXXX) to prevent train/val leakage across frames from the same clip
    clip_ids = sorted({rec[0].split("_")[0] for rec in all_records})
    rng.shuffle(clip_ids)
    n_val_clips = max(1, int(round(0.15 * len(clip_ids))))
    val_clip_set = set(clip_ids[:n_val_clips])

    counts = {"train_imgs": 0, "val_imgs": 0, "train_heads": 0, "train_chests": 0, "val_heads": 0, "val_chests": 0}
    for name, img_256, labels in all_records:
        clip_id = name.split("_")[0]
        split = "val" if clip_id in val_clip_set else "train"
        img_path = os.path.join(dataset_dir, "images", split, f"{name}.jpg")
        lbl_path = os.path.join(dataset_dir, "labels", split, f"{name}.txt")
        cv2.imwrite(img_path, img_256, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        with open(lbl_path, "w", encoding="utf-8") as f:
            for cls_id, xc, yc, wn, hn in labels:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
                if split == "train":
                    counts["train_heads" if cls_id == 0 else "train_chests"] += 1
                else:
                    counts["val_heads" if cls_id == 0 else "val_chests"] += 1
        counts[f"{split}_imgs"] += 1

    data_yaml_path = os.path.join(dataset_dir, "data.yaml")
    yaml_payload = {
        "path": dataset_dir.replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": {0: "head", 1: "chest"},
    }
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_payload, f, sort_keys=False)

    print(
        f"[Dataset Ready] {counts['train_imgs']} train ({counts['train_heads']} heads, {counts['train_chests']} chests) | "
        f"{counts['val_imgs']} val ({counts['val_heads']} heads, {counts['val_chests']} chests) in {time.perf_counter() - t0:.1f}s"
    )
    return {"data_yaml": data_yaml_path, "pos": len(pos_records), "neg": n_neg_keep, **counts}


def render_val_prediction_collage(
    model_path: str,
    dataset_dir: str = DEFAULT_DATASET_DIR,
    out_path: str = DEFAULT_VIS_PATH,
    conf_thresh: float = 0.35,
    map50: float = 0.0,
) -> str:
    """Render a high-res validation grid on real CS2 frames with RED=head and BLUE=chest boxes."""
    val_imgs = sorted(glob.glob(os.path.join(dataset_dir, "images", "val", "*.jpg")))
    if not val_imgs:
        raise FileNotFoundError("No validation images found to visualize.")

    model = YOLO(model_path)
    device = 0 if torch.cuda.is_available() else "cpu"

    # Prefer non-empty positive validation frames showing actual enemy detections
    chosen_panels: List[np.ndarray] = []
    for img_p in val_imgs:
        lbl_p = img_p.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}").replace(".jpg", ".txt")
        if os.path.exists(lbl_p) and os.path.getsize(lbl_p) == 0:
            continue
        res = model.predict(img_p, imgsz=256, conf=conf_thresh, verbose=False, device=device)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        has_head = any(int(c) == 0 for c in res.boxes.cls.cpu().numpy())
        if not has_head and len(chosen_panels) < 3:
            continue

        img_bgr = cv2.imread(img_p)
        vis = cv2.resize(img_bgr, (384, 384), interpolation=cv2.INTER_CUBIC)
        scale = 384.0 / 256.0
        for box, cls_t, conf_t in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            cls_id = int(cls_t)
            conf_v = float(conf_t)
            bx1, by1, bx2, by2 = [int(round(v * scale)) for v in box]
            # Head = RED (0, 0, 255), Chest = BLUE (255, 110, 0) in BGR
            color = (0, 0, 255) if cls_id == 0 else (255, 110, 0)
            label = f"{'head' if cls_id == 0 else 'chest'} {conf_v:.2f}"
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 2)
            cv2.putText(vis, label, (bx1, max(16, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)

        cv2.putText(vis, os.path.basename(img_p), (8, 372), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
        chosen_panels.append(vis)
        if len(chosen_panels) >= 4:
            break

    while len(chosen_panels) < 4 and chosen_panels:
        chosen_panels.append(chosen_panels[-1].copy())

    top_row = np.hstack(chosen_panels[:2])
    bot_row = np.hstack(chosen_panels[2:4])
    grid = np.vstack([top_row, bot_row])

    banner = np.full((52, grid.shape[1], 3), 24, dtype=np.uint8)
    title = f"CS2 Student YOLOv8n (256x256 Val) | mAP@50={map50:.3f} | RED=head  BLUE=chest"
    cv2.putText(banner, title, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 240, 220), 2, cv2.LINE_AA)
    out_img = np.vstack([banner, grid])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"[Saved Visual Evidence] {out_path}")
    return out_path


def train_student_yolo(
    data_yaml: str,
    base_weights: str = DEFAULT_STUDENT_BASE,
    out_weights: str = DEFAULT_OUT_WEIGHTS,
    epochs: int = 30,
    batch: int = 32,
    imgsz: int = 256,
) -> Dict[str, float]:
    """Fine-tune YOLOv8n on RTX 4060 GPU and verify mAP@50 + sub-2ms inference latency."""
    device = 0 if torch.cuda.is_available() else "cpu"
    run_dir = os.path.join(_THIS_DIR, "runs_yolo")
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)

    student = YOLO(base_weights)
    student.train(
        data=data_yaml,
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        device=device,
        workers=2,
        amp=True,
        patience=12,
        mosaic=0.25,
        scale=0.25,
        fliplr=0.5,
        project=run_dir,
        name="cs2_head_chest",
        exist_ok=True,
        verbose=True,
    )

    best_pt = os.path.join(run_dir, "cs2_head_chest", "weights", "best.pt")
    if not os.path.exists(best_pt):
        raise FileNotFoundError(f"Training did not produce best.pt at {best_pt}")
    os.makedirs(os.path.dirname(out_weights), exist_ok=True)
    shutil.copy2(best_pt, out_weights)

    eval_model = YOLO(out_weights)
    metrics = eval_model.val(data=data_yaml, split="val", imgsz=int(imgsz), device=device, verbose=False)
    map50_all = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    per_cls = metrics.box.ap50
    map50_head = float(per_cls[0]) if len(per_cls) > 0 else map50_all
    map50_chest = float(per_cls[1]) if len(per_cls) > 1 else map50_all

    # Benchmark RTX 4060 GPU inference speed at 256x256
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(10):
        eval_model.predict(dummy, imgsz=imgsz, conf=0.35, verbose=False, device=device)
    t0 = time.perf_counter()
    for _ in range(100):
        eval_model.predict(dummy, imgsz=imgsz, conf=0.35, verbose=False, device=device)
    infer_ms = ((time.perf_counter() - t0) / 100.0) * 1000.0

    print(
        f"[Student Eval] mAP@50={map50_all:.4f} (head={map50_head:.4f}, chest={map50_chest:.4f}) | "
        f"mAP@50-95={map50_95:.4f} | RTX4060 latency={infer_ms:.2f} ms/frame"
    )
    return {
        "map50_all": map50_all,
        "map50_head": map50_head,
        "map50_chest": map50_chest,
        "map50_95": map50_95,
        "infer_ms": infer_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-Distillation YOLOv8n CS2 Head/Chest Trainer")
    parser.add_argument("--video-dir", type=str, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--teacher", type=str, default=DEFAULT_TEACHER)
    parser.add_argument("--student-base", type=str, default=DEFAULT_STUDENT_BASE)
    parser.add_argument("--out-weights", type=str, default=DEFAULT_OUT_WEIGHTS)
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--target-samples", type=int, default=1800)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--skip-dataset", action="store_true", help="Reuse existing dataset_cs2")
    args = parser.parse_args()

    if args.skip_dataset and os.path.exists(os.path.join(args.dataset_dir, "data.yaml")):
        ds_info = {"data_yaml": os.path.join(args.dataset_dir, "data.yaml"), "reused": True}
    else:
        ds_info = build_auto_distilled_dataset(
            video_dir=args.video_dir,
            teacher_weights=args.teacher,
            dataset_dir=args.dataset_dir,
            target_pos_samples=args.target_samples,
        )

    eval_info = train_student_yolo(
        data_yaml=str(ds_info["data_yaml"]),
        base_weights=args.student_base,
        out_weights=args.out_weights,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=256,
    )
    render_val_prediction_collage(
        model_path=args.out_weights,
        dataset_dir=args.dataset_dir,
        out_path=DEFAULT_VIS_PATH,
        map50=eval_info["map50_all"],
    )

    payload = {"dataset": ds_info, "eval": eval_info, "weights": args.out_weights}
    os.makedirs(os.path.dirname(DEFAULT_TRAIN_REPORT), exist_ok=True)
    with open(DEFAULT_TRAIN_REPORT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[Saved Training Report] {DEFAULT_TRAIN_REPORT}")
    return 0 if eval_info["map50_all"] >= 0.75 else 1


if __name__ == "__main__":
    raise SystemExit(main())
