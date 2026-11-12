"""
unidirectional_approach.py — 单向趋近精修（不修改基线 rect_manifold_calibrator.py）

v2 安全版（2026-11-12 撞屏边事故后）：
  - 越界急停：读数超出 ±BOUND_RATIO*W/H，或单拍位移 > JUMP_ABORT_CNT -> driver.center() 并抛 SafetyAbort
  - max_retry 默认 0：过冲直接接受结果并记录，不再退回重来
  - 预备点被限制在 ±0.95 W/H 内
"""
import time
from typing import Dict

import numpy as np

from rect_manifold_calibrator import HardwareRectangleTrainer, WS63ServoDriver

BOUND_RATIO = 1.05
JUMP_ABORT_CNT = 50.0


class SafetyAbort(RuntimeError):
    pass


def _guard(trainer, p: np.ndarray, p_prev: np.ndarray = None):
    m = trainer.manifold
    bad = abs(p[0]) > BOUND_RATIO * m.W or abs(p[1]) > BOUND_RATIO * m.H
    jump = p_prev is not None and float(np.hypot(*(p - p_prev))) > JUMP_ABORT_CNT
    if bad or jump:
        try:
            trainer.driver.center(settle_s=0.4)
        finally:
            raise SafetyAbort(f"pos={np.round(p, 1).tolist()} prev={None if p_prev is None else np.round(p_prev, 1).tolist()} "
                              f"{'OUT_OF_BOUND' if bad else 'JUMP'}")


def _send(trainer, u_kin: np.ndarray, kick: np.ndarray = None):
    trainer.manifold.pi_net.last_u_kin = u_kin.copy()
    u_cmd = u_kin + trainer.manifold.pi_net.play_g
    lo, hi = WS63ServoDriver.SAFE_MIN_US, WS63ServoDriver.SAFE_MAX_US
    if kick is not None and np.any(kick != 0.0):
        trainer.driver.send_raw_pulses(float(np.clip(u_cmd[0] + kick[0], lo, hi)),
                                       float(np.clip(u_cmd[1] + kick[1], lo, hi)))
        time.sleep(0.028)
    trainer.driver.send_raw_pulses(float(np.clip(u_cmd[0], lo, hi)), float(np.clip(u_cmd[1], lo, hi)))


def move_unidirectional(
    trainer: HardwareRectangleTrainer,
    x_target: float,
    y_target: float,
    tol_counts: float = 1.2,
    approach_off: float = 15.0,
    overshoot_tol: float = 2.0,
    max_trim: int = 24,
    max_retry: int = 0,
) -> Dict:
    m = trainer.manifold
    x_tgt = float(np.clip(x_target, -0.95 * m.W, 0.95 * m.W))
    y_tgt = float(np.clip(y_target, -0.95 * m.H, 0.95 * m.H))
    p_target = np.array([x_tgt, y_tgt], dtype=float)
    p_start = trainer.get_rect_position()
    _guard(trainer, p_start)
    t0 = time.perf_counter()

    K_trim = m.K_trim
    du_dir = np.sign(np.diag(K_trim))
    min_err = 1e9
    retries = 0
    trims_total = 0
    overshoot = False
    ballistic_ms = 0.0
    r_pre = None

    while True:
        pre = p_target - approach_off
        pre[0] = float(np.clip(pre[0], -0.95 * m.W, 0.95 * m.W))
        pre[1] = float(np.clip(pre[1], -0.95 * m.H, 0.95 * m.H))
        r_pre = trainer.move_point_to_point(float(pre[0]), float(pre[1]), tol_counts=3.0)
        if ballistic_ms == 0.0:
            ballistic_ms = r_pre["ballistic_ms"]
        cur_u_kin = np.array(r_pre["u_kin_final"], dtype=float)

        p_prev = trainer.get_rect_position()
        _guard(trainer, p_prev)
        stall = np.zeros(2)
        overshoot = False
        for k in range(max_trim):
            p = trainer.get_rect_position()
            _guard(trainer, p, p_prev if k > 0 else None)
            e = p_target - p
            err = float(np.hypot(e[0], e[1]))
            min_err = min(min_err, err)
            if err <= tol_counts:
                break
            if np.any(e < -overshoot_tol):
                overshoot = True
                break
            moved = np.abs(p - p_prev) if k > 0 else np.full(2, 9.0)
            p_prev = p.copy()
            for i in (0, 1):
                if e[i] > 1.0 and moved[i] < 1.0:
                    stall[i] += 1
                elif moved[i] >= 1.5:
                    stall[i] = 0

            e_pos = np.where(e > 0.6, e, 0.0)
            gain = 0.55 * (1.0 + 0.30 * np.minimum(stall, 4))
            max_du = max(2.0, min(12.0, 0.40 * err))
            du = np.clip(gain * (K_trim @ e_pos), -max_du, max_du)
            du = np.where(np.sign(du) == du_dir, du, 0.0)
            if not np.any(du):
                break
            cur_u_kin = cur_u_kin + du
            kick = np.where((stall >= 1) & (np.abs(du) > 0.18), np.sign(du) * (4.5 + np.minimum(stall, 3)), 0.0)
            _send(trainer, cur_u_kin, kick)
            trainer.wait_until_settled(min_wait_s=0.065, max_wait_s=0.110, vel_tol_cnt=0.8)
            trims_total += 1

        if overshoot and retries < max_retry:
            retries += 1
            continue
        break

    p_final = trainer.get_rect_position()
    _guard(trainer, p_final)
    final_err = float(np.hypot(x_tgt - p_final[0], y_tgt - p_final[1]))
    min_err = min(min_err, final_err)
    return {
        "start": (round(float(p_start[0]), 1), round(float(p_start[1]), 1)),
        "target": (round(x_tgt, 1), round(y_tgt, 1)),
        "final": (round(float(p_final[0]), 1), round(float(p_final[1]), 1)),
        "err_counts": round(final_err, 2),
        "err_um": round(final_err * 7.9375, 1),
        "straight_dev_cnt": r_pre["straight_dev_cnt"],
        "straight_dev_pct": r_pre["straight_dev_pct"],
        "ballistic_ms": round(ballistic_ms, 1),
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "u_kin_final": cur_u_kin.tolist(),
        "retries": retries,
        "trims": trims_total,
        "overshoot": int(overshoot),
        "min_err_seen": round(min_err, 2),
    }
