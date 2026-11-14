"""
telemetry_referee.py - Module 4: Millisecond Ring-Buffer Referee & 4-Fault Auto-Attribution Engine
==================================================================================================
Maintains a 5-second high-rate ring buffer and evaluates every shot k at firing instant t_k:
1. Actual Miss Error E_actual(t_k) = ||h_{i*}(t_k) - c(t_k)||_2
2. Counterfactual Hand-Only Error E_hand(t_k) = ||h_{i*}(t_k) - (c(t_k) - p_stage(t_k))||_2
3. Net Assist Gain Delta_E_k = E_hand(t_k) - E_actual(t_k)
4. Hitbox Verdict: HEAD_HIT | BODY_HIT | WHIFF
5. 4-Fault Mathematical Attribution & Gradient Directions:
   - FAULT_LAG           -> increase k_lead
   - FAULT_CONFLICT      -> decrease lambda_0 or narrow sigma_v
   - FAULT_SATURATION    -> decrease washout time constant tau_w
   - FAULT_HANDOVER_DELAY-> increase beta or decrease P_switch
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional
import numpy as np


FAULT_LAG = "FAULT_LAG"
FAULT_CONFLICT = "FAULT_CONFLICT"
FAULT_SATURATION = "FAULT_SATURATION"
FAULT_HANDOVER_DELAY = "FAULT_HANDOVER_DELAY"


@dataclass
class RingFrame:
    t_s: float
    c_cnt: np.ndarray
    p_stage_cnt: np.ndarray
    v_hand_cnt_s: np.ndarray
    active_target_id: int
    intended_target_id: int
    err_vec_cnt: np.ndarray


@dataclass
class ShotVerdict:
    shot_idx: int
    t_s: float
    scenario: str
    active_target_id: int
    intended_target_id: int
    anchor_mode: str
    hitbox: str                         # "HEAD_HIT" | "BODY_HIT" | "WHIFF"
    e_vec_cnt: np.ndarray
    e_actual_cnt: float
    e_actual_px: float
    e_hand_cnt: float
    delta_e_cnt: float                  # >0 good assist, <0 machine hurt
    p_stage_cnt: np.ndarray
    head_dist_cnt: float
    chest_dist_cnt: float
    conflict_flag: bool
    saturation_flag: bool
    faults: List[str] = field(default_factory=list)


class TelemetryReferee:
    """Black-box telemetry recorder, shot-instant evaluator, and 4-fault attribution engine."""

    def __init__(
        self,
        counts_per_px: float = 5.6,
        head_radius_cnt: float = 10.5,
        chest_radius_cnt: float = 21.0,
        hard_limit_cnt: float = 88.0,
        conflict_thresh_cnt: float = -1.5,
        maxlen: int = 5000,
    ):
        self.counts_per_px = float(counts_per_px)
        self.head_radius_cnt = float(head_radius_cnt)
        self.chest_radius_cnt = float(chest_radius_cnt)
        self.hard_limit_cnt = float(hard_limit_cnt)
        self.conflict_thresh_cnt = float(conflict_thresh_cnt)
        self.ring: Deque[RingFrame] = deque(maxlen=maxlen)
        self.verdicts: List[ShotVerdict] = []

    def reset(self) -> None:
        self.ring.clear()
        self.verdicts.clear()

    def push_frame(
        self,
        t_s: float,
        c_cnt: np.ndarray,
        p_stage_cnt: np.ndarray,
        v_hand_cnt_s: np.ndarray,
        active_target_id: int,
        intended_target_id: int,
        err_vec_cnt: np.ndarray,
    ) -> None:
        self.ring.append(
            RingFrame(
                t_s=float(t_s),
                c_cnt=np.asarray(c_cnt, dtype=float).copy(),
                p_stage_cnt=np.asarray(p_stage_cnt, dtype=float).copy(),
                v_hand_cnt_s=np.asarray(v_hand_cnt_s, dtype=float).copy(),
                active_target_id=int(active_target_id),
                intended_target_id=int(intended_target_id),
                err_vec_cnt=np.asarray(err_vec_cnt, dtype=float).copy(),
            )
        )

    def _find_frame_ago(self, t_now_s: float, lookback_s: float = 0.080) -> Optional[RingFrame]:
        target_t = float(t_now_s) - float(lookback_s)
        best: Optional[RingFrame] = None
        for fr in self.ring:
            if fr.t_s <= target_t:
                best = fr
            else:
                break
        return best

    def evaluate_shot(
        self,
        shot_idx: int,
        t_s: float,
        scenario: str,
        c_cnt: np.ndarray,
        p_stage_cnt: np.ndarray,
        head_true_cnt: np.ndarray,
        chest_true_cnt: np.ndarray,
        v_target_cnt_s: np.ndarray,
        v_hand_cnt_s: np.ndarray,
        active_target_id: int,
        intended_target_id: int,
        anchor_mode: str,
        handover_delay_ms: float = 0.0,
        head_radius_override: Optional[float] = None,
    ) -> ShotVerdict:
        """Snapshot referee evaluation at bullet firing instant t_k."""
        c = np.asarray(c_cnt, dtype=float)
        p_st = np.asarray(p_stage_cnt, dtype=float)
        h_head = np.asarray(head_true_cnt, dtype=float)
        h_chest = np.asarray(chest_true_cnt, dtype=float)
        v_tgt = np.asarray(v_target_cnt_s, dtype=float)
        v_h = np.asarray(v_hand_cnt_s, dtype=float)

        # Counterfactual hand-only crosshair (what human alone would aim at)
        c_hand = c - p_st

        # Reference target point aligns with hitbox priority
        ref_pt = h_chest if anchor_mode == "CHEST" else h_head
        e_vec = ref_pt - c
        e_actual_cnt = float(np.hypot(e_vec[0], e_vec[1]))
        e_actual_px = e_actual_cnt / self.counts_per_px
        e_hand_cnt = float(np.hypot(ref_pt[0] - c_hand[0], ref_pt[1] - c_hand[1]))
        delta_e_cnt = e_hand_cnt - e_actual_cnt

        # Hitbox check: does crosshair land within physical head OR chest radius?
        head_dist = float(np.hypot(h_head[0] - c[0], h_head[1] - c[1]))
        chest_dist = float(np.hypot(h_chest[0] - c[0], h_chest[1] - c[1]))
        r_head = float(head_radius_override) if head_radius_override is not None else self.head_radius_cnt
        if head_dist <= r_head:
            hitbox = "HEAD_HIT"
        elif chest_dist <= self.chest_radius_cnt:
            hitbox = "BODY_HIT"
        else:
            hitbox = "WHIFF"

        # 4-Fault Mathematical Attribution
        faults: List[str] = []

        # 1. FAULT_LAG: residual aligns > 60% with target velocity when moving
        v_tgt_norm = float(np.hypot(v_tgt[0], v_tgt[1]))
        if v_tgt_norm > 15.0 and e_actual_cnt > 1.0:
            cos_align = float(np.dot(e_vec, v_tgt)) / (e_actual_cnt * v_tgt_norm)
            if cos_align > 0.60 and (hitbox == "WHIFF" or delta_e_cnt < -self.counts_per_px):
                faults.append(FAULT_LAG)

        # 2. FAULT_CONFLICT: zero-crossing overshoot OR significant negative assist (Delta_E < -1.5 cnt)
        prev_fr = self._find_frame_ago(t_s, lookback_s=0.080)
        conflict_flag = delta_e_cnt < self.conflict_thresh_cnt
        if prev_fr is not None:
            dp_stage_win = p_st - prev_fr.p_stage_cnt
            dp_hand_win = v_h * 0.080
            same_sign_drive = float(np.dot(dp_hand_win, dp_stage_win)) > 0.0
            sign_flip = float(np.dot(e_vec, prev_fr.err_vec_cnt)) < 0.0
            if (sign_flip and same_sign_drive and hitbox == "WHIFF" and delta_e_cnt < 0.0) or conflict_flag:
                conflict_flag = True
                if FAULT_CONFLICT not in faults:
                    faults.append(FAULT_CONFLICT)

        # 3. FAULT_SATURATION: stage reaches >= 88 counts mechanical boundary
        saturation_flag = max(abs(float(p_st[0])), abs(float(p_st[1]))) >= self.hard_limit_cnt
        if saturation_flag:
            faults.append(FAULT_SATURATION)

        # 4. FAULT_HANDOVER_DELAY: wrong target locked after > 40 ms hand swipe intent
        if (active_target_id != intended_target_id) or (handover_delay_ms > 40.0):
            if hitbox == "WHIFF" or active_target_id != intended_target_id:
                faults.append(FAULT_HANDOVER_DELAY)

        verdict = ShotVerdict(
            shot_idx=int(shot_idx),
            t_s=float(t_s),
            scenario=str(scenario),
            active_target_id=int(active_target_id),
            intended_target_id=int(intended_target_id),
            anchor_mode=anchor_mode,
            hitbox=hitbox,
            e_vec_cnt=e_vec,
            e_actual_cnt=round(e_actual_cnt, 3),
            e_actual_px=round(e_actual_px, 4),
            e_hand_cnt=round(e_hand_cnt, 3),
            delta_e_cnt=round(delta_e_cnt, 3),
            p_stage_cnt=p_st.copy(),
            head_dist_cnt=round(head_dist, 3),
            chest_dist_cnt=round(chest_dist, 3),
            conflict_flag=bool(conflict_flag),
            saturation_flag=bool(saturation_flag),
            faults=faults,
        )
        self.verdicts.append(verdict)
        return verdict

    def summarize(self) -> Dict:
        n = max(1, len(self.verdicts))
        e_actual = [v.e_actual_cnt for v in self.verdicts]
        e_hand = [v.e_hand_cnt for v in self.verdicts]
        delta_e = [v.delta_e_cnt for v in self.verdicts]
        heads = sum(1 for v in self.verdicts if v.hitbox == "HEAD_HIT")
        bodies = sum(1 for v in self.verdicts if v.hitbox == "BODY_HIT")
        whiffs = sum(1 for v in self.verdicts if v.hitbox == "WHIFF")
        conflicts = sum(1 for v in self.verdicts if v.conflict_flag)
        sats = sum(1 for v in self.verdicts if v.saturation_flag)

        fault_counts = {
            FAULT_LAG: sum(1 for v in self.verdicts if FAULT_LAG in v.faults),
            FAULT_CONFLICT: sum(1 for v in self.verdicts if FAULT_CONFLICT in v.faults),
            FAULT_SATURATION: sum(1 for v in self.verdicts if FAULT_SATURATION in v.faults),
            FAULT_HANDOVER_DELAY: sum(1 for v in self.verdicts if FAULT_HANDOVER_DELAY in v.faults),
        }

        max_stage = max(
            (max(abs(float(v.p_stage_cnt[0])), abs(float(v.p_stage_cnt[1]))) for v in self.verdicts),
            default=0.0,
        )

        # Recommend gradient directions based on dominant faults
        grad_hints: List[str] = []
        if fault_counts[FAULT_LAG] > 0:
            grad_hints.append("+k_lead (target motion lag detected)")
        if fault_counts[FAULT_CONFLICT] > 0:
            grad_hints.append("-sigma_v / same-direction damping (human-stage flick conflict)")
        if fault_counts[FAULT_SATURATION] > 0:
            grad_hints.append("-tau_w (washout decay too slow -> stage hit boundary)")
        if fault_counts[FAULT_HANDOVER_DELAY] > 0:
            grad_hints.append("+beta, -P_switch (slow multi-target handover)")

        return {
            "total_shots": len(self.verdicts),
            "mean_e_actual_cnt": round(float(np.mean(e_actual)) if e_actual else 0.0, 3),
            "mean_e_hand_cnt": round(float(np.mean(e_hand)) if e_hand else 0.0, 3),
            "mean_delta_e_cnt": round(float(np.mean(delta_e)) if delta_e else 0.0, 3),
            "head_hit_rate": round(heads / n, 4),
            "effective_hit_rate": round((heads + bodies) / n, 4),
            "whiff_rate": round(whiffs / n, 4),
            "conflict_rate": round(conflicts / n, 4),
            "saturation_count": int(sats),
            "saturation_rate": round(sats / n, 4),
            "max_stage_disp_cnt": round(max_stage, 2),
            "fault_counts": fault_counts,
            "gradient_hints": grad_hints,
        }
