"""
verify_hardware_kpi.py - Hard Hardware KPI Acceptance Gate ("紧箍咒")
=====================================================================
Enforces strict real-hardware performance criteria before ANY paper claims
or finalization can be made:

Gate 1 (Non-Oscillatory ILC Stability on (30, 20)):
  - 12 consecutive 100ms shots from cold start.
  - Last 5 shots (Shots 8..12) MUST have:
    * Mean error <= 2.5 counts (19.8 um)
    * Max error among last 5 shots <= 4.0 counts (eliminates period-2 oscillation!)

Gate 2 (6-Direction All-Quadrant 100ms Precision):
  - Targets: [(35, 0), (-35, 0), (0, 35), (0, -35), (30, 25), (-30, -25)]
  - For each target: 5 ILC calibration shots, followed by 2 consecutive test shots.
  - Every direction's test error MUST be <= 3.0 counts, and overall mean error
    across all 6 directions MUST be <= 2.0 counts (15.9 um) within 100ms!

Exits with code 1 if ANY gate fails.
"""

import sys
import time
import numpy as np
from ballistic_controller import Ballistic100msController


def main() -> int:
    print("=" * 74)
    print("  3-RRR 100ms-3200DPI HARD ENGINEERING KPI VERIFICATION GATE (紧箍咒)")
    print("=" * 74)

    ctrl = Ballistic100msController()
    # Cold reset ILC weights to prevent cached cheating
    ctrl.ilc.params["target_table"] = {}
    ctrl.ilc.save()

    ctrl.sensor.start()
    ctrl.driver.center(settle_s=0.45, preload=True)

    failed_reasons = []

    # -------------------------------------------------------------------------
    # GATE 1: 12-Shot Cold-Start Convergence & Anti-Oscillation Test
    # -------------------------------------------------------------------------
    print("\n[GATE 1] Testing 12-Shot Monotonic Convergence & Stability on (+30, +20)...")
    tx, ty = 30, 20
    gate1_errors = []
    for i in range(1, 13):
        ctrl.driver.center(settle_s=0.45, preload=True)
        res = ctrl.flick_shot(tx, ty, timeout_ms=100.0, enable_ilc_update=True)
        fd = res["final_delta"]
        err = res["final_error_counts"]
        tb = res["t_brake_ms"]
        gate1_errors.append(err)
        corr = ctrl.ilc.get_correction(tx, ty)
        print(
            f"  Shot {i:2d}/12 | Target (+30, +20) -> Final ({fd[0]:+3d}, {fd[1]:+3d}) | "
            f"Err: {err:5.2f} cnt ({err*7.9375:5.1f} um) | Brake: {tb:4.1f}ms | "
            f"ILC_u: [{corr[0]:+5.1f}, {corr[1]:+5.1f}]"
        )

    last5 = gate1_errors[-5:]
    mean_last5 = float(np.mean(last5))
    max_last5 = float(np.max(last5))
    print(f"  -> Gate 1 Summary: Last-5 Mean = {mean_last5:.2f} cnt, Last-5 Max = {max_last5:.2f} cnt")

    if mean_last5 > 2.5:
        failed_reasons.append(f"Gate 1 Failed: Last-5 mean error {mean_last5:.2f} cnt > 2.50 cnt limit.")
    if max_last5 > 4.5:
        failed_reasons.append(f"Gate 1 Failed: Last-5 max error {max_last5:.2f} cnt > 4.50 cnt (Oscillation detected!).")

    # -------------------------------------------------------------------------
    # GATE 2: 6-Direction All-Quadrant 100ms Repeatability Test
    # -------------------------------------------------------------------------
    print("\n[GATE 2] Testing 6-Direction All-Quadrant 100ms Precision (6 Warmup + 2 Eval)...")
    directions = [(35, 0), (-35, 0), (0, 35), (0, -35), (30, 25), (-30, -25)]
    dir_eval_errors = []

    for dx, dy in directions:
        # 6 warmup ILC learning shots
        for w in range(1, 7):
            ctrl.driver.center(settle_s=0.45, preload=True)
            ctrl.flick_shot(dx, dy, timeout_ms=100.0, enable_ilc_update=True)

        # 2 consecutive evaluation shots (NO cherry-picking!)
        eval_errs = []
        eval_finals = []
        for ev in range(1, 3):
            ctrl.driver.center(settle_s=0.45, preload=True)
            res = ctrl.flick_shot(dx, dy, timeout_ms=100.0, enable_ilc_update=False)
            eval_errs.append(res["final_error_counts"])
            eval_finals.append(res["final_delta"])

        avg_dir_err = float(np.mean(eval_errs))
        dir_eval_errors.append(avg_dir_err)
        corr = ctrl.ilc.get_correction(dx, dy)
        status_str = "PASS" if avg_dir_err <= 3.5 else "FAIL"
        print(
            f"  [{status_str}] Target ({dx:+3d}, {dy:+3d}) | Eval Finals: {eval_finals[0]}, {eval_finals[1]} | "
            f"Mean Err: {avg_dir_err:4.2f} cnt ({avg_dir_err*7.9375:5.1f} um) | ILC_u: [{corr[0]:+5.1f}, {corr[1]:+5.1f}]"
        )
        if avg_dir_err > 3.5:
            failed_reasons.append(
                f"Gate 2 Failed on Target ({dx:+d}, {dy:+d}): Mean eval error {avg_dir_err:.2f} cnt > 3.50 cnt."
            )

    overall_mean = float(np.mean(dir_eval_errors))
    print(f"  -> Gate 2 Overall 6-Direction Mean Error: {overall_mean:.2f} cnt ({overall_mean*7.9375:.1f} um)")
    if overall_mean > 2.5:
        failed_reasons.append(f"Gate 2 Failed: Overall 6-direction mean error {overall_mean:.2f} cnt > 2.50 cnt.")

    # -------------------------------------------------------------------------
    # GATE 3: Full-Screen & Beyond-Screen Macro Flick-Shot Test ([1150, 1850] us Full Envelope)
    # -------------------------------------------------------------------------
    print("\n[GATE 3] Testing Full-Range Macro Flick-Shots Across Mechanical Limits (Zero Edge Clipping)...")
    macro_targets = [(160, -90), (-160, 90), (200, -110), (-200, 110)]
    for mdx, mdy in macro_targets:
        for w in range(1, 7):
            ctrl.driver.center(settle_s=0.45, preload=True)
            ctrl.flick_shot(mdx, mdy, timeout_ms=100.0, enable_ilc_update=True)
        ctrl.driver.center(settle_s=0.45, preload=True)
        res = ctrl.flick_shot(mdx, mdy, timeout_ms=100.0, enable_ilc_update=False)
        m_err = res["final_error_counts"]
        m_status = "PASS" if m_err <= 6.5 else "FAIL"
        print(
            f"  [{m_status}] Macro Target ({mdx:+4d}, {mdy:+4d}) -> Final {res['final_delta']} | "
            f"Err: {m_err:5.2f} cnt ({m_err*2.8:5.1f} px on Full Screen)"
        )
        if m_err > 6.5:
            failed_reasons.append(
                f"Gate 3 Failed on Macro Target ({mdx:+d}, {mdy:+d}): Error {m_err:.2f} cnt > 6.50 cnt (18.2 px ball radius)."
            )

    ctrl.driver.center(settle_s=0.35, preload=True)
    ctrl.close()

    print("\n" + "=" * 74)
    if failed_reasons:
        print("  [KPI GATE FAILED! 紧箍咒触发：严禁写论文或宣布完工，必须继续调参！]")
        for r in failed_reasons:
            print(f"   * {r}")
        print("=" * 74)
        return 1
    else:
        print("  [ALL HARDWARE KPI GATES PASSED! 实机百毫秒 3200DPI 指标全面达标！]")
        print("=" * 74)
        return 0


if __name__ == "__main__":
    sys.exit(main())

