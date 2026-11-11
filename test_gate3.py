import time
import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.ilc.params["target_table"] = {}
ctrl.sensor.start()

print("=" * 72)
print("  TESTING GATE 3 (Full-Screen Macro Flick-Shots: +-160, +-90 counts)")
print("=" * 72)

macro_targets = [(160, -90), (-160, 90)]
for mdx, mdy in macro_targets:
    print(f"\nMacro Target ({mdx:+4d}, {mdy:+4d}):")
    for w in range(1, 7):
        ctrl.driver.center(settle_s=0.40, preload=True)
        res = ctrl.flick_shot(mdx, mdy, timeout_ms=100.0, enable_ilc_update=True)
        fd = res["final_delta"]
        err = res["final_error_counts"]
        corr = ctrl.ilc.get_correction(mdx, mdy)
        print(f"  Warmup {w}: Final ({fd[0]:+4d}, {fd[1]:+4d}) | Err: {err:5.2f} cnt | ILC: [{corr[0]:+5.1f}, {corr[1]:+5.1f}]")
        
    ctrl.driver.center(settle_s=0.40, preload=True)
    res = ctrl.flick_shot(mdx, mdy, timeout_ms=100.0, enable_ilc_update=False)
    m_err = res["final_error_counts"]
    m_status = "PASS" if m_err <= 6.5 else "FAIL"
    print(
        f"  [{m_status}] Macro Target ({mdx:+4d}, {mdy:+4d}) -> Final {res['final_delta']} | "
        f"Err: {m_err:5.2f} cnt ({m_err*2.8:5.1f} px on Full Screen)"
    )

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

