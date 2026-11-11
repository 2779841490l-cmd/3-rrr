import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.ilc.params["target_table"] = {}
ctrl.sensor.start()

macro_targets = [(160, -90), (-160, 90)]
print("Testing Gate 3 Macro Targets...")

for mdx, mdy in macro_targets:
    for w in range(1, 7):
        ctrl.driver.center(settle_s=0.40, preload=True)
        res_w = ctrl.flick_shot(mdx, mdy, timeout_ms=100.0, enable_ilc_update=True)
        corr = ctrl.ilc.get_correction(mdx, mdy)
        print(f"  Warmup {w}: Target ({mdx:+4d}, {mdy:+4d}) -> Final {res_w['final_delta']} | Err: {res_w['final_error_counts']:5.1f} cnt | Brake: {res_w['t_brake_ms']:4.1f}ms | ILC_u: [{corr[0]:+5.1f}, {corr[1]:+5.1f}]")

    ctrl.driver.center(settle_s=0.40, preload=True)
    res = ctrl.flick_shot(mdx, mdy, timeout_ms=100.0, enable_ilc_update=False)
    m_err = res["final_error_counts"]
    m_status = "PASS" if m_err <= 6.5 else "FAIL"
    print(f"[{m_status}] Macro Target ({mdx:+4d}, {mdy:+4d}) -> Final {res['final_delta']} | Err: {m_err:5.2f} cnt ({m_err*2.8:5.1f} px)")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()
