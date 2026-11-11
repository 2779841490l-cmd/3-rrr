import time
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.sensor.start()

print("Testing trigger reasons for 5 shots on (+30, +20):")
for shot in range(5):
    ctrl.driver.center(settle_s=0.35, preload=True)
    res = ctrl.flick_shot(30, 20, timeout_ms=100.0, enable_ilc_update=False)
    traj = res['trajectory']
    b_idx = [i for i, p in enumerate(traj) if p['phase'] == 'LOCK']
    b0 = traj[b_idx[0]] if b_idx else None
    pre_b = traj[b_idx[0]-1] if b_idx and b_idx[0] > 0 else None
    print(f"Shot {shot+1}: final={res['final_delta']}, err={res['final_error_counts']}")
    if b0 and pre_b:
        print(f"   Pre-Lock: active={pre_b['active_ms']:.1f}ms, pos=({pre_b['x']},{pre_b['y']}), v=({pre_b['vx']},{pre_b['vy']})")
        print(f"   Lock:     active={b0['active_ms']:.1f}ms, pos=({b0['x']},{b0['y']}), v=({b0['vx']},{b0['vy']})")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

