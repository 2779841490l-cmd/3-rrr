from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.ilc.params['target_table'] = {}
ctrl.sensor.start()
ctrl.driver.center(settle_s=0.35, preload=True)
res = ctrl.flick_shot(30, 20, timeout_ms=100.0, enable_ilc_update=False)

print(f"Final: {res['final_delta']}, Err: {res['final_error_counts']}, Brake: {res['t_brake_ms']}ms, Onset: {res['t_onset_ms']}ms")
for step in res['trajectory']:
    if step['active_ms'] >= 5.0 and step['active_ms'] <= 50.0:
        print(f"act={step['active_ms']:4.1f}ms: pos=({step['x']:2d}, {step['y']:2d}), v=({step['vx']:6.1f}, {step['vy']:6.1f}), err={step['err_counts']:4.1f}, phase={step['phase']}")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()
