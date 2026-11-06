"""
run_phase_4.py
Phase 4: Pedestrian Dead Reckoning — Live Runner.
Main loop ported exactly from Pedestrian_Dead_Reckoning.py.
Uses modular core for UDP, calibration, and config.
"""

import os
import sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import math
import time
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation

from core.network import UDPReceiver
from core.config_loader import load_config
from core.calibration import CalibrationManager
from core.math_utils import get_rpy_swing_safe, wrap
from GUI import gui_phase4_pedestrian as gui_module
from estimation.ESKF_12_state import ESKF_Trajectory
from estimation.ESKF_6_state import ESKF_6_State
from estimation.step_detector import StepDetector, ActivityDetector
from estimation.mechanization import SavageMechanization
from estimation.serial_KF_noise import IMUPreprocessor


def main():
    config       = load_config('phase_4')
    calib_manager = CalibrationManager()

    # ── Pull calibration values from CalibrationManager ───────────────────
    MAG_BIAS   = calib_manager.MAG_BIAS
    MAG_AINV   = calib_manager.MAG_AINV
    ACCEL_BIAS = calib_manager.ACCEL_BIAS
    ACCEL_AINV = calib_manager.ACCEL_AINV
    GYRO_BIAS  = calib_manager.GYRO_BIAS
    R_imu      = calib_manager.R_imu

    # ── Pull config values ─────────────────────────────────────────────────
    G                    = config.get("gravity", 9.80665)
    MAX_DT               = config.get("max_dt", 0.2)
    CALIBRATION_TIME     = config.get("calibration_time", 10.0)
    CALIBRATE_IMU        = config.get("calibrate_imu", True)

    STEADY_ACC_VAR_THRESH  = config.get("steady_acc_var_thresh",  0.005)
    STEADY_GYRO_VAR_THRESH = config.get("steady_gyro_var_thresh", 0.0005)
    FLOOR_ACC_MIN          = STEADY_ACC_VAR_THRESH  * 0.1
    FLOOR_GYRO_MIN         = STEADY_GYRO_VAR_THRESH * 0.1

    FLOOR_EMA_ALPHA      = config.get("floor_ema_alpha",      0.01)
    MDR_ACC_ENTER_RATIO  = config.get("mdr_acc_enter_ratio",  1.6)
    MDR_ACC_EXIT_RATIO   = config.get("mdr_acc_exit_ratio",   2.5)
    MDR_GYRO_ENTER_RATIO = config.get("mdr_gyro_enter_ratio", 3.0)
    MDR_GYRO_EXIT_RATIO  = config.get("mdr_gyro_exit_ratio",  5.0)
    STEADY_MIN_SEC       = config.get("steady_min_sec",        0.15)

    MARU_ENABLED         = config.get("maru_enabled",          True)
    MARU_R_YAW_SCALE     = config.get("maru_r_yaw_scale",      0.05)
    ZARU_SIGMA           = config.get("zaru_sigma",             0.01)
    ZUPT_ENABLED         = config.get("zupt_enabled",           True)
    ZUPT_SIGMA           = config.get("zupt_sigma",             0.02)

    ADAPTIVE_KF_R_ENABLED = config.get("adaptive_kf_r_enabled", True)
    KF_Q_ACC              = config.get("kf_q_acc",   1e-4)
    KF_Q_GYRO             = config.get("kf_q_gyro",  1e-5)
    SIGMA_ACCEL_VRW_SCALE = config.get("sigma_accel_vrw_scale", 1.0)

    ZDU_ENABLED          = config.get("zdu_enabled",      True)
    ZDU_MIN_HOLD_SEC     = config.get("zdu_min_hold_sec", 0.3)
    ZDU_RAMP_SEC         = config.get("zdu_ramp_sec",     1.0)
    ZDU_MAX_GAIN         = config.get("zdu_max_gain",     0.15)

    # ── GUI ────────────────────────────────────────────────────────────────
    gui = gui_module.setup_gui()
    import matplotlib.pyplot as plt
    plt.show(block=False)

    # ── Instantiate both filters (exactly as in monolith) ─────────────────
    eskf_traj = ESKF_Trajectory(calib_manager)   # 12-state ESKF (navigation)
    eskf_att  = ESKF_6_State()                   # 6-state ESKF  (attitude / RPY)

    savage            = SavageMechanization(g=G)
    imu_pre           = IMUPreprocessor(calib_manager, config, calibrate=CALIBRATE_IMU)
    step_detector     = StepDetector()
    activity_detector = ActivityDetector()

    # ── Trajectory save buffers ────────────────────────────────────────────
    t_save     = []
    pdr_e_save = []
    pdr_n_save = []
    pos_d_save = []

    # ── GUI history buffers ────────────────────────────────────────────────
    p_hist   = []
    t_hist   = deque(maxlen=gui_module.TS_HISTORY_MAXLEN)
    bx_hist  = deque(maxlen=gui_module.TS_HISTORY_MAXLEN)
    by_hist  = deque(maxlen=gui_module.TS_HISTORY_MAXLEN)
    bz_hist  = deque(maxlen=gui_module.TS_HISTORY_MAXLEN)
    vel_hist = deque(maxlen=gui_module.TS_HISTORY_MAXLEN)

    prev_imu_ts  = None
    initialized  = False

    acc_floor   = STEADY_ACC_VAR_THRESH
    gyro_floor  = STEADY_GYRO_VAR_THRESH
    was_trans_steady     = False
    was_fully_steady     = False
    acc_norm_window      = deque(maxlen=10)
    gyro_norm_window     = deque(maxlen=10)
    steady_candidate_since = None

    stationary_since      = None
    stationary_anchor_pos = None

    acc_history    = deque(maxlen=20)
    pdr_e          = 0.0
    pdr_n          = 0.0
    step_k         = 0.72
    total_distance = 0.0
    total_steps    = 0
    is_idle        = False

    start_time     = time.time()
    last_plot_time = 0.0

    calibration_done  = False
    calib_start_time  = time.time()
    calib_samples     = 0
    acc_bias_zupt     = np.zeros(3)
    gyro_bias_zupt    = np.zeros(3)

    mag_cal                = np.zeros(3)
    rpy_traj               = np.zeros(3)
    rpy_att                = np.zeros(3)
    yaw_rad                = 0.0
    curr_pos_fake          = np.zeros(3)
    steady_flight_confirmed = False
    vel                    = np.zeros(3)
    initial_heading_att    = 0.0

    # ── UDP ────────────────────────────────────────────────────────────────
    network = UDPReceiver(
        port=config.get("udp_port", 5005),
        packet_format=config.get("packet_format", "<II6fI3f")
    )
    network.start()

    print("Running Live Tracker with Phase02 Attitude ESKF...")
    try:
        while True:
            if not plt.fignum_exists(gui["fig"].number):
                break

            packets_to_process = network.get_all_packets()
            if len(packets_to_process) == 0:
                gui_module.keep_windows_responsive()
                continue

            for pkt in packets_to_process:
                (packet_id, imu_ts_raw, ax_raw, ay_raw, az_raw,
                 gx_raw, gy_raw, gz_raw, mag_ts, mx_raw, my_raw, mz_raw) = pkt

                imu_ts = imu_ts_raw / 1000.0

                if prev_imu_ts is None:
                    prev_imu_ts = imu_ts
                    continue

                dt = imu_ts - prev_imu_ts
                prev_imu_ts = imu_ts
                if dt <= 0 or dt > MAX_DT:
                    continue

                acc_raw  = np.array([ax_raw, ay_raw, az_raw])
                gyro_raw = np.array([gx_raw, gy_raw, gz_raw])
                mag_raw  = np.array([mx_raw, my_raw, mz_raw])

                acc_cal_pre, gyro_cal_pre = imu_pre.process(
                    acc_raw, gyro_raw,
                    eskf_traj.accel_b if initialized else None
                )

                # ── Calibration ────────────────────────────────────────────
                if not calibration_done:
                    elapsed = time.time() - calib_start_time
                    if elapsed < CALIBRATION_TIME:
                        acc_bias_zupt  += acc_cal_pre
                        gyro_bias_zupt += gyro_cal_pre
                        calib_samples  += 1
                        print(f"Calibrating... {elapsed:.2f} / {CALIBRATION_TIME:.2f} sec", end="\r")
                        continue
                    if calib_samples > 0:
                        acc_bias_zupt  /= calib_samples
                        gyro_bias_zupt /= calib_samples
                        acc_bias_zupt[2] -= 1.0   # Gravity subtraction
                    calibration_done = True
                    print("\nCalibration complete.")

                acc_cal  = acc_cal_pre  - acc_bias_zupt
                gyro_cal = gyro_cal_pre - gyro_bias_zupt

                mag_cal        = MAG_AINV @ (mag_raw - MAG_BIAS)
                mag_cal_scaled = mag_cal / 100.0
                gyro_rad       = gyro_cal * (np.pi / 180.0)

                # ── Initialisation (both filters) ─────────────────────────
                if not initialized:
                    eskf_traj.initialize(acc_cal, mag_cal_scaled)
                    initial_yaw_traj = eskf_traj.yaw_initial
                    eskf_traj.q = Rotation.from_euler('Z', initial_yaw_traj) * eskf_traj.q
                    eskf_traj.yaw_initial = 0.0

                    eskf_att.initialize(acc_cal, mag_cal_scaled)
                    initial_heading_att = eskf_att.yaw_initial

                    initialized = True
                    continue

                # ── Steady-state detection ────────────────────────────────
                acc_norm  = np.linalg.norm(acc_cal)
                gyro_norm = np.linalg.norm(gyro_rad)
                acc_norm_window.append(acc_norm)
                gyro_norm_window.append(gyro_norm)
                acc_history.append(acc_norm)

                is_steady    = False
                fully_steady = False

                if len(acc_norm_window) == acc_norm_window.maxlen:
                    acc_var   = float(np.var(acc_norm_window))
                    gyro_var  = float(np.var(gyro_norm_window))
                    mdr_acc   = acc_var  / max(acc_floor,  FLOOR_ACC_MIN)
                    mdr_gyro  = gyro_var / max(gyro_floor, FLOOR_GYRO_MIN)
                    acc_ratio_thresh  = MDR_ACC_EXIT_RATIO   if was_trans_steady   else MDR_ACC_ENTER_RATIO
                    gyro_ratio_thresh = MDR_GYRO_EXIT_RATIO  if was_fully_steady   else MDR_GYRO_ENTER_RATIO

                    no_translation = mdr_acc < acc_ratio_thresh
                    fully_steady   = no_translation and (mdr_gyro < gyro_ratio_thresh)

                    if fully_steady:
                        acc_floor  = (1.0 - FLOOR_EMA_ALPHA) * acc_floor  + FLOOR_EMA_ALPHA * acc_var
                        gyro_floor = (1.0 - FLOOR_EMA_ALPHA) * gyro_floor + FLOOR_EMA_ALPHA * gyro_var

                    is_steady        = no_translation
                    was_trans_steady = is_steady
                    was_fully_steady = fully_steady

                    if ADAPTIVE_KF_R_ENABLED:
                        imu_pre.set_accel_R(acc_floor)
                        imu_pre.set_gyro_R(gyro_floor)
                    eskf_traj.set_accel_noise(acc_floor)

                if is_steady:
                    if steady_candidate_since is None:
                        steady_candidate_since = imu_ts
                    steady_flight_confirmed = ((imu_ts - steady_candidate_since) >= STEADY_MIN_SEC)
                else:
                    steady_candidate_since  = None
                    steady_flight_confirmed = False

                # ── Update both filters ───────────────────────────────────
                eskf_traj.update(
                    gyro_rad, acc_cal, mag_cal_scaled, dt,
                    steady_flight=(fully_steady and MARU_ENABLED),
                    maru_r_scale=MARU_R_YAW_SCALE
                )
                eskf_att.update(gyro_rad, acc_cal, mag_cal_scaled, dt)

                # ── Savage mechanisation (uses trajectory attitude) ────────
                pos, vel = savage.update(dt, gyro_rad, acc_cal, eskf_traj.q.as_matrix())

                # ── ZUPT ─────────────────────────────────────────────────
                if fully_steady:
                    eskf_traj.update_zaru(gyro_rad, ZARU_SIGMA ** 2)
                    if ZUPT_ENABLED:
                        dv = eskf_traj.update_zupt(savage.vel, R_zupt=ZUPT_SIGMA ** 2)
                        savage.apply_velocity_correction(dv)
                        pos, vel = savage.pos, savage.vel

                # ── ZDU ──────────────────────────────────────────────────
                if ZDU_ENABLED:
                    if steady_flight_confirmed:
                        if stationary_since is None:
                            stationary_since      = time.time()
                            stationary_anchor_pos = savage.pos.copy()
                        hold_time = time.time() - stationary_since
                        if hold_time >= ZDU_MIN_HOLD_SEC:
                            ramp = min(1.0, (hold_time - ZDU_MIN_HOLD_SEC) / ZDU_RAMP_SEC)
                            savage.pos = savage.pos + ZDU_MAX_GAIN * ramp * (stationary_anchor_pos - savage.pos)
                            pos = savage.pos
                    else:
                        stationary_since      = None
                        stationary_anchor_pos = None

                # ── Extract RPY from both filters ─────────────────────────
                rpy_traj = get_rpy_swing_safe(eskf_traj.q.as_matrix())
                rpy_att  = get_rpy_swing_safe(eskf_att.q.as_matrix())
                yaw_rad  = rpy_traj[2]

                # ── Step detection ─────────────────────────────────────────
                is_step = step_detector.update(acc_norm, imu_ts)
                if is_step and len(acc_history) >= 5:
                    total_steps += 1
                    w_arr = list(acc_history)
                    sl    = step_k * ((np.max(w_arr) - np.min(w_arr)) ** 0.25)
                    step_len = float(np.clip(sl, 0.45, 0.95))
                    total_distance += step_len
                    pdr_e += step_len * math.sin(yaw_rad)
                    pdr_n += step_len * math.cos(yaw_rad)

                curr_pos_fake = np.array([pdr_e, pdr_n, pos[2]])
                p_hist.append(curr_pos_fake)

                t_save.append(imu_ts)
                pdr_e_save.append(pdr_e)
                pdr_n_save.append(pdr_n)
                pos_d_save.append(pos[2])

            # ── GUI update at 45 Hz ────────────────────────────────────────
            now = time.time()
            if now - last_plot_time >= 1.0 / 45.0:
                mag_intensity_ut = np.linalg.norm(mag_cal)

                # Attitude display (from eskf_att – Phase02)
                roll_deg_raw, pitch_deg_raw, yaw_rel_deg = np.rad2deg(rpy_att)
                yaw_abs_deg = np.rad2deg(wrap(rpy_att[2] + initial_heading_att))

                roll_disp, yaw_disp = gui_module.compute_damped_display_angles(
                    roll_deg_raw, pitch_deg_raw, yaw_rel_deg, gui, now)
                pitch_disp = pitch_deg_raw

                gui_module.update_horizon(gui, roll_disp, pitch_disp)
                gui_module.update_rpy_panel(gui, roll_disp, pitch_disp, yaw_disp)

                # Compass — absolute heading from attitude filter
                abs_heading_deg = yaw_abs_deg
                gui_module.update_compass(gui, np.rad2deg(wrap(np.deg2rad(abs_heading_deg))))

                # Map — uses trajectory position and relative yaw
                gui_module.update_map(gui, curr_pos_fake, p_hist, yaw_rad)

                # Pedestrian activity lights
                is_idle = activity_detector.update(steady_flight_confirmed, now)
                gui_module.update_pedestrian(gui, step_detector, is_idle)

                # Gyro bias and velocity from trajectory filter
                elapsed_time = now - start_time
                t_hist.append(elapsed_time)
                bx_hist.append(np.rad2deg(eskf_traj.b[0]))
                by_hist.append(np.rad2deg(eskf_traj.b[1]))
                bz_hist.append(np.rad2deg(eskf_traj.b[2]))
                vel_hist.append(np.linalg.norm(vel))
                gui_module.update_gyro_and_vel(gui, t_hist, bx_hist, by_hist, bz_hist, vel_hist)

                gui_module.update_magnetic(gui, mag_intensity_ut)

                gui["fig"].canvas.draw_idle()
                last_plot_time = now

            gui_module.keep_windows_responsive()

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        network.stop()
        gui_module.save_gui_plot(gui)
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.close('all')


if __name__ == "__main__":
    main()