
import os
import sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


"""
Inertial Dead Reckoning with ZUPT + ZARU
GUI       : Phase_3_trial_42_arghya.py  (red dot body, blue trajectory,
                                          auto-rescaling axes)
Logic     : Phase_3_Trial_35.py         (adaptive ZUPT/MDR, MARU, ZARU, ZDU)
True Heading / Compass : Phase3_20_arghya.py
    - R_mag = np.eye(3)  -- magnetometer stays in raw sensor frame so the
      tilt-compensated magnetic bearing fed to the ESKF is Earth-referenced.
    - gui_module.true_heading_deg() and gui_module.update_compass() carry the full mathematical
      commentary from Phase3_20 explaining the heading derivation.
"""

import socket
import struct
import threading
import time
from collections import deque
from pathlib import Path


from core.config_loader import load_config
from core.network import UDPReceiver
from core.calibration import CalibrationManager
from core.math_utils import wrap, skew, quat_normalize, get_rpy_swing_safe
from estimation.mechanization import SavageMechanization
from estimation.ESKF_6_state import ESKF_6_State
from GUI import gui_phase3_trajectory as gui_module
from estimation.serial_KF_noise import IMUPreprocessor
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from scipy.spatial.transform import Rotation

# ============================================================
# CONFIG
# ============================================================

config = load_config("phase_3")
calib_manager = CalibrationManager()

def main():
    eskf = ESKF_6_State()
    savage = SavageMechanization(g=config.get("gravity", 9.80665))
    imu_preproc = IMUPreprocessor(calib_manager, config, calibrate=config.get("calibrate_imu", True))

    p_hist = []
    bias_hist = deque(maxlen=300)
    t_hist = deque(maxlen=300)
    vel_hist = deque(maxlen=300)
    bx_hist = deque(maxlen=300)
    by_hist = deque(maxlen=300)
    bz_hist = deque(maxlen=300)

    start_time = time.time()
    prev_packet_id = None
    prev_imu_ts = None
    initialized = False

    calibration_done = not config.get("calibrate", True)
    calib_start_time = time.time()
    calib_samples = 0
    acc_bias_zupt = np.zeros(3)
    gyro_bias_zupt = np.zeros(3)

    # ZUPT windows
    acc_norm_window = deque(maxlen=10)
    gyro_norm_window = deque(maxlen=10)

    # Adaptive ZUPT/ZARU (MDR) noise-floor state
    acc_floor = config.get("zupt_acc_var_thresh", 0.005)
    gyro_floor = config.get("zupt_gyro_var_thresh", 0.0005)
    was_stationary = False

    # Adaptive ZDU state
    stationary_since = None
    stationary_anchor_pos = None

    # Debounce state for config.get("zupt_min_stationary_sec", 0.1)
    zupt_candidate_since = None
    zupt_confirmed = False

    # ---- Diagnostic CSV logging ----
    log_file = None
    log_rows_since_flush = 0
    if config.get("log_enabled", False):
        Path(config.get("log_dir", "logs")).mkdir(parents=True, exist_ok=True)
        log_path = Path(config.get("log_dir", "logs")) / f"run_{int(time.time())}.csv"
        log_file = open(log_path, "w")
        log_file.write(
            "t,acc_var,gyro_var,mdr_acc,mdr_gyro,is_stationary,fully_static,"
            "zupt_confirmed,vel_x,vel_y,vel_z,pos_x,pos_y,pos_z\n"
        )
        print(f"[LOG] Writing ZUPT/trajectory diagnostics to {log_path}")

    gui = gui_module.setup_gui()
    plt.show(block=False)
    last_plot_time = 0.0

    network = UDPReceiver(port=config.get("udp_port", 5005), packet_format=config.get("packet_format", "<II6fI3f"))
    network.start()

    try:
        while True:
            if not plt.fignum_exists(gui["fig"].number):
                break
            data = network.get_latest_data()

            if data is None:
                plt.pause(config.get("ui_pause", 0.001))
                time.sleep(0.001)
                continue

            (packet_id,
             gyro_ts, ax_, ay_, az_,
             gx, gy, gz,
             mag_ts, mx, my, mz) = data
            imu_ts = gyro_ts

            if packet_id == prev_packet_id:
                gui_module.keep_windows_responsive()
                time.sleep(0.001)
                continue
            prev_packet_id = packet_id

            if prev_imu_ts is None:
                prev_imu_ts = imu_ts
                continue

            dt = (imu_ts - prev_imu_ts) / 1000.0
            prev_imu_ts = imu_ts
            if dt <= 0 or dt > config.get("max_dt", 0.2):
                continue

            acc_raw = np.array([ax_, ay_, az_])
            gyro_raw = np.array([gx, gy, gz])
            mag_raw = np.array([mx, my, mz])

            acc_cal, gyro_cal = imu_preproc.process(acc_raw, gyro_raw)
            mag_cal = mag_raw
            if config.get("calibrate_imu", True):
                mag_cal = calib_manager.MAG_AINV @ (mag_raw - calib_manager.MAG_BIAS)

            if not calibration_done:
                elapsed = time.time() - calib_start_time
                if elapsed < config.get("calibration_time", 10.0):
                    acc_bias_zupt += acc_cal
                    gyro_bias_zupt += gyro_cal
                    calib_samples += 1
                    print(f"Calibrating... {elapsed:.2f} / {config.get("calibration_time", 10.0):.2f} sec", end="\r")
                    gui_module.keep_windows_responsive()
                    continue
                if calib_samples > 0:
                    acc_bias_zupt /= calib_samples
                    gyro_bias_zupt /= calib_samples
                    acc_bias_zupt[2] -= 1.0   # remove gravity
                calibration_done = True
                print("\nCalibration complete.")

            acc_cal = acc_cal - acc_bias_zupt
            gyro_cal = gyro_cal - gyro_bias_zupt

            gyro_rad = gyro_cal * np.pi / 180.0

            if not initialized:
                eskf.initialize(acc_cal, mag_cal)
                initialized = True
                continue

            # ---- ZUPT / ZARU / MARU detection ----
            # Computed BEFORE eskf.update() so the stationarity verdict can
            # be handed INTO eskf.update() for MARU -- the magnetometer yaw
            # fusion needs to know "are we at rest?" in the same call that
            # fuses it, not after the fact.
            acc_norm = np.linalg.norm(acc_cal)
            gyro_norm = np.linalg.norm(gyro_rad)
            acc_norm_window.append(acc_norm)
            gyro_norm_window.append(gyro_norm)

            is_stationary = False
            fully_static = False
            acc_var = gyro_var = mdr_acc = mdr_gyro = float("nan")
            if len(acc_norm_window) == acc_norm_window.maxlen:
                acc_var = np.var(acc_norm_window)
                gyro_var = np.var(gyro_norm_window)

                if config.get("adaptive_zupt_enabled", True):
                    # --- Adaptive threshold via Motion Detection Ratio ---
                    mdr_acc = acc_var / max(acc_floor, (config.get("steady_acc_var_thresh", 0.005) * 0.1))
                    mdr_gyro = gyro_var / max(gyro_floor, (config.get("steady_gyro_var_thresh", 0.0005) * 0.1))

                    acc_ratio_thresh = (config.get("mdr_acc_exit_ratio", 2.5) if was_stationary
                                        else config.get("mdr_acc_enter_ratio", 1.6))
                    gyro_ratio_thresh = (config.get("mdr_gyro_exit_ratio", 5.0) if was_stationary
                                         else config.get("mdr_gyro_enter_ratio", 3.0))

                    no_translation = mdr_acc < acc_ratio_thresh
                    fully_static = no_translation and mdr_gyro < gyro_ratio_thresh

                    # Only refine the noise floor while confidently static
                    if fully_static:
                        acc_floor = ((1.0 - config.get("floor_ema_alpha", 0.01)) * acc_floor
                                     + config.get("floor_ema_alpha", 0.01) * acc_var)
                        gyro_floor = ((1.0 - config.get("floor_ema_alpha", 0.01)) * gyro_floor
                                      + config.get("floor_ema_alpha", 0.01) * gyro_var)
                else:
                    # Legacy fixed-threshold gating
                    no_translation = acc_var < config.get("zupt_acc_var_thresh", 0.005)
                    fully_static = no_translation and gyro_var < config.get("zupt_gyro_var_thresh", 0.0005)

                is_stationary = no_translation
                was_stationary = is_stationary

            # ---- Debounce before trusting a "stationary" verdict ----
            if is_stationary:
                if zupt_candidate_since is None:
                    zupt_candidate_since = time.time()
                zupt_confirmed = (time.time() - zupt_candidate_since) >= config.get("zupt_min_stationary_sec", 0.1)
            else:
                zupt_candidate_since = None
                zupt_confirmed = False

            # Attitude update with MARU
            eskf.update(
                gyro_rad, acc_cal, mag_cal, dt,
                stationary=(fully_static and config.get("maru_enabled", True)),
                maru_r_scale=config.get("maru_r_yaw_scale", 0.05),
            )

            # ZARU: Zero Angular Rate Update
            if fully_static:
                eskf.update_zaru(gyro_rad, config.get("zaru_sigma", 0.01) ** 2)

            # Velocity/position integration with in-mechanization ZUPT
            pos, vel = savage.update(
                dt, gyro_rad, acc_cal, eskf.q.as_matrix(), zupt=zupt_confirmed
            )

            # ---- Adaptive ZDU (Zero Distance/Displacement Update) ----
            if config.get("zdu_enabled", True):
                if zupt_confirmed:
                    if stationary_since is None:
                        stationary_since = time.time()
                        stationary_anchor_pos = savage.pos.copy()
                    hold_time = time.time() - stationary_since
                    if hold_time >= config.get("zdu_min_hold_sec", 0.3):
                        ramp = min(1.0, (hold_time - config.get("zdu_min_hold_sec", 0.3)) / config.get("zdu_ramp_sec", 1.0))
                        gain = config.get("zdu_max_gain", 0.15) * ramp
                        savage.pos = (savage.pos
                                      + gain * (stationary_anchor_pos - savage.pos))
                        pos = savage.pos
                else:
                    stationary_since = None
                    stationary_anchor_pos = None

            p_hist.append(pos.copy())
            bias_hist.append(eskf.b.copy())

            if log_file is not None:
                log_file.write(
                    f"{time.time() - start_time:.4f},{acc_var},{gyro_var},"
                    f"{mdr_acc},{mdr_gyro},{int(is_stationary)},{int(fully_static)},"
                    f"{int(zupt_confirmed)},{vel[0]},{vel[1]},{vel[2]},"
                    f"{pos[0]},{pos[1]},{pos[2]}\n"
                )
                log_rows_since_flush += 1
                if log_rows_since_flush >= config.get("log_flush_every", 100):
                    log_file.flush()
                    log_rows_since_flush = 0

            rpy_e = get_rpy_swing_safe(eskf.q.as_matrix())

            now = time.time()
            if now - last_plot_time >= config.get("plot_interval", 0.05):
                roll_deg, pitch_deg, yaw_deg = np.rad2deg(rpy_e)

                # Update left panel: RPY text, horizon (damped/anchored
                # display only -- see update_yaw_display_damped docstring)
                gui_module.update_yaw_display_damped(gui, roll_deg, pitch_deg, yaw_deg)

                # Compass: true heading relative to Earth's magnetic poles,
                # always live/undamped regardless of pure-roll or high-pitch
                # state. Uses gui_module.true_heading_deg() which adds yaw_initial back
                # to the ESKF's internal yaw state (see docstring for the
                # full mathematical derivation).
                gui_module.update_compass(gui, gui_module.true_heading_deg(eskf, rpy_e[2]))

                # Update map
                gui_module.update_map(gui, pos, p_hist, rpy_e[2])

                # Update timeseries (right panel)
                elapsed_time = now - start_time
                t_hist.append(elapsed_time)
                bx_hist.append(np.rad2deg(eskf.b[0]))
                by_hist.append(np.rad2deg(eskf.b[1]))
                bz_hist.append(np.rad2deg(eskf.b[2]))
                vel_hist.append(np.linalg.norm(vel))
                gui_module.update_timeseries(gui, t_hist, bx_hist, by_hist, bz_hist, vel_hist)

                # Update table
                gui_module.update_table(gui, pos, vel)

                gui["fig"].canvas.draw_idle()
                last_plot_time = now

            gui_module.keep_windows_responsive()

    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    finally:
        if log_file is not None:
            log_file.flush()
            log_file.close()
        print("\nGenerating final trajectory plot...")
        plt.ioff()
        plt.close('all')
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111)
        ax.set_title("FINAL TRAJECTORY")
        ax.set_xlabel("X (m, East/local)")
        ax.set_ylabel("Y (m, North/local)")
        ax.grid(True)
        ax.set_aspect('equal')
        if len(p_hist) > 0:
            px = [p[0] for p in p_hist]
            py = [p[1] for p in p_hist]
            ax.plot(px, py, 'r-', lw=2, label="Sensor Trajectory")
            ax.legend()
        plt.show(block=True)

if __name__ == "__main__":
    main()
