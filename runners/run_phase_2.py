
import os
import sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import sys
import os
import time
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config_loader import load_config
from core.network import UDPReceiver
from core.calibration import CalibrationManager
from core.math_utils import get_rpy_swing_safe
from estimation.ESKF_6_state import ESKF_6_State
import GUI.gui_phase2_attitude as gui_module

def main():
    config = load_config('phase_2')
    print("Starting Phase 2: Attitude Characterization")
    print(f"UDP Port: {config.get('udp_port')}")
    print(f"Filter Type: {config.get('filter_type')}")
    
    # Initialize Core Modules
    network = UDPReceiver(ip="0.0.0.0", port=config.get('udp_port', 5005), 
                          packet_format=config.get('packet_format', '<II6fI3f'))
    calibration = CalibrationManager()
    
    eskf = ESKF_6_State()
    
    MAX_POINTS = 300
    PLOT_HZ = 20.0
    PLOT_INTERVAL = 1.0 / PLOT_HZ
    MAX_DT = 0.2
    CALIBRATION_TIME = 10.0
    CALIBRATE = True

    t_hist = deque(maxlen=MAX_POINTS)
    bias_hist = deque(maxlen=MAX_POINTS)

    start_time = time.time()
    prev_packet_id = None
    prev_imu_ts = None
    initialized = False

    calibration_done = not CALIBRATE
    calib_start_time = time.time()
    calib_samples = 0
    acc_bias_zupt = np.zeros(3)
    gyro_bias_zupt = np.zeros(3)

    gui = gui_module.setup_gui()
    plt.show(block=False)
    last_plot_time = 0.0

    network.start()
    
    try:
        while True:
            data = network.get_latest_data()
            if data is None:
                plt.pause(0.001)
                time.sleep(0.001)
                continue

            (packet_id,
             gyro_ts, ax_, ay_, az_,
             gx, gy, gz,
             mag_ts, mx, my, mz) = data
             
            imu_ts = gyro_ts

            if packet_id == prev_packet_id:
                plt.pause(0.001)
                time.sleep(0.001)
                continue
            prev_packet_id = packet_id

            if prev_imu_ts is None:
                prev_imu_ts = imu_ts
                continue

            dt = (imu_ts - prev_imu_ts) / 1000.0
            prev_imu_ts = imu_ts
            if dt <= 0 or dt > MAX_DT:
                continue

            acc_raw = np.array([ax_, ay_, az_])
            gyro_raw = np.array([gx, gy, gz])
            mag_raw = np.array([mx, my, mz])

            acc_cal, gyro_cal, mag_cal = calibration.preprocess_sensor_data(acc_raw, gyro_raw, mag_raw)

            if not calibration_done:
                elapsed = time.time() - calib_start_time
                if elapsed < CALIBRATION_TIME:
                    acc_bias_zupt += acc_cal
                    gyro_bias_zupt += gyro_cal
                    calib_samples += 1
                    print(f"Calibrating... {elapsed:.2f} / {CALIBRATION_TIME:.2f} sec", end="\r")
                    plt.pause(0.001)
                    continue
                if calib_samples > 0:
                    acc_bias_zupt /= calib_samples
                    gyro_bias_zupt /= calib_samples
                    acc_bias_zupt[2] -= 1.0
                calibration_done = True
                print("\nCalibration complete.")

            acc_cal = acc_cal - acc_bias_zupt
            gyro_cal = gyro_cal - gyro_bias_zupt

            gyro_rad = gyro_cal # preprocess_sensor_data already converts to rad/s

            if not initialized:
                eskf.initialize(acc_cal, mag_cal)
                initialized = True
                continue

            eskf.update(gyro_rad, acc_cal, mag_cal, dt)

            rpy_e = get_rpy_swing_safe(eskf.q.as_matrix())

            t_now = time.time() - start_time
            t_hist.append(t_now)
            bias_hist.append(eskf.b.copy())

            gui_module.update_3d(gui, eskf.q.as_matrix())

            now = time.time()
            if now - last_plot_time >= PLOT_INTERVAL:
                gui_module.update_table(gui, rpy_e, eskf.b)
                gui_module.update_timeseries(gui, t_hist, bias_hist)
                last_plot_time = now
                
    except KeyboardInterrupt:
        print("\nStopping Phase 2 Runner.")
    finally:
        network.stop()

if __name__ == '__main__':
    main()
