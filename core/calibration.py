"""
calibration.py
Loads and applies sensor calibration values from sensor_calibration.json.
Exposes both structured and flat-key accessors so all runners can use a
consistent interface.
"""

import json
import os
import numpy as np


class CalibrationManager:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), '..', 'config', 'sensor_calibration.json'))
        self.config_path = config_path
        self.calib_data  = self._load_calibration()

        # ── Flat-key accessors (used directly in ESKF / runners) ──────────
        d = self.calib_data
        self.ACCEL_BIAS = np.array(d['accelerometer']['bias'])
        self.ACCEL_AINV = np.array(d['accelerometer']['scale_factor'])
        self.GYRO_BIAS  = np.array(d['gyroscope']['bias'])
        self.MAG_BIAS   = np.array(d['magnetometer']['bias'])
        self.MAG_AINV   = np.array(d['magnetometer']['scale_factor'])
        self.R_imu      = np.array(d.get('imu_rotation',
                            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))

    def _load_calibration(self):
        if not os.path.exists(self.config_path):
            print(f"Calibration file not found at {self.config_path}, using defaults.")
            return self._default_calibration()
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading calibration: {e}. Using defaults.")
            return self._default_calibration()

    def _default_calibration(self):
        return {
            "accelerometer": {
                "bias":         [0.0, 0.0, 0.0],
                "scale_factor": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            },
            "gyroscope": {
                "bias":         [0.0, 0.0, 0.0],
                "scale_factor": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            },
            "magnetometer": {
                "bias":         [0.0, 0.0, 0.0],
                "scale_factor": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            },
            "imu_rotation": [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        }

    def save_calibration(self, new_data):
        self.calib_data = new_data
        with open(self.config_path, 'w') as f:
            json.dump(self.calib_data, f, indent=4)

    # ── Convenience calibration methods ───────────────────────────────────
    def apply_accel_calibration(self, acc_raw, calibrate=True):
        if not calibrate:
            return acc_raw.copy()
        return self.ACCEL_AINV @ (acc_raw - self.ACCEL_BIAS)

    def apply_gyro_calibration(self, gyro_raw_deg_s, calibrate=True):
        if not calibrate:
            return gyro_raw_deg_s.copy() * np.pi / 180.0
        return (gyro_raw_deg_s - self.GYRO_BIAS) * np.pi / 180.0

    def apply_mag_calibration(self, mag_raw, calibrate=True):
        if not calibrate:
            return mag_raw.copy()
        return self.MAG_AINV @ (mag_raw - self.MAG_BIAS)

    def preprocess_sensor_data(self, acc_raw, gyro_raw_deg_s, mag_raw, calibrate=True):
        acc_cal  = self.apply_accel_calibration(acc_raw, calibrate)
        gyro_cal = self.apply_gyro_calibration(gyro_raw_deg_s, calibrate)   # rad/s
        mag_cal  = self.apply_mag_calibration(mag_raw, calibrate)
        return self.R_imu @ acc_cal, self.R_imu @ gyro_cal, mag_cal
