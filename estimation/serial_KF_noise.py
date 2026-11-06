"""
serial_KF_noise.py
Serial Kalman Filter for IMU noise reduction, and IMUPreprocessor.
Ported exactly from Pedestrian_Dead_Reckoning.py.
Calibration values are loaded from CalibrationManager (sensor_calibration.json).
"""

import numpy as np


class SerialKF_NoiseReduction:
    """1-D scalar Kalman filter for sensor noise smoothing."""
    def __init__(self, Q=1.0, R=100.0):
        self.Q = Q
        self.R = R
        self.x = None
        self.p = 1.0

    def set_R(self, R):
        self.R = max(R, 1e-12)

    def filter(self, measurement):
        if self.x is None:
            self.x = measurement
            return self.x
        x_pred = self.x
        p_pred = self.p + self.Q
        k      = p_pred / (p_pred + self.R)
        self.x = x_pred + k * (measurement - x_pred)
        self.p = (1.0 - k) * p_pred
        return self.x


class IMUPreprocessor:
    """
    IMU noise pre-filter using per-axis SerialKF_NoiseReduction filters.
    Calibration (ACCEL_BIAS, ACCEL_AINV, GYRO_BIAS, R_imu) is pulled
    from a CalibrationManager instance so no values are ever hardcoded.
    """
    def __init__(self, calib_manager, config, calibrate=True):
        self.calib_manager = calib_manager
        self.config        = config
        self.calibrate     = calibrate

        kf_q_acc  = config.get('kf_q_acc',  1e-4)
        kf_q_gyro = config.get('kf_q_gyro', 1e-5)
        self.steady_acc_var_thresh  = config.get('steady_acc_var_thresh',  0.005)
        self.steady_gyro_var_thresh = config.get('steady_gyro_var_thresh', 0.0005)
        self.floor_acc_min  = self.steady_acc_var_thresh  * 0.1
        self.floor_gyro_min = self.steady_gyro_var_thresh * 0.1

        self.kf_acc = [
            SerialKF_NoiseReduction(Q=kf_q_acc,  R=self.steady_acc_var_thresh)
            for _ in range(3)
        ]
        self.kf_gyro = [
            SerialKF_NoiseReduction(Q=kf_q_gyro, R=self.steady_gyro_var_thresh)
            for _ in range(3)
        ]

    def set_accel_R(self, r):
        r = max(r, self.floor_acc_min)
        for kf in self.kf_acc:
            kf.set_R(r)

    def set_gyro_R(self, r):
        r = max(r, self.floor_gyro_min)
        for kf in self.kf_gyro:
            kf.set_R(r)

    def process(self, acc_raw, gyro_raw, dynamic_accel_bias=None):
        """
        Apply calibration then filter each axis.
        dynamic_accel_bias: if supplied, overrides the static ACCEL_BIAS
                            (used after ESKF accel bias estimation).
        """
        cm = self.calib_manager

        if self.calibrate:
            bias = dynamic_accel_bias if dynamic_accel_bias is not None else cm.ACCEL_BIAS
            acc_raw  = cm.ACCEL_AINV @ (acc_raw - bias)
            gyro_raw = gyro_raw - cm.GYRO_BIAS

        acc_filtered  = np.array([self.kf_acc[i].filter(acc_raw[i])   for i in range(3)])
        gyro_filtered = np.array([self.kf_gyro[i].filter(gyro_raw[i]) for i in range(3)])

        return cm.R_imu @ acc_filtered, cm.R_imu @ gyro_filtered
