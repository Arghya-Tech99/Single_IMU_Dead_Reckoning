"""
NASA Coding Guidelines Level Architecture: IMU Preprocessor
Handles noise reduction and calibration for incoming sensor data.
"""

import numpy as np
from config.config import (
    KF_Q_ACC, KF_Q_GYRO, STEADY_ACC_VAR_THRESH, STEADY_GYRO_VAR_THRESH,
    FLOOR_ACC_MIN, FLOOR_GYRO_MIN, ACCEL_BIAS, ACCEL_AINV, GYRO_BIAS, R_imu
)

class SerialKF_NoiseReduction:
    """Serial Kalman Filter for noise reduction."""
    
    def __init__(self, Q=1.0, R=100.0):
        self.Q = Q
        self.R = R
        self.x = None
        self.p = 1.0

    def set_R(self, R: float) -> None:
        """Dynamically update measurement noise variance."""
        self.R = max(R, 1e-12)

    def filter(self, measurement: float) -> float:
        """Applies 1D Kalman filter to the measurement."""
        if self.x is None:
            self.x = measurement
            return self.x
        x_pred = self.x
        p_pred = self.p + self.Q
        k = p_pred / (p_pred + self.R)
        self.x = x_pred + k * (measurement - x_pred)
        self.p = (1.0 - k) * p_pred
        return self.x

class IMUPreprocessor:
    """Pre-processes and calibrates raw IMU data."""
    
    def __init__(self, calibrate=False):
        self.calibrate = calibrate
        self.kf_acc = [
            SerialKF_NoiseReduction(Q=KF_Q_ACC, R=STEADY_ACC_VAR_THRESH)
            for _ in range(3)
        ]
        self.kf_gyro = [
            SerialKF_NoiseReduction(Q=KF_Q_GYRO, R=STEADY_GYRO_VAR_THRESH)
            for _ in range(3)
        ]

    def set_accel_R(self, r: float) -> None:
        r = max(r, FLOOR_ACC_MIN)
        for kf in self.kf_acc:
            kf.set_R(r)

    def set_gyro_R(self, r: float) -> None:
        r = max(r, FLOOR_GYRO_MIN)
        for kf in self.kf_gyro:
            kf.set_R(r)

    def process(self, acc_raw: np.ndarray, gyro_raw: np.ndarray, dynamic_accel_bias: np.ndarray = None):
        """Applies calibration and noise filtering to IMU readings."""
        if dynamic_accel_bias is None:
            dynamic_accel_bias = ACCEL_BIAS

        if self.calibrate:
            acc_raw = ACCEL_AINV @ (acc_raw - dynamic_accel_bias)
            gyro_raw = gyro_raw - GYRO_BIAS

        acc_filtered = np.array([self.kf_acc[i].filter(acc_raw[i]) for i in range(3)])
        gyro_filtered = np.array([self.kf_gyro[i].filter(gyro_raw[i]) for i in range(3)])

        return R_imu @ acc_filtered, R_imu @ gyro_filtered
