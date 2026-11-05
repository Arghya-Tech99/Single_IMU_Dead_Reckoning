"""
NASA Coding Guidelines Level Architecture: ESKF Trajectory
12-state Error-State Kalman Filter for tracking trajectory.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from config.config import ACCEL_BIAS, ACCEL_AINV, FLOOR_ACC_MIN, G, SIGMA_ACCEL_VRW_SCALE, ZUPT_SIGMA
from utils.math_utils import skew, quat_normalize, get_rpy_swing_safe, wrap

class ESKF_Trajectory:
    kOneG = 1.0
    _PURE_ROLL_ENTRY_ROLL  = 12.0
    _PURE_ROLL_ENTRY_YAW   =  9.0
    _PURE_ROLL_EXIT_ROLL   =  8.0
    _PURE_ROLL_EXIT_YAW    =  9.0
    _COUPLED_ENTRY_ROLL    =  9.0
    _COUPLED_ENTRY_YAW     =  9.0
    _COUPLED_EXIT_ROLL     =  7.0
    _COUPLED_EXIT_YAW      =  7.0

    N = 12
    
    def __init__(self):
        self.sigma_gyro = 1e-3
        self.sigma_bias = 1e-6
        self.sigma_accel_vrw = 1e-3
        self.sigma_accel_bias = 1e-6
        self.R_acc  = 0.01
        self.R_yaw  = 0.002
        self.accel_gate = 0.3
        self.mag_gate   = 1.5
        self.mag_reject_recovery_count = 30
        self._mag_consecutive_rejects  = 0
        self._gate_state = "default"
        self.q = Rotation.identity()
        self.b = np.zeros(3)
        self.accel_b = np.copy(ACCEL_BIAS)
        self.P = np.zeros((self.N, self.N))
        self.P[0:3, 0:3]   = np.eye(3) * 0.0076
        self.P[3:6, 3:6]   = np.eye(3) * 1e-6
        self.P[6:9, 6:9]   = np.eye(3) * 1e-2
        self.P[9:12, 9:12] = np.eye(3) * 1e-6
        self.yaw_initial = 0.0

    def set_accel_noise(self, acc_var_g2: float) -> None:
        var_ms2 = max(acc_var_g2, FLOOR_ACC_MIN) * (G ** 2)
        self.sigma_accel_vrw = var_ms2 * SIGMA_ACCEL_VRW_SCALE

    def _evaluate_gate(self, roll_deg: float, yaw_deg: float) -> str:
        ar, ay = abs(roll_deg), abs(yaw_deg)
        if self._gate_state == "pure_roll":
            if ar < self._PURE_ROLL_EXIT_ROLL or ay >= self._PURE_ROLL_EXIT_YAW:
                if ar >= self._COUPLED_ENTRY_ROLL and ay >= self._COUPLED_ENTRY_YAW:
                    self._gate_state = "coupled"
                else:
                    self._gate_state = "default"
        elif self._gate_state == "coupled":
            if ar < self._COUPLED_EXIT_ROLL and ay < self._COUPLED_EXIT_YAW:
                self._gate_state = "default"
        else:
            if ar >= self._PURE_ROLL_ENTRY_ROLL and ay < self._PURE_ROLL_ENTRY_YAW:
                self._gate_state = "pure_roll"
            elif ar >= self._COUPLED_ENTRY_ROLL and ay >= self._COUPLED_ENTRY_YAW:
                self._gate_state = "coupled"
        return self._gate_state

    def initialize(self, ab: np.ndarray, mb: np.ndarray) -> bool:
        phi   = np.arctan2(ab[1], -ab[2])
        theta = np.arctan2(ab[0], np.sqrt(ab[1]**2 + ab[2]**2))
        self.q = (Rotation.from_rotvec([0, 0, 0]) *
                  Rotation.from_rotvec([0, theta, 0]) *
                  Rotation.from_rotvec([phi, 0, 0]))
        self.q = quat_normalize(self.q)
        qx, qy, qz, qw = self.q.as_quat()
        tw = np.sqrt(qw**2 + qz**2)
        if tw > 1e-6:
            q_yaw = Rotation.from_quat([0, 0, qz / tw, qw / tw])
            m_ti  = (q_yaw.inv() * self.q).as_matrix() @ mb
            self.yaw_initial = (np.arctan2(-m_ti[1], m_ti[0]) -
                                2.0 * np.arctan2(qz / tw, qw / tw))
        else:
            self.yaw_initial = 0.0
        self.b = np.zeros(3)
        self.accel_b = np.copy(ACCEL_BIAS)
        self.P = np.zeros((self.N, self.N))
        self.P[0:3, 0:3]   = np.eye(3) * 0.0076
        self.P[3:6, 3:6]   = np.eye(3) * 1e-4
        self.P[6:9, 6:9]   = np.eye(3) * 1e-2
        self.P[9:12, 9:12] = np.eye(3) * 1e-4
        self._gate_state  = "default"
        self._mag_consecutive_rejects = 0
        return True

    def predict(self, gyro_rad: np.ndarray, dt: float, acc_g: np.ndarray = None, R_b2n: np.ndarray = None):
        w = gyro_rad - self.b
        angle = np.linalg.norm(w) * dt
        if angle > 1e-10:
            self.q = self.q * Rotation.from_rotvec((w / np.linalg.norm(w)) * angle)
        else:
            self.q = self.q * Rotation.from_quat(
                [w[0]*dt*0.5, w[1]*dt*0.5, w[2]*dt*0.5, 1.0])
        self.q = quat_normalize(self.q)

        F = np.eye(self.N)
        F[0:3, 0:3] = np.eye(3) - skew(w) * dt
        F[0:3, 3:6] = -np.eye(3) * dt
        if acc_g is not None and R_b2n is not None:
            f_body = acc_g * G
            F[6:9, 0:3] = -R_b2n @ skew(f_body) * dt
            F[6:9, 9:12] = -R_b2n @ ACCEL_AINV * G * dt

        Q = np.zeros((self.N, self.N))
        Q[0:3, 0:3]   = np.eye(3) * (self.sigma_gyro * dt)
        Q[3:6, 3:6]   = np.eye(3) * (self.sigma_bias * dt)
        Q[6:9, 6:9]   = np.eye(3) * (self.sigma_accel_vrw * dt)
        Q[9:12, 9:12] = np.eye(3) * (self.sigma_accel_bias * dt)

        self.P = F @ self.P @ F.T + Q

    def _apply_correction(self, dx: np.ndarray, K: np.ndarray, H: np.ndarray, R_eff):
        dtheta = dx[0:3]
        self.q = self.q * Rotation.from_quat(
            [dtheta[0]*0.5, dtheta[1]*0.5, dtheta[2]*0.5, 1.0])
        self.q = quat_normalize(self.q)
        self.b = np.clip(self.b + dx[3:6], -0.02, 0.02)
        dv = dx[6:9]
        self.accel_b += dx[9:12]
        IKH = np.eye(self.N) - K @ H if K.ndim == 2 else np.eye(self.N) - np.outer(K, H)
        if K.ndim == 2:
            self.P = IKH @ self.P @ IKH.T + K @ R_eff @ K.T
        else:
            self.P = IKH @ self.P @ IKH.T + np.outer(K, K) * R_eff
        return dv

    def update_accel(self, ab: np.ndarray, steady_flight: bool = True):
        if not steady_flight: return
        if abs(np.linalg.norm(ab) - self.kOneG) > self.accel_gate: return
        bRw = self.q.inv().as_matrix()
        Ra  = (1.0 + 20.0 * abs(bRw[0, 2])) * self.R_acc
        g_w = np.array([0.0, 0.0, -self.kOneG])
        a_pred = bRw @ g_w
        r = ab - a_pred
        H = np.zeros((3, self.N))
        H[0:3, 0:3] = skew(a_pred)
        H[0:3, 9:12] = ACCEL_AINV
        S = H @ self.P @ H.T + np.eye(3) * Ra
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        K = self.P @ H.T @ S_inv
        dx = K @ r
        self._apply_correction(dx, K, H, np.eye(3) * Ra)

    def update_mag(self, mb: np.ndarray, r_scale: float = 1.0):
        R_b2n = self.q.as_matrix()
        m_nav = R_b2n @ mb
        h_norm_sq = m_nav[0]**2 + m_nav[1]**2
        if h_norm_sq < 1e-6:
            return
        r_psi = wrap(np.arctan2(-m_nav[1], m_nav[0]) - self.yaw_initial)
        H_body = R_b2n.T @ np.array([0.0, 0.0, 1.0])
        H = np.zeros(self.N)
        H[0:3] = H_body
        R_eff = max(1e-4, self.R_yaw * r_scale)
        S = float(H @ self.P @ H.T) + R_eff
        if S < 1e-10:
            return
        K = (self.P @ H) / S
        dx = K * r_psi
        self._apply_correction(dx, K, H, R_eff)

    def update_zaru(self, gyro_rad: np.ndarray, R_zaru: float):
        z = gyro_rad - self.b
        H = np.zeros((3, self.N))
        H[0:3, 3:6] = np.eye(3)
        R = np.eye(3) * R_zaru
        S = H @ self.P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        K = self.P @ H.T @ S_inv
        dx = K @ z
        self._apply_correction(dx, K, H, R)

    def update_zupt(self, vel_nav: np.ndarray, R_zupt: float = ZUPT_SIGMA ** 2):
        H = np.zeros((3, self.N))
        H[0:3, 6:9] = np.eye(3)
        R = np.eye(3) * R_zupt
        innovation = vel_nav
        S = H @ self.P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return np.zeros(3)
        K = self.P @ H.T @ S_inv
        dx = K @ innovation
        dv = self._apply_correction(dx, K, H, R)
        return dv

    def update(self, gyro_rad, acc, mag, dt, steady_flight=False, maru_r_scale=1.0):
        R_b2n_prev = self.q.as_matrix()
        self.predict(gyro_rad, dt, acc_g=acc, R_b2n=R_b2n_prev)
        self.update_accel(acc, steady_flight)
        rpy      = get_rpy_swing_safe(self.q.as_matrix())
        roll_deg = np.rad2deg(rpy[0])
        yaw_deg  = np.rad2deg(rpy[2])
        gate = self._evaluate_gate(roll_deg, yaw_deg)
        r_scale = maru_r_scale if steady_flight else 1.0
        if gate != "pure_roll":
            self.update_mag(mag, r_scale=r_scale)
        return self.q
