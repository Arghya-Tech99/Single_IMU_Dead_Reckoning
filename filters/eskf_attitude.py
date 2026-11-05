"""
NASA Coding Guidelines Level Architecture: ESKF Attitude
6-state Error-State Kalman Filter for tracking absolute magnetic heading.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from utils.math_utils import skew, quat_normalize, get_rpy_swing_safe, wrap

class ESKF_Attitude:
    kOneG = 1.0

    _PURE_ROLL_ENTRY_ROLL  = 12.0
    _PURE_ROLL_ENTRY_YAW   =  9.0
    _PURE_ROLL_EXIT_ROLL   =  8.0
    _PURE_ROLL_EXIT_YAW    =  9.0
    _COUPLED_ENTRY_ROLL    =  9.0
    _COUPLED_ENTRY_YAW     =  9.0
    _COUPLED_EXIT_ROLL     =  7.0
    _COUPLED_EXIT_YAW      =  7.0

    N = 6

    def __init__(self):
        self.sigma_gyro = 5e-3
        self.sigma_bias = 1e-4
        self.R_acc  = 0.05
        self.R_yaw  = 0.004
        self.accel_gate = 0.15
        self.mag_gate   = 1.5
        self.mag_reject_recovery_count = 30
        self._mag_consecutive_rejects  = 0
        self._gate_state = "default"

        self.q = Rotation.identity()
        self.b = np.zeros(3)
        self.P = np.zeros((self.N, self.N))
        self.P[0:3, 0:3] = np.eye(3) * 0.0076
        self.P[3:6, 3:6] = np.eye(3) * 1e-4
        self.yaw_initial = 0.0

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
        phi   = np.arctan2(ab[1], ab[2])
        theta = np.arctan2(-ab[0], np.sqrt(ab[1]**2 + ab[2]**2))
        self.q = (Rotation.from_rotvec([0, 0, 0]) *
                  Rotation.from_rotvec([0, theta, 0]) *
                  Rotation.from_rotvec([phi, 0, 0]))
        self.q = quat_normalize(self.q)

        qx, qy, qz, qw = self.q.as_quat()
        tw = np.sqrt(qw**2 + qz**2)
        if tw > 1e-6:
            q_yaw = Rotation.from_quat([0, 0, qz / tw, qw / tw])
            m_ti  = (q_yaw.inv() * self.q).as_matrix() @ mb
            self.yaw_initial = (-np.arctan2(m_ti[1], m_ti[0]) -
                                2.0 * np.arctan2(qz / tw, qw / tw))
        else:
            self.yaw_initial = 0.0

        self.b = np.zeros(3)
        self.P = np.zeros((self.N, self.N))
        self.P[0:3, 0:3] = np.eye(3) * 0.0076
        self.P[3:6, 3:6] = np.eye(3) * 1e-4
        self._gate_state  = "default"
        self._mag_consecutive_rejects = 0
        return True

    def predict(self, gyro_rad: np.ndarray, dt: float) -> None:
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

        Q = np.zeros((self.N, self.N))
        Q[0:3, 0:3] = np.eye(3) * (self.sigma_gyro * dt)
        Q[3:6, 3:6] = np.eye(3) * (self.sigma_bias * dt)

        self.P = F @ self.P @ F.T + Q

    def _apply_correction(self, dx: np.ndarray, K: np.ndarray, H: np.ndarray, R_eff: float) -> None:
        dtheta = dx[0:3]
        self.q = self.q * Rotation.from_quat(
            [dtheta[0]*0.5, dtheta[1]*0.5, dtheta[2]*0.5, 1.0])
        self.q = quat_normalize(self.q)
        self.b += dx[3:6]

        if K.ndim == 2:
            IKH = np.eye(self.N) - K @ H
            self.P = IKH @ self.P @ IKH.T + K @ R_eff @ K.T
        else:
            IKH = np.eye(self.N) - np.outer(K, H)
            self.P = IKH @ self.P @ IKH.T + np.outer(K, K) * R_eff

    def update_accel(self, ab: np.ndarray) -> None:
        if abs(np.linalg.norm(ab) - self.kOneG) > self.accel_gate:
            return
        bRw = self.q.inv().as_matrix()
        Ra  = (1.0 + 20.0 * abs(bRw[0, 2])) * self.R_acc
        g_w = np.array([0.0, 0.0, self.kOneG])
        a_pred = bRw @ g_w
        r = ab - a_pred

        H = np.zeros((3, self.N))
        H[0:3, 0:3] = skew(a_pred)

        S = H @ self.P @ H.T + np.eye(3) * Ra
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        K = self.P @ H.T @ S_inv
        dx = K @ r
        self._apply_correction(dx, K, H, np.eye(3) * Ra)

    def update_mag(self, mb: np.ndarray) -> None:
        qx, qy, qz, qw = self.q.as_quat()
        tw = np.sqrt(qw**2 + qz**2)
        if tw < 1e-6:
            return
        tw2 = tw * tw

        q_yaw = Rotation.from_quat([0, 0, qz / tw, qw / tw])
        m_tilt = (q_yaw.inv() * self.q).as_matrix() @ mb

        h = m_tilt[0]**2 + m_tilt[1]**2
        if h < 1e-6:
            return

        mag_norm_sq = mb @ mb
        if mag_norm_sq < 1e-12:
            return
        obs_ratio = h / mag_norm_sq
        if obs_ratio < 0.1:
            return

        bz_world = 2.0 * tw2 - 1.0
        if bz_world < 0.12:
            return

        psi_mag = -np.arctan2(m_tilt[1], m_tilt[0])
        psi_nom = 2.0 * np.arctan2(qz / tw, qw / tw)
        r_psi = wrap(psi_mag - psi_nom)

        if abs(r_psi) > self.mag_gate:
            self._mag_consecutive_rejects += 1
            if self._mag_consecutive_rejects < self.mag_reject_recovery_count:
                return
            self._mag_consecutive_rejects = 0
            R_eff = self.R_yaw * 50.0
        else:
            self._mag_consecutive_rejects = 0
            R_eff = self.R_yaw / (obs_ratio * bz_world)

        H_psi = np.zeros(self.N)
        H_psi[0] = (qx * qz - qw * qy) / tw2
        H_psi[1] = (qy * qz + qw * qx) / tw2
        H_psi[2] = 1.0

        S = float(H_psi @ self.P @ H_psi.T) + R_eff
        if S < 1e-10:
            return
        K = (self.P @ H_psi) / S
        dx = K * r_psi
        self._apply_correction(dx, K, H_psi, R_eff)

    def update(self, gyro_rad: np.ndarray, acc: np.ndarray, mag: np.ndarray, dt: float) -> Rotation:
        self.predict(gyro_rad, dt)
        self.update_accel(acc)

        rpy = get_rpy_swing_safe(self.q.as_matrix())
        roll_deg = np.rad2deg(rpy[0])
        yaw_deg  = np.rad2deg(rpy[2])
        gate = self._evaluate_gate(roll_deg, yaw_deg)

        if gate != "pure_roll":
            self.update_mag(mag)
        return self.q
