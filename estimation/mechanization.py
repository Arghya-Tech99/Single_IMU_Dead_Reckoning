"""
mechanization.py
Savage 2-speed mechanization for strapdown INS.
Ported exactly from Pedestrian_Dead_Reckoning.py.
"""

import numpy as np

G = 9.80665


class SavageMechanization:
    """
    Savage 2-speed strapdown mechanization.
    Exactly as implemented in Pedestrian_Dead_Reckoning.py.
    """
    def __init__(self, g=G):
        self.g = g
        self.g_vec = np.array([0.0, 0.0, self.g])
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.d_theta_prev = np.zeros(3)
        self.d_vel_prev = np.zeros(3)

    def update(self, dt, gyro_rad_s, acc_g, R_b2n, zupt=False):
        d_theta = gyro_rad_s * dt
        f_body  = acc_g * self.g
        d_vel   = f_body * dt
        term1 = 0.5 * np.cross(d_theta, d_vel)
        term2 = (1.0 / 12.0) * (np.cross(self.d_theta_prev, d_vel) + np.cross(self.d_vel_prev, d_theta))
        d_vel_scull = d_vel + term1 + term2
        d_pos_rot_body = -(dt / 12.0) * np.cross(d_theta, d_vel)
        self.d_theta_prev = d_theta.copy()
        self.d_vel_prev   = d_vel.copy()
        d_vel_nav     = R_b2n @ d_vel_scull
        d_pos_rot_nav = R_b2n @ d_pos_rot_body
        vel_prev = self.vel.copy()
        vel_new  = self.vel + d_vel_nav + self.g_vec * dt

        if zupt:
            vel_new = np.zeros(3)
            d_pos_rot_nav = np.zeros(3)

        self.vel = vel_new
        self.pos = self.pos + 0.5 * (vel_prev + self.vel) * dt + d_pos_rot_nav
        return self.pos, self.vel

    def apply_velocity_correction(self, dv):
        self.vel = self.vel - dv
        return self.vel
