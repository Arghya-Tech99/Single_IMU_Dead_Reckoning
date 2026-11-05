"""
NASA Coding Guidelines Level Architecture: Configuration
This file contains all constants and static parameters, adhering to Rule 3 
(no dynamic allocation after initialization) by pre-allocating constants.
"""

import struct
import numpy as np

# Network Constants
UDP_IP = "0.0.0.0"
UDP_PORT = 5005
PACKET_FORMAT = "<II6fI3f"
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)

# Physical Constants
MAX_DT = 0.2
G = 9.80665

# ZUPT thresholds
ZUPT_ACC_VAR_THRESH = 0.005
ZUPT_GYRO_VAR_THRESH = 0.0005
STEADY_ACC_VAR_THRESH  = 0.01
STEADY_GYRO_VAR_THRESH = 0.001
ADAPTIVE_ZUPT_ENABLED = True

FLOOR_EMA_ALPHA      = 0.01
MDR_ACC_ENTER_RATIO  = 2.0
MDR_ACC_EXIT_RATIO   = 2.5
MDR_GYRO_ENTER_RATIO = 4.0
MDR_GYRO_EXIT_RATIO  = 5.0
STEADY_MIN_SEC       = 0.10
FLOOR_ACC_MIN        = STEADY_ACC_VAR_THRESH * 0.1
FLOOR_GYRO_MIN       = STEADY_GYRO_VAR_THRESH * 0.1

MARU_ENABLED     = True
MARU_R_YAW_SCALE = 0.05
ZARU_SIGMA       = 0.01
ZUPT_ENABLED     = True
ZUPT_SIGMA       = 0.02

ADAPTIVE_KF_R_ENABLED = True
KF_Q_ACC  = 1e-4
KF_Q_GYRO = 1e-5

SIGMA_ACCEL_VRW_SCALE = 1.0
CALIBRATE_IMU = True

CALIBRATE = True
CALIBRATION_TIME = 10.0

MAG_BIAS = np.array([0.24675210, -0.14440901, 0.90685429])
MAG_AINV = np.array([
    [1.01095845, 0.00293556, -0.00765120],
    [0.00293556, 0.99673875, -0.01952619],
    [-0.00765120, -0.01952619, 0.99842304],
])

ACCEL_BIAS = np.array([-0.00356944, 0.00173197, 0.00804703])
ACCEL_AINV = np.array([
    [ 1.00089818e+00, -2.00093885e-03, -9.68772054e-04],
    [-2.00093885e-03,  1.00126044e+00,  1.50347363e-04],
    [-9.68772054e-04,  1.50347363e-04,  1.00254840e+00]
])

GYRO_BIAS = np.array([0.0, 0.0, 0.0])

R_imu = np.array([
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
])

ZDU_ENABLED = True
ZDU_MIN_HOLD_SEC = 0.3
ZDU_RAMP_SEC = 1.0
ZDU_MAX_GAIN = 0.15

STEP_THRESH_HIGH   = 1.12
STEP_THRESH_LOW    = 0.90
STEP_MIN_INTERVAL  = 0.30
STEP_CADENCE_WINDOW = 10
STEP_TIMEOUT_SEC   = 3.0

IDLE_HOLD_SEC = 1.0

MAG_THRESH_LOW_UT  = 50.0
MAG_THRESH_MID_UT  = 55.0
MAG_THRESH_HIGH_UT = 60.0

FIG_SIZE = (16, 9)
COLOR_ESKF = "#1f77b4"
TS_HISTORY_MAXLEN = 300
TS_LINE_COLORS = ["#ca1a0e", "#38ec50", "#3274d8", "#ff8c00"]
BOLD_LW = 2.4
BOLD_LINE_LW = 1.5

TL_RED_ON     = "#ff2222"
TL_RED_DIM    = "#661111"
TL_YELLOW_ON  = "#ffcc00"
TL_YELLOW_DIM = "#665500"
TL_GREEN_ON   = "#22ff44"
TL_GREEN_DIM  = "#116622"
TL_OFF        = "#1a1a1a"

DISPLAY_MIRROR_X = -1.0
UI_PAUSE = 0.001

PITCH_HIGH_DEG = 75.0
PITCH_MID_DEG  = 15.0
PURE_ROLL_HOLD_ROLL_DEG  = 45.0   
PURE_ROLL_HOLD_PITCH_DEG = 20.0   
PURE_ROLL_HOLD_SECONDS   = 3.0    
PURE_ROLL_YAW_DAMPING_RATE = 0.3  
