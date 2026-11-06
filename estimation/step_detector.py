import numpy as np
import time
from collections import deque

class StepDetector:
    def __init__(self, thresh_high=1.12,
                 thresh_low=0.90,
                 min_interval=0.30,
                 cadence_window=10,
                 timeout=3.0):
        self.thresh_high  = thresh_high
        self.thresh_low   = thresh_low
        self.min_interval = min_interval
        self.timeout      = timeout
        self.step_count    = 0
        self.last_step_time = 0.0
        self.state         = "below"
        self.step_times    = deque(maxlen=cadence_window)
        self.cadence_spm   = 0.0

    def update(self, acc_norm, current_time):
        step_detected = False
        if self.state == "below":
            if acc_norm > self.thresh_high:
                self.state = "above"
        elif self.state == "above":
            if acc_norm < self.thresh_low:
                self.state = "below"
                if current_time - self.last_step_time >= self.min_interval:
                    self.step_count += 1
                    self.step_times.append(current_time)
                    self.last_step_time = current_time
                    step_detected = True
                    if len(self.step_times) >= 2:
                        intervals = [self.step_times[i] - self.step_times[i - 1]
                                     for i in range(1, len(self.step_times))]
                        avg_interval = np.mean(intervals)
                        if avg_interval > 0.05:
                            self.cadence_spm = 60.0 / avg_interval
        if current_time - self.last_step_time > self.timeout and self.last_step_time > 0:
            self.cadence_spm = 0.0
        return step_detected
    
# Activity detector

class ActivityDetector:
    def __init__(self, hold_sec=1.0):
        self.hold_sec         = hold_sec
        self.stationary_start = None
        self.is_idle          = False

    def update(self, is_stationary, current_time):
        if is_stationary:
            if self.stationary_start is None:
                self.stationary_start = current_time
            self.is_idle = (current_time - self.stationary_start >= self.hold_sec)
        else:
            self.stationary_start = None
            self.is_idle = False
        return self.is_idle

# GUI setup

