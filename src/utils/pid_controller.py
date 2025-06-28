# utils/pid_controller.py
import numpy as np

class PID:
    def __init__(self, kp, ki, kd):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.cte_prev = None
        self.p_error = 0.0
        self.i_error = 0.0
        self.d_error = 0.0
        self.i_min = -10
        self.i_max = 10

    def pid_control(self, cte, dt=1.0):
        if self.cte_prev is None:
            self.cte_prev = cte

        self.d_error = (cte - self.cte_prev) / dt
        self.p_error = cte
        self.i_error += cte * dt
        self.i_error = max(min(self.i_error, self.i_max), self.i_min)
        self.cte_prev = cte

        return self.Kp * self.p_error + self.Ki * self.i_error + self.Kd * self.d_error
