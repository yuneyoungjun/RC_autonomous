# utils/lane_follower.py

import math
import numpy as np
from utils.pid import PID
from utils import hough

class LaneFollower:
    def __init__(self, width=640, height=480):
        self.WIDTH = width
        self.HEIGHT = height
        self.View_Center = self.WIDTH // 2

        self.angle_pid = PID(kp=1.1, ki=0.001, kd=0.5)
        self.prev_angle = 0
        self.prev_speed = 20
        self.missing_count = 0
        self.flag = 0

    def get_control(self, image):
        found, x_left, x_right = hough.lane_detect(image)

        if found:
            self.missing_count = 0
            x_midpoint = (x_left + x_right) // 2
            corner_shift = self.prev_angle * 0.5
            raw_angle = (x_midpoint - (self.View_Center + corner_shift)) * 0.7
            new_angle = self.angle_pid.pid_control(raw_angle)

            if abs(new_angle) < 20:
                new_speed = 100
                self.flag = 0
            elif abs(new_angle) < 40:
                self.flag += 1
                if self.flag < 5:
                    new_speed = 0
                elif self.flag < 10:
                    new_speed = -50
                else:
                    new_speed = 60
            else:
                new_speed = 15

            self.prev_angle = new_angle
            self.prev_speed = new_speed

        else:
            self.missing_count += 1
            if self.missing_count < 10:
                new_angle = self.prev_angle
                new_speed = self.prev_speed
            else:
                new_angle = self.prev_angle
                new_speed = 0

        return new_angle, new_speed
