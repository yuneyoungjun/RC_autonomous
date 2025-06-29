# utils/sonic_driver.py

import rospy
from std_msgs.msg import Int32MultiArray, Bool
from utils.ultrasonic_polar_plotter import UltrasonicPolarPlotter
from utils.filters import FilterWithQueue
from utils.pid_controller import PID  # 기존 구현한 PID 클래스
from utils.parameters import *

class SonicDriver:
    def __init__(self):
        self.ultra_msg = None
        self.april_tag_detected = False
        self.polar_plotter = UltrasonicPolarPlotter()
        # 필터 인스턴스 생성
        self.left_filter = FilterWithQueue(queue_len=5, alpha=0.2, threshold=30)
        self.right_filter = FilterWithQueue(queue_len=5, alpha=0.2, threshold=30)
        self.steering_pid = PID(SONIC_PID_KP, SONIC_PID_KI, SONIC_PID_KD)
        rospy.on_shutdown(self.shutdown_hook)
        rospy.Subscriber("xycar_ultrasonic", Int32MultiArray, self.ultra_callback)
        rospy.Subscriber("april_tag_detected", Bool, self.april_tag_callback)

        rospy.wait_for_message("xycar_ultrasonic", Int32MultiArray)
        print("Ultrasonic Corner Driver Ready ----------")


    def april_tag_callback(self, data):
        self.april_tag_detected = data.data
        if self.april_tag_detected:
            rospy.loginfo("🎯 AprilTag 비콘 감지됨!")
        else:
            rospy.loginfo("AprilTag 비콘 미감지...")

    def ultra_callback(self, data):
        raw_left = data.data[1]
        raw_right = data.data[3]

        self.filtered_left, left_outlier = self.left_filter.update(raw_left)
        self.filtered_right, right_outlier = self.right_filter.update(raw_right)

        if left_outlier:
            rospy.logwarn(f"[Left] 이상치 감지됨: {raw_left}")
        if right_outlier:
            rospy.logwarn(f"[Right] 이상치 감지됨: {raw_right}")

        current_ultra_values = list(data.data)
        current_ultra_values[1] = int(self.filtered_left)
        current_ultra_values[3] = int(self.filtered_right)
        self.ultra_msg = current_ultra_values

        self.polar_plotter.update_plot(self.ultra_msg)

    def get_drive_values(self):
        if self.ultra_msg is None:
            return 0, 0, False

        left = self.filtered_left
        right = self.filtered_right

        # 장애물 판단
        direction = 0  # 0: 없음, 1: 좌측 회피, 2: 우측 회피
        target_distance = 45

        if 0 < left < target_distance and left < right:
            direction = 1
            cte = target_distance - left
        elif 0 < right < target_distance and left > right:
            direction = 2
            cte = target_distance - right
        else:
            return 0, 30, self.april_tag_detected  # 기본 전진

        # PID 계산
        steering_correction = self.steering_pid.pid_control(cte)
        speed_correction = cte

        # 방향에 따라 조정
        if direction == 1:
            angle = +steering_correction
        else:
            angle = -steering_correction

        speed = max(10, min(30, 100 - abs(speed_correction)))

        return angle, speed, self.april_tag_detected

    def shutdown_hook(self):
        self.polar_plotter.close_plot()
        print("SonicDriver 노드 종료됨. Plot 창 닫기 완료.")
