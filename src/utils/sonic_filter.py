# utils/sonic_driver.py

import rospy
from std_msgs.msg import Int32MultiArray, Bool
from ultrasonic_polar_plotter import UltrasonicPolarPlotter
from filters import FilterWithQueue

class SonicDriver:
    def __init__(self):
        self.ultra_msg = None
        self.april_tag_detected = False

        rospy.Subscriber("xycar_ultrasonic", Int32MultiArray, self.ultra_callback)
        rospy.Subscriber("april_tag_detected", Bool, self.april_tag_callback)

        rospy.wait_for_message("xycar_ultrasonic", Int32MultiArray)
        print("Ultrasonic Corner Driver Ready ----------")

        # 필터 인스턴스 생성
        self.left_filter = FilterWithQueue(queue_len=5, alpha=0.2, threshold=30)
        self.right_filter = FilterWithQueue(queue_len=5, alpha=0.2, threshold=30)

        self.polar_plotter = UltrasonicPolarPlotter()
        rospy.on_shutdown(self.shutdown_hook)

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

        ok = 0
        if 0 < left < 45 and left < right:
            ok = 1
        elif 0 < right < 45 and left > right:
            ok = 2

        transition_to_next_stage = self.april_tag_detected

        if ok == 1:
            speed = min(left, 20)
            angle = 100 - 1.0 * left
            return angle, speed, transition_to_next_stage
        elif ok == 2:
            speed = min(right, 20)
            angle = 100 - 1.0 * right
            return -angle, speed, transition_to_next_stage
        else:
            return 0, 30, transition_to_next_stage

    def shutdown_hook(self):
        self.polar_plotter.close_plot()
        print("SonicDriver 노드 종료됨. Plot 창 닫기 완료.")
