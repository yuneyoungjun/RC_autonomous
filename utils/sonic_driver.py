# sonic_driver.py
import rospy
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Bool # 💡 AprilTag 감지 여부를 Bool 메시지로 받는다고 가정!
import time
import numpy as np

from utils.ultrasonic_polar_plotter import UltrasonicPolarPlotter 

class SonicDriver:
    def __init__(self):
        self.ultra_msg = None
        self.april_tag_detected = False # 💡 AprilTag 감지 여부 플래그 추가!

        rospy.Subscriber("xycar_ultrasonic", Int32MultiArray, self.ultra_callback)
        rospy.Subscriber("april_tag_detected", Bool, self.april_tag_callback) # 💡 AprilTag 토픽 구독 추가!

        rospy.wait_for_message("xycar_ultrasonic", Int32MultiArray)
        print("Ultrasonic Corner Driver Ready ----------")


        # 2. 칼만 필터 초기화
        self.kf_left = self._initialize_kalman_filter()
        self.kf_right = self._initialize_kalman_filter()

        # 💡 UltrasonicPolarPlotter 객체 생성
        self.polar_plotter = UltrasonicPolarPlotter() 
        
        rospy.on_shutdown(self.shutdown_hook)

    def _initialize_kalman_filter(self):
        A = np.array([[1.0]]) 
        H = np.array([[1.0]])
        Q = np.array([[0.1]]) 
        R = np.array([[10.0]]) 
        P = np.array([[100.0]])
        x = np.array([[0.0]])
        return {'A': A, 'H': H, 'Q': Q, 'R': R, 'P': P, 'x': x}

    def _kalman_filter_update(self, kf_params, measurement):
        # 예측 (Predict)
        kf_params['x'] = kf_params['A'] @ kf_params['x']
        kf_params['P'] = kf_params['A'] @ kf_params['P'] @ kf_params['A'].T + kf_params['Q']

        # 업데이트 (Update)
        y = measurement - kf_params['H'] @ kf_params['x']
        S = kf_params['H'] @ kf_params['P'] @ kf_params['H'].T + kf_params['R']
        K = kf_params['P'] @ kf_params['H'].T @ np.linalg.inv(S)

        kf_params['x'] = kf_params['x'] + K @ y
        kf_params['P'] = (np.eye(kf_params['P'].shape[0]) - K @ kf_params['H']) @ kf_params['P']
        
        return kf_params['x'][0][0]

    def april_tag_callback(self, data):
        # 💡 AprilTag 감지 여부 업데이트!
        self.april_tag_detected = data.data
        if self.april_tag_detected:
            print("AprilTag 비콘 감지됨! 🎯")
        else:
            print("AprilTag 비콘 미감지...")

    def ultra_callback(self, data):
        raw_left = data.data[1]
        raw_right = data.data[3]

        self.filtered_left = self._kalman_filter_update(self.kf_left, raw_left)
        self.filtered_right = self._kalman_filter_update(self.kf_right, raw_right)
        
        current_ultra_values = list(data.data)
        current_ultra_values[1] = int(self.filtered_left)
        current_ultra_values[3] = int(self.filtered_right)
        self.ultra_msg = current_ultra_values

        self.polar_plotter.update_plot(self.ultra_msg)
        

    def get_drive_values(self):
        if self.ultra_msg is None:
            # 💡 초음파 데이터가 아직 없으면 다음 단계로 넘어갈 수 없음
            return 0, 0, False 
        
        # 필터링된 값 사용
        left = self.ultra_msg[1]
        right = self.ultra_msg[3]

        ok = 0
        if 0 < left < 45 and left < right:
            ok = 1
        elif 0 < right < 45 and left > right:
            ok = 2

        # 💡 다음 단계로 넘어갈 조건:
        # 1. 초음파 센서 상 4초 이상 장애물이 없다고 판단되었고 (self.no_obstacle_flag)
        # 2. AprilTag 비콘이 감지되었을 때 (self.april_tag_detected)
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
            return 0, 30, transition_to_next_stage # 장애물 없음 (직진)

    def shutdown_hook(self):
        self.polar_plotter.close_plot()
        print("SonicDriver node shutting down and closing plotter.")

if __name__ == '__main__':
    try:
        driver = SonicDriver()
        rospy.spin()
    except rospy.ROSInterruptException:
        print("SonicDriver Node Shut down.")
        pass
