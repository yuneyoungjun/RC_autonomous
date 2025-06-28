import rospy
from utils.lane_detector import LaneDetector

class LaneChanger:
    def __init__(self):
        self.detector = LaneDetector()
        self.step = 0
        self.done = False
        self.active = False
        self.direction = None  # 'left' or 'right'

    def start(self, image):
        _, _, yellow_right = self.detector.get_steering_angle(image)
        self.direction = 'left' if yellow_right else 'right'
        self.step = 0
        self.done = False
        self.active = True
        rospy.loginfo(f"[LaneChanger] 차선 변경 시작! 방향: {self.direction.upper()}")

    def is_done(self):
        return self.done

    def is_active(self):
        return self.active and not self.done

    def get_control(self, image):
        angle, speed, yellow_right = self.detector.get_steering_angle(image)

        if not self.active:
            return angle, speed, False  # 활성화되지 않음

        if self.direction == 'left':
            if self.step == 0:
                if yellow_right:
                    rospy.loginfo("[LaneChanger] 왼쪽 변경: 노란선 감지됨 → 이동 시작")
                    angle = -20
                    speed = 25
                    self.step = 1
            elif self.step == 1:
                if not yellow_right:
                    rospy.loginfo("[LaneChanger] 왼쪽 변경: 노란선 사라짐 → 진입 중")
                    angle = -15
                    speed = 25
                    self.step = 2
            elif self.step == 2:
                if not yellow_right:
                    rospy.loginfo("[LaneChanger] 왼쪽 변경 완료!")
                    angle = 0
                    speed = 20
                    self.done = True
                    self.active = False

        elif self.direction == 'right':
            if self.step == 0:
                if not yellow_right:
                    rospy.loginfo("[LaneChanger] 오른쪽 변경: 노란선이 왼쪽 → 이동 시작")
                    angle = 20
                    speed = 25
                    self.step = 1
            elif self.step == 1:
                if yellow_right:
                    rospy.loginfo("[LaneChanger] 오른쪽 변경: 노란선 진입 중")
                    angle = 15
                    speed = 25
                    self.step = 2
            elif self.step == 2:
                if yellow_right:
                    rospy.loginfo("[LaneChanger] 오른쪽 변경 완료!")
                    angle = 0
                    speed = 20
                    self.done = True
                    self.active = False

        return angle, speed, self.done
