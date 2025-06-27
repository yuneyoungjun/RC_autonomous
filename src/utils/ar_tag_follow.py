import numpy as np
import cv2
import math
import apriltag

class ARTagFollow:
    def __init__(self):
        self.detector = apriltag.Detector()
        self.tag_size = 9.5  # 단위: cm
        self.camera_matrix = np.array([
            [371.42821, 0., 310.49805],
            [0., 372.60371, 235.74201],
            [0., 0., 1.]
        ])
        self.retry_count = 0
        self.fix_speed = 30

    def detect_tags(self, image):
        """
        이미지에서 AR 태그들을 탐지하고 거리 및 좌우 위치를 추정하여 반환.
        """
        ar_msg = {"ID": [], "DX": [], "DZ": []}
        if image is None:
            return ar_msg

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)

        for det in detections:
            center = det.center
            left = abs(det.corners[0][1] - det.corners[3][1])
            right = abs(det.corners[1][1] - det.corners[2][1])
            tag_size_px = (left + right) / 2.0

            focal_length = self.camera_matrix[0, 0]
            distance = (focal_length * self.tag_size) / tag_size_px * 100 / 126
            z_cm = distance * 1.1494 - 14.94

            x_offset_px = center[0] - 320  # assume image width 640
            x_cm = (x_offset_px * self.tag_size) / tag_size_px

            ar_msg["ID"].append(det.tag_id)
            ar_msg["DX"].append(round(x_cm, 1))
            ar_msg["DZ"].append(round(z_cm, 1))

        return ar_msg

    def get_closest_tag(self, ar_data):
        if not ar_data["ID"]:
            return 99, 500, 500

        closest_index = ar_data["DZ"].index(min(ar_data["DZ"]))
        return (
            ar_data["ID"][closest_index],
            ar_data["DZ"][closest_index],
            ar_data["DX"][closest_index]
        )

    def get_drive_command(self, image):
        """
        이미지 입력으로부터 (angle, speed, 도착 여부) 반환
        """
        ar_data = self.detect_tags(image)
        ar_ID, z_pos, x_pos = self.get_closest_tag(ar_data)

        if ar_ID == 99:
            self.retry_count += 1
            if self.retry_count < 5:
                return -20, self.fix_speed, False
            elif self.retry_count < 10:
                return -20, self.fix_speed, False
            else:
                return 0, 0, False  # 정지

        # AR 태그 감지 시 주행 제어
        self.retry_count = 0
        distance = math.sqrt(z_pos ** 2 + x_pos ** 2)

        if distance > 100:
            x_pos += 300 / (distance + 1)
            angle = int(x_pos * 1)
        elif distance > 50:
            x_pos += max(50 / (distance + 1), 5)
            angle = int(x_pos * 2)
        else:
            x_pos += 40
            angle = int(x_pos * 3)
            if distance < 20:
                return 0, 0, True  # 가까우면 도착으로 판단

        return angle, self.fix_speed, False
