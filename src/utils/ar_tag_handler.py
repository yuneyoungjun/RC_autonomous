import numpy as np
import cv2
import apriltag  # 설치 필요: pip install apriltag

class ARTagHandler:
    def __init__(self):
        self.detector = apriltag.Detector()
        self.tag_size = 9.5  # cm (태그 실제 크기)
        self.camera_matrix = np.array([
            [371.42821, 0., 310.49805],
            [0., 372.60371, 235.74201],
            [0., 0., 1.]
        ])

    def detect_ar_tags(self, image):
        """
        AprilTag를 탐지하여 거리(z), 좌우 위치(x), ID 값을 반환.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)

        tags = []
        for detection in detections:
            center = detection.center
            left = abs(detection.corners[0][1] - detection.corners[3][1])
            right = abs(detection.corners[1][1] - detection.corners[2][1])
            tag_size_px = (left + right) / 2.0

            # 거리 추정
            focal_length = self.camera_matrix[0, 0]
            distance = (focal_length * self.tag_size) / tag_size_px * 100 / 126
            z_cm = distance * 1.1494 - 14.94

            # 좌우 오차 추정
            x_offset_px = center[0] - 320  # assuming 640px width
            x_cm = (x_offset_px * self.tag_size) / tag_size_px

            tags.append({
                "id": detection.tag_id,
                "x": round(x_cm, 1),
                "z": round(z_cm, 1)
            })

        return tags

    def get_drive_command(self, image):
        """
        이미지에서 ARTag 탐지 후 조향각, 속도, 종료 여부 반환
        - return: (angle, speed, parking_complete_flag)
        """
        tags = self.detect_ar_tags(image)
        if not tags:
            return 0, 0, False  # 탐지 실패 시

        # 가장 가까운 태그 선택
        closest = min(tags, key=lambda t: t["z"])
        x, z = closest["x"], closest["z"]

        if z > 30:
            angle = int(x * 1.0)
            speed = 12
            return angle, speed, False
        else:
            return 0, 0, True
