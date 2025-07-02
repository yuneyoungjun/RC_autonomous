#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================================
#
# FILE: hull_cluster_driver_ghost_test.py
#
# AUTHOR: Gemini AI
#
# DESCRIPTION:
#   ['유령 차선' 로직 추가]
#   코너링 시 한쪽 차선이 시야에서 사라지는 경우, 보이는 차선을 기준으로
#   반대편에 가상의 '유령 차선'을 생성하여 경로를 유지합니다.
#   이를 통해 인코스/아웃코스 주행 시 안정성을 크게 향상시킵니다.
#
# CHANGELOG (User Request):
#   - 한쪽 차선만 인식될 경우를 처리하는 '유령 차선' 로직 추가
#   - LANE_WIDTH 상수 추가 및 튜닝 가이드
#   - [MODIFIED]: Removed rospy.init_node, rospy.wait_for_message, and `run` method
#                 to integrate into an external main loop.
#   - [MODIFIED]: `vision_drive` now takes `frame` as an explicit argument.
#   - [MODIFIED]: Changed lane detection color from RED to WHITE.
#   - [MODIFIED]: Converted to a standalone script for testing with a single image file.
# =================================================================================

# rospy 대신 print()를 사용하며, ROS 관련 의존성을 주석 처리합니다.
# cv_bridge, xycar_motor, sensor_msgs 등 ROS 메시지 타입은 사용하지 않습니다.

import cv2
import numpy as np
# from sensor_msgs.msg import Image # ROS 의존성
# from cv_bridge import CvBridge   # ROS 의존성
# from xycar_msgs.msg import xycar_motor # ROS 의존성
# import rospy # ROS 의존성

# scikit-learn은 그대로 사용
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from collections import defaultdict

# rospy.loginfo 대신 일반 print 함수를 사용하기 위한 래퍼
class SimpleLogger:
    def info(self, msg):
        print(f"[INFO] {msg}")
    def warn(self, msg):
        print(f"[WARN] {msg}")
    def logerr(self, msg):
        print(f"[ERROR] {msg}")

# 글로벌 로거 인스턴스 (rospy.loginfo 대체)
rospy = SimpleLogger()


class HullClusterDriver:
    def __init__(self):
        # self.bridge = CvBridge() # ROS 의존성 제거
        self.image = np.empty(shape=[0]) 
        self.last_cx = 320
        self.is_paused = False 
        self.angle = 0
        self.speed = 0 
        self.last_error = 0
        
        # ? [핵심] 차선 폭(픽셀 단위) 변수. 주행 환경에 맞게 튜닝 필요
        self.LANE_WIDTH = 400 

        self.CLUSTER_COLORS = [
            (0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
            (0, 128, 255), (255, 128, 0)
        ]
        self.CENTER_LINE_COLOR = (255, 0, 0) # 파란색 (BGR)

        self.last_successful_slope = 0.0
        self.last_successful_intercept = 0.0
        self.is_path_detected_recently = False
        
        # self.motor_pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1) # ROS 의존성 제거
        
        rospy.info("RANSAC Hull Cluster Driver (Ghost Lane) Initialized for testing.")

    def get_lane_center(self, frame_for_processing):
        self.image = frame_for_processing.copy() 

        h, w = self.image.shape[:2]
        roi_top_y = int(h * 0.6) # 이미지 상단 60%부터 하단까지 ROI 설정

        mask = np.zeros((h, w), dtype=np.uint8)
        roi_vertices = np.array([[(0, roi_top_y), (w, roi_top_y), (w, h), (0, h)]], dtype=np.int32)
        cv2.fillPoly(mask, roi_vertices, 255) # ROI 영역을 마스크로 채움
        
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # ==================================================================
        # ✅ 흰색 차선 인식을 위한 HSV 임계값 설정 (튜닝 필요)
        # H (Hue): 색상 (0-179)
        # S (Saturation): 채도 (0-255) - 흰색은 채도가 낮음
        # V (Value): 명도 (0-255) - 흰색은 명도가 높음
        # ==================================================================
        lower_white = np.array([0, 0, 200])   # 낮은 채도, 높은 명도로 흰색 영역 정의
        upper_white = np.array([179, 30, 255]) # 높은 채도는 제외하여 회색빛까지 포함 가능
        
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        # ==================================================================
        
        final_mask = cv2.bitwise_and(white_mask, white_mask, mask=mask) # ROI와 흰색 마스크 결합

        pixel_coords = np.argwhere(final_mask > 0) # 마스크에서 0이 아닌 픽셀의 좌표를 가져옴
        
        # 픽셀이 너무 많으면 샘플링하여 처리 속도 향상
        if len(pixel_coords) > 500:
            indices = np.random.choice(len(pixel_coords), 500, replace=False)
            sampled_coords = pixel_coords[indices]
        else:
            sampled_coords = pixel_coords

        if len(sampled_coords) < 10: # 충분한 픽셀이 없으면 경로 감지 실패
            self.is_path_detected_recently = False
        else:
            X = sampled_coords[:, ::-1] # (row, col) -> (x, y) 형태로 변환
            
            # 이전 성공적인 선을 기준으로 좌우 차선 픽셀 분리
            dividing_line_x = self.last_successful_slope * X[:, 1] + self.last_successful_intercept
            left_points = X[X[:, 0] < dividing_line_x]
            right_points = X[X[:, 0] >= dividing_line_x]

            left_lane_cluster, right_lane_cluster = None, None

            # 왼쪽 차선 클러스터링 (DBSCAN)
            if len(left_points) > 10:
                db_left = DBSCAN(eps=80, min_samples=10).fit(left_points)
                unique_labels, counts = np.unique(db_left.labels_[db_left.labels_ != -1], return_counts=True)
                if len(counts) > 0:
                    left_lane_cluster = left_points[db_left.labels_ == unique_labels[np.argmax(counts)]]

            # 오른쪽 차선 클러스터링 (DBSCAN)
            if len(right_points) > 10:
                db_right = DBSCAN(eps=80, min_samples=10).fit(right_points)
                unique_labels, counts = np.unique(db_right.labels_[db_right.labels_ != -1], return_counts=True)
                if len(counts) > 0:
                    right_lane_cluster = right_points[db_right.labels_ == unique_labels[np.argmax(counts)]]

            # 클러스터링된 차선 볼록 껍질 시각화
            if left_lane_cluster is not None and len(left_lane_cluster) >= 3:
                hull = cv2.convexHull(left_lane_cluster); cv2.polylines(self.image, [hull], True, self.CLUSTER_COLORS[0], 2)
            if right_lane_cluster is not None and len(right_lane_cluster) >= 3:
                hull = cv2.convexHull(right_lane_cluster); cv2.polylines(self.image, [hull], True, self.CLUSTER_COLORS[1], 2)

            center_path_points = []
            
            # ? [핵심] 차선 감지 상태에 따른 분기 처리 (유령 차선 로직)
            # 1. 양쪽 차선이 모두 보일 때 (기존 로직)
            if left_lane_cluster is not None and right_lane_cluster is not None:
                left_y_to_x, right_y_to_x = defaultdict(list), defaultdict(list)
                for x, y in left_lane_cluster: left_y_to_x[y].append(x)
                for x, y in right_lane_cluster: right_y_to_x[y].append(x)
                common_y = set(left_y_to_x.keys()) & set(right_y_to_x.keys()) # 공통 Y 좌표
                for y in sorted(list(common_y)):
                    # 양쪽 차선의 X 좌표 중간 지점을 중심 경로로 사용
                    center_path_points.append((int((max(left_y_to_x[y]) + min(right_y_to_x[y])) / 2), y))
            
            # 2. 왼쪽 차선만 보일 때 -> 오른쪽 유령 차선 생성
            elif left_lane_cluster is not None and right_lane_cluster is None:
                rospy.info("Ghost Lane: 왼쪽 차선만 감지됨. 오른쪽 유령 차선 생성.")
                left_y_to_x = defaultdict(list)
                for x, y in left_lane_cluster: left_y_to_x[y].append(x)
                for y, x_coords in sorted(left_y_to_x.items()):
                    # 왼쪽 차선의 최대 X 값에서 차선 폭의 절반만큼 더한 지점을 중심 경로로 사용
                    center_x = max(x_coords) + self.LANE_WIDTH // 2
                    center_path_points.append((center_x, y))
                    # 선택 사항: 유령 차선 시각화 (주황색)
                    ghost_x = max(x_coords) + self.LANE_WIDTH
                    cv2.circle(self.image, (int(ghost_x), y), 3, (255, 100, 0), -1) 

            # 3. 오른쪽 차선만 보일 때 -> 왼쪽 유령 차선 생성 (인코스 주행 시 핵심)
            elif left_lane_cluster is None and right_lane_cluster is not None:
                rospy.info("Ghost Lane: 오른쪽 차선만 감지됨. 왼쪽 유령 차선 생성.")
                right_y_to_x = defaultdict(list)
                for x, y in right_lane_cluster: right_y_to_x[y].append(x)
                for y, x_coords in sorted(right_y_to_x.items()):
                    # 오른쪽 차선의 최소 X 값에서 차선 폭의 절반만큼 뺀 지점을 중심 경로로 사용
                    center_x = min(x_coords) - self.LANE_WIDTH // 2
                    center_path_points.append((center_x, y))
                    # 선택 사항: 유령 차선 시각화 (주황색)
                    ghost_x = min(x_coords) - self.LANE_WIDTH
                    cv2.circle(self.image, (int(ghost_x), y), 3, (255, 100, 0), -1) 

            # RANSAC을 이용한 중심 경로선 추정
            if len(center_path_points) > 10:
                y_vals = np.array([p[1] for p in center_path_points]).reshape(-1, 1)
                x_vals = np.array([p[0] for p in center_path_points])
                ransac = RANSACRegressor()
                try:
                    ransac.fit(y_vals, x_vals)
                    alpha = 0.6 # EMA(Exponential Moving Average) 계수
                    current_slope = ransac.estimator_.coef_[0]
                    current_intercept = ransac.estimator_.intercept_
                    # EMA를 적용하여 선 추정 값 안정화
                    self.last_successful_slope = alpha * current_slope + (1 - alpha) * self.last_successful_slope
                    self.last_successful_intercept = alpha * current_intercept + (1 - alpha) * self.last_successful_intercept
                    self.is_path_detected_recently = True
                except ValueError: 
                    self.is_path_detected_recently = False
            else:
                self.is_path_detected_recently = False

        # 추정된 중심 경로선 그리기
        y1 = roi_top_y
        x1 = int(self.last_successful_slope * y1 + self.last_successful_intercept)
        y2 = h - 1
        x2 = int(self.last_successful_slope * y2 + self.last_successful_intercept)
        cv2.line(self.image, (x1, y1), (x2, y2), self.CENTER_LINE_COLOR, 3)

        # 로봇이 따라갈 최종 목표 X 좌표 계산 (특정 높이에서)
        y_horizon = int(h * 0.65) # 0.65 비율의 높이에서 목표 X 좌표 계산
        final_target_cx = self.last_successful_slope * y_horizon + self.last_successful_intercept
        self.last_cx = final_target_cx # 시각화 및 디버깅을 위해 저장
        
        return final_target_cx

    def vision_drive(self, frame):
        """
        이미지 프레임을 처리하여 조향 각도와 속도를 결정합니다.
        Returns:
            tuple: (angle, speed) - 모터 제어를 위한 조향 각도와 속도
        """
        if frame.size == 0:
            rospy.warn("HullClusterDriver: vision_drive에 빈 프레임이 수신되었습니다.")
            return self.angle, self.speed # 이전 값 또는 0 반환

        target_cx = self.get_lane_center(frame) # 차선 중심 X 좌표 계산 (시각화도 이 안에서 이루어짐)
        frame_center = frame.shape[1] // 2 # 이미지의 중앙 X 좌표

        error = target_cx - frame_center # 이미지 중앙과 목표 X 좌표 간의 오차
        k_p = 0.4 # 비례 이득
        d_error = error - self.last_error # 미분 오차
        k_d = 0.6 # 미분 이득
        
        self.angle = (k_p * error) + (k_d * d_error) # PD 제어
        self.last_error = error # 다음 루프를 위해 현재 오차 저장
        
        # 조향 각도에 따른 속도 제어 (부드러운 코너링을 위해 튜닝됨)
        if abs(self.angle) < 10: self.speed = 35
        elif abs(self.angle) < 30: self.speed = 20
        else: self.speed = 15
            
        # 조향 각도와 속도 값을 모터 제어 범위로 클램핑
        self.angle = max(min(self.angle, 100.0), -100.0) 
        self.speed = max(min(self.speed, 50), 0) 
        
        return self.angle, self.speed

    def drive(self, angle, speed):
        """
        주어진 각도와 속도 값을 xycar_motor 토픽으로 발행합니다.
        (이 함수는 테스트 스크립트에서는 실제 발행하지 않습니다.)
        """
        # ROS 환경이 아니므로 실제 motor_pub.publish는 동작하지 않습니다.
        # 테스트를 위해 print로 대체
        print(f"Drive command: Angle={angle:.2f}, Speed={speed:.2f}")

    def get_processed_image(self):
        """
        차선과 중심 경로가 그려진 이미지(시각화용)를 반환합니다.
        """
        return self.image

# =================================================================================
# 테스트 실행을 위한 메인 코드 (ROS 환경이 아닌 일반 Python에서 실행)
# =================================================================================

class TestRunner:
    def __init__(self, image_path):
        self.driver = HullClusterDriver()
        self.image_path = image_path
        print(f"테스트 이미지 경로: {self.image_path}")

    def run_test(self):
        # 이미지 로드
        frame = cv2.imread(self.image_path)
        if frame is None:
            print(f"[ERROR] 이미지를 로드할 수 없습니다. 경로를 확인해주세요: {self.image_path}")
            return

        print(f"이미지 로드 성공: {self.image_path}, 크기: {frame.shape}")

        # HullClusterDriver의 vision_drive 메서드 호출
        # 이 메서드 내부에서 get_lane_center를 호출하고 self.image에 시각화 결과를 저장합니다.
        angle, speed = self.driver.vision_drive(frame)
        
        # 처리된 이미지 가져오기 (차선 및 경로가 그려진 이미지)
        processed_image = self.driver.get_processed_image()

        # 결과 출력 (터미널)
        print(f"\n--- 처리 결과 ---")
        print(f"계산된 조향 각도: {angle:.2f}")
        print(f"계산된 속도: {speed:.2f}")
        print(f"경로 감지 성공 여부: {self.driver.is_path_detected_recently}")

        # 이미지 시각화 (새 창에 표시)
        if processed_image.size > 0:
            # 이미지에 조향 각도와 속도 정보 추가 (디버깅 편의)
            cv2.putText(processed_image, f"Angle: {angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(processed_image, f"Speed: {speed:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Processed Lane Image (Press any key to exit)", processed_image)
            cv2.waitKey(0) # 아무 키나 누를 때까지 창 유지
        else:
            print("[ERROR] 처리된 이미지가 비어 있습니다. 시각화 불가.")
        
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # ⚠️ 여기에 테스트할 이미지 파일 경로를 입력하세요.
    # 예시: 'test_image_straight_lane.png', 'test_image_curve_left.jpg', 'test_image_single_lane.jpeg' 등
    # 파일이 없으면 에러가 발생합니다.
    test_image_path = "test_img/track01.jpg" 
    # 예시: test_image_path = "/home/user/Pictures/test_lane.jpg"
    # 또는 코드와 같은 디렉토리에 이미지를 두었다면:
    # test_image_path = "my_lane_test.png" 

    tester = TestRunner(test_image_path)
    tester.run_test()