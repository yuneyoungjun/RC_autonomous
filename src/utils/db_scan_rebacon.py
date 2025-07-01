#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================================
#
# FILE: hull_cluster_driver_ghost.py
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
#                 to integrate into an external main loop.
#   - [MODIFIED]: `vision_drive` now takes `frame` as an explicit argument.
# =================================================================================

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from xycar_msgs.msg import xycar_motor
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from collections import defaultdict

class HullClusterDriver:
    def __init__(self):
        self.bridge = CvBridge()
        # self.image will now be passed directly to vision_drive,
        # no internal subscriber needed if main_controller handles it.
        # However, we keep it for drawing visualization.
        self.image = np.empty(shape=[0]) 
        self.last_cx = 320
        self.is_paused = False # This state will be managed by the main_controller
        self.angle = 0
        self.speed = 0 # Initialize speed
        self.last_error = 0
        
        # ? [핵심] 차선 폭(픽셀 단위) 변수. 주행 환경에 맞게 튜닝 필요
        self.LANE_WIDTH = 400 

        self.CLUSTER_COLORS = [
            (0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
            (0, 128, 255), (255, 128, 0)
        ]
        self.CENTER_LINE_COLOR = (255, 0, 0)

        self.last_successful_slope = 0.0
        self.last_successful_intercept = 0.0
        self.is_path_detected_recently = False

        # Removed rospy.init_node and rospy.wait_for_message
        # as they should be handled by the main controller.
        self.motor_pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
        # Removed internal rospy.Subscriber, image will be passed explicitly.
        
        rospy.loginfo("RANSAC Hull Cluster Driver (Ghost Lane) Initialized.")

    # Removed usbcam_callback as image is now passed directly to vision_drive

    def get_lane_center(self, frame_for_processing):
        # We need to set the internal self.image to the frame being processed
        # so that visualization (cv2.line, cv2.circle) draws on the correct image.
        self.image = frame_for_processing.copy() # Make a copy to avoid modifying the original frame passed in

        h, w = self.image.shape[:2]
        roi_top_y = int(h * 0.6)

        mask = np.zeros((h, w), dtype=np.uint8)
        roi_vertices = np.array([[(0, roi_top_y), (w, roi_top_y), (w, h), (0, h)]], dtype=np.int32)
        cv2.fillPoly(mask, roi_vertices, 255)
        
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 60, 40])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([158, 36, 28])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
        final_mask = cv2.bitwise_and(red_mask, red_mask, mask=mask)

        pixel_coords = np.argwhere(final_mask > 0)
        if len(pixel_coords) > 500:
            indices = np.random.choice(len(pixel_coords), 500, replace=False)
            sampled_coords = pixel_coords[indices]
        else:
            sampled_coords = pixel_coords

        if len(sampled_coords) < 10:
            self.is_path_detected_recently = False
        else:
            X = sampled_coords[:, ::-1]
            # Use last_successful_slope/intercept for initial splitting, 
            # as it provides a stable reference point for left/right differentiation.
            # It's crucial for the ghost lane logic.
            dividing_line_x = self.last_successful_slope * X[:, 1] + self.last_successful_intercept
            left_points = X[X[:, 0] < dividing_line_x]
            right_points = X[X[:, 0] >= dividing_line_x]

            left_lane_cluster, right_lane_cluster = None, None

            if len(left_points) > 10:
                db_left = DBSCAN(eps=80, min_samples=10).fit(left_points)
                unique_labels, counts = np.unique(db_left.labels_[db_left.labels_ != -1], return_counts=True)
                if len(counts) > 0:
                    left_lane_cluster = left_points[db_left.labels_ == unique_labels[np.argmax(counts)]]

            if len(right_points) > 10:
                db_right = DBSCAN(eps=80, min_samples=10).fit(right_points)
                unique_labels, counts = np.unique(db_right.labels_[db_right.labels_ != -1], return_counts=True)
                if len(counts) > 0:
                    right_lane_cluster = right_points[db_right.labels_ == unique_labels[np.argmax(counts)]]

            if left_lane_cluster is not None and len(left_lane_cluster) >= 3:
                hull = cv2.convexHull(left_lane_cluster); cv2.polylines(self.image, [hull], True, self.CLUSTER_COLORS[0], 2)
            if right_lane_cluster is not None and len(right_lane_cluster) >= 3:
                hull = cv2.convexHull(right_lane_cluster); cv2.polylines(self.image, [hull], True, self.CLUSTER_COLORS[1], 2)

            center_path_points = []
            
            # ? [핵심] 차선 감지 상태에 따른 분기 처리
            # 1. 양쪽 차선이 모두 보일 때 (기존 로직)
            if left_lane_cluster is not None and right_lane_cluster is not None:
                left_y_to_x, right_y_to_x = defaultdict(list), defaultdict(list)
                for x, y in left_lane_cluster: left_y_to_x[y].append(x)
                for x, y in right_lane_cluster: right_y_to_x[y].append(x)
                common_y = set(left_y_to_x.keys()) & set(right_y_to_x.keys())
                for y in sorted(list(common_y)):
                    center_path_points.append((int((max(left_y_to_x[y]) + min(right_y_to_x[y])) / 2), y))
            
            # 2. 왼쪽 차선만 보일 때 -> 오른쪽 유령 차선 생성
            elif left_lane_cluster is not None and right_lane_cluster is None:
                rospy.loginfo("Ghost Lane: Left lane only detected, creating right ghost lane.")
                left_y_to_x = defaultdict(list)
                for x, y in left_lane_cluster: left_y_to_x[y].append(x)
                for y, x_coords in sorted(left_y_to_x.items()):
                    center_x = max(x_coords) + self.LANE_WIDTH // 2
                    center_path_points.append((center_x, y))
                    # Optional: Draw the ghost lane for visualization
                    ghost_x = max(x_coords) + self.LANE_WIDTH
                    cv2.circle(self.image, (int(ghost_x), y), 3, (255, 100, 0), -1) # Orange for ghost lane

            # 3. 오른쪽 차선만 보일 때 -> 왼쪽 유령 차선 생성 (인코스 주행 시 핵심)
            elif left_lane_cluster is None and right_lane_cluster is not None:
                rospy.loginfo("Ghost Lane: Right lane only detected, creating left ghost lane.")
                right_y_to_x = defaultdict(list)
                for x, y in right_lane_cluster: right_y_to_x[y].append(x)
                for y, x_coords in sorted(right_y_to_x.items()):
                    center_x = min(x_coords) - self.LANE_WIDTH // 2
                    center_path_points.append((center_x, y))
                    # Optional: Draw the ghost lane for visualization
                    ghost_x = min(x_coords) - self.LANE_WIDTH
                    cv2.circle(self.image, (int(ghost_x), y), 3, (255, 100, 0), -1) # Orange for ghost lane


            if len(center_path_points) > 10:
                y_vals = np.array([p[1] for p in center_path_points]).reshape(-1, 1)
                x_vals = np.array([p[0] for p in center_path_points])
                ransac = RANSACRegressor()
                try:
                    ransac.fit(y_vals, x_vals)
                    alpha = 0.6
                    current_slope = ransac.estimator_.coef_[0]
                    current_intercept = ransac.estimator_.intercept_
                    # Apply exponential moving average to stabilize line prediction
                    self.last_successful_slope = alpha * current_slope + (1 - alpha) * self.last_successful_slope
                    self.last_successful_intercept = alpha * current_intercept + (1 - alpha) * self.last_successful_intercept
                    self.is_path_detected_recently = True
                except ValueError: # RANSAC might fail if not enough inliers
                    self.is_path_detected_recently = False
            else:
                self.is_path_detected_recently = False

        # Draw the predicted center line using the smoothed slope and intercept
        y1 = roi_top_y
        x1 = int(self.last_successful_slope * y1 + self.last_successful_intercept)
        y2 = h - 1
        x2 = int(self.last_successful_slope * y2 + self.last_successful_intercept)
        cv2.line(self.image, (x1, y1), (x2, y2), self.CENTER_LINE_COLOR, 3)

        y_horizon = int(h * 0.65) # A fixed horizon line to calculate the target X
        final_target_cx = self.last_successful_slope * y_horizon + self.last_successful_intercept
        self.last_cx = final_target_cx # Store for visualization or debugging
        
        return final_target_cx

    def vision_drive(self, frame):
        """
        Processes the image frame to determine the steering angle and speed.
        Returns:
            tuple: (angle, speed) for the motor control.
        """
        # Ensure frame is not empty before processing
        if frame.size == 0:
            rospy.logwarn("HullClusterDriver: Received empty frame in vision_drive.")
            return self.angle, self.speed # Return last known values or zeros

        target_cx = self.get_lane_center(frame) # This will also set self.image for visualization
        frame_center = frame.shape[1] // 2

        error = target_cx - frame_center
        k_p = 0.4
        d_error = error - self.last_error
        k_d = 0.6
        
        self.angle = (k_p * error) + (k_d * d_error)
        self.last_error = error
        
        # Speed control based on angle (tuned for smoother turns)
        if abs(self.angle) < 10: self.speed = 35
        elif abs(self.angle) < 30: self.speed = 20
        else: self.speed = 15
            
        self.angle = max(min(self.angle, 100.0), -100.0) # Clamp angle to motor limits
        self.speed = max(min(self.speed, 50), 0) # Clamp speed to motor limits
        
        return self.angle, self.speed

    def drive(self, angle, speed):
        """
        Publishes the given angle and speed to the xycar_motor topic.
        """
        motor_msg = xycar_motor()
        motor_msg.angle = int(angle)
        motor_msg.speed = int(speed)
        self.motor_pub.publish(motor_msg)

    def get_processed_image(self):
        """
        Returns the image with lane lines and center path drawn for visualization.
        """
        return self.image

# The `run` method is removed as the main_controller will manage the loop.
# The `if __name__ == "__main__":` block is also removed as this file is now a module.