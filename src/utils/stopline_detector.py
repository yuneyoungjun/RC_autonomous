# stopline_detector.py (다시 이 파일로 저장해줘!)

import cv2
import numpy as np

class StoplineDetector:
    def __init__(self):
        self.stopline_detected_flag = False # 정지선 감지 여부 플래그
        
        # 정지선 검출에 필요한 상수 정의
        self.ROI_Y_START = 300
        self.ROI_Y_END = 480
        self.ROI_X_START = 0
        self.ROI_X_END = 640
        self.STOPLINE_CHECK_Y_START = 100
        self.STOPLINE_CHECK_Y_END = 120
        self.STOPLINE_CHECK_X_START = 200
        self.STOPLINE_CHECK_X_END = 440
        self.STOPLINE_WHITE_THRESHOLD = 180 # HSV V채널 하한값
        self.STOPLINE_COUNT_THRESHOLD = 2500 # 흰색 픽셀 개수 기준치
        self.GREEN_COLOR = (0, 255, 0) # 녹색 (BGR)

        print("Stopline Detector Initialized.")

    def get_control_values(self, image):
        """
        주어진 이미지에서 정지선을 검출하고, 그 결과에 따라 목표 angle과 speed를 반환합니다.
        한 번 정지선이 감지되면, 해당 플래그는 True로 유지됩니다.
        
        Args:
            image (np.array): 카메라 이미지 (BGR 포맷).
            current_angle (float): 차선 감지 등 다른 로직에서 계산된 현재 조향각.
            
        Returns:
            tuple: (target_angle, target_speed, stopline_detected_persistent_flag)
        """
        if image is None or image.size == 0:
            print("Warning: Input image is empty or None in get_control_values.")
            # 이미지가 유효하지 않으면, 현재 플래그 상태를 유지하며 기본값 반환
            return 0, 0, self.stopline_detected_flag 

        # --- 정지선 감지 로직 ---
        # 이 로직은 self.stopline_detected_flag가 아직 False일 때만 실행됩니다.
        # 한 번 True가 되면 이 부분은 더 이상 self.stopline_detected_flag를 변경하지 않습니다.
        if not self.stopline_detected_flag: # 💡 이 조건이 핵심! 한 번 감지되면 이 로직은 더 이상 플래그를 False로 만들지 않아.
            # 1. ROI 설정 (관심 영역)
            roi_img = image[self.ROI_Y_START:self.ROI_Y_END, self.ROI_X_START:self.ROI_X_END]

            # 2. HSV 변환 및 이진화
            hsv_image = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV) 
            lower_white = np.array([0, 0, self.STOPLINE_WHITE_THRESHOLD])
            upper_white = np.array([255, 255, 255])
            binary_img = cv2.inRange(hsv_image, lower_white, upper_white)

            # 3. 정지선 체크 영역 추출
            stopline_check_img = binary_img[self.STOPLINE_CHECK_Y_START:self.STOPLINE_CHECK_Y_END, 
                                            self.STOPLINE_CHECK_X_START:self.STOPLINE_CHECK_X_END]
            
            # 4. 흰색 픽셀 개수 카운트
            stopline_count = cv2.countNonZero(stopline_check_img)
            
            # 5. 정지선 감지 여부 업데이트 (클래스 내부 플래그)
            if stopline_count > self.STOPLINE_COUNT_THRESHOLD:
                print("Stopline Detected! Initiating stop sequence (and will remain detected).")
                self.stopline_detected_flag = True
            # else: 정지선이 감지되지 않으면 플래그는 여전히 False.
            # 이 로직은 self.stopline_detected_flag가 True가 된 후에는 다시 False로 만들지 않습니다.
        # --- 정지선 감지 로직 끝 ---

        # 💡 정지선 감지 여부에 따라 최종 angle과 speed 결정
        if self.stopline_detected_flag:
            # 정지선이 감지되면 즉시 정지
            target_angle = 0 # 정지 시에는 조향각도 0으로
            target_speed = 0 # 완전히 정지
            print(f"Target: Angle={target_angle}, Speed={target_speed} (Stopline Detected)")
        else:
            # 정지선이 감지되지 않으면 기존 angle 유지, speed 15 고정
            target_angle = 0 # 차선 감지 등 다른 로직에서 계산된 angle을 그대로 사용
            target_speed = 15 # 고정 속도
            #print(f"Target: Angle={target_angle}, Speed={target_speed} (Driving)")
            
        # 💡 최종 반환 값에 self.stopline_detected_flag 추가!
        return target_angle, target_speed, self.stopline_detected_flag
