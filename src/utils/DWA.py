import numpy as np
import cv2
from utils.pid_controller import PID
from utils.parameters import *
import math


class LaneDetector:
    def __init__(self):
        self.prev_l_pos = 0
        self.prev_r_pos = LANE_IMG_WIDTH
        self.prev_l_detected = False
        self.prev_r_detected = False
        self.Width = LANE_IMG_WIDTH
        self.Height = LANE_IMG_HEIGHT

        self.M = cv2.getPerspectiveTransform(LANE_WARP_SRC_POINTS, LANE_WARP_DST_POINTS)
        self.Minv = cv2.getPerspectiveTransform(LANE_WARP_DST_POINTS, LANE_WARP_SRC_POINTS)
        self.pid = PID(kp=LANE_PID_KP, ki=LANE_PID_KI, kd=LANE_PID_KD)
        
        self.prev_angle = 0
        self.outlier_threshold = 60  # 이전 angle과 비교해 100도 이상 튀면 이상치로 간주
        self.outlier_threshold_speed=20
        self.outlier_count = 0
        self.max_outlier_count = 10  # 5회 이상 연속 튀면 그 값을 수용

        # 색상 감지 결과 저장을 위한 변수 (시각화용)
        self.is_left_yellow = False
        self.is_right_yellow = False
        self.is_left_white = False
        self.is_right_white = False
        self.pre_speed=30

    def preprocess(self, img):
        # 1. 가우시안 블러 (노이즈 감소)
        blur = cv2.GaussianBlur(img, LANE_GAUSSIAN_BLUR_KERNEL_SIZE, 0)
        # cv2.imshow('1_GaussianBlur', blur) 

        # 2. HLS 및 LAB 색 공간 분리 (L 채널 추출 - 밝기/휘도)
        hls_l = cv2.split(cv2.cvtColor(blur, cv2.COLOR_BGR2HLS))[1]
        lab_l = cv2.split(cv2.cvtColor(blur, cv2.COLOR_BGR2LAB))[0]
        # cv2.imshow('2_HLS_L_Channel', hls_l) 
        # cv2.imshow('3_LAB_L_Channel', lab_l) 

        # 3. 적응형 임계값 적용 함수
        def adaptive_th(img_channel):
            img_norm = cv2.normalize(img_channel, None, 0, 255, cv2.NORM_MINMAX)
            th = cv2.adaptiveThreshold(img_norm.astype(np.uint8), 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, # 흰색 차선이 밝으므로 THRESH_BINARY_INV 사용
                                       LANE_ADAPTIVE_THRESH_BLOCK_SIZE,
                                       LANE_ADAPTIVE_THRESH_C_CONSTANT)
            return th

        hls_th = adaptive_th(hls_l)
        lab_th = adaptive_th(lab_l)
        combined = cv2.bitwise_and(hls_th, lab_th)
        cv2.imshow('6_Combined_Threshold', combined) # 이진화 결과 시각화
        
        # 4. 모폴로지 닫기(CLOSE) 연산 (노이즈 제거 및 끊어진 선 연결)
        kernel = np.ones(LANE_MORPHOLOGY_KERNEL_SIZE, np.uint8)
        processed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow('7_Final_Processed_Binary', processed) 
        return processed

    def warp_image(self, binary):
        # 5. 원근 변환 (버드 뷰)
        warped = cv2.warpPerspective(binary, self.M, (self.Width, self.Height))
        # cv2.imshow('8_Warped_Binary', warped) 
        return warped

    def detect_lane_pixels(self, binary_warped):
        # 6. 히스토그램 기반 슬라이딩 윈도우 방식으로 차선 픽셀 감지
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0) 
        midpoint = int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        window_height = int(binary_warped.shape[0] / LANE_SLIDING_WINDOWS_COUNT)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(LANE_SLIDING_WINDOWS_COUNT):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height

            win_xleft_low = leftx_current - LANE_SLIDING_WINDOW_MARGIN
            win_xleft_high = leftx_current + LANE_SLIDING_WINDOW_MARGIN
            win_xright_low = rightx_current - LANE_SLIDING_WINDOW_MARGIN
            win_xright_high = rightx_current + LANE_SLIDING_WINDOW_MARGIN

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > LANE_SLIDING_WINDOW_MIN_PIXELS:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > LANE_SLIDING_WINDOW_MIN_PIXELS:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty

    def fit_polynomial(self, leftx, lefty, rightx, righty):
        try:
            # 7. 2차 다항식으로 좌우 차선 피팅
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            self.prev_l_detected = True
            self.prev_r_detected = True
            self.prev_l_pos = left_fit[0] * (self.Height - 1)**2 + left_fit[1] * (self.Height - 1) + left_fit[2]
            self.prev_r_pos = right_fit[0] * (self.Height - 1)**2 + right_fit[1] * (self.Height - 1) + right_fit[2]
        except:
            # 차선 감지 실패 시 이전 위치 기반으로 가상의 직선 생성
            left_fit = np.array([0, 0, self.prev_l_pos]) 
            right_fit = np.array([0, 0, self.prev_r_pos])
            self.prev_l_detected = False
            self.prev_r_detected = False
        return left_fit, right_fit

    def draw_lane(self, original_img, binary_img, left_fit, right_fit):
        ploty = np.linspace(0, binary_img.shape[0]-1, binary_img.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        warp_zero = np.zeros_like(binary_img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # 8. 차선 영역 녹색으로 채우기
        cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0)) 
        
        # 9. 역원근 변환하여 원본 이미지에 오버레이
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (self.Width, self.Height))
        result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0) 

        # 차선 중앙점 계산 (가장 아래쪽 Y좌표 기준)
        return result, int((left_fitx[-1] + right_fitx[-1]) // 2) 

    def detect_lane_color_position(self, img):
        # 10. HSV 색 공간에서 노란색/흰색 차선 감지
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 노란색 차선 HSV 범위 (환경에 맞게 튜닝 필요)
        # H: 색상 (Yellow: 20-40), S: 채도 (보통 100 이상), V: 명도 (보통 100 이상)
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([35, 255, 255])
        
        # 흰색 차선 HSV 범위 (환경에 맞게 튜닝 필요)
        # H: 어떤 색상도 가능 (0-179), S: 채도 낮음 (0-30), V: 명도 높음 (180-255)
        white_lower = np.array([0, 0, 180])
        white_upper = np.array([200, 45, 255])
        yellow_lower = np.array([0, 0, 180])
        yellow_upper = np.array([200, 45, 255])
        
        mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
        mask_white = cv2.inRange(hsv, white_lower, white_upper)

        # 이미지의 왼쪽/오른쪽 절반에서 각 색상의 픽셀 수 계산
        left_yellow_pixels = cv2.countNonZero(mask_yellow[:, :self.Width//2])
        right_yellow_pixels = cv2.countNonZero(mask_yellow[:, self.Width//2:])
        left_white_pixels = cv2.countNonZero(mask_white[:, :self.Width//2])
        right_white_pixels = cv2.countNonZero(mask_white[:, self.Width//2:])
        
        # 색상 감지 플래그 업데이트
        self.is_left_yellow = left_yellow_pixels > 100 # 임계값은 튜닝 필요
        self.is_right_yellow = right_yellow_pixels > 100
        self.is_left_white = left_white_pixels > 100
        self.is_right_white = right_white_pixels > 100

        # 이 함수는 직접적으로 차선 위치를 반환하지 않고, 클래스 변수를 업데이트

    def compute_lookahead_error(self, left_fit, right_fit, y_vals):
        """여러 y 좌표에 대한 차선 중심 좌표를 기반으로 평균 error 계산"""
        lane_center_xs = []
        car_center_x = self.Width // 2
        
        for y in y_vals:
            left_x = left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]
            right_x = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]
            
            # 유령 차선 로직처럼, 한쪽 차선이 감지되지 않았을 때 예상되는 차선 너비 적용
            # 이 부분은 `fit_polynomial`에서 이전 값을 사용하는 것과 보완적으로 작동
            if not self.prev_l_detected and self.prev_r_detected: # 왼쪽 차선이 없으면
                left_x = right_x - LANE_WIDTH # 오른쪽 차선에서 왼쪽으로 차선 폭만큼 이동
            elif not self.prev_r_detected and self.prev_l_detected: # 오른쪽 차선이 없으면
                right_x = left_x + LANE_WIDTH # 왼쪽 차선에서 오른쪽으로 차선 폭만큼 이동

            lane_center_x = (left_x + right_x) / 2
            lane_center_xs.append(lane_center_x)
        
        mean_error = np.mean([car_center_x - cx for cx in lane_center_xs])
        
        return mean_error, lane_center_xs  # error 값과 시각화를 위한 x좌표 리스트

    def get_control(self, frame):
        # PID 게인 스케줄링에 사용할 상수 (parameters.py에서 가져옴)
        # 이 값들은 parameters.py에 정의되어 있어야 합니다.
        LANE_PID_KP_LOW_ERROR = 0.4

        LANE_PID_KI_LOW_ERROR = 0.00

        LANE_PID_KD_LOW_ERROR = 0.2



        LANE_PID_KP_MID_ERROR = 1.0

        LANE_PID_KI_MID_ERROR = 0.5

        LANE_PID_KD_MID_ERROR = 0.4



        LANE_PID_KP_HIGH_ERROR = 0.9

        LANE_PID_KI_HIGH_ERROR = 1.0

        LANE_PID_KD_HIGH_ERROR = 0.6
        # 1. 전처리 파이프라인 실행
        proc = self.preprocess(frame)
        warped = self.warp_image(proc)

        # 2. 차선 픽셀 감지
        leftx, lefty, rightx, righty = self.detect_lane_pixels(warped)
        
        # 3. 차선 다항식 피팅
        left_fit, right_fit = self.fit_polynomial(leftx, lefty, rightx, righty)
        
        # 4. 차선 그리기 및 중앙점 계산
        drawn_img, _ = self.draw_lane(frame, warped, left_fit, right_fit) 

        # 5. lookahead 선 기반 오차 계산
        # 0.4*Height ~ Height-1 범위의 80개 Y 좌표에 대해 오차 계산
        lookahead_ys = np.linspace(int(self.Height * 0.4), self.Height - 1, 80)
        error, center_xs = self.compute_lookahead_error(left_fit, right_fit, lookahead_ys)
        abs_error = abs(error)

        # 6. PID 게인 스케줄링: 오차의 크기에 따라 Kp, Ki, Kd 값 동적 변경
        if abs_error < 40:
            self.pid.set_gains(kp=LANE_PID_KP_LOW_ERROR, ki=LANE_PID_KI_LOW_ERROR, kd=LANE_PID_KD_LOW_ERROR)
        elif abs_error < 80:
            self.pid.set_gains(kp=LANE_PID_KP_MID_ERROR, ki=LANE_PID_KI_MID_ERROR, kd=LANE_PID_KD_MID_ERROR)
        else:
            self.pid.set_gains(kp=LANE_PID_KP_HIGH_ERROR, ki=LANE_PID_KI_HIGH_ERROR, kd=LANE_PID_KD_HIGH_ERROR)

        # 7. PID 컨트롤러를 사용하여 조향 각도 계산
        raw_angle = int(np.clip(self.pid.pid_control(error, dt=0.1), -100, 100)) # raw_angle은 -100 ~ 100
        
        # 8. 각도 필터링 (EMA)
        alpha = 0.5 # 필터링 강도 (0에 가까울수록 이전 값에 더 큰 가중치, 1에 가까울수록 현재 값에 더 큰 가중치)
        filtered_angle = int(alpha * raw_angle + (1 - alpha) * self.prev_angle)

        # 9. 이상치(Outlier) 제거 로직
        # 현재 계산된 각도와 이전 각도의 차이가 outlier_threshold보다 크면 이상치로 간주
        if abs(filtered_angle - self.prev_angle) > self.outlier_threshold:
            self.outlier_count += 1
            if self.outlier_count >= self.max_outlier_count: # 연속된 이상치 횟수가 임계값 초과 시 수용
                angle = filtered_angle
                self.prev_angle = angle
                self.outlier_count = 0
            else: # 임계값 미만이면 이전 각도 유지
                angle = self.prev_angle
        else: # 정상 범위면 현재 각도 수용
            angle = filtered_angle
            self.prev_angle = angle
            self.outlier_count = 0

        # 조향 각도를 Xycar 모터의 허용 범위(-50 ~ 50)로 클램핑
        angle = np.clip(angle, -100, 100) 
        speed_alpha = 0.2
        # 10. 속도 계산: 조향 각도에 따라 속도 감소 (코너링 시 감속)
        speed = min(max(15, 100 - 30*math.sqrt(abs(angle)^2)),100) # 최소 속도 8, 각도에 따라 최대 40까지 감소
        speed = int(speed_alpha * speed + (1 - speed_alpha) * self.pre_speed)
        if abs(speed - self.pre_speed) > self.outlier_threshold_speed:
            self.outlier_count += 1
            if self.outlier_count >= self.max_outlier_count: # 연속된 이상치 횟수가 임계값 초과 시 수용
                speed = speed
                self.speed = speed
                self.outlier_count = 0
            else: # 임계값 미만이면 이전 각도 유지
                speed = self.pre_speed
        else: # 정상 범위면 현재 각도 수용
            speed = speed
            self.pre_speed = speed
            self.outlier_count = 0
        self.pre_speed=speed
        # 11. 차선 색상 감지 (클래스 변수 업데이트)
        self.detect_lane_color_position(frame) 
        
        # 12. lookahead 선 시각화 (보라색 점과 노란색 선)
        for x, y in zip(center_xs, lookahead_ys):
            cv2.circle(drawn_img, (int(x), int(y)), 5, (255, 0, 255), -1)  # 보라색 점 (BGR)

        for i in range(len(center_xs) - 1):
            pt1 = (int(center_xs[i]), int(lookahead_ys[i]))
            pt2 = (int(center_xs[i + 1]), int(lookahead_ys[i + 1]))
            cv2.line(drawn_img, pt1, pt2, (0, 255, 255), 2)  # 노란색 선 (BGR)

        # 13. 최종 drawn_img에 Error, Angle, Speed, Lane Color 정보 표시
        cv2.putText(drawn_img, f"Error: {error:.2f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # 빨간색 (Error)
        cv2.putText(drawn_img, f"Angle: {angle}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2) # 파란색 (Angle)
        cv2.putText(drawn_img, f"Speed: {speed}", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # 초록색 (Speed)
        
        # 차선 색상 감지 결과 표시
        lane_color_status = ""
        if self.is_left_yellow: lane_color_status += "L_Yellow "
        if self.is_right_yellow: lane_color_status += "R_Yellow "
        if self.is_left_white: lane_color_status += "L_White "
        if self.is_right_white: lane_color_status += "R_White "
        
        cv2.putText(drawn_img, f"Lane Colors: {lane_color_status.strip()}", (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1) # 흰색 텍스트

        # 모든 중간 시각화 창 업데이트 및 최종 이미지 표시
        cv2.imshow('Final Lane Detection Result', drawn_img)
        cv2.waitKey(1) # 1ms 대기하며 모든 cv2.imshow 창을 업데이트

        # 최종 조향 각도(반전), 속도, 시각화된 이미지를 반환
        return -angle, speed, drawn_img