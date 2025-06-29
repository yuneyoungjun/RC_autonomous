import numpy as np
import cv2
from utils.pid_controller import PID  # ✅ PID 클래스 가져오기
from utils.parameters import *
from utils.pid_controller import PID  # ✅ PID 클래스 가져오기
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






    def preprocess(self, img):
        blur = cv2.GaussianBlur(img, LANE_GAUSSIAN_BLUR_KERNEL_SIZE, 0)
        hls_l = cv2.split(cv2.cvtColor(blur, cv2.COLOR_BGR2HLS))[1]
        lab_l = cv2.split(cv2.cvtColor(blur, cv2.COLOR_BGR2LAB))[0]

        def adaptive_th(img):
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            th = cv2.adaptiveThreshold(img.astype(np.uint8), 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV,
                                       LANE_ADAPTIVE_THRESH_BLOCK_SIZE,
                                       LANE_ADAPTIVE_THRESH_C_CONSTANT)
            return th

        hls_th = adaptive_th(hls_l)
        lab_th = adaptive_th(lab_l)
        combined = cv2.bitwise_and(hls_th, lab_th)

        kernel = np.ones(LANE_MORPHOLOGY_KERNEL_SIZE, np.uint8)
        processed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        return processed








    def warp_image(self, binary):
        return cv2.warpPerspective(binary, self.M, (self.Width, self.Height))








    def detect_lane_pixels(self, binary_warped):
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
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            self.prev_l_detected = True
            self.prev_r_detected = True
        except:
            left_fit = np.polyfit([0, self.Height-1], [self.prev_l_pos]*2, 2)
            right_fit = np.polyfit([0, self.Height-1], [self.prev_r_pos]*2, 2)
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

        cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (self.Width, self.Height))
        result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)
        return result, int((left_fitx[-1] + right_fitx[-1]) // 2)














    def detect_lane_color_position(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 25, 255])
        yellow_lower = np.array([0, 0, 200])
        yellow_upper = np.array([180, 25, 255])

        mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
        mask_white = cv2.inRange(hsv, white_lower, white_upper)

        left_half = mask_yellow[:, :self.Width//2]
        right_half = mask_yellow[:, self.Width//2:]

        left_yellow = cv2.countNonZero(left_half)
        right_yellow = cv2.countNonZero(right_half)

        return right_yellow > left_yellow















    def get_steering_angle(self, frame):
        proc = self.preprocess(frame)
        warped = self.warp_image(proc)
        leftx, lefty, rightx, righty = self.detect_lane_pixels(warped)
        left_fit, right_fit = self.fit_polynomial(leftx, lefty, rightx, righty)
        drawn_img, mid_pos = self.draw_lane(frame, warped, left_fit, right_fit)

        lane_center = mid_pos
        car_center = self.Width // 2
        error = car_center - lane_center

        # ✅ PID 제어기 사용
        angle = int(self.pid.pid_control(error, dt=0.1))  # 0.1초 주기로 동작한다고 가정
        angle = np.clip(angle, -50, 50)

        speed = max(10, 30 - abs(angle) // 3)

        self.prev_l_pos = left_fit[2]
        self.prev_r_pos = right_fit[2]

        yellow_right = self.detect_lane_color_position(frame)

        return angle, speed, yellow_right