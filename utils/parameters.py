# utils/constants.py

import numpy as np
#main parameter
# ==============================================================================
# 전역 미션 제어 모드 정의 (State Machine)
# 로봇의 현재 상태를 나타내며, 각 모드에서 수행할 동작이 정의됩니다.
# ==============================================================================
MODE_INITIAL_LANE_FOLLOW = 0
MODE_STOPLINE_HALT = 1
MODE_LEBACON_AVOIDANCE = 2
MODE_AR_DRIVE = 3
MODE_AR_FOLLOW = 4
MODE_LANE_CHANGE = 5
MODE_FINAL_LANE_FOLLOW = 6
MODE_MISSION_COMPLETE = 7

MODE_MISSION_COMPLETE = 99   # 미션 완료 모드: 모든 미션을 성공적으로 마치고 정지합니다.



##PID gains
#################차선 주행
LANE_PID_KP = 0.4
LANE_PID_KI = 0.0
LANE_PID_KD = 0.1

#################AR테그 추종
AR_PID_KP = 0.3
AR_PID_KI = 0.0
AR_PID_KD = 0.1
#########################


################레바콘 주행
SONIC_PID_KP = 1.2
SONIC_PID_KI = 0.02
SONIC_PID_KD = 0.4
###############################################




##Filters gain
FILTER_QUEUE_LENGTH = 3       # 큐 길이
FILTER_IIR_ALPHA = 0.2        # IIR 필터 계수
FILTER_OUTLIER_THRESHOLD = 30 # 이상치 판단 기준 (절댓값 차이 또는 2*표준편차 중 큰 값)
#########################











#Lane parameter
# ==============================================================================
# LaneDetector (차선 감지기) 파라미터 정의
# 차선 감지 알고리즘의 동작을 세밀하게 조정하기 위한 상수들입니다.
# ==============================================================================

# --- 1. 이미지 및 기본 설정 ---
LANE_IMG_WIDTH = 640  # 차선 감지에 사용될 이미지의 너비 (픽셀)
LANE_IMG_HEIGHT = 480 # 차선 감지에 사용될 이미지의 높이 (픽셀)

# --- 2. 원근 변환 (Perspective Transform) 파라미터 ---
# 원본 이미지에서 차선 영역을 "위에서 내려다보는" 형태로 변환하기 위한 설정입니다.
# 이를 통해 차선이 평행하게 보여서 감지하기 쉬워집니다.

# LANE_WARP_SRC_POINTS: 원본 이미지에서 변환할 영역의 4개 꼭지점 (좌표 순서: 상단 좌, 하단 좌, 상단 우, 하단 우)
# 이 점들은 일반적으로 차량 전방의 도로를 사다리꼴 형태로 지정합니다.
LANE_WARP_SRC_POINTS = np.float32([
    [0, 346],  # 픽셀 (0, 346): 원본 이미지의 좌측 상단 (차량 시야에서 도로의 왼쪽 위)
    [0, 396],  # 픽셀 (0, 396): 원본 이미지의 좌측 하단 (차량 시야에서 도로의 왼쪽 아래)
    [640, 346], # 픽셀 (640, 346): 원본 이미지의 우측 상단 (차량 시야에서 도로의 오른쪽 위)
    [640, 396]  # 픽셀 (640, 396): 원본 이미지의 우측 하단 (차량 시야에서 도로의 오른쪽 아래)
])

# LANE_WARP_DST_POINTS: 변환된 이미지에서 LANE_WARP_SRC_POINTS가 매핑될 4개 꼭지점
# 이 점들은 일반적으로 직사각형 형태로 지정하여, 차선이 평행하게 보이도록 합니다.
# LANE_IMG_WIDTH와 LANE_IMG_HEIGHT를 사용하여 변환된 이미지의 크기를 정의합니다.
LANE_WARP_DST_POINTS = np.float32([
    [0, 0],             # 픽셀 (0, 0): 변환된 이미지의 좌측 상단
    [0, LANE_IMG_HEIGHT], # 픽셀 (0, 480): 변환된 이미지의 좌측 하단
    [LANE_IMG_WIDTH, 0], # 픽셀 (640, 0): 변환된 이미지의 우측 상단
    [LANE_IMG_WIDTH, LANE_IMG_HEIGHT] # 픽셀 (640, 480): 변환된 이미지의 우측 하단
])


# --- 3. 이미지 전처리 (Preprocessing) 파라미터 ---
# 차선 감지를 위해 이미지를 노이즈 제거 및 이진화하는 과정에 사용됩니다.

# LANE_GAUSSIAN_BLUR_KERNEL_SIZE: 가우시안 블러(Gaussian Blur) 필터의 커널 크기
# 이미지의 노이즈를 줄여 차선을 더 명확하게 만듭니다. (가로, 세로) 튜플 형태이며, 홀수여야 합니다.
LANE_GAUSSIAN_BLUR_KERNEL_SIZE = (5, 5)

# LANE_ADAPTIVE_THRESH_BLOCK_SIZE: 적응형 이진화(Adaptive Thresholding)의 블록 크기
# 주변 픽셀의 평균을 계산하는 영역의 크기입니다. 이 영역 내에서 임계값을 동적으로 결정합니다. 홀수여야 합니다.
LANE_ADAPTIVE_THRESH_BLOCK_SIZE = 45

# LANE_ADAPTIVE_THRESH_C_CONSTANT: 적응형 이진화 시 평균에서 빼는 상수 (C 값)
# 블록 내 평균에서 이 상수를 뺀 값이 임계값이 됩니다. 이 값을 조절하여 이진화의 민감도를 조절합니다.
LANE_ADAPTIVE_THRESH_C_CONSTANT = 15

# LANE_MORPHOLOGY_KERNEL_SIZE: 모폴로지 연산(Morphological Operation)의 커널 크기
# 이미지의 작은 구멍을 메우거나 끊어진 선을 연결하는 데 사용됩니다 (예: cv2.MORPH_CLOSE).
LANE_MORPHOLOGY_KERNEL_SIZE = (5, 5)


# --- 4. 차선 픽셀 감지 (Sliding Window) 파라미터 ---
# 원근 변환된 이진 이미지에서 차선 픽셀을 찾기 위한 슬라이딩 윈도우 알고리즘에 사용됩니다.

# LANE_SLIDING_WINDOWS_COUNT: 차선 픽셀을 검색할 슬라이딩 윈도우의 개수
# 이미지 하단에서 상단으로 몇 개의 윈도우를 사용하여 차선을 추적할지 결정합니다.
LANE_SLIDING_WINDOWS_COUNT = 9

# LANE_SLIDING_WINDOW_MARGIN: 슬라이딩 윈도우의 마진 (너비)
# 현재 차선 중심에서 좌우로 얼마나 넓게 픽셀을 검색할지 정의합니다.
LANE_SLIDING_WINDOW_MARGIN = 50

# LANE_SLIDING_WINDOW_MIN_PIXELS: 윈도우를 재중심화하기 위한 최소 픽셀 수
# 윈도우 내에서 이 값 이상의 픽셀이 감지되어야 윈도우의 중심을 해당 픽셀들의 평균 위치로 이동시킵니다.
LANE_SLIDING_WINDOW_MIN_PIXELS = 50
