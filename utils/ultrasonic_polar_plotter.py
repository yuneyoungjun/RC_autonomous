# ultrasonic_polar_plotter.py
import matplotlib.pyplot as plt
import numpy as np
import math

class UltrasonicPolarPlotter:
    def __init__(self):
        # 플롯 초기화
        plt.ion() # 인터랙티브 모드 켜기 (실시간 업데이트)
        self.fig, self.ax = plt.subplots(subplot_kw={'projection': 'polar'}) # 극좌표계로 설정!
        self.ax.set_theta_zero_location("N") # 0도를 북쪽(위)으로 설정
        self.ax.set_theta_direction(-1) # 시계방향으로 각도 증가 (차량 전방 기준)
        self.ax.set_rlabel_position(180) # 반지름 레이블 위치
        self.ax.set_rlim(0, 250) # 반지름(거리) 축 범위 설정 (0cm ~ 250cm)
        self.ax.set_rticks([50, 100, 150, 200, 250]) # 반지름 축 눈금
        self.ax.set_title("Ultrasonic Sensor Data (Polar Plot)", va='bottom')
        
        # 초음파 센서의 상대적인 각도 설정 (xycar_ultrasonic 메시지 순서에 맞춰서)
        # 이 각도들은 실제 네 차의 센서 배치에 맞게 정확히 수정해야 해!
        self.sensor_angles_deg = {
            0: 0,    # 전방 (가운데)
            1: -30,  # 전방 좌측
            2: 30,   # 전방 우측
            3: -120, # 후방 좌측
            4: 120,  # 후방 우측
            5: -90,  # 좌측 중앙
            6: 90,   # 우측 중앙
            7: 180,  # 후방 중앙
            8: -60,  # 전방 좌측 대각선
            9: 60    # 전방 우측 대각선
        }
        self.sensor_angles_rad = {idx: math.radians(angle) for idx, angle in self.sensor_angles_deg.items()}

        # 플롯 데이터 초기화
        self.scatter_plot = self.ax.scatter([], [], s=100, c='red', alpha=0.7) # 센서 데이터 점

        # 플롯 창이 바로 나타나도록
        plt.show(block=False) 

    def update_plot(self, ultra_values):
        # 플로팅할 각도와 거리 데이터
        angles = [] # 라디안
        distances = [] # cm
        
        # 메시지에서 각 센서 데이터를 추출하고 각도와 매핑
        for i, dist in enumerate(ultra_values):
            if i in self.sensor_angles_rad: # 정의된 센서 각도에 포함되는 경우만
                # 거리가 0이거나 너무 크면 (장애물 감지 안됨) 플로팅하지 않거나 최대 거리로 설정
                if dist == 0 or dist >= 250: # 0이거나 250 이상이면 감지 안된 것으로 간주 (조절 가능)
                    continue 
                
                angles.append(self.sensor_angles_rad[i])
                distances.append(dist) # 초음파 센서는 보통 cm 단위로 반환 (확인 필요)

        # 플롯 업데이트
        if angles and distances:
            # np.c_는 두 배열을 열 방향으로 합쳐줘서 (angle, distance) 쌍을 만들어줘.
            self.scatter_plot.set_offsets(np.c_[angles, distances]) 
        else: # 데이터가 없으면 플롯 비우기
            self.scatter_plot.set_offsets(np.c_[:, :])

        self.fig.canvas.draw_idle() # 플롯 다시 그리기 요청
        self.fig.canvas.flush_events() # 이벤트 처리 (화면 업데이트)

    def close_plot(self):
        plt.close(self.fig) # 플롯 창 닫기