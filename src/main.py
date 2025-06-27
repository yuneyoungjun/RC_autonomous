#!/usr/bin/env python

import rospy
import cv2
import numpy as np 

from utils.basic_controller import BasicController
from utils.traffic_light_detector import TrafficLightDetector
from utils.sonic_driver import SonicDriver
from utils.lane_detector import LaneDetector
from utils.stopline_detector import StoplineDetector
from utils.camera_handler import CameraHandler 
from utils.parameters import * # 💡 상수들 임포트 
from utils.ar_tag_handler import ARTagHandler  # 추가된 클래스
from utils.ar_tag_follow import ARTagFollow  # 추가된 클래스
#=============================================
# 메인 함수
#=============================================
if __name__ == '__main__':
    rospy.init_node('main_controller')
    controller = BasicController()
    traffic_detector = TrafficLightDetector()
    sonic_driver = SonicDriver()
    lane_detector = LaneDetector()
    stopline_detector = StoplineDetector()
    camera_handler = CameraHandler() # 💡 CameraHandler 인스턴스 생성!
    ar_handler=ARTagHandler()
    ar_follow=ARTagFollow()
    rospy.loginfo("메인 컨트롤러 시작! 초기 모드: MODE_INITIAL_LANE_FOLLOW")
    rate = rospy.Rate(10)  # 10Hz
    # 💡 현재 모드 (constants.py에서 정의)
    current_mode = MODE_INITIAL_LANE_FOLLOW 




    while not rospy.is_shutdown():
        if not camera_handler.is_image_available():
            rospy.logwarn("이미지 데이터 수신 대기 중...")
            rate.sleep()
            continue
        current_image = camera_handler.get_latest_image()




        if current_mode == MODE_INITIAL_LANE_FOLLOW:
            # 🚗 초기 차선 주행 (정지선까지)
            rospy.loginfo("현재 모드: MODE_INITIAL_LANE_FOLLOW (정지선 찾기)")
            final_angle, final_speed, stopline_is_detected_permanently = stopline_detector.get_control_values(current_image)
            controller.drive(final_angle, final_speed)
            rospy.loginfo(f"주행 중: Angle={final_angle}, Speed={final_speed}")
            if stopline_is_detected_permanently and final_speed == 0:
                rospy.loginfo("🛑 정지선 영구 감지 및 정지 완료! MODE_STOPLINE_HALT 모드로 전환.")
                current_mode = MODE_STOPLINE_HALT # 정지선 정지 모드로 전환




        elif current_mode == MODE_STOPLINE_HALT:
            rospy.loginfo("현재 모드: MODE_STOPLINE_HALT (신호등 대기)")
            controller.drive(0, 0) # 계속 정지
            is_green_detected, _ = traffic_detector.get_traffic_light_status(current_image) # 💡 이미지 전달
            if is_green_detected:
                rospy.loginfo("🟢 초록불 감지! MODE_LEBACON_AVOIDANCE 모드로 전환.")
                controller.drive(0, 30)
                rospy.sleep(0.5)
                current_mode = MODE_LEBACON_AVOIDANCE # 레바콘 회피 모드로 전환
            else:
                rospy.loginfo("초록불 대기 중...")




        elif current_mode == MODE_AR_DRIVE:
            rospy.loginfo("현재 모드: MODE_AR_DRIVE (AR 태그 기반 주차)")
            ar_angle, ar_speed, is_parking_complete = ar_handler.get_drive_command(current_image)
            if ar_angle is None:
                rospy.loginfo("AR 태그 탐색 중...")
                continue
            controller.drive(ar_angle, ar_speed)
            rospy.loginfo(f"AR 주행: Angle={ar_angle}, Speed={ar_speed}")
            if is_parking_complete:
                rospy.loginfo("🅿️ 주차 완료 → MODE_MISSION_COMPLETE 전환")
                controller.drive(ar_angle, 0)
                rospy.sleep(0.5)
                current_mode = MODE_AR_FOLLOW


        elif current_mode == MODE_AR_FOLLOW:
            rospy.loginfo("현재 모드: MODE_AR_FOLLOW (AR 태그 추종 중)")
            follow_angle, follow_speed, follow_complete = ar_follow.get_follow_command(current_image)

            controller.drive(follow_angle, follow_speed)
            rospy.loginfo(f"AR 추종: Angle={follow_angle}, Speed={follow_speed}")

            if follow_complete:
                rospy.loginfo("✅ AR 태그 추종 완료 → MODE_FINAL_LANE_FOLLOW 전환")
                controller.drive(0, 15)
                rospy.sleep(0.5)
                current_mode = MODE_FINAL_LANE_FOLLOW

        
        elif current_mode == MODE_LEBACON_AVOIDANCE:
            rospy.loginfo("현재 모드: MODE_LEBACON_AVOIDANCE (레바콘 회피 중)")
            leba_angle, leba_speed, lebacon_section_complete = sonic_driver.get_drive_values(current_image) # 💡 이미지 전달
            controller.drive(leba_angle, leba_speed)
            rospy.loginfo(f"레바콘 회피: Angle={leba_angle}, Speed={leba_speed}")
            if lebacon_section_complete: # 레바콘 구간이 끝났다는 플래그가 올라오면
                rospy.loginfo("✅ 레바콘 구간 완료! MODE_FINAL_LANE_FOLLOW 모드로 전환.")
                current_mode = MODE_FINAL_LANE_FOLLOW # 최종 라인 트래킹 모드로 전환
                controller.drive(0, 15) # 부드럽게 재출발
                rospy.sleep(0.5) # 잠시 대기




        elif current_mode == MODE_FINAL_LANE_FOLLOW:
            rospy.loginfo("현재 모드: MODE_FINAL_LANE_FOLLOW (최종 라인 트래킹)")
            lane_angle, lan_speed, mission_complete_flag = lane_detector.get_steering_angle(current_image) 
            controller.drive(lane_angle, lan_speed)
            rospy.loginfo(f"최종 주행: Angle={lane_angle}, Speed={final_speed}")
            if mission_complete_flag: # lane_detector에서 미션 완료 플래그를 반환한다고 가정
                rospy.loginfo("🎉 미션 완료! MODE_MISSION_COMPLETE 모드로 전환.")
                current_mode = MODE_MISSION_COMPLETE # 미션 완료 모드로 전환




        elif current_mode == MODE_MISSION_COMPLETE:
            # 🎉 미션 완료 모드
            rospy.loginfo("현재 모드: MODE_MISSION_COMPLETE. 모든 미션 완료!")
            controller.drive(0, 0) # 최종 정지
            rospy.signal_shutdown("미션 완료") # ROS 노드 종료



        rate.sleep() # 루프 주기 유지
