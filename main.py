#!/usr/bin/env python

import rospy
import cv2
import numpy as np 



from utils.basic_controller import BasicController
from utils.traffic_light_detector import TrafficLightDetector
from utils.sonic_filter import SonicDriver
from utils.lane_detector import LaneDetector
from utils.lane_changer import LaneChanger  # ✅ 차선 변경 모듈
from utils.stopline_detector import StoplineDetector
from utils.camera_handler import CameraHandler 
from utils.parameters import *  # ✅ MODE_* 상수들
from utils.ar_tag_handler import ARTagHandler
from utils.ar_tag_follow import ARTagFollow



if __name__ == '__main__':
    rospy.init_node('main_controller')
    controller = BasicController()
    camera_handler = CameraHandler()
    rospy.loginfo("메인 컨트롤러 시작! 초기 모드: MODE_INITIAL_LANE_FOLLOW")
    rate = rospy.Rate(10)  # 10Hz
    current_mode = MODE_FINAL_LANE_FOLLOW
    traffic_detector = TrafficLightDetector()
    lane_detector=LaneDetector()
    ar_follow = ARTagFollow()
    lane_changer=LaneChanger()
    ar_handler = ARTagHandler()




    while not rospy.is_shutdown():
        if not camera_handler.is_image_available():
            rospy.logwarn("이미지 데이터 수신 대기 중...")
            rate.sleep()
            continue
        current_image = camera_handler.get_latest_image()









        if current_mode == MODE_INITIAL_LANE_FOLLOW:
            stopline_detector = StoplineDetector()
            rospy.loginfo("현재 모드: MODE_INITIAL_LANE_FOLLOW (정지선 찾기)")
            final_angle, final_speed, stopline_is_detected_permanently = stopline_detector.get_control_values(current_image)
            controller.drive(final_angle, final_speed)
            rospy.loginfo(f"주행 중: Angle={final_angle}, Speed={final_speed}")
            if stopline_is_detected_permanently and final_speed == 0:
                rospy.loginfo("🛑 정지선 영구 감지 및 정지 완료 → MODE_STOPLINE_HALT")
                current_mode = MODE_STOPLINE_HALT







        if current_mode == MODE_STOPLINE_HALT:
            rospy.loginfo("현재 모드: MODE_STOPLINE_HALT (신호등 대기)")
            controller.drive(0, 0)
            # is_green_detected, _ = traffic_detector.is_green_light_detected(current_image)
            is_green_detected = traffic_detector.is_green_light_detected()
            if is_green_detected:
                rospy.loginfo("🟢 초록불 감지 → MODE_LEBACON_AVOIDANCE")
                controller.drive(0, 30)
                rospy.sleep(0.2)
                current_mode = MODE_LEBACON_AVOIDANCE
            else:
                rospy.loginfo("초록불 대기 중...")






        elif current_mode == MODE_LEBACON_AVOIDANCE:
            sonic_driver = SonicDriver()
            rospy.loginfo("현재 모드: MODE_LEBACON_AVOIDANCE (레바콘 회피 중)")
            leba_angle, leba_speed, lebacon_section_complete = sonic_driver.get_drive_values()
            controller.drive(leba_angle, leba_speed)
            rospy.loginfo(f"레바콘 회피: Angle={leba_angle}, Speed={leba_speed}")
            if lebacon_section_complete:
                rospy.loginfo("✅ 레바콘 구간 완료 → MODE_AR_DRIVE")
                controller.drive(0, 15)
                rospy.sleep(0.5)
                current_mode = MODE_AR_DRIVE









        elif current_mode == MODE_AR_DRIVE:
            rospy.loginfo("현재 모드: MODE_AR_DRIVE (AR 태그 기반 주차)")
            ar_angle, ar_speed, is_parking_complete = ar_handler.get_drive_command()
            if ar_angle is None:
                rospy.loginfo("AR 태그 탐색 중...")
                continue
            controller.drive(ar_angle, ar_speed)
            rospy.loginfo(f"AR 주행: Angle={ar_angle}, Speed={ar_speed}")
            if is_parking_complete:
                rospy.loginfo("🅿️ 주차 완료 → MODE_AR_FOLLOW")
                controller.drive(ar_angle, 0)
                rospy.sleep(0.5)
                current_mode = MODE_AR_FOLLOW









        elif current_mode == MODE_AR_FOLLOW:
            rospy.loginfo("현재 모드: MODE_AR_FOLLOW (AR 태그 추종 중)")
            follow_angle, follow_speed, follow_complete = ar_follow.get_follow_command()
            controller.drive(follow_angle, follow_speed)
            rospy.loginfo(f"AR 추종: Angle={follow_angle}, Speed={follow_speed}")
            if follow_complete:
                rospy.loginfo("✅ AR 태그 추종 완료 → MODE_LANE_CHANGE")
                controller.drive(0, 15)
                rospy.sleep(0.5)
                lane_changer.start(current_image)  # ✅ 차선 변경 시작 준비
                current_mode = MODE_LANE_CHANGE









        elif current_mode == MODE_LANE_CHANGE:
            rospy.loginfo("현재 모드: MODE_LANE_CHANGE (차선 변경 중)")
            angle, speed, lane_change_done = lane_changer.get_control()
            controller.drive(angle, speed)
            rospy.loginfo(f"차선 변경 중: Angle={angle}, Speed={speed}")
            if lane_change_done:
                rospy.loginfo("✅ 차선 변경 완료 → MODE_FINAL_LANE_FOLLOW")
                controller.drive(0, 15)
                rospy.sleep(0.5)
                current_mode = MODE_FINAL_LANE_FOLLOW









        elif current_mode == MODE_FINAL_LANE_FOLLOW:
            rospy.loginfo("현재 모드: MODE_FINAL_LANE_FOLLOW (최종 라인 트래킹)")
            lane_angle, lane_speed, mission_complete_flag = lane_detector.get_steering_angle(current_image)
            controller.drive(lane_angle, lane_speed)
            rospy.loginfo(f"최종 주행: Angle={lane_angle}, Speed={lane_speed}")
            if mission_complete_flag:
                rospy.loginfo("🎉 미션 완료 → MODE_MISSION_COMPLETE")
                current_mode = MODE_MISSION_COMPLETE









        elif current_mode == MODE_MISSION_COMPLETE:
            rospy.loginfo("현재 모드: MODE_MISSION_COMPLETE. 모든 미션 완료!")
            controller.drive(0, 0)
            rospy.signal_shutdown("미션 완료")








        rate.sleep()
