# utils/camera_handler.py

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class CameraHandler:
    def __init__(self):
        self.cv_bridge = CvBridge()
        self.current_image = np.empty(shape=[0])
        
        # 카메라 토픽 구독 설정
        rospy.Subscriber("/usb_cam/image_raw/", Image, self._usbcam_callback, queue_size=1)
        
        # 첫 이미지가 수신될 때까지 대기하여 카메라가 준비되었는지 확인
        rospy.wait_for_message("/usb_cam/image_raw/", Image)
        rospy.loginfo("CameraHandler: 카메라 준비 완료!")

    def _usbcam_callback(self, data):
        """
        USB 카메라 토픽의 콜백 함수.
        최신 프레임으로 current_image를 업데이트합니다.
        """
        try:
            self.current_image = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CameraHandler: CvBridgeError 발생: {e}")

    def get_latest_image(self):
        """
        가장 최근에 수신된 카메라 이미지를 반환합니다.
        """
        return self.current_image

    def is_image_available(self):
        """
        유효한 이미지가 수신되었는지 확인합니다.
        """
        return self.current_image.size > 0
