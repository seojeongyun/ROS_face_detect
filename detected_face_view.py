#!/usr/bin/env python3

import cv2
import sys
import rospy
import numpy as np

from function import imgmsg_to_cv2
from sensor_msgs.msg import Image
from ring_buffer import CircularBuffer

class detected_face_subscriber:
    def __init__(self):
        self.color_sub = rospy.Subscriber("/face_detector/image_result", Image, self.callback)
        self.buffer = CircularBuffer(50)

    def callback(self, data):
        try:
            data.encoding = 'bgr8'
            cv_image = imgmsg_to_cv2(data)
            # self.buffer.enqueue(cv_image)
            cv2.imshow('FACE_DETECT', cv_image)
            cv2.waitKey(int(1000/10))

        except:
            print("FUCKING ERROR SIBAL\n")


def main():
    print('\033[96m' + '[START] ROS NODE: FACE IMAGE SUBSCRIBER' + '\033[0m')
    img_node = detected_face_subscriber()
    
    print('\033[96m' + f"[ INIT ] ROS NODE: FACE IMAGE SUBSCRIBER" + '\033[0m')
    rospy.init_node('detected_face', anonymous=True)

    try:
        rospy.spin()
    
    except KeyboardInterrupt:
        print("shutting down")
        cv2.destroyAllWindows()

if __name__=='__main__':
    main()
