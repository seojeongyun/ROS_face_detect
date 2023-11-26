#!/usr/bin/env python3

import cv2
import sys

import rospy
from sensor_msgs.msg import Image

class fps_controller:
    def __init__(self):
        self.color_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        self.result_pub = rospy.Publisher('/fps_controller/image_raw', Image, queue_size=1)
        self.count = 0

    def callback(self, data):
        try:
            self.count += 1
            if self.count % 3 == 0:
                self.result_pub.publish(data)
            elif self.count >= 100:
                self.count = 0
        except:
            print("FUCKING ERROR SIBAL\n")


def main():
    print('\033[96m' + '[START] ROS NODE: FPS_CONTROLLER' + '\033[0m')
    img_node = fps_controller()
    
    print('\033[96m' + f"[ INIT ] ROS NODE: FPS_CONTROLLER" + '\033[0m')
    rospy.init_node('fps_controller', anonymous=True)

    try:
        rospy.spin()
    
    except KeyboardInterrupt:
        print("shutting down")
        cv2.destroyAllWindows()

if __name__=='__main__':
    main()

