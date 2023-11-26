#!/usr/bin/env python2

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import matplotlib.pyplot as plt

class cam_image_publisher:
    def __init__(self):
        self.bridge = CvBridge()
        self.color_pub = rospy.Publisher("/camera/color/image_raw", Image, self.callback_rgb)
    
    def callback_rgb(self, data):
        try:        
            image = self.circular_buff.dequeue()
            cv2.imshow('rgb', cv_image)
            cv2.waitKey(1)

        except:
            print("FUCKING ERROR SIBAL\n")


def main():
    img_publisher = cam_image_publisher()
    rospy.init_node('cam_image_publisher', anonymous=True)
    img_publisher.color_pub.publish()
    try:
        rospy.spin()
    
    except KeyboardInterrupt:
        print("shutting down")
        cv2.destroyAllWindows()

if __name__=='__main__':
    main()
