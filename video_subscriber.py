#!/usr/bin/env python3

import sys
import rospy
import cv2
from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import numpy as np

def imgmsg_to_cv2(img_msg):
    if img_msg.encoding != "bgr8":
        rospy.logerr("This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
    dtype = np.dtype("uint8") # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                    dtype=dtype, buffer=img_msg.data)
    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
        
    return image_opencv

class CircularBuffer:
    def __init__(self, max_size=10):
        self.cell = np.zeros((480,640,3), dtype='float32')
        self.buffer = [self.cell] * max_size

        self.head = 0
        self.tail = 0
        self.numItems = 0

        self.max_size = max_size

    def __str__(self):
        items = ['{!r}'.format(item) for item in self.buffer]
        return '[' + ', '.join(items) + ']'
    
    def size(self):
        if self.tail >= self.head:
            return self.tail - self.head
        return self.max_size - self.head - self.tail
    
    def is_empty(self):
        # return self.buffer[self.head] == None and self.buffer[self.tail] == None
        # return self.tail == self.head
        return self.numItems == 0
    
    def is_full(self):
        # return None not in self.buffer
        # return self.tail == (self.head-1) % self.max_size
        return self.numItems == self.max_size
    
    def enqueue(self, item):
        if self.is_full():
            raise OverflowError(
                "CircularBuffer is full, unable to enqueue item")
        self.buffer[self.tail] = item
        self.tail = (self.tail+1) % self.max_size
        self.numItems += 1

    def front(self):
        return self.buffer[self.head]
    
    def dequeue(self):
        if self.is_empty():
            raise IndexError("CircularBuffer is empty, unable to dequeue")
        item = self.buffer[self.head]
        self.buffer[self.head] = None
        self.head = (self.head + 1) % self.max_size
        self.numItems -= 1

        return item
    
class cam_image_subscriber:
    def __init__(self):
        # self.bridge = CvBridge()
        self.color_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback_rgb)
        self.circular_buff = CircularBuffer(30)
    
    def callback_rgb(self, data):
        try:
            # cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            data.encoding = 'bgr8'
            cv_image = imgmsg_to_cv2(data)
            cv_image = cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
            self.circular_buff.enqueue(cv_image)
            image = self.circular_buff.dequeue()

            cv2.imshow('rgb', cv_image)
            cv2.waitKey(1)

        except:
            print("FUCKING ERROR SIBAL\n")


def main():
    img_publisher = cam_image_subscriber()
    rospy.init_node('cam_image_subscriber', anonymous=True)

    try:
        rospy.spin()
    
    except KeyboardInterrupt:
        print("shutting down")
        cv2.destroyAllWindows()

if __name__=='__main__':
    main()

