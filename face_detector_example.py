#!/usr/bin/env python3

import cv2
import sys
sys.path.append("/home/parksungjun/PycharmProjects/")
sys.path.append("/home/parksungjun/PycharmProjects/YOLOv6/")

import torch
import rospy
import matplotlib.pyplot as plt
import numpy as np

from sensor_msgs.msg import Image
from YOLOv6.yolov6.core.demo import demo
from function import imgmsg_to_cv2
from ring_buffer import CircularBuffer

# ==== CONFIGURATION ====
HALF = True
MAX_BATCH_SIZE = 10
RESIZE_SIZE = 160

ONNX = False
WEIGHT_PATH = "demo_weight/6s_hs_960_sc.onnx" if ONNX else "/home/parksungjun/Downloads/ours_SW_150.pt"
CONF_THRES = 0.35
IOU_THRES = 0.5
NMS_DET = 100
GPU_ID = '0'
# DEVICE = torch.device(f'cuda:{GPU_ID}') if GPU_ID != '' else torch.device('cpu')
DEVICE = torch.device('cuda:0')
VIEW = True

FACE_DETECTOR = demo(WEIGHT_PATH, RESIZE_SIZE, DEVICE, ONNX,
                     CONF_THRES, IOU_THRES, NMS_DET)

    
class cam_image_subscriber:
    def __init__(self):
        self.color_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback_rgb)
        self.circular_buff = CircularBuffer(30)
    
    def callback_rgb(self, data):
        try:
            data.encoding = 'bgr8'
            cv_image = imgmsg_to_cv2(data)
            cv_image = cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
            video_img = [cv_image]
    
            bboxes, out_boxes, ori_size, resized_size, ratios, pads = FACE_DETECTOR.run(video_img)
            img_w_box = FACE_DETECTOR.post_processing(bboxes, video_img)

            img = img_w_box[0]

            cv2.imshow('rgb', img)
            cv2.waitKey(int(1000/32))

        except:
            print("FUCKING ERROR SIBAL\n")


def main():
    print('\033[96m' + '[START] FACE ROS NODE: CAM IMAGE SUBSCRIBER' + '\033[0m')
    img_publisher = cam_image_subscriber()
    print('\033[96m' + f"[ INIT ] ROS NODE: CAM IMAGE SUBSCRIBER" + '\033[0m')
    rospy.init_node('cam_image_subscriber', anonymous=True)

    try:
        rospy.spin()
    
    except KeyboardInterrupt:
        print("shutting down")
        cv2.destroyAllWindows()

if __name__=='__main__':
    main()

