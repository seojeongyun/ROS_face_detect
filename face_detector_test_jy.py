# ==== Library ====
import torch
import glob
from yolov6.core.demo import demo
from yolov6.data.data_load import create_dataloader
from tqdm import tqdm
import cv2
from yolov6.utils.fn_for_test import *

# ==== CONFIGURATION ====
HALF = True
MAX_BATCH_SIZE = 50
RESIZE_SIZE = 960
ONNX = False
# WEIGHT_PATH = "demo_weight/6s_hs_960_sc.onnx" if ONNX else "demo_weight/6s_hs_960.pt"
# WEIGHT_PATH = "demo_weight/6s_hs_960_sc.onnx" if ONNX else "/storage/jysuh/YOLOv6_weights/960/6s_hs_SW_960/weights/epoch : 100 .pt"
# WEIGHT_PATH = "demo_weight/6s_hs_960_sc.onnx" if ONNX else "/storage/jysuh/YOLOv6_weights/960/6s_hs_SW_960/weights/epoch130.pt"
WEIGHT_PATH = "demo_weight/6s_hs_960_sc.onnx" if ONNX else "./runs/train/6s_hs_SW_960/weights/epoch130.pt"
CONF_THRES = 0.35
IOU_THRES = 0.5
NMS_DET = 300
GPU_ID = '1'
DEVICE = torch.device(f'cuda:{GPU_ID}') if GPU_ID != '' else torch.device('cpu')
VIEW = True

# ==== NETWORK_MODULE ====
FACE_DETECTOR = demo(WEIGHT_PATH, RESIZE_SIZE, DEVICE, ONNX,
                     CONF_THRES, IOU_THRES, NMS_DET)

class get_frame_from_video:
    def __init__(self):
        self.mosaic = True
        self.save = False
        self.show = True
        self.video_capture = True
        self.rotate = True
        #
        self.video_path = '/storage/jysuh/Video/ParkSeongJun_1.mp4'
        self.name = self.get_Name()
        self.save_path = self.get_save_path()
        self.count = self.get_imgNum()

    def get_Name(self):
        id = self.video_path.split('/')[4].split('_')[0]
        return id

    def get_save_path(self):
        save_path = '/storage/jysuh/gallery/' + self.name
        return save_path

    def get_imgNum(self):
        count = len(glob.glob(self.save_path + '/' + '/*'))
        return count

    def make_gallery(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

    def capture(self):
        capture = cv2.VideoCapture(self.video_path)
        #
        while self.video_capture:
            success, self.vid_img = capture.read()
            self.vid_img = [cv2.resize(self.vid_img, dsize=(960,960))]
            fps = capture.get(cv2.CAP_PROP_FPS)
            #
            if self.rotate:
                self.vid_img[0] = cv2.rotate(self.vid_img[0], cv2.ROTATE_90_CLOCKWISE)
            #
            if success:
                if self.show:
                    self.img_show()
                    print(fps)
                #
                self.count += 1
            else:
                cv2.destroyAllWindows()
                break

    def img_save(self, img):
        cv2.imwrite(self.save_path + '/' + self.name + '_{}.jpg'.format(self.count), img)


    def img_crop(self, out_boxes):
        if len(out_boxes[0]) != 0:
            box = out_boxes[0][0][:4].cpu().numpy()
            box = box.round().astype(np.int32).tolist()
            img_w, img_h = self.vid_img[0].shape[:2]
            x1 = max(0, box[0])
            y1 = max(0, box[1])
            x2 = min(img_w, box[2])
            y2 = min(img_h, box[3])
            img = self.vid_img[0][y1: y2, x1: x2]
            return img
        else:
            pass

    def face_mosaic(self, out_boxes):
        for idx in range(len(out_boxes[0])):
            box = out_boxes[0][idx][:4].cpu().numpy()
            box = box.round().astype(np.int32).tolist()
            img_w, img_h = self.vid_img[0].shape[:2]
            x1 = max(0, box[0])
            y1 = max(0, box[1])
            x2 = min(img_w, box[2])
            y2 = min(img_h, box[3])
            h, w = y2-y1, x2-x1
            img = self.vid_img[0][y1: y2, x1: x2]
            img = cv2.resize(img, dsize=(8,8))
            img = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
            self.vid_img[0][y1: y2, x1: x2] = img
        return self.vid_img[0]

    def img_show(self):
        bboxes, out_boxes, ori_size, resized_size, ratios, pads = FACE_DETECTOR.run(self.vid_img)
        img_w_box = FACE_DETECTOR.post_processing(bboxes, self.vid_img)
        if self.mosaic:
            img = self.face_mosaic(out_boxes)

        else:
            img = img_w_box[0]


        cv2.imshow('detected faces', img)
        cv2.waitKey(5)

        if self.save and len(out_boxes[0]) != 0:
            img = self.img_crop(out_boxes)
            self.img_save(img)


if __name__ == '__main__':
    img = get_frame_from_video()
    img.make_gallery()
    img.capture()
