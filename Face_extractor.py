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
WEIGHT_PATH = "demo_weight/6s_hs_960_sc.onnx" if ONNX else "/storage/jysuh/YOLOv6_weights/960/6s_hs_SW_960/weights/coco_150.pt"
# WEIGHT_PATH = "demo_weight/6s_hs_960_sc.onnx" if ONNX else "./runs/train/6s_hs_SW_960/weights/coco_150.pt"
CONF_THRES = 0.35
IOU_THRES = 0.5
NMS_DET = 300
GPU_ID = '1'
DEVICE = 'cpu' # torch.device(f'cuda:{GPU_ID}') if GPU_ID != '' else torch.device('cpu')
VIEW = True

# ==== NETWORK_MODULE ====
FACE_DETECTOR = demo(WEIGHT_PATH, RESIZE_SIZE, DEVICE, ONNX,
                     CONF_THRES, IOU_THRES, NMS_DET)

class get_frame_from_video:
    def __init__(self):
        self.save = True
        self.show = True
        #
        self.img_path = '/home/jysuh/Downloads/psj_1.JPG'
        self.name = self.get_Name()
        self.save_path = self.get_save_path()
        self.count = self.get_imgNum()

    def get_Name(self):
        id = self.img_path.split('/')[4].split('_')[0]
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
        if self.show:
            self.img_show()
        else:
            cv2.destroyAllWindows()

    def img_save(self, img_list):
        for idx in range (len(img_list)):
            cv2.imwrite(self.save_path + '/' + self.name + '_{}.jpg'.format(self.count), img_list[idx])
            self.count += 1


    def img_crop(self, out_boxes):
        img_list = []
        for idx in range(len(out_boxes[0])):
            box = out_boxes[0][idx][:4].cpu().numpy()
            box = box.round().astype(np.int32).tolist()
            img_w, img_h = self.img[0].shape[:2]
            x1 = max(0, box[0])
            y1 = max(0, box[1])
            x2 = min(img_w, box[2])
            y2 = min(img_h, box[3])
            img = self.img[0][y1: y2, x1: x2]
            img_list.append(img)
        return img_list

    def img_show(self):
        self.img = cv2.imread(self.img_path)
        self.img = [cv2.resize(self.img, dsize=(960,960))]
        bboxes, out_boxes, ori_size, resized_size, ratios, pads = FACE_DETECTOR.run(self.img)
        img_w_box = FACE_DETECTOR.post_processing(bboxes, self.img)

        cv2.imshow('detected faces', img_w_box[0])
        cv2.waitKey()

        if self.save and len(out_boxes[0]) != 0:
            img_list = self.img_crop(out_boxes)
            self.img_save(img_list)




if __name__ == '__main__':
    img = get_frame_from_video()
    img.make_gallery()
    img.capture()
