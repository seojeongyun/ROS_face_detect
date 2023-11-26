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
RESIZE_SIZE = 640
ONNX = False
# WEIGHT_PATH = "demo_weight/6s_hs_960_sc.onnx" if ONNX else "demo_weight/6s_hs_960.pt"
# WEIGHT_PATH = "demo_weight/6s_hs_960_sc.onnx" if ONNX else "/storage/jysuh/YOLOv6_weights/960/6s_hs_SW_960/weights/epoch : 100 .pt"
WEIGHT_PATH = "demo_weight/6s_hs_960_sc.onnx" if ONNX else "/storage/jysuh/YOLOv6_weights/960/6s_hs_SC_960/weights/last_ckpt.pt"
# WEIGHT_PATH = "demo_weight/6s_hs_960_sc.onnx" if ONNX else "./runs/train/6s_hs_SW_960/weights/coco_150.pt"
CONF_THRES = 0.05
IOU_THRES = 0.6
NMS_DET = 12
GPU_ID = '1'
DEVICE = 'cpu' # torch.device(f'cuda:{GPU_ID}') if GPU_ID != '' else torch.device('cpu')
VIEW = True

# ==== NETWORK_MODULE ====
FACE_DETECTOR = demo(WEIGHT_PATH, RESIZE_SIZE, DEVICE, ONNX,
                     CONF_THRES, IOU_THRES, NMS_DET)

class get_frame_from_video:
    def __init__(self):
        self.show = True
        self.mosaic = False
        #
        self.img_path = '/home/jysuh/Downloads/hard_3_.jpeg'
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


    def face_mosaic(self, out_boxes):
        for idx in range(len(out_boxes[0])):
            box = out_boxes[0][idx][:4].cpu().numpy()
            box = box.round().astype(np.int32).tolist()
            img_w, img_h = self.img[0].shape[:2]
            x1 = max(0, box[0])
            y1 = max(0, box[1])
            x2 = min(img_w, box[2])
            y2 = min(img_h, box[3])
            h, w = y2-y1, x2-x1
            img = self.img[0][y1: y2, x1: x2]
            img = cv2.resize(img, dsize=(13,13))
            img = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
            self.img[0][y1: y2, x1: x2] = img
        return self.img[0]

    def img_show(self):
        self.img = cv2.imread(self.img_path)
        self.img = [cv2.resize(self.img, dsize=(275,183))]
        bboxes, out_boxes, ori_size, resized_size, ratios, pads = FACE_DETECTOR.run(self.img)
        img_w_box = FACE_DETECTOR.post_processing(bboxes, self.img)

        if self.mosaic:
            img = self.face_mosaic(out_boxes)

        else:
            self.img_save(img_w_box)
            img = img_w_box[0]


        if self.show:
            cv2.imshow('detected faces', img)
            cv2.waitKey()






if __name__ == '__main__':
    img = get_frame_from_video()
    img.make_gallery()
    img.capture()
