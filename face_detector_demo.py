# ==== Library ====
import torch
from yolov6.core.demo import demo

# ==== CONFIGURATION ====
HALF = True
MAX_BATCH_SIZE = 200
RESIZE_SIZE = 416
ONNX = False
WEIGHT_PATH = "demo_weight/6s_hs_960_sc.onnx" if ONNX else "demo_weight/6s_hs_960_cw.pt"
CONF_THRES = 0.7
IOU_THRES = 0.65
NMS_DET = 100
GPU_ID = '0'
DEVICE = torch.device(f'cuda:{GPU_ID}') if GPU_ID != '' else torch.device('cpu')


# ==== NETWORK_MODULE ====
FACE_DETECTOR = demo(WEIGHT_PATH, RESIZE_SIZE, DEVICE, ONNX,
                     CONF_THRES, IOU_THRES, NMS_DET)

if __name__ == '__main__':
    from torch.utils.data.dataloader import DataLoader
    from yolov6.data.VGG_face_loader import data_loader
    from tqdm import tqdm
    import cv2
    #
    demo_loader = DataLoader(data_loader(path='/storage/sjpark/VGGFace2/', type='test'),
                             batch_size=MAX_BATCH_SIZE, collate_fn=data_loader.collate_fn)
    #
    cv2.namedWindow('detected faces', flags=cv2.WINDOW_AUTOSIZE)
    #
    for step, data in tqdm(enumerate(demo_loader), total=len(demo_loader)):
        #
        bboxes = FACE_DETECTOR.run(data[0])
        #
        imgs_w_box = FACE_DETECTOR.post_processing(bboxes, data[0])
        #
        for img_w_box in imgs_w_box:
            cv2.imshow('detected faces', cv2.cvtColor(img_w_box, cv2.COLOR_BGR2RGB))
            cv2.waitKey(int(1000/30))
    #
    cv2.destroyAllWindow()


