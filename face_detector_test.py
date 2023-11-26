# ==== Library ====
import torch
from yolov6.core.demo import demo

# ==== CONFIGURATION ====
HALF = True
MAX_BATCH_SIZE = 30
RESIZE_SIZE = 960
ONNX = False
WEIGHT_PATH = "demo_weight/6s_hs_960_sc.onnx" if ONNX else "demo_weight/6s_hs_960_cw.pt"
CONF_THRES = 0.7
IOU_THRES = 0.8
NMS_DET = 1000
GPU_ID = '0'
DEVICE = torch.device(f'cuda:{GPU_ID}') if GPU_ID != '' else torch.device('cpu')
VIEW = False

# ==== NETWORK_MODULE ====
FACE_DETECTOR = demo(WEIGHT_PATH, RESIZE_SIZE, DEVICE, ONNX,
                     CONF_THRES, IOU_THRES, NMS_DET)

if __name__ == '__main__':
    from yolov6.data.data_load import create_dataloader
    from tqdm import tqdm
    import cv2

    #
    demo_loader = create_dataloader('/storage/hrlee/widerface/images/val',
                                    RESIZE_SIZE, MAX_BATCH_SIZE, 32,
                                    pad=0.5, workers=0, check_images=False,
                                    check_labels=False,
                                    data_dict={'anno_path': '/storage/hrlee/widerface/annotations/instances_val.json',
                                               'is_coco': False, 'nc': 1, 'names': ['traffic']},
                                    task='val')

    #
    cv2.namedWindow('detected faces', flags=cv2.WINDOW_AUTOSIZE)
    #
    for step, data in tqdm(enumerate(demo_loader), total=len(demo_loader)):
        #
        imgs = list(data[0].permute(0, 2, 3, 1).numpy())
        #
        bboxes, out_boxes, ori_size, resized_size, ratios, pads = FACE_DETECTOR.run(imgs)
        #
        if VIEW:
            imgs_w_box = FACE_DETECTOR.post_processing(bboxes, imgs)
            for img_w_box in imgs_w_box:
                cv2.imshow('detected faces', cv2.cvtColor(img_w_box, cv2.COLOR_BGR2RGB))
                cv2.waitKey(int(30000 / 30))
        #
        # ===== Get Validation Metrics =====
        # Make shape for COCO mAP rescaling
        #
    #
    # ===== Calculate pr metrics =====
    #
    cv2.destroyAllWindow()
