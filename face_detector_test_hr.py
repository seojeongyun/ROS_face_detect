# ==== Library ====
import torch
from yolov6.core.demo import demo

# ==== CONFIGURATION ====
HALF = True
MAX_BATCH_SIZE = 50
RESIZE_SIZE = 320
ONNX = False
# WEIGHT_PATH = "demo_weight/6s_hs_960_sc.onnx" if ONNX else "demo_weight/6s_hs_960.pt"
# WEIGHT_PATH = "demo_weight/6s_hs_960_sc.onnx" if ONNX else "/storage/jysuh/YOLOv6_weights/640/6s_hs_SW_640/weights/last_ckpt.pt"
WEIGHT_PATH = "demo_weight/6s_hs_960_sc.onnx" if ONNX else "./runs/train/6n/weights/coco_150.pt" #map50 :4.6532098337616256e-05
CONF_THRES = 0.3
IOU_THRES = 0.65
NMS_DET = 300
GPU_ID = '0'
DEVICE = torch.device(f'cuda:{GPU_ID}') if GPU_ID != '' else torch.device('cpu')
VIEW = False
NET_TYPE = '6n'  # [6s, 6s-ours, 6n]

# ==== NETWORK_MODULE ====
FACE_DETECTOR = demo(WEIGHT_PATH, RESIZE_SIZE, DEVICE, ONNX,
                     CONF_THRES, IOU_THRES, NMS_DET, NET_TYPE)

if __name__ == '__main__':
    # Check the RESIZE_SIZE !

    from yolov6.data.data_load import create_dataloader
    from tqdm import tqdm
    import cv2
    from yolov6.utils.fn_for_test import *
    from setproctitle import *

    setproctitle('YOLOv6 : map50')
    #
    demo_loader = create_dataloader('/storage/hrlee/widerface/images/val',
                                    RESIZE_SIZE, MAX_BATCH_SIZE, 32,
                                    pad=0, workers=0, check_images=False,
                                    check_labels=False,
                                    data_dict={'anno_path': '/storage/hrlee/widerface/annotations/instances_val.json',
                                               'is_coco': False, 'nc': 1, 'names': ['traffic']},
                                    task='val')

    #
    cv2.namedWindow('detected faces', flags=cv2.WINDOW_AUTOSIZE)
    #
    stats = dict(stats=[], seen=0)
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
                # cv2.waitKey(0)

        # ===== Get Validation Metrics =====
        # Make shape for COCO mAP rescaling
        shapes = []
        for batch_idx in range(len(imgs)):
            temp = (ori_size[batch_idx][0], ori_size[batch_idx][1]), (
                (resized_size[batch_idx][0] * ratios[batch_idx] / ori_size[batch_idx][0],
                 resized_size[batch_idx][1] * ratios[batch_idx] / ori_size[batch_idx][1]), pads[batch_idx])
            shapes.append(temp)

        # There is a bug in the code loading labels...
        targets = data[1]
        targets[:, 2] = targets[:, 2] + targets[:, 4] / 2
        targets[:, 3] = targets[:, 3] + targets[:, 5] / 2
        #
        imgs = np.stack(imgs, axis=0).transpose(0, 3, 1, 2)
        buffer_statistics(imgs, out_boxes, targets, shapes, stats)
    #
    # ===== Calculate pr metrics =====
    #
    import os
    save_dir = 'test_result/{}/'.format(WEIGHT_PATH.split('/')[-1].split('.')[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    calc_pr_metrics(stats, save_dir=save_dir)
    #
    cv2.destroyAllWindows()
