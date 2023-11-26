#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import copy
import time
import sys
import os
import torch
import torch.nn as nn
import onnx

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.models.yolo import *
from yolov6.models.effidehead import Detect
from yolov6.layers.common import *
from yolov6.utils.events import LOGGER
from yolov6.utils.checkpoint import load_checkpoint
from io import BytesIO


# --- ONNX RUNTIME VERSION ---
# python onnx_export.py --end2end --simplify --topk-all 1000 --iou-thres 0.3 --conf-thres 0.35 --img-size 640 640 --dynamic-batch --ort --weights '/storage/jysuh/YOLOv6_logdir/6s_hs_SW_960/weights/coco_150.pt'
# python onnx_export.py --end2end --simplify --topk-all 1000 --iou-thres 0.3 --conf-thres 0.35 --img-size 640 640 --weights ./runs/train/6s_face2/weights/best_ckpt.pt
# ----------------------------

# --- TensorRT VERSION ---
# python onnx_export.py --end2end --batch 1 --simplify --topk-all 1000 --iou-thres 0.3 --conf-thres 0.35 \
#                       --img-size 640 640 \
#                       --trt-version 8 \
#                       --weights ./runs/train/6s_face2/weights/best_ckpt.pt
# ----------------------------


def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/storage/jysuh/YOLOv6_logdir/6s_hs_SW_960/weights/coco_150.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size, the order is: height width')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set Detect() inplace=True')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--dynamic-batch', action='store_true', help='export dynamic batch onnx model')
    parser.add_argument('--end2end', action='store_true', help='export end2end onnx')
    parser.add_argument('--trt-version', type=int, default=8, help='tensorrt version')
    parser.add_argument('--ort', action='store_true', help='export onnx for onnxruntime')
    parser.add_argument('--with-preprocess', action='store_true', help='export bgr2rgb and normalize')
    parser.add_argument('--topk-all', type=int, default=100, help='topk objects for every images')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='iou threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='conf threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    args.img_size *= 2 if len(args.img_size) == 1 else 1  # expand
    print(args)
    return args


def to_numpy(tensor):
    return tensor.detach().float().cpu().numpy() if tensor.requires_grad else tensor.float().cpu().numpy()


if __name__ == '__main__':
    #
    t = time.time()

    # Get configuration parameters
    args = get_arg()

    # Check device
    cuda = args.device != 'cpu' and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.device}' if cuda else 'cpu')
    assert not (device.type == 'cpu' and args.half), '--half only compatible with GPU export, i.e. use --device 0'

    # Load PyTorch model
    # load FP32 (float precision) model
    model = load_checkpoint(args.weights, map_location=device, inplace=True, fuse=True)
    for layer in model.modules():
        if isinstance(layer, RepVGGBlock):
            layer.switch_to_deploy()

    # Input
    img = torch.zeros(args.batch_size, 3, *args.img_size).to(device)

    # Update model
    # --- to FP16 (float precision)
    if args.half:
        img, model = img.half(), model.half()
    model.eval()
    # --- assign export-friendly activations
    for k, m in model.named_modules():
        if isinstance(m, ConvModule):
            if hasattr(m, 'act') and isinstance(m.act, nn.SiLU):
                m.act = SiLU()
            elif isinstance(m, Detect):
                m.inplace = args.inplace

    # Set dynamic axes
    dynamic_axes = None
    if args.dynamic_batch:
        args.batch_size = 'batch'
        dynamic_axes = {
            'images': {0: 'batch'},
        }
        if args.end2end:
            output_axes = {
                'num_dets': {0: 'batch'},
                'det_boxes': {0: 'batch'},
                'det_scores': {0: 'batch'},
                'det_classes': {0: 'batch'},
            }
        else:
            output_axes = {
                'outputs': {0: 'batch'},
            }
        dynamic_axes.update(output_axes)

    if args.end2end:
        from yolov6.models.end2end import End2End

        model = End2End(model,
                        max_obj=args.topk_all, iou_thres=args.iou_thres, score_thres=args.conf_thres,
                        device=device,
                        ort=args.ort, trt_version=args.trt_version,
                        with_preprocess=args.with_preprocess)

    print("=============")
    print(model)
    print("=============")

    # ONNX export
    x = torch.randn(1, 3, 640, 640, requires_grad=True).to(device)
    if args.half:
        x = x.half()
        model = model.half()
        in_torch = torch.tensor(to_numpy(x)).half().to(device)
    else:
        in_torch = x
    y = model(in_torch)
    # y[0] - num_dets:      [1, 1]      : means the number of object in every image in its batch .
    # y[1] - det_boxes:     [1, 100, 4] : means topk(100) object's location about [x0,y0,x1,y1] .
    # y[2] - det_scores:    [1, 100]    : means the confidence score of every topk(100) objects .
    # y[3] - det_classes:   [1, 100]    : means the category of every topk(100) objects .
    try:
        LOGGER.info('\nStarting to export ONNX...')
        export_file = args.weights.replace('.pt', '.onnx')
        torch.onnx.export(model, img, export_file,
                          verbose=False, opset_version=12,
                          export_params=True,  # save model's weights in the saved file or not
                          training=torch.onnx.TrainingMode.EVAL,  # export the model in inference mode
                          do_constant_folding=True,
                          input_names=['images'],
                          output_names=['num_dets', 'det_boxes', 'det_scores', 'det_classes'] if args.end2end else [
                              'outputs'],
                          dynamic_axes=dynamic_axes)

        # Check
        onnx_model = onnx.load(export_file)
        onnx.checker.check_model(onnx_model)
        # Fix output shape
        if args.end2end and not args.ort:
            shapes = [args.batch_size, 1, args.batch_size, args.topk_all, 4,
                      args.batch_size, args.topk_all, args.batch_size, args.topk_all]
            for i in onnx_model.graph.output:
                for j in i.type.tensor_type.shape.dim:
                    j.dim_param = str(shapes.pop(0))

        if args.simplify:
            try:
                import onnxsim

                LOGGER.info('\nStarting to simplify ONNX...')
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, 'assert check failed'
            except Exception as e:
                LOGGER.info(f'Simplifier failure: {e}')

        onnx.save(onnx_model, export_file)
        LOGGER.info(f'ONNX export success, saved as {export_file}')
    except Exception as e:
        LOGGER.info(f'ONNX export failure: {e}')

    # Finish
    LOGGER.info('\nExport complete (%.2fs)' % (time.time() - t))
    if args.end2end:
        if not args.ort:
            info = f'trtexec --onnx={export_file} --saveEngine={export_file.replace(".onnx", ".engine")}'
            if args.dynamic_batch:
                LOGGER.info('Dynamic batch export should define min/opt/max batchsize\n' +
                            'We set min/opt/max = 1/16/32 default!')
                wandh = 'x'.join(list(map(str, args.img_size)))
                info += (f' --minShapes=images:1x3x{wandh}' +
                         f' --optShapes=images:16x3x{wandh}' +
                         f' --maxShapes=images:32x3x{wandh}' +
                         f' --shapes=images:16x3x{wandh}')
            LOGGER.info('\nYou can export tensorrt engine use trtexec tools.\nCommand is:')
            LOGGER.info(info)
