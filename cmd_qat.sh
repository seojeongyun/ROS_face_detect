#!/bin/bash

clear

# ----- YOLOv6 RepOpt - Hyperparameter search -----
# For yolov6s
# --- COMMAND IN TERMINAL ---
# python core_train.py --batch-size 16 --data-path ./data/WIDER_FACE.yaml --conf-file ./configs/repopt/yolov6s_hs.py --output-dir ./runs/train --name 6s_hs --gpu-id 1 --worker 8
# -------------------------------------------------

# ----- YOLOv6 RepOpt - Target network training -----
# For yolov6s
# Note that the file path of a weight file obtained by hyperparameter search is included in the configuration file, "./config/repopt/yolov6s_opt.py".
# --- COMMAND IN TERMINAL ---
# python core_train.py --batch-size 4 --data-path ./data/WIDER_FACE.yaml --conf-file ./configs/repopt/yolov6s_opt.py --output-dir ./runs/train --name 6s_opt --gpu-id 0 --worker 8
# -------------------------------------------------


# ----- YOLOv6 PTQ -----
# For yolov6s
# In configuration file, yolov6s_opt_qat.py, the dictionary ptq has a key, "calib_output_path", which is the save folder for a weight quantized by PTQ.
# --- COMMAND IN TERMINAL ---
# python core_train.py --batch-size 16 --data-path ./data/WIDER_FACE.yaml --conf-file ./configs/repopt/yolov6s_opt_qat.py --output-dir ./weights/train_v6s_ptq --quant --calib --worker 0
# -------------------------------------------------


# ----- YOLOv6 QAT -----
# For yolov6s
# --- COMMAND IN TERMINAL ---
python train.py --batch-size 8 --data-path ./data/WIDER_FACE.yaml --conf-file ./configs/repopt/yolov6s_opt_qat.py --output-dir ./weights/train_v6s_qat --quant --distill --distill_feat --epochs 10 --workers 8 --teacher_model_path ./weights/opt/yolov6s_opt.pt --gpu-id 1 
# -------------------------------------------------

