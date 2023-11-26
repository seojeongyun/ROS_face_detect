#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import os

import torch
import yaml
import os.path as osp
from pathlib import Path

from yolov6.core.engine import Trainer
from yolov6.utils.config import Config
from yolov6.utils.events import LOGGER, save_yaml
from yolov6.utils.envs import select_device
from yolov6.utils.general import increment_name, find_latest_checkpoint, check_img_size
from cmd_in import get_args_parser

## python core_train.py --batch-size 4 --conf-file ./configs/repopt/yolov6s_hs.py --img-size 416 --check-images --check-labels --output-dir ./runs/train --name 6s_hs_CW_416 --gpu-id 0 --fuse_ab
# check resume in ./cmd_in.py
# check attention, use_se in ./configs/repopt/yolov6s_hs.py
# check set_device in  ./core_train.py
# check setproctitle in ./core_train.py

def check_and_init(args):
    '''check config files.'''
    # img size additionally is added
    args.img_size = check_img_size(args.img_size, 32, floor=256)

    # Check files
    checkpoint_path = args.resume
    gpu_id = args.gpu_id
    if checkpoint_path:
        # args.resume can be a checkpoint file path or a boolean value.
        assert isinstance(checkpoint_path, str) and os.path.isfile(checkpoint_path), f'Resume should be string...{args.resume} and the checkpoint path should exist {checkpoint_path}'
        LOGGER.info(f'Resume training from the checkpoint file :{checkpoint_path}')
        # Ex)
        # checkpoint_path = '/home/hrlee/PycharmProjects/YOLOv6/runs/train/exp1/weights/last_a.pt'
        # Path(checkpoint_path).parent.parent = '/home/hrlee/PycharmProjects/YOLOv6/runs/train/exp1/'
        resume_opt_file_path = Path(checkpoint_path).parent.parent / 'args.yaml'
        if osp.exists(resume_opt_file_path):
            with open(resume_opt_file_path) as f:
                args = argparse.Namespace(**yaml.safe_load(f))  # load args value from args.yaml
                args.gpu_id = gpu_id
                args.resume = checkpoint_path
        else:
            LOGGER.warning(f'We can not find the path of {Path(checkpoint_path).parent.parent / "args.yaml"},' \
                           f' we will save exp log to {Path(checkpoint_path).parent.parent}')
            LOGGER.warning(f'In this case, make sure to provide configuration, such as data, batch size.')
            args.save_dir = str(Path(checkpoint_path).parent.parent)
    else:
        args.save_dir = str(increment_name(osp.join(args.output_dir, args.name)))
        os.makedirs(args.save_dir)

    # Load configuration file
    cfg = Config.fromfile(args.conf_file)
    if not hasattr(cfg, 'training_mode'):
        setattr(cfg, 'training_mode', 'repvgg')

    # Save args
    save_yaml(vars(args), osp.join(args.save_dir, 'args.yaml'))

    return cfg, args


if __name__ == '__main__':
    torch.cuda.set_device(1)

    args = get_args_parser().parse_args()

    from setproctitle import *
    setproctitle('ours_320 : None')
    # Setup
    cfg, args = check_and_init(args)
    LOGGER.info(f'training args are: {args}\n')

    # device
    # device = torch.device('cuda:{}'.format(args.gpu_id)) if args.gpu_id is not None else torch.device('cpu')
    device = torch.device('cpu')
    # Get trainer
    trainer = Trainer(args, cfg, device)

    # PTQ
    if args.quant and args.calib:
        trainer.calibrate(cfg)
        exit()

    # Start training
    trainer.train()
