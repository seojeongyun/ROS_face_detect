#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import time
from copy import deepcopy
import os.path as osp

from tqdm import tqdm

import cv2
import numpy as np
import math
import torch
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter

import core_eval as eval
from yolov6.data.data_load import create_dataloader
from yolov6.models.yolo import build_model
from yolov6.models.yolo_lite import build_model as build_lite_model

from yolov6.models.losses.loss import ComputeLoss as ComputeLoss
from yolov6.models.losses.loss_fuseab import ComputeLoss as ComputeLoss_ab
from yolov6.models.losses.loss_distill import ComputeLoss as ComputeLoss_distill
from yolov6.models.losses.loss_distill_ns import ComputeLoss as ComputeLoss_distill_ns

from yolov6.utils.events import LOGGER, NCOLS, load_yaml, write_tblog, write_tbimg, write_tbloss
from yolov6.utils.ema_single import ModelEMA
from yolov6.utils.checkpoint import load_state_dict, strip_optimizer
from yolov6.solver.build import build_optimizer, build_lr_scheduler
from yolov6.utils.RepOptimizer import extract_scales, RepVGGOptimizer
from yolov6.utils.nms import xywh2xyxy


class Trainer:
    def __init__(self, args, cfg, device):
        self.pretrained = False
        #
        self.args = args
        self.cfg = cfg
        self.device = device
        self.save_dir = args.save_dir

        # ===== Configuration of Data Loader =====
        self.data_dict = load_yaml(args.data_path)
        self.num_classes = self.data_dict['nc']

        # ===== Model =====
        # self.distill_ns = True if self.args.distill and self.cfg.model.type in ['YOLOv6n', 'YOLOv6s'] else False
        self.distill_ns = False
        self.model = self.get_model(args, cfg, self.num_classes, device)
        self.model.nc, self.model.names = self.data_dict['nc'], self.data_dict['names']

        # ===== Optimizer =====
        if cfg.training_mode == 'repopt_':
            scales = self.load_scale_from_pretrained_models(cfg, device)
            reinit = False if cfg.model.pretrained is not None else True
            self.optimizer = RepVGGOptimizer(self.model, scales, args, cfg, reinit=reinit, device=self.device)
        else:
            self.optimizer = self.get_optimizer(args, cfg, self.model)

        # ===== Distill =====
        if self.args.distill:
            if self.args.fuse_ab:
                LOGGER.error('ERROR in: Distill models should turn off the fuse_ab.\n')
                exit()
            self.teacher_model = self.get_teacher_model(args, cfg, self.num_classes, device)

        # ===== Quantization =====
        if self.args.quant:
            self.quant_setup(self.model, cfg, device)

        # ===== Scheduler =====
        self.scheduler, self.lf = self.get_lr_scheduler(args, cfg, self.optimizer)

        # ===== EMA =====
        # self.ema is like a container that maintains moving-averaged trained weights
        self.ema = ModelEMA(self.model)

        # ===== tensorboard =====
        self.tblogger = SummaryWriter(self.save_dir)
        self.start_epoch = 0

        # ===== resume =====
        if not self.pretrained:
            if args.resume:
                self.ckpt = torch.load(args.resume, map_location='cpu')
            if hasattr(self, "ckpt"):
                resume_state_dict = self.ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
                self.model.load_state_dict(resume_state_dict, strict=True)  # load
                self.start_epoch = self.ckpt['epoch'] + 1
                self.optimizer.load_state_dict(self.ckpt['optimizer'])
                self.ema.ema.load_state_dict(self.ckpt['ema'].float().state_dict())
                self.ema.updates = self.ckpt['updates']

        elif self.pretrained:
            if args.resume:
                self.ckpt = torch.load(args.resume, map_location='cpu')

            if hasattr(self, "ckpt"):
                new_model_dict = {}

                resume_state_dict = self.ckpt['model'].float().state_dict()
                for key, value in resume_state_dict.items():
                    if key.count('detect.cls_preds') != 0:
                        pass
                    elif key.count('detect.reg_preds') != 0:
                        pass
                    else:
                        new_model_dict[key] = value

                self.model.load_state_dict(new_model_dict, strict=False)  # load

        else:
            raise print('not implmented')

        # ===== DataLoader =====
        self.img_size = args.img_size
        self.train_loader, self.val_loader = self.get_data_loader(self.args, self.cfg, self.data_dict)

        # set loss
        self.loss_num = 6
        self.loss_info = ['Epoch', 'iou_loss', 'dfl_loss', 'cls_loss', 'rgt_loss', 'rbox_loss', 'lk_loss']

        if self.args.distill:
            self.loss_num += 1
            self.loss_info += ['cwd_loss']
        self.set_criterion()

        #
        self.max_epoch = args.epochs
        self.max_stepnum = len(self.train_loader)
        self.batch_size = args.batch_size
        self.vis_imgs_list = []
        self.write_trainbatch_tb = args.write_trainbatch_tb

        # set color for classnames
        if args.data_path.split('/')[-1].split('.')[0] == 'WIDER_FACE':
            self.color = [(24, 252, 0)]
        else:
            self.color = [tuple(np.random.choice(range(256), size=3)) for _ in range(self.model.nc)]

    def set_criterion(self):
        self.compute_loss = ComputeLoss(num_classes=self.data_dict['nc'],
                                        ori_img_size=self.img_size,
                                        warmup_epoch=self.cfg.model.head.atss_warmup_epoch,
                                        use_dfl=self.cfg.model.head.use_dfl,
                                        reg_max=self.cfg.model.head.reg_max,
                                        iou_type=self.cfg.model.head.iou_type,
                                        fpn_strides=self.cfg.model.head.strides,
                                        device=self.device)

        if self.args.fuse_ab:
            self.compute_loss_ab = ComputeLoss_ab(num_classes=self.data_dict['nc'],
                                                  ori_img_size=self.img_size,
                                                  warmup_epoch=0,
                                                  use_dfl=False,
                                                  reg_max=0,
                                                  iou_type=self.cfg.model.head.iou_type,
                                                  fpn_strides=self.cfg.model.head.strides,
                                                  device=self.device)
        if self.args.distill:
            if self.cfg.model.type in ['YOLOv6n', 'YOLOv6s']:
                # Loss_distill_func = ComputeLoss_distill_ns
                Loss_distill_func = ComputeLoss_distill
            else:
                Loss_distill_func = ComputeLoss_distill

            self.compute_loss_distill = Loss_distill_func(num_classes=self.data_dict['nc'],
                                                          ori_img_size=self.img_size,
                                                          fpn_strides=self.cfg.model.head.strides,
                                                          warmup_epoch=self.cfg.model.head.atss_warmup_epoch,
                                                          use_dfl=self.cfg.model.head.use_dfl,
                                                          reg_max=self.cfg.model.head.reg_max,
                                                          iou_type=self.cfg.model.head.iou_type,
                                                          distill_weight=self.cfg.model.head.distill_weight,
                                                          distill_feat=self.args.distill_feat,
                                                          device=self.device
                                                          )

    def get_model(self, args, cfg, nc, device):
        if 'YOLOv6-lite' in cfg.model.type:
            assert not args.fuse_ab, 'ERROR in: YOLOv6-lite models not support fuse_ab mode.'
            assert not args.distill, 'ERROR in: YOLOv6-lite models not support distill mode.'
            model = build_lite_model(cfg, nc, device)
        else:
            model = build_model(cfg, nc, device, fuse_ab=args.fuse_ab, distill_ns=self.distill_ns, use_se=cfg.model.backbone.use_se, attention=cfg.model.backbone.attention)
        weights = cfg.model.pretrained
        if weights:  # finetune if pretrained model is set
            if not os.path.exists(weights):
                LOGGER.error(f'There is no weight {weights} for fine-tuning...')
                raise
            LOGGER.info(f'Loading state_dict from {weights} for fine-tuning...')
            model = load_state_dict(weights, model, map_location=device)
        LOGGER.info('Model: {}'.format(model))
        return model

    @staticmethod
    def load_scale_from_pretrained_models(cfg, device):
        weights = cfg.model.scales
        scales = None
        if not weights:
            LOGGER.error("ERROR: No scales provided to init RepOptimizer!")
        else:
            ckpt = torch.load(weights, map_location=device)
            scales = extract_scales(ckpt)
        return scales

    def get_optimizer(self, args, cfg, model):
        accumulate = max(1, round(64 / args.batch_size))
        # If args.batch_size < 64, (args.batch_size * accumulate / 64) ~= 1.
        # If args.batch_size > 64, (args.batch_size * accumulate / 64) ~= (args.batch_size / 64)
        # Hence, If batch_size is larger than 64, weight_decay becomes larger.
        cfg.solver.weight_decay *= args.batch_size * accumulate / 64
        # cfg.solver.lr0 *= args.batch_size / (self.world_size * args.bs_per_gpu)
        optimizer = build_optimizer(cfg, model)
        return optimizer

    def get_teacher_model(self, args, cfg, nc, device):
        teacher_fuse_ab = False if cfg.model.head.num_layers != 3 or not args.fuse_ab else True
        model = build_model(cfg, nc, device, fuse_ab=teacher_fuse_ab)
        weights = args.teacher_model_path
        if weights == args.teacher_model_path:
            LOGGER.info(f'loading state_dict from {weights} for teacher')
            model = load_state_dict(weights, model, map_location=device)
        LOGGER.info('Model: {}'.format(model))
        # Do not update running means and running vars
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.track_running_stats = False
        return model

    @staticmethod
    def get_lr_scheduler(args, cfg, optimizer):
        epochs = args.epochs
        lr_scheduler, lf = build_lr_scheduler(cfg, optimizer, epochs)
        return lr_scheduler, lf

    @staticmethod
    def get_data_loader(args, cfg, data_dict):
        train_path, val_path = data_dict['train'], data_dict['val']
        # check data
        nc = int(data_dict['nc'])
        class_names = data_dict['names']
        assert len(class_names) == nc, f'the length of class names does not match the number of classes defined'
        grid_size = max(int(max(cfg.model.head.strides)), 32)
        # create train dataloader
        train_loader = create_dataloader(train_path, args.img_size, args.batch_size, grid_size,
                                         hyp=dict(cfg.data_aug), augment=True,
                                         workers=args.workers, shuffle=True, check_images=args.check_images,
                                         check_labels=args.check_labels, data_dict=data_dict, task='train')
        # create val dataloader
        val_loader = create_dataloader(val_path, args.img_size, args.batch_size, grid_size,
                                       hyp=dict(cfg.data_aug), pad=0.5,
                                       workers=args.workers, check_images=args.check_images,
                                       check_labels=args.check_labels, data_dict=data_dict, task='val')

        return train_loader, val_loader

    def update_data_loader(self):
        # stop strong aug like mosaic and mixup from last n epoch by recreate dataloader
        if self.epoch == self.max_epoch - self.args.stop_aug_last_n_epoch:
            self.cfg.data_aug.mosaic = 0.0
            self.cfg.data_aug.mixup = 0.0
            # ==> NO DATA AUGMENTATION
            self.train_loader, self.val_loader = self.get_data_loader(self.args, self.cfg, self.data_dict)
        else:
            # Use previous data loader
            pass

    @staticmethod
    def prepro_data(batch_data, device):
        images = batch_data[0].to(device, non_blocking=True).float() / 255
        targets = batch_data[1].to(device)
        return images, targets

    def train(self):
        # Initialization of parameters for training
        self.start_time = time.time()
        self.warmup_stepnum = max(round(self.cfg.solver.warmup_epochs * self.max_stepnum),
                                  1000) if self.args.quant is False else 0
        self.scheduler.last_epoch = self.start_epoch - 1
        self.last_opt_step = -1
        self.scaler = amp.GradScaler(enabled=self.device != 'cpu')

        self.best_ap, self.ap = 0.0, 0.0
        self.best_stop_strong_aug_ap = 0.0  # strong augmentation like mosaic and mixup
        self.evaluate_results = (0, 0)


        try:
            LOGGER.info(f'Training start...')
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.train_one_epoch(self.epoch)
            LOGGER.info(f'\nTraining completed in {(time.time() - self.start_time) / 3600:.3f} hours.')

            # strip optimizers for saved pt model
            LOGGER.info(f'Strip optimizer from the saved pt model...')
            strip_optimizer(osp.join(self.save_dir, 'weights'), self.epoch)
        except Exception as _:
            LOGGER.error('ERROR in training loop or eval/save model.')
            raise
        finally:
            if self.device != 'cpu':
                torch.cuda.empty_cache()

    def train_one_epoch(self, epoch_num):
        try:
            self.mean_loss = torch.zeros(self.loss_num, device=self.device)
            #
            self.update_data_loader()
            self.pbar = tqdm(enumerate(self.train_loader), total=self.max_stepnum, ncols=NCOLS,
                             bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            for self.step, self.batch_data in self.pbar:
                # Training mode
                self.model.train()

                # Data and label
                images = self.batch_data[0].to(self.device, non_blocking=True).float() / 255
                targets = self.batch_data[1].to(self.device)

                with amp.autocast(enabled=self.device != 'cpu'):
                    # Forward
                    preds, s_featmaps = self.model(images)
                    # Calculate Loss
                    if self.args.distill:
                        with torch.no_grad():
                            t_preds, t_featmaps = self.teacher_model(images)
                        total_loss, loss_items = self.compute_loss_distill(
                            outputs=preds, t_outputs=t_preds, s_featmaps=s_featmaps, t_featmaps=t_featmaps,
                            targets=targets, epoch_num=epoch_num, max_epoch=self.max_epoch,
                            temperature=self.args.temperature, step_num=self.step
                        )
                    elif self.args.fuse_ab:
                        # preds[0]: model_output
                        # preds[1]: cls_score_ab, preds[2]: reg_dist_ab, (ab: anchor based)
                        # preds[3]: cls_score, preds[4]: reg_dist (af: anchor free)
                        total_loss, loss_items = self.compute_loss((preds[0], preds[3], preds[4]), targets, epoch_num, self.step)
                        total_loss_ab, loss_items_ab = self.compute_loss_ab(preds[:3], targets, epoch_num, self.step)
                        total_loss += total_loss_ab
                        loss_items += loss_items_ab
                    else:
                        total_loss, loss_items = self.compute_loss(preds, targets, epoch_num, self.step)  # YOLOv6_af

                # backward and update
                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.update_optimizer()

                # Print losses
                self.loss_items = loss_items
                self.mean_loss = (self.mean_loss * self.step + self.loss_items) / (self.step + 1)

                #
                if ((self.epoch * self.max_stepnum) + self.step) % 20 == 0:
                    LOGGER.info(('\n\n' + '%10s' * (self.loss_num + 1)) % (*self.loss_info,))
                    self.pbar.set_description(('%10s' + '%10.4g' * self.loss_num)
                                              % (f'{self.epoch + 1}/{self.max_epoch}', *(self.mean_loss)))
                    write_tbloss(self.tblogger, losses=self.loss_items,
                                 step=(self.epoch * self.max_stepnum) + self.step)

                # plot train_batch and save to tensorboard once an epoch
                if self.write_trainbatch_tb and self.step == 0:
                    vis_train_batch = self.plot_train_batch(images, targets)
                    write_tbimg(self.tblogger, imgs=vis_train_batch,
                                step=(self.step + self.max_stepnum * self.epoch),
                                type='train')
            # Update the learning rate of scheduler
            self.scheduler.step()

            #
        except Exception as _:
            LOGGER.error('ERROR in training steps')
            raise
        #
        try:
            # --- After one epoch ---
            # Evaluate the model
            # self.eval()
            # Save the model
            self.save()
        except Exception as _:
            LOGGER.error('ERROR in evaluate and save model.')
            raise

    def update_optimizer(self):
        curr_step = self.step + self.max_stepnum * self.epoch
        self.accumulate = max(1, round(64 / self.batch_size))
        if curr_step <= self.warmup_stepnum:
            self.accumulate = max(1, np.interp(curr_step, [0, self.warmup_stepnum], [1, 64 / self.batch_size]).round())
            for k, param in enumerate(self.optimizer.param_groups):
                warmup_bias_lr = self.cfg.solver.warmup_bias_lr if k == 2 else 0.0
                param['lr'] = np.interp(curr_step, [0, self.warmup_stepnum],
                                        [warmup_bias_lr, param['initial_lr'] * self.lf(self.epoch)])
                if 'momentum' in param:
                    param['momentum'] = np.interp(curr_step, [0, self.warmup_stepnum],
                                                  [self.cfg.solver.warmup_momentum, self.cfg.solver.momentum])
        if curr_step - self.last_opt_step >= self.accumulate:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.ema:
                self.ema.update(self.model)
            self.last_opt_step = curr_step

    # QAT
    def quant_setup(self, model, cfg, device):
        if self.args.quant:
            from tools.qat.qat_utils import qat_init_model_manu, skip_sensitive_layers
            qat_init_model_manu(model, cfg, self.args)
            # workaround
            model.neck.upsample_enable_quant(cfg.ptq.num_bits, cfg.ptq.calib_method)
            if self.args.calib is False:
                if cfg.qat.sensitive_layers_skip:
                    skip_sensitive_layers(model, cfg.qat.sensitive_layers_list)
                # QAT flow load calibrated model
                assert cfg.qat.calib_pt is not None, 'Please provide calibrated model'
                model.load_state_dict(torch.load(cfg.qat.calib_pt)['model'].float().state_dict())
            model.to(device)

    # PTQ
    def calibrate(self, cfg):
        def save_calib_model(model, cfg):
            # Save calibrated checkpoint
            output_model_path = os.path.join(cfg.ptq.calib_output_path, '{}_calib_{}.pt'.
                                             format(os.path.splitext(os.path.basename(cfg.model.pretrained))[0],
                                                    cfg.ptq.calib_method))
            if cfg.ptq.sensitive_layers_skip is True:
                output_model_path = output_model_path.replace('.pt', '_partial.pt')
            LOGGER.info('Saving calibrated model to {}... '.format(output_model_path))
            if not os.path.exists(cfg.ptq.calib_output_path):
                os.mkdir(cfg.ptq.calib_output_path)
            torch.save({'model': deepcopy(model).half()}, output_model_path)

        assert self.args.quant is True and self.args.calib is True
        from tools.qat.qat_utils import ptq_calibrate
        ptq_calibrate(self.model, self.train_loader, cfg)
        self.epoch = 0
        self.eval_model()
        write_tbimg(self.tblogger, self.vis_imgs_list, self.epoch, type='val')
        save_calib_model(self.model, cfg)

    def eval(self):
        # update attributes for ema model
        self.ema.update_attr(self.model, include=['nc', 'names', 'stride'])

        # self.epoch is start from 0
        remaining_epochs = (self.max_epoch - 1) - self.epoch

        # evaluating every epoch for last such epochs
        # If self.args.heavy_eval_range == 50, the model will be evaluated per 3 epochs during the last 50 epochs.
        eval_interval = self.args.eval_interval
        if remaining_epochs < self.args.heavy_eval_range:
            eval_interval = 3

        # If self.args.eval_final_only == True, evaluation will be conducted at the last epoch
        is_val_epoch = (remaining_epochs == 0) or \
                       ((not self.args.eval_final_only) and ((self.epoch + 1) % eval_interval == 0))
        if is_val_epoch:
            self.eval_model()
            self.ap = self.evaluate_results[1]
            self.best_ap = max(self.ap, self.best_ap)

        if self.epoch >= self.max_epoch - self.args.stop_aug_last_n_epoch:
            # self.best_stop_strong_aug_ap is the best ap value during stop_aug_last_n_epoch
            if self.best_stop_strong_aug_ap < self.ap:
                self.best_stop_strong_aug_ap = max(self.ap, self.best_stop_strong_aug_ap)

        # log for learning rate
        lr = [x['lr'] for x in self.optimizer.param_groups]
        self.evaluate_results = list(self.evaluate_results) + lr

        # log for tensorboard
        write_tblog(self.tblogger, self.epoch, self.evaluate_results, self.mean_loss)

        # save validation predictions to tensorboard
        write_tbimg(self.tblogger, self.vis_imgs_list, self.epoch, type='val')

    def save(self):
        # Check save directory
        save_ckpt_dir = osp.join(self.save_dir, 'weights')
        if not osp.exists(save_ckpt_dir):
            os.makedirs(save_ckpt_dir)

        # Make checkpoint dictionary
        ckpt = {
            'model': deepcopy(self.model).half(),
            'ema': deepcopy(self.ema.ema).half(),
            'updates': self.ema.updates,
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
        }

        # Save last checkpoint
        # In the folder, run/train/exp/wegiths, last and best ckpts are saved.
        torch.save(ckpt, osp.join(save_ckpt_dir, 'coco_150.pt'))

        # Save best checkpoint
        if self.ap == self.best_ap:
            # The case that this weight is best
            # ap is average precision, which is a performance metric
            torch.save(ckpt, osp.join(save_ckpt_dir, 'best_ckpt.pt'))

        # Save best ap ckpt in stop strong aug epochs
        #                      (Stop strong data augmentation such as mosaic during last n epoch)
        if (self.epoch >= self.max_epoch - self.args.stop_aug_last_n_epoch) \
                and (self.best_stop_strong_aug_ap == self.ap):
            torch.save(ckpt, osp.join(save_ckpt_dir, 'best_stop_aug_ckpt.pt'))

        # Save checkpoints during last n epoch even not best or last
        if self.epoch >= self.max_epoch - self.args.save_ckpt_on_last_n_epoch:
            torch.save(ckpt, osp.join(save_ckpt_dir, f'{self.epoch}_ckpt.pt'))

        del ckpt

    def eval_model(self):
        if not hasattr(self.cfg, "eval_params"):
            results, vis_outputs, vis_paths = \
                eval.run(self.data_dict,
                         batch_size=self.batch_size,
                         img_size=self.img_size,
                         model=self.ema.ema if self.args.calib is False else self.model,
                         conf_thres=0.03,
                         dataloader=self.val_loader,
                         save_dir=self.save_dir,
                         task='train')
        else:
            def get_cfg_value(cfg_dict, value_str, default_value):
                if value_str in cfg_dict:
                    if isinstance(cfg_dict[value_str], list):
                        return cfg_dict[value_str][0] if cfg_dict[value_str][0] is not None else default_value
                    else:
                        return cfg_dict[value_str] if cfg_dict[value_str] is not None else default_value
                else:
                    return default_value

            eval_img_size = get_cfg_value(self.cfg.eval_params, "img_size", self.img_size)
            results, vis_outputs, vis_paths = \
                eval.run(self.data_dict,
                         batch_size=get_cfg_value(self.cfg.eval_params, "batch_size", self.batch_size),
                         img_size=eval_img_size,
                         model=self.ema.ema if self.args.calib is False else self.model,
                         conf_thres=get_cfg_value(self.cfg.eval_params, "conf_thres", 0.03),
                         dataloader=self.val_loader,
                         save_dir=self.save_dir,
                         task='train',
                         test_load_size=get_cfg_value(self.cfg.eval_params, "test_load_size", eval_img_size),
                         letterbox_return_int=get_cfg_value(self.cfg.eval_params, "letterbox_return_int", False),
                         force_no_pad=get_cfg_value(self.cfg.eval_params, "force_no_pad", False),
                         not_infer_on_rect=get_cfg_value(self.cfg.eval_params, "not_infer_on_rect", False),
                         scale_exact=get_cfg_value(self.cfg.eval_params, "scale_exact", False),
                         verbose=get_cfg_value(self.cfg.eval_params, "verbose", False),
                         do_coco_metric=get_cfg_value(self.cfg.eval_params, "do_coco_metric", True),
                         do_pr_metric=get_cfg_value(self.cfg.eval_params, "do_pr_metric", False),
                         plot_curve=get_cfg_value(self.cfg.eval_params, "plot_curve", False),
                         plot_confusion_matrix=get_cfg_value(self.cfg.eval_params, "plot_confusion_matrix", False),
                         )

        LOGGER.info(f"Epoch: {self.epoch} | mAP@0.5: {results[0]} | mAP@0.50:0.95: {results[1]}")
        self.evaluate_results = results[:2]
        # plot validation predictions
        self.plot_val_pred(vis_outputs, vis_paths)

    def plot_train_batch(self, images, targets, max_size=1920, max_subplots=16):
        # Plot train_batch with labels
        if isinstance(images, torch.Tensor):
            images = images.cpu().float().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if np.max(images[0]) <= 1:
            images *= 255  # de-normalise (optional)
        bs, _, h, w = images.shape  # batch size, _, height, width
        bs = min(bs, max_subplots)  # limit plot images
        ns = np.ceil(bs ** 0.5)  # number of subplots (square)
        paths = self.batch_data[2]  # image paths
        # Build Image
        mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
        for i, im in enumerate(images):
            if i == max_subplots:  # if last batch has fewer images than we expect
                break
            x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
            im = im.transpose(1, 2, 0)
            mosaic[y:y + h, x:x + w, :] = im
        # Resize (optional)
        scale = max_size / ns / max(h, w)
        if scale < 1:
            h = math.ceil(scale * h)
            w = math.ceil(scale * w)
            mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))
        for i in range(bs):
            x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
            cv2.rectangle(mosaic, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)  # borders
            cv2.putText(mosaic, f"{os.path.basename(paths[i])[:40]}", (x + 5, y + 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, color=(220, 220, 220), thickness=1)  # filename
            if len(targets) > 0:
                ti = targets[targets[:, 0] == i]  # image targets
                boxes = xywh2xyxy(ti[:, 2:6]).T
                classes = ti[:, 1].astype('int')
                labels = ti.shape[1] == 6  # labels if no conf column
                if boxes.shape[1]:
                    if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                        boxes[[0, 2]] *= w  # scale to pixels
                        boxes[[1, 3]] *= h
                    elif scale < 1:  # absolute coords need scale if image scales
                        boxes *= scale
                boxes[[0, 2]] += x
                boxes[[1, 3]] += y
                for j, box in enumerate(boxes.T.tolist()):
                    box = [int(k) for k in box]
                    cls = classes[j]
                    color = tuple([int(x) for x in self.color[cls]])
                    cls = self.data_dict['names'][cls] if self.data_dict['names'] else cls
                    if labels:
                        label = f'{cls}'
                        cv2.rectangle(mosaic, (box[0], box[1]), (box[2], box[3]), color, thickness=1)
                        cv2.putText(mosaic, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, color,
                                    thickness=1)
        self.vis_train_batch = mosaic.copy()
        return mosaic.copy()

    def plot_val_pred(self, vis_outputs, vis_paths, vis_conf=0.3, vis_max_box_num=5):
        # plot validation predictions
        self.vis_imgs_list = []
        for (vis_output, vis_path) in zip(vis_outputs, vis_paths):
            vis_output_array = vis_output.cpu().numpy()  # xyxy
            ori_img = cv2.imread(vis_path)
            for bbox_idx, vis_bbox in enumerate(vis_output_array):
                x_tl = int(vis_bbox[0])
                y_tl = int(vis_bbox[1])
                x_br = int(vis_bbox[2])
                y_br = int(vis_bbox[3])
                box_score = vis_bbox[4]
                cls_id = int(vis_bbox[5])
                # draw top n bbox
                if box_score < vis_conf or bbox_idx > vis_max_box_num:
                    break
                cv2.rectangle(ori_img, (x_tl, y_tl), (x_br, y_br), tuple([int(x) for x in self.color[cls_id]]),
                              thickness=3)
                cv2.putText(ori_img, f"{self.data_dict['names'][cls_id]}: {box_score:.2f}", (x_tl, y_tl - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, tuple([int(x) for x in self.color[cls_id]]), thickness=1)
            self.vis_imgs_list.append(torch.from_numpy(ori_img[:, :, ::-1].copy()))


if __name__ == '__main__':
    from cmd_in import get_args_parser

    args = get_args_parser().parse_args()

    # Setup
    from core_train import check_and_init

    cfg, args = check_and_init(args)
    LOGGER.info(f'training args are: {args}\n')

    #
    args.data_path = '/home/hrlee/PycharmProjects/YOLOv6/data/WIDER_FACE.yaml'

    # Start
    trainer = Trainer(args, cfg, torch.device('cpu'))
    trainer.train()
