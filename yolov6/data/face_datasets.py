#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import glob
import shutil
from io import UnsupportedOperation
import os
import os.path as osp
import random
import json
import time
import hashlib
from pathlib import Path

from multiprocessing.pool import Pool

import cv2
import numpy as np
import torch
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import Dataset
from tqdm import tqdm

from yolov6.data.data_augment_face import (
    augment_hsv,
    letterbox,
    mixup,
    random_affine,
    mosaic_augmentation,
)
from yolov6.utils.events import LOGGER

# Parameters
IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
VID_FORMATS = ["mp4", "mov", "avi", "mkv"]
IMG_FORMATS.extend([f.upper() for f in IMG_FORMATS])
VID_FORMATS.extend([f.upper() for f in VID_FORMATS])
# Get orientation exif tag
# exif is a metadata of a image captured by a camera,
# including image length, image name, camera information, and etc..
for k, v in ExifTags.TAGS.items():
    if v == "Orientation":
        ORIENTATION = k
        break


class TrainValDataset(Dataset):
    '''
        YOLOv6 train_loader/val_loader,
        loads images and labels for training and validation.
    '''

    def __init__(self,
                 img_dir,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 check_images=False,
                 check_labels=False,
                 stride=32,
                 pad=0.0,
                 data_dict=None,
                 task="train"):
        t1 = time.time()
        assert task.lower() in ("train", "val", "test", "speed"), f"Not supported task: {task}"
        # locals() returns all the inputs of def __init__ function.
        # For example, inputs of def __init__ function are height=300 and width=300.
        # Then, self.__dict__.update(locals()) creates self.height=300 and self.width=300.
        self.__dict__.update(locals())
        self.task = self.task.capitalize()
        self.class_names = data_dict["names"]
        self.img_paths, self.labels = self.get_imgs_labels(self.img_dir)
        #
        LOGGER.info(f"%.1fs for dataset initialization." % (time.time() - t1))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        """Fetching a data sample for a given key.
        This function applies mosaic and mixup augments during training.
        During validation, letterbox augment is applied.
        """
        # Mosaic Augmentation
        if self.augment and random.random() < self.hyp["mosaic"]:
            img, labels = self.get_mosaic(index)
            shapes = None

            # MixUp augmentation
            if random.random() < self.hyp["mixup"]:
                img_other, labels_other = self.get_mosaic(random.randint(0, len(self.img_paths) - 1))
                img, labels = mixup(img, labels, img_other, labels_other)
        else:
            # Load image
            if self.hyp and "test_load_size" in self.hyp:
                img, (h0, w0), (h, w) = self.load_image(index, self.hyp["test_load_size"])
            else:
                img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            shape = (self.img_size)  # final letterboxed shape
            if self.hyp and "letterbox_return_int" in self.hyp:
                img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment,
                                            return_int=self.hyp["letterbox_return_int"])
            else:
                img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h * ratio / h0, w * ratio / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:
                w *= ratio
                h *= ratio
                # Create new boxes
                # -- box corners
                boxes = np.copy(labels[:, 1:])
                boxes[:, 0] = (w * (labels[:, 1] - labels[:, 3] / 2) + pad[0])  # top left x
                boxes[:, 1] = (h * (labels[:, 2] - labels[:, 4] / 2) + pad[1])  # top left y
                boxes[:, 2] = (w * (labels[:, 1] + labels[:, 3] / 2) + pad[0])  # bottom right x
                boxes[:, 3] = (h * (labels[:, 2] + labels[:, 4] / 2) + pad[1])  # bottom right y
                # -- traffic landmarks
                boxes[:, 4] = np.array(labels[:, 5] > 0, dtype=np.int32) * (w * labels[:, 5] + pad[0]) + (
                        np.array(labels[:, 5] > 0, dtype=np.int32) - 1)
                boxes[:, 5] = np.array(labels[:, 6] > 0, dtype=np.int32) * (h * labels[:, 6] + pad[1]) + (
                        np.array(labels[:, 6] > 0, dtype=np.int32) - 1)
                boxes[:, 6] = np.array(labels[:, 7] > 0, dtype=np.int32) * (w * labels[:, 7] + pad[0]) + (
                        np.array(labels[:, 7] > 0, dtype=np.int32) - 1)
                boxes[:, 7] = np.array(labels[:, 8] > 0, dtype=np.int32) * (h * labels[:, 8] + pad[1]) + (
                        np.array(labels[:, 8] > 0, dtype=np.int32) - 1)
                boxes[:, 8] = np.array(labels[:, 5] > 0, dtype=np.int32) * (w * labels[:, 9] + pad[0]) + (
                        np.array(labels[:, 9] > 0, dtype=np.int32) - 1)
                boxes[:, 9] = np.array(labels[:, 5] > 0, dtype=np.int32) * (h * labels[:, 10] + pad[1]) + (
                        np.array(labels[:, 10] > 0, dtype=np.int32) - 1)
                boxes[:, 10] = np.array(labels[:, 11] > 0, dtype=np.int32) * (w * labels[:, 11] + pad[0]) + (
                        np.array(labels[:, 11] > 0, dtype=np.int32) - 1)
                boxes[:, 11] = np.array(labels[:, 12] > 0, dtype=np.int32) * (h * labels[:, 12] + pad[1]) + (
                        np.array(labels[:, 12] > 0, dtype=np.int32) - 1)
                boxes[:, 12] = np.array(labels[:, 13] > 0, dtype=np.int32) * (w * labels[:, 13] + pad[0]) + (
                        np.array(labels[:, 13] > 0, dtype=np.int32) - 1)
                boxes[:, 13] = np.array(labels[:, 14] > 0, dtype=np.int32) * (h * labels[:, 14] + pad[1]) + (
                        np.array(labels[:, 14] > 0, dtype=np.int32) - 1)
                # Set labels
                labels[:, 1:] = boxes

            if self.augment:
                img, labels = random_affine(
                    img,
                    labels,
                    degrees=self.hyp["degrees"],
                    translate=self.hyp["translate"],
                    scale=self.hyp["scale"],
                    shear=self.hyp["shear"],
                    new_shape=(self.img_size, self.img_size),
                )

        if len(labels):
            h, w = img.shape[:2]

            labels[:, [1, 3]] = labels[:, [1, 3]].clip(0, w - 1e-3)  # x1, x2
            labels[:, [2, 4]] = labels[:, [2, 4]].clip(0, h - 1e-3)  # y1, y2

            boxes = np.copy(labels[:, 1:])
            boxes[:, 0] = ((labels[:, 1] + labels[:, 3]) / 2) / w  # x center
            boxes[:, 1] = ((labels[:, 2] + labels[:, 4]) / 2) / h  # y center
            boxes[:, 2] = (labels[:, 3] - labels[:, 1]) / w  # width
            boxes[:, 3] = (labels[:, 4] - labels[:, 2]) / h  # height
            labels[:, 1:] = boxes

            labels[:, [5, 7, 9, 11, 13]] /= img.shape[1]  # normalized landmark x 0-1
            labels[:, [5, 7, 9, 11, 13]] = np.where(labels[:, [5, 7, 9, 11, 13]] < 0, -1,
                                                    labels[:, [5, 7, 9, 11, 13]])
            labels[:, [6, 8, 10, 12, 14]] /= img.shape[0]  # normalized landmark y 0-1
            labels[:, [6, 8, 10, 12, 14]] = np.where(labels[:, [6, 8, 10, 12, 14]] < 0, -1,
                                                     labels[:, [6, 8, 10, 12, 14]])

        # --- Debugging ---
        # If you want to see the bounding boxes and landmark on images,
        # please use the lower function.
        # self.showlabels(img, labels[:, 1:5], labels[:, 5:15])
        # -----------------

        if self.augment:
            img, labels = self.general_augment(img, labels.copy())
        #
        labels_out = torch.zeros((len(labels), 16))
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_paths[index], shapes

    def showlabels(self, img, boxs, landmarks):
        b, g, r = cv2.split(img)
        # print(boxs)
        new_image1 = cv2.merge([r, g, b])
        for box in boxs:
            x, y, w, h = box[0] * img.shape[1], box[1] * img.shape[0], box[2] * img.shape[1], box[3] * img.shape[0]
            # cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

            cv2.rectangle(new_image1, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0),
                          2)

        for landmark in landmarks:
            # cv2.circle(img,(60,60),30,(0,0,255))
            for i in range(5):
                cv2.circle(new_image1, (int(landmark[2 * i] * img.shape[1]), int(landmark[2 * i + 1] * img.shape[0])),
                           3, (0, 0, 255), -1)
        import matplotlib.pyplot as plt
        plt.imshow(new_image1)
        plt.show()

    def load_image(self, index, force_load_size=None):
        """Load image.
        This function loads image by cv2, resize original image to target shape(img_size) with keeping ratio.

        Returns:
            Image, original shape of image, resized image shape
        """
        path = self.img_paths[index]
        try:
            im = cv2.imread(path)
            assert im is not None, f"opencv cannot read image correctly or {path} not exists"
        except:
            im = cv2.cvtColor(np.asarray(Image.open(path)), cv2.COLOR_RGB2BGR)
            assert im is not None, f"Image Not Found {path}, workdir: {os.getcwd()}"

        h0, w0 = im.shape[:2]  # origin shape
        if force_load_size:
            r = force_load_size / max(h0, w0)
        else:
            r = self.img_size / max(h0, w0)
        if r != 1:
            im = cv2.resize(
                im,
                (int(w0 * r), int(h0 * r)),
                interpolation=cv2.INTER_AREA
                if r < 1 and not self.augment
                else cv2.INTER_LINEAR,
            )
        return im, (h0, w0), im.shape[:2]

    @staticmethod
    def collate_fn(batch):
        """Merges a list of samples to form a mini-batch of Tensor(s)"""
        img, label, path, shapes = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    def get_imgs_labels(self, img_dir):

        assert osp.exists(img_dir), f"{img_dir} is an invalid directory path!"
        valid_img_record = osp.join(osp.dirname(img_dir), "." + osp.basename(img_dir) + ".json")
        NUM_THREADS = min(8, os.cpu_count())

        img_paths = glob.glob(osp.join(img_dir, "**/*"), recursive=True)
        # Remove directories, that is, Remain only image files.
        img_paths = sorted(p for p in img_paths if p.split(".")[-1].lower() in IMG_FORMATS and os.path.isfile(p))
        assert img_paths, f"No images found in {img_dir}."

        img_hash = self.get_hash(img_paths)
        if osp.exists(valid_img_record):
            with open(valid_img_record, "r") as f:
                cache_info = json.load(f)
                if "image_hash" in cache_info and cache_info["image_hash"] == img_hash:
                    img_info = cache_info["information"]
                else:
                    self.check_images = True
        else:
            self.check_images = True

        # check images
        if self.check_images:
            img_info = {}
            nc, msgs = 0, []  # number corrupt, messages
            LOGGER.info(f"{self.task}: Checking formats of images with {NUM_THREADS} process(es): ")
            with Pool(NUM_THREADS) as pool:
                pbar = tqdm(pool.imap(TrainValDataset.check_image, img_paths), total=len(img_paths), )
                for img_path, shape_per_img, nc_per_img, msg in pbar:
                    if nc_per_img == 0:  # not corrupted
                        img_info[img_path] = {"shape": shape_per_img}
                    nc += nc_per_img
                    if msg:
                        msgs.append(msg)
                    pbar.desc = f"{nc} image(s) corrupted"
            pbar.close()
            if msgs:
                LOGGER.info("\n".join(msgs))

            cache_info = {"information": img_info, "image_hash": img_hash}
            # save valid image paths.
            with open(valid_img_record, "w") as f:
                json.dump(cache_info, f)

        # check and load anns
        base_dir = osp.basename(img_dir)
        if base_dir != "":
            label_dir = osp.join(osp.dirname(osp.dirname(img_dir)), "labels", osp.basename(img_dir))
            assert osp.exists(label_dir), f"{label_dir} is an invalid directory path!"
        else:
            sub_dirs = []
            label_dir = img_dir
            for rootdir, dirs, files in os.walk(label_dir):
                for subdir in dirs:
                    sub_dirs.append(subdir)
            assert "labels" in sub_dirs, f"Could not find a labels directory!"

        # Look for labels in the save relative dir that the images are in
        def _new_rel_path_with_ext(img_base_path: str, img_full_path: str, label_base_path: str, new_ext: str):
            rel_path = osp.relpath(img_full_path, img_base_path)
            #
            dir_path = osp.join(label_base_path, osp.dirname(rel_path))
            if not osp.exists(osp.join(label_base_path, osp.join(osp.dirname(rel_path),
                                                                 osp.splitext(osp.basename(rel_path))[0] + new_ext))):
                if osp.exists(osp.join(label_base_path, osp.splitext(osp.basename(rel_path))[0] + new_ext)):
                    if not osp.exists(dir_path):
                        os.makedirs(dir_path)
                    shutil.move(osp.join(label_base_path, osp.splitext(osp.basename(rel_path))[0] + new_ext),
                                osp.join(label_base_path, osp.join(osp.dirname(rel_path),
                                                                   osp.splitext(osp.basename(rel_path))[0] + new_ext)))
                else:
                    LOGGER.error(
                        f'There is no label txt file {osp.join(label_base_path, osp.splitext(osp.basename(rel_path))[0] + new_ext)}')
            return osp.join(osp.dirname(rel_path), osp.splitext(osp.basename(rel_path))[0] + new_ext)

        img_paths = list(img_info.keys())
        label_paths = sorted(
            osp.join(label_dir, _new_rel_path_with_ext(img_dir, p, label_dir, ".txt"))
            for p in img_paths
        )
        assert label_paths, f"No labels found in {label_dir}."
        label_hash = self.get_hash(label_paths)
        if "label_hash" not in cache_info or cache_info["label_hash"] != label_hash:
            self.check_labels = True

        if self.check_labels:
            cache_info["label_hash"] = label_hash
            nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number corrupt, messages
            LOGGER.info(
                f"{self.task}: Checking formats of labels with {NUM_THREADS} process(es): "
            )
            with Pool(NUM_THREADS) as pool:
                pbar = pool.imap(
                    TrainValDataset.check_label_files, zip(img_paths, label_paths)
                )
                pbar = tqdm(pbar, total=len(label_paths))
                for (
                        img_path,
                        labels_per_file,
                        nc_per_file,
                        nm_per_file,
                        nf_per_file,
                        ne_per_file,
                        msg,
                ) in pbar:
                    if nc_per_file == 0:
                        img_info[img_path]["labels"] = labels_per_file
                    else:
                        img_info.pop(img_path)
                    nc += nc_per_file
                    nm += nm_per_file
                    nf += nf_per_file
                    ne += ne_per_file
                    if msg:
                        msgs.append(msg)
                    pbar.desc = f"{nf} label(s) found, {nm} label(s) missing, {ne} label(s) empty, {nc} invalid label files"
            pbar.close()
            with open(valid_img_record, "w") as f:
                json.dump(cache_info, f)
            if msgs:
                LOGGER.info("\n".join(msgs))
            if nf == 0:
                LOGGER.warning(
                    f"WARNING: No labels found in {osp.dirname(img_paths[0])}. "
                )

        if self.task.lower() == "val":
            if self.data_dict.get("is_coco", False):  # use original json file when evaluating on coco dataset.
                assert osp.exists(self.data_dict[
                                      "anno_path"]), "Eval on coco dataset must provide valid path of the annotation file in config file: data/coco.yaml"
            else:
                assert (
                    self.class_names), "Class names is required when converting labels to coco format for evaluating."
                save_dir = osp.join(osp.dirname(osp.dirname(img_dir)), "annotations")
                if not osp.exists(save_dir):
                    os.mkdir(save_dir)
                save_path = osp.join(save_dir, "instances_" + osp.basename(img_dir) + ".json")
                TrainValDataset.generate_coco_format_labels(img_info, self.class_names, save_path)

        img_paths, labels = list(
            zip(
                *[
                    (
                        img_path,
                        np.array(info["labels"], dtype=np.float32)
                        if info["labels"]
                        else np.zeros((0, 15), dtype=np.float32),
                    )
                    for img_path, info in img_info.items()
                ]
            )
        )
        self.img_info = img_info
        LOGGER.info(f"{self.task}: Final numbers of valid images: {len(img_paths)}/ labels: {len(labels)}. ")
        return img_paths, labels

    def get_mosaic(self, index):
        """Gets images and labels after mosaic augments"""
        indices = [index] + random.choices(
            range(0, len(self.img_paths)), k=3
        )  # 3 additional image indices
        random.shuffle(indices)
        imgs, hs, ws, labels = [], [], [], []
        for index in indices:
            img, _, (h, w) = self.load_image(index)
            labels_per_img = self.labels[index]
            imgs.append(img)
            hs.append(h)
            ws.append(w)
            labels.append(labels_per_img)
        img, labels = mosaic_augmentation(self.img_size, imgs, hs, ws, labels, self.hyp)
        # For debugging
        # import matplotlib.pyplt as plt
        # plt.imshow(img)
        # plt.show()
        return img, labels

    def general_augment(self, img, labels):
        """Gets images and labels after general augment
        This function applies hsv, random ud-flip and random lr-flips augments.
        """
        nl = len(labels)

        # HSV color-space
        augment_hsv(
            img,
            hgain=self.hyp["hsv_h"],
            sgain=self.hyp["hsv_s"],
            vgain=self.hyp["hsv_v"],
        )

        # Flip up-down
        if random.random() < self.hyp["flipud"]:
            img = np.flipud(img)
            if nl:
                labels[:, 2] = 1 - labels[:, 2]

                labels[:, 6] = np.where(labels[:, 6] < 0, -1, 1 - labels[:, 6])
                labels[:, 8] = np.where(labels[:, 8] < 0, -1, 1 - labels[:, 8])
                labels[:, 10] = np.where(labels[:, 10] < 0, -1, 1 - labels[:, 10])
                labels[:, 12] = np.where(labels[:, 12] < 0, -1, 1 - labels[:, 12])
                labels[:, 14] = np.where(labels[:, 14] < 0, -1, 1 - labels[:, 14])

        # Flip left-right
        if random.random() < self.hyp["fliplr"]:
            img = np.fliplr(img)
            if nl:
                labels[:, 1] = 1 - labels[:, 1]

                labels[:, 5] = np.where(labels[:, 5] < 0, -1, 1 - labels[:, 5])
                labels[:, 7] = np.where(labels[:, 7] < 0, -1, 1 - labels[:, 7])
                labels[:, 9] = np.where(labels[:, 9] < 0, -1, 1 - labels[:, 9])
                labels[:, 11] = np.where(labels[:, 11] < 0, -1, 1 - labels[:, 11])
                labels[:, 13] = np.where(labels[:, 13] < 0, -1, 1 - labels[:, 13])

                eye_left = np.copy(labels[:, [5, 6]])
                mouth_left = np.copy(labels[:, [11, 12]])
                labels[:, [5, 6]] = labels[:, [7, 8]]
                labels[:, [7, 8]] = eye_left
                labels[:, [11, 12]] = labels[:, [13, 14]]
                labels[:, [13, 14]] = mouth_left

        return img, labels

    def sort_files_shapes(self):
        '''Sort by aspect ratio.'''
        batch_num = self.batch_indices[-1] + 1
        s = self.shapes  # wh
        ar = s[:, 1] / s[:, 0]  # aspect ratio
        irect = ar.argsort()
        self.img_paths = [self.img_paths[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        self.shapes = s[irect]  # wh
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * batch_num
        for i in range(batch_num):
            ari = ar[self.batch_indices == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]
        self.batch_shapes = (
                np.ceil(np.array(shapes) * self.img_size / self.stride + self.pad).astype(
                    np.int_
                )
                * self.stride
        )

    @staticmethod
    def check_image(im_file):
        '''Verify an image.'''
        nc, msg = 0, ""
        try:
            im = Image.open(im_file)
            im.verify()  # PIL verify
            im = Image.open(im_file)  # need to reload the image after using verify()
            shape = im.size  # (width, height)
            try:
                im_exif = im._getexif()
                if im_exif and ORIENTATION in im_exif:
                    rotation = im_exif[ORIENTATION]
                    if rotation in (6, 8):
                        shape = (shape[1], shape[0])
            except:
                im_exif = None
            if im_exif and ORIENTATION in im_exif:
                rotation = im_exif[ORIENTATION]
                if rotation in (6, 8):
                    shape = (shape[1], shape[0])

            assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
            assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
            if im.format.lower() in ("jpg", "jpeg"):
                with open(im_file, "rb") as f:
                    f.seek(-2, 2)
                    if f.read() != b"\xff\xd9":  # corrupt JPEG
                        ImageOps.exif_transpose(Image.open(im_file)).save(
                            im_file, "JPEG", subsampling=0, quality=100
                        )
                        msg += f"WARNING: {im_file}: corrupt JPEG restored and saved"
            return im_file, shape, nc, msg
        except Exception as e:
            nc = 1
            msg = f"WARNING: {im_file}: ignoring corrupt image: {e}"
            return im_file, None, nc, msg

    @staticmethod
    def check_label_files(args):
        img_path, lb_path = args
        nm, nf, ne, nc, msg = 0, 0, 0, 0, ""  # number (missing, found, empty, message
        try:
            if osp.exists(lb_path):
                nf = 1  # label found
                with open(lb_path, "r") as f:
                    labels = [
                        x.split() for x in f.read().strip().splitlines() if len(x)
                    ]
                    labels = np.array(labels, dtype=np.float32)
                if len(labels):
                    assert all(
                        len(l) == 15 for l in labels
                    ), f"{lb_path}: wrong label format."
                    assert (
                            labels[:, : 5] >= 0
                    ).all(), f"{lb_path}: Label values error: all values in label file must > 0"
                    assert (
                            labels[:, 1:5] <= 1
                    ).all(), f"{lb_path}: Label values error: all coordinates must be normalized"

                    _, indices = np.unique(labels, axis=0, return_index=True)
                    if len(indices) < len(labels):  # duplicate row check
                        labels = labels[indices]  # remove duplicates
                        msg += f"WARNING: {lb_path}: {len(labels) - len(indices)} duplicate labels removed"
                    labels = labels.tolist()
                else:
                    ne = 1  # label empty
                    labels = []
            else:
                nm = 1  # label missing
                labels = []

            return img_path, labels, nc, nm, nf, ne, msg
        except Exception as e:
            nc = 1
            msg = f"WARNING: {lb_path}: ignoring invalid labels: {e}"
            return img_path, None, nc, nm, nf, ne, msg

    @staticmethod
    def generate_coco_format_labels(img_info, class_names, save_path):
        # for evaluation with pycocotools
        dataset = {"categories": [], "annotations": [], "images": []}
        for i, class_name in enumerate(class_names):
            dataset["categories"].append(
                {"id": i, "name": class_name, "supercategory": ""}
            )

        ann_id = 0
        LOGGER.info(f"Convert to COCO format")
        for i, (img_path, info) in enumerate(tqdm(img_info.items())):
            labels = info["labels"] if info["labels"] else []
            img_id = osp.splitext(osp.basename(img_path))[0]
            img_w, img_h = info["shape"]
            dataset["images"].append(
                {
                    "file_name": os.path.basename(img_path),
                    "id": img_id,
                    "width": img_w,
                    "height": img_h,
                }
            )
            if labels:
                for label in labels:
                    c, x, y, w, h = label[:5]
                    # convert x,y,w,h to x1,y1,x2,y2
                    x1 = (x - w / 2) * img_w
                    y1 = (y - h / 2) * img_h
                    x2 = (x + w / 2) * img_w
                    y2 = (y + h / 2) * img_h
                    # cls_id starts from 0
                    cls_id = int(c)
                    w = max(0, x2 - x1)
                    h = max(0, y2 - y1)
                    dataset["annotations"].append(
                        {
                            "area": h * w,
                            "bbox": [x1, y1, w, h],
                            "category_id": cls_id,
                            "id": ann_id,
                            "image_id": img_id,
                            "iscrowd": 0,
                            # mask
                            "segmentation": [],
                        }
                    )
                    ann_id += 1

        with open(save_path, "w") as f:
            json.dump(dataset, f)
            LOGGER.info(
                f"Convert to COCO format finished. Resutls saved in {save_path}"
            )

    @staticmethod
    def get_hash(paths):
        """Get the hash value of paths"""
        assert isinstance(paths, list), "Only support list currently."
        h = hashlib.md5("".join(paths).encode())
        return h.hexdigest()


if __name__ == '__main__':
    print('Debugging')
    import glob

    # Data class and Loader
    from yolov6.utils.events import load_yaml
    from yolov6.utils.config import Config

    cfg = Config.fromfile('../../configs/traffic/yolov6l_finetune.py')
    data_dict = load_yaml('../../data/WIDER_FACE.yaml')
    path = data_dict['train']
    nc = int(data_dict['nc'])
    class_names = data_dict['names']
    assert len(class_names) == nc, f'the length of class_names should match the number of classes defined'
    grid_size = max(int(max(cfg.model.head.strides)), 32)

    img_dir = path
    img_size = 640
    batch_size = 16
    stride = 32
    hyp = dict(cfg.data_aug)
    augment = True
    workers = 8
    shuffle = True
    check_images = True
    check_labels = True
    task = 'train'

    dst_object = TrainValDataset(
        img_dir=path,
        img_size=640,
        batch_size=16,
        augment=augment,
        hyp=hyp,
        check_images=False,
        check_labels=False,
        stride=int(stride),
        pad=0.0,
        data_dict=data_dict,
        task=task,
    )

    dst_object.__getitem__(101)