"""
This script is used for evaluating the performance of YOLOv6 TensorRT models.
"""
import os
import sys

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tensorrt_eval_fns import *
from yolov6.utils.Processor import Processor


def main():
    # --- Arguments ---
    args = parse_args()
    check_args(args)

    # --- File names ---
    # Ex of model prefix: yolov6s, yolov6l ...
    model_prefix = args.model.replace('.trt', '').split('/')[-1]
    results_file = './eval/tensorrt/results_{}.json'.format(model_prefix)

    # --- Set a processor to perform inference of a tensorRT engine ---
    processor = Processor(model=args.model,
                          scale_exact=args.scale_exact,
                          return_int=args.letterbox_return_int,
                          force_no_pad=args.force_no_pad,
                          is_end2end=args.is_end2end)

    # --- Set data and info. of data---
    valid_images, imgname2id = trim_valid_data(args.imgs_dir, args.annotations, args.labels_dir)
    data_class = [1]
    model_names = ['traffic']

    # Perform tensorRT
    stats, seen, confusion_matrix = generate_results(data_class, model_names, args.do_pr_metric,
                                                     args.plot_confusion_matrix, processor,
                                                     args.imgs_dir, args.labels_dir, valid_images, results_file,
                                                     args.conf_thres,
                                                     args.iou_thres, args.is_coco, batch_size=args.batch_size,
                                                     test_load_size=args.test_load_size,
                                                     visualize=args.visualize,
                                                     num_imgs_to_visualize=args.num_imgs_to_visualize,
                                                     imgname2id=imgname2id)

    # Run COCO mAP evaluation
    # Reference: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    cocoGt = COCO(args.annotations)
    cocoDt = cocoGt.loadRes(results_file)
    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # Run PR_metric evaluation
    if args.do_pr_metric:
        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            from yolov6.utils.metrics import ap_per_class
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=args.plot_curve, save_dir=args.save_dir,
                                                  names=model_names)
            AP50_F1_max_idx = len(f1.mean(0)) - f1.mean(0)[::-1].argmax() - 1
            LOGGER.info(f"IOU 50 best mF1 thershold near {AP50_F1_max_idx / 1000.0}.")
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p[:, AP50_F1_max_idx].mean(), r[:, AP50_F1_max_idx].mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=len(model_names))  # number of targets per class

            # Print results
            s = ('%-16s' + '%12s' * 7) % (
                'Class', 'Images', 'Labels', 'P@.5iou', 'R@.5iou', 'F1@.5iou', 'mAP@.5', 'mAP@.5:.95')
            LOGGER.info(s)
            pf = '%-16s' + '%12i' * 2 + '%12.3g' * 5  # print format
            LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, f1.mean(0)[AP50_F1_max_idx], map50, map))

            pr_metric_result = (map50, map)
            print("pr_metric results:", pr_metric_result)

            # Print results per class
            if args.verbose and len(model_names) > 1:
                for i, c in enumerate(ap_class):
                    LOGGER.info(pf % (model_names[c], seen, nt[c], p[i, AP50_F1_max_idx], r[i, AP50_F1_max_idx],
                                      f1[i, AP50_F1_max_idx], ap50[i], ap[i]))

            if args.plot_confusion_matrix:
                confusion_matrix.plot(save_dir=args.save_dir, names=list(model_names))
        else:
            LOGGER.info("Calculate metric failed, might check dataset.")
            pr_metric_result = (0.0, 0.0)


if __name__ == '__main__':
    main()
