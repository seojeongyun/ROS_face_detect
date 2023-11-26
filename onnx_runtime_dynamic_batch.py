import random

import cv2
import onnxruntime as ort

import numpy as np


# Functions
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Preprocess image. For details, see:
    https://github.com/meituan/YOLOv6/issues/613
    """

    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


def get_arg():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='/storage/sjpark/VGGFace2', type=str, help='path of dataset')
    parser.add_argument('--half', action='store_true', help='FP16')
    args = parser.parse_args()
    print(args)

    return args


if __name__ == '__main__':
    args = get_arg()
    # Configuration
    cuda = True
    w = "/storage/jysuh/YOLOv6_weights/6s_hs_{}_960/weights/last_ckpt.onnx"

    # Test images
    imgList = [cv2.imread('/home/jysuh/Downloads/test.jpg')]
    imgList *= 7
    imgList = imgList[:32]  # For dynamic batch, 32 is the maximum batch_size ?

    # Create session of onnx model
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    session = ort.InferenceSession(w, providers=providers)

    # Class name and class color
    colors = {'face': (24, 252, 0)}

    # Preprocessing images
    origin_RGB = []
    resize_data = []
    for img in imgList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        origin_RGB.append(img)
        image = img.copy()
        image, ratio, dwdh = letterbox(image)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        if args.half:
            im = image.astype(np.float16)
        else:
            im = image.astype(np.float32)
        resize_data.append((im, ratio, dwdh))

    # Concate data
    np_batch = np.concatenate([data[0] for data in resize_data])

    #
    outname = [i.name for i in session.get_outputs()]
    inname = [i.name for i in session.get_inputs()]

    # Infer 1 batch
    im = np.ascontiguousarray(np_batch[0:1, ...] / 255)
    out1 = session.run(outname, {'images': im})

    # Infer 4 batches
    im = np.ascontiguousarray(np_batch[0:4, ...] / 255)
    out4 = session.run(outname, {'images': im})

    # Infer 5 batches
    im = np.ascontiguousarray(np_batch[0:5, ...] / 255)
    out5 = session.run(outname, {'images': im})

    # Infer all batches
    im = np.ascontiguousarray(np_batch / 255)
    out_al = session.run(outname, {'images': im})

    # Results
    for i in range(out_al[0].shape[0]):
        obj_num = out_al[0][i]
        boxes = out_al[1][i]
        scores = out_al[2][i]
        cls_id = out_al[3][i]
        image = origin_RGB[i]
        img_h, img_w = image.shape[:2]
        ratio, dwdh = resize_data[i][1:]
        for num in range(obj_num[0]):
            box = boxes[num]
            score = round(float(scores[num]), 3)
            if score > 0:
                obj_name = 'face'
                box -= np.array(dwdh * 2)
                box /= ratio
                box = box.round().astype(np.int32).tolist()
                x1 = max(0, box[0])
                y1 = max(0, box[1])
                x2 = min(img_w, box[2])
                y2 = min(img_h, box[3])
                color = colors[obj_name]
                obj_name += ' ' + str(score)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, obj_name, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255],
                            thickness=2)

        import matplotlib.pyplot as plt

        plt.imshow(image)
        plt.show()
