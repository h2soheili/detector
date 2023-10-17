from typing import List

import cv2
import numpy as np
import torch


def get_device(device="cuda"):
    use_cuda = device == "cuda" and torch.cuda.is_available()
    use_mps = device == "mps" and torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device(device)
    elif use_mps:
        device = torch.device(device)
    else:
        device = torch.device("cpu")
    return device


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scale_fill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        # cv2.resize expects (H,W,C)
        # print("  shape[::-1]", shape[::-1], "   new_shape,", new_shape)
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def resize_and_mask_img(img: np.array,
                        boundary: List[np.array] = None,
                        img_size=(640, 640),
                        stride=32,
                        auto=True):
    # h: height, w: width, c: channel, b: batch
    apply_mask = boundary is not None
    # img = img[..., ::-1]  # BGR to RGB
    resized_img = letterbox(img, img_size, stride=stride, auto=auto)[0]
    if apply_mask:
        img_copy = img.copy()
        # https://stackoverflow.com/a/48301735
        mask = np.zeros(img_copy.shape[:2], np.uint8)
        cv2.drawContours(mask, boundary, -1, (255, 255, 255), -1, cv2.LINE_AA)
        dst = cv2.bitwise_and(img_copy, img_copy, mask=mask)
        dst = letterbox(dst, img_size, stride=stride, auto=auto)[0]
    else:
        dst = letterbox(img, img_size, stride=stride, auto=auto)[0]
    # dst = dst.transpose((2, 0, 1))  # HWC to CHW
    dst = np.ascontiguousarray(dst)  # contiguous
    return dst, resized_img
