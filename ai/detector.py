import os
from typing import Any, List

import cv2
import numpy as np
import torch

from ai.models.common import DetectMultiBackend
from ai.utils.dataloaders import LoadStreams
from ai.utils.general import (Profile,
                              non_max_suppression, scale_boxes)
from backend.schemas import StreamInDB


class Detector:
    def __init__(self, weights: str = 'yolov5s.pt',
                 device: Any = torch.device('cpu'), data: str = 'ai/data/coco128.yaml', half: bool = False,
                 dnn: bool = False):
        print(os.getcwd())
        super().__init__()
        self.weights = weights
        self.device = device
        self.dnn = dnn
        self.data = data
        self.half = half
        self.model = None
        self.stride, self.names, self.pt = None, None, None

    def load_model(self):
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt

    def send_result(self, stream_object: StreamInDB, results):
        print("result of stream id ", stream_object.id)
        print(results)
        # pass

    def crop_img(self, images: np.array, boundary: np.array = None):
        # hwc
        # height, width, channel
        if boundary is None:
            return images
        final_images = []
        # https://stackoverflow.com/a/48301735
        ## (1) Crop the bounding rect
        for i in range(images.shape[0]):
            img = images[i]
            shape = img.shape
            img = img.reshape(shape[2], shape[1], shape[0])
            # rect = cv2.boundingRect(boundary)
            # x, y, w, h = rect
            croped = img.copy()
            # croped = img[y:y + h, x:x + w].copy()
            ## (2) make mask
            pts = boundary
            # pts = boundary - boundary.min(axis=0)
            mask = np.zeros(croped.shape[:2], np.uint8)
            cv2.drawContours(mask, [boundary], -1, (255, 255, 255), -1, cv2.LINE_AA)
            ## (3) do bit-op
            dst = cv2.bitwise_and(croped, croped, mask=mask)
            # dst = np.expand_dims(dst, axis=(shape[2], shape[1], shape[0]))
            dst = dst.reshape(shape[0], shape[1], shape[2])
            final_images.append(dst)
        return np.array(final_images)
        # img = imges
        # shape = img.shape
        # # img = img.reshape(shape[2], shape[1], shape[0])
        # rect = cv2.boundingRect(boundary)
        # x, y, w, h = rect
        # croped = img[y:y + h, x:x + w].copy()
        # ## (2) make mask
        # pts = boundary - boundary.min(axis=0)
        # mask = np.zeros(croped.shape[:2], np.uint8)
        # cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        # ## (3) do bit-op
        # dst = cv2.bitwise_and(croped, croped, mask=mask)
        # return dst

    def crop_img2(self, images: np.array, boundary: np.array = None):
        # hwc
        # height, width, channel
        if boundary is None:
            return images
        img = images
        shape = img.shape
        # img = img.reshape(shape[2], shape[1], shape[0])
        # rect = cv2.boundingRect(boundary)
        # x, y, w, h = rect
        # croped = img[y:y + h, x:x + w].copy()
        croped = img.copy()
        ## (2) make mask
        pts = boundary
        # pts = boundary - boundary.min(axis=0)
        mask = np.zeros(croped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        ## (3) do bit-op
        dst = cv2.bitwise_and(croped, croped, mask=mask)
        return dst

    def detect(self, stream: LoadStreams, stream_object: StreamInDB, stream_data: List[Any]):
        print('process_stream >>>>', stream_object.id)
        return
        conf_thres = 0.25  # confidence threshold
        iou_thres = 0.45  # NMS IOU threshold
        max_det = 1000  # maximum detections per image
        agnostic_nms = False  # class-agnostic NMS

        path, im, im0s, vid_cap, s = stream_data

        boundary = stream_object.boundary
        classes = stream_object.classes  # filter by class: --class 0, or --class 0 2 3
        if isinstance(classes, list) and len(classes) == 0:
            classes = None
        # Run inference
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        if isinstance(path, list) and len(path) > 0:
            path = path[0]
        base_img_shape = im.shape
        # print(base_img_shape, )
        with dt[0]:
            # im = self.crop_img(im, boundary)
            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = self.model(im, augment=True, visualize=False)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image

            seen += 1

            if len(stream) >= 1:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), stream.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(stream, 'frame', 0)
            # _p = p
            # im2222 = self.crop_img(im0, points)
            # p = Path(p)  # to Path
            # im0 = self.crop_img2(im0, points)
            # annotator = Annotator(im0, line_width=1, example=str(self.names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # for c in det[:, 5].unique():
                #     n = (det[:, 5] == c).sum()  # detections per class
                #     s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                results = []
                for *xyxy, conf, cls in reversed(det):
                    # print(len(xyxy))
                    image_cls = int(cls)  # integer class
                    label = self.names[image_cls]
                    confidence = float(conf)
                    results.append({"class": image_cls,
                                    "label": label,
                                    "cords": [point.item() for point in xyxy],
                                    "confidence": confidence})
                # annotator.box_label(xyxy, label, color=colors(image_cls, True))
                # print(1)
                self.send_result(stream_object, results)
            # Stream results
            # im0 = annotator.result()
            # # if boundary is not None:
            # #     cv2.polylines(im0, [boundary], True, (255, 0, 0), 2)
            # if platform.system() == 'Linux' and p not in windows:
            #     windows.append(p)
            #     cv2.namedWindow(str(p),
            #                     cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            #     cv2.resizeWindow(str(p), base_img_shape.shape[1], base_img_shape.shape[0])
            # cv2.imshow(str(p), im0)
            # cv2.waitKey(1)
