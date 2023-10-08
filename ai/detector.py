import os
import platform
from typing import Any, List

import cv2
import numpy as np
import torch
from ultralytics.utils.plotting import Annotator, colors

from ai.models.common import DetectMultiBackend
from ai.utils.augmentations import letterbox
from ai.utils.dataloaders import LoadStreams2
from ai.utils.general import (Profile,
                              non_max_suppression, scale_boxes)
from backend.schemas import StreamInDB


class Detector:
    def __init__(self,
                 show_stream=True,
                 weights: str = 'yolov5s.pt',
                 device: Any = torch.device('cpu'),
                 data: str = 'ai/data/coco128.yaml',
                 half: bool = False,
                 dnn: bool = False):
        print(os.getcwd())
        super().__init__()
        self.show_stream = show_stream
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
        print("result of stream with id: ", stream_object.id, results)
        # pass

    def resize_and_mask_img(self,
                            images: List[np.array],
                            boundary: List[np.array] = None,
                            img_size=(640, 640), stride=32,
                            auto=True, ):
        # h: height, w: width, c: channel, b: batch
        apply_mask = boundary is not None
        final_images = []
        for i in range(len(images)):
            img = images[i]  # whc
            img = img[..., ::-1]  # BGR to RGB
            if apply_mask:
                # https://stackoverflow.com/a/48301735
                img_copy = img.copy()
                mask = np.zeros(img_copy.shape[:2], np.uint8)
                cv2.drawContours(mask, boundary, -1, (255, 255, 255), -1, cv2.LINE_AA)
                dst = cv2.bitwise_and(img_copy, img_copy, mask=mask)
                dst = letterbox(dst, img_size, stride=stride, auto=auto)[0]
            else:
                dst = letterbox(img, img_size, stride=stride, auto=auto)[0]
            final_images.append(dst)
        final_images = np.array(final_images)
        final_images = final_images.transpose((0, 3, 1, 2))  # BHWC to BCHW
        final_images = np.ascontiguousarray(final_images)  # contiguous
        return final_images

    def detect(self, stream: LoadStreams2, stream_object: StreamInDB, stream_data: List[Any]):
        # print('process_stream >>>>', stream_object.id)
        # return
        conf_thres = 0.25  # confidence threshold
        iou_thres = 0.45  # NMS IOU threshold
        max_det = 1000  # maximum detections per image
        agnostic_nms = False  # class-agnostic NMS

        path, im0s = stream_data
        im = self.resize_and_mask_img(im0s, stream_object.boundary,
                                      img_size=stream_object.img_size,
                                      stride=stream_object.stride,
                                      auto=stream_object.auto)
        classes = stream_object.classes  # filter by class: --class 0, or --class 0 2 3
        if isinstance(classes, list) and len(classes) == 0:
            classes = None
        # Run inference
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        if isinstance(path, list) and len(path) > 0:
            path = path[0]
        with dt[0]:
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
            else:
                p, im0, frame = path, im0s.copy(), getattr(stream, 'frame', 0)
            annotator = None
            if self.show_stream:
                annotator = Annotator(im0, line_width=2, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                results = []
                for *xyxy, conf, cls in reversed(det):
                    image_cls = int(cls)  # integer class
                    label = self.names[image_cls]
                    confidence = float(conf)
                    results.append({"class": image_cls,
                                    "label": label,
                                    "cords": [point.item() for point in xyxy],
                                    "confidence": confidence})
                    if self.show_stream:
                        label = f'{label} {confidence:.2f}'
                        annotator.box_label(xyxy, label, color=colors(image_cls, True))
                # Stream results
                self.send_result(stream_object, results)
            if self.show_stream:
                # print(p)
                im0 = annotator.result()
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])

                if stream_object.boundary:
                    cv2.polylines(im0, stream_object.boundary, True, (255, 0, 0), 2)
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
