import os
import platform
from datetime import datetime
from typing import Any, List, Dict

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
        # last_detected_objects - > {"stream_id": {"class_id": {"time": timestamp}}}
        self.last_detected_objects: Dict[int, Dict[int, Any]] = {}

    def load_model(self):
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt

    def send_result(self, stream_object: StreamInDB, results):
        # print("result of stream with id: ", stream_object.id, results)
        pass

    def resize_and_mask_img(self,
                            images: List[np.array],
                            boundary: List[np.array] = None,
                            img_size=(640, 640), stride=32,
                            auto=True):
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

    """
    basic object enter/exit detection
    """

    def on_detect(self, stream_object: StreamInDB, detection_results: List[Any]):
        stream_id = stream_object.id
        if not stream_id in self.last_detected_objects:
            self.last_detected_objects[stream_id] = {}
        now = datetime.now().timestamp()
        for result in detection_results:
            cls = result["class"]
            # object enters
            if not cls in list(self.last_detected_objects[stream_id].keys()):
                self.last_detected_objects[stream_id][cls] = {
                    "time": now,
                    "sent_enter_notif": False,
                    "sent_exit_notif": False,
                }
            else:
                self.last_detected_objects[stream_id][cls]["time"] = now

        entered_classes = []
        exited_classes = []
        for cls, last_config in list(self.last_detected_objects[stream_id].items()):
            time_diff = now - last_config["time"]
            # object not seen in last ? ms, so we detect it as exited
            # print("----time_diff", time_diff, "  ", now, "  t:", last_config["time"], last_config)
            if time_diff > 0.02:
                if last_config["sent_exit_notif"] == False:
                    self.last_detected_objects[stream_id][cls]["sent_exit_notif"] = True
                    self.last_detected_objects[stream_id][cls]["sent_enter_notif"] = False
                    exited_classes.append({
                        "class": cls,
                        "label": self.names[cls],
                        "time": now
                    })
            else:
                if last_config["sent_enter_notif"] == False:
                    self.last_detected_objects[stream_id][cls]["sent_enter_notif"] = True
                    self.last_detected_objects[stream_id][cls]["sent_exit_notif"] = False
                    entered_classes.append({
                        "class": cls,
                        "label": self.names[cls],
                        "time": now
                    })
        if len(entered_classes) > 0 or len(exited_classes) > 0:
            self.notif_enter_and_exits(stream_object, entered_classes, exited_classes)

    def notif_enter_and_exits(self, stream_object: StreamInDB, entered, exited):
        print("entered  ", entered, "   exited", exited)

    def get_detection_filters(self, stream_object: StreamInDB):
        boundaries = None
        include_classes = None
        if stream_object.configs is not None:
            for config in stream_object.configs:
                if config.boundary:
                    if boundaries is None:
                        boundaries = []
                    boundaries.append(np.array(config.boundary))
                if config.include_classes:
                    if include_classes is None:
                        include_classes = []
                    include_classes.append(config.include_classes)
        return boundaries, include_classes

    def detect(self, stream: LoadStreams2, stream_object: StreamInDB, stream_data: List[Any]):

        # print('process_stream >>>>', stream_object.id)
        # return
        max_det = 1000  # maximum detections per image
        agnostic_nms = False  # class-agnostic NMS
        """
        # boundaries -> for  detect inside polygons
        # include_classes -> filter by class: --class 0, or --class 0 2 3
        """
        boundaries, include_classes = self.get_detection_filters(stream_object)

        path, im0s = stream_data
        im = self.resize_and_mask_img(im0s, boundaries,
                                      img_size=stream_object.img_size,
                                      stride=stream_object.stride,
                                      auto=stream_object.auto)

        # Run inference
        windows, dt = [], (Profile(), Profile(), Profile())
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
            pred = non_max_suppression(pred,
                                       stream_object.confidence_threshold,
                                       stream_object.iou_threshold,
                                       include_classes, agnostic_nms,
                                       max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image

            if len(stream) >= 1:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), stream.count
            else:
                p, im0, frame = path, im0s.copy(), getattr(stream, 'frame', 0)
            annotator = None
            if self.show_stream:
                annotator = Annotator(im0, line_width=2, example=str(self.names))
            detection_results = []
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # Write results

                for *xyxy, conf, cls in reversed(det):
                    image_cls = int(cls)  # integer class
                    label = self.names.get(image_cls, "Unknown")
                    confidence = float(conf)
                    cords = [point.cpu().item() for point in xyxy]
                    detection_results.append({"class": image_cls,
                                              "label": label,
                                              "cords": cords,
                                              "confidence": confidence})
                    if self.show_stream:
                        label = f'{label} {confidence:.2f}'
                        annotator.box_label(cords, label, color=colors(image_cls, True))
                # Stream results
                self.send_result(stream_object, detection_results)
            self.on_detect(stream_object, detection_results)
            if self.show_stream:
                # print(p)
                im0 = annotator.result()
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])

                if boundaries:
                    cv2.polylines(im0, boundaries, True, (255, 0, 0), 2)
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
