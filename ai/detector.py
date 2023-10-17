import os
from typing import Any, List, Dict

import cv2
import matplotlib
import numpy as np
import torch

from ai.helpers import letterbox
from backend.schemas import StreamInDB

matplotlib.use('Agg')


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
        self.exited_items: Dict[int, Any] = {}

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

    def on_detect(self, msg: Any, ):
        # print('main process')
        # def on_detect(self, msg):
        # print("on_detect", msg)
        # return
        stream_id = msg["id"]
        detection_results: Dict[int, Dict[int, Any]] = msg["results"]

        if stream_id not in self.last_detected_objects:
            self.last_detected_objects[stream_id] = {}

        entered: Dict[int, Dict[int, Any]] = {}
        exited: Dict[int, Dict[int, Any]] = {}

        for item_class, class_objects in detection_results.items():
            if item_class not in self.last_detected_objects[stream_id]:
                entered[item_class] = class_objects
            else:
                last_class_objects = self.last_detected_objects[stream_id][item_class]
                last_ids = len(last_class_objects.keys())
                new_ids = len(class_objects.keys())
                copy_of_last_class_objects = last_class_objects.copy()
                copy_of_class_objects = class_objects.copy()
                new_class_objects_is_base = True if new_ids > last_ids else False
                bigger_dict = copy_of_class_objects if new_class_objects_is_base else copy_of_last_class_objects

                for item_id, item in bigger_dict.items():
                    if new_class_objects_is_base:
                        if item_id not in copy_of_last_class_objects:
                            entered[item_id] = item
                        else:
                            del copy_of_last_class_objects[item_id]
                    else:
                        if item_id not in copy_of_class_objects:
                            exited[item_id] = item
                        else:
                            del copy_of_class_objects[item_id]

                if new_class_objects_is_base:
                    for item_id, item in copy_of_last_class_objects.items():
                        exited[item_id] = item
                else:
                    for item_id, item in copy_of_class_objects.items():
                        entered[item_id] = item

        self.last_detected_objects[stream_id] = detection_results
        self.notif_enter_and_exits(stream_id, entered, exited)

        # now = datetime.now().timestamp()
        # for result in detection_results:
        #     cls = result["class"]
        #     # img = msg["image"]
        #     # img = np.array(img, dtype=np.float32)
        #     # img = img.transpose((2, 1, 0))
        #     # cv2.imshow("webcam", img)
        #     # object enters
        #     # print(1)
        #     if not cls in self.last_detected_objects[stream_id]:
        #         self.last_detected_objects[stream_id][cls] = {
        #             "time": msg["timestamp"],
        #             "sent_enter_notif": False,
        #             "sent_exit_notif": False,
        #             "class": result["class"],
        #             "label": result["label"],
        #             "confidence": result["confidence"],
        #             "cords": result["cords"],
        #         }
        #     else:
        #         self.last_detected_objects[stream_id][cls]["time"] = now
        #
        # entered_classes = []
        # exited_classes = []
        # for cls, last_config in list(self.last_detected_objects[stream_id].items()):
        #     time_diff = now - last_config["time"]
        #     # object not seen in last ? ms, so we detect it as exited
        #     # print("----time_diff", time_diff, "  ", now, "  t:", last_config["time"], last_config)
        #     if time_diff > 0.01:
        #         if last_config["sent_exit_notif"] == False:
        #             self.last_detected_objects[stream_id][cls]["sent_exit_notif"] = True
        #             self.last_detected_objects[stream_id][cls]["sent_enter_notif"] = False
        #             exited_classes.append({
        #                 "class": cls,
        #                 "label": last_config["label"],
        #                 "time": now
        #             })
        #     else:
        #         if last_config["sent_enter_notif"] == False:
        #             self.last_detected_objects[stream_id][cls]["sent_enter_notif"] = True
        #             self.last_detected_objects[stream_id][cls]["sent_exit_notif"] = False
        #             entered_classes.append({
        #                 "class": cls,
        #                 "label": last_config["label"],
        #                 "time": now
        #             })
        # if len(entered_classes) > 0 or len(exited_classes) > 0:
        #     self.notif_enter_and_exits(stream_id, entered_classes, exited_classes)

    def notif_enter_and_exits(self, stream_id: Any, entered, exited):
        print(f"stream_id {stream_id}  entered:", entered, "   exited:", exited)

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
