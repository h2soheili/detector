from math import floor
from typing import Dict, Any, Optional

import cv2
import numpy as np
import torch
import torch.multiprocessing as torch_mp
from shared_memory_dict import SharedMemoryDict
from ultralytics import YOLO
from ultralytics.engine.results import Results

from ai.helpers import resize_and_mask_img


def get_detection_filters(stream_object: Any):
    boundaries = None
    include_classes = None
    if stream_object["configs"] is not None:
        for config in stream_object["configs"]:
            if config["boundary"]:
                if boundaries is None:
                    boundaries = []
                boundaries.append(np.array(config["boundary"]))
            if config["include_classes"]:
                if include_classes is None:
                    include_classes = []
                include_classes.append(config["include_classes"])
    return boundaries, include_classes


def get_batch(q: torch_mp.Queue, batch: int, shared_data_for_stream_config: SharedMemoryDict,
              stream_filters_dict: Dict[int, Any]):
    result = {}
    for i in range(batch):
        msg = None
        try:
            msg = q.get(block=True, timeout=1)
        except:
            continue
        if msg is None:
            continue
        stream_id = msg["id"]
        original_image = msg["image"]  # BGR
        stream_config = shared_data_for_stream_config[str(stream_id)]
        if stream_id not in stream_filters_dict:
            stream_filters_dict[stream_id] = get_detection_filters(stream_config)
        boundaries, include_classes = stream_filters_dict[stream_id]
        masked_and_resized_image, resized_image = resize_and_mask_img(original_image,
                                                                      boundary=boundaries,
                                                                      img_size=stream_config["img_size"])

        masked_and_resized_image = cv2.cvtColor(masked_and_resized_image, cv2.COLOR_BGR2RGB)  # BGR to RGB
        masked_and_resized_image = masked_and_resized_image.transpose((2, 0, 1))  # hwc to chw
        masked_and_resized_image = masked_and_resized_image / 255
        masked_and_resized_image = masked_and_resized_image.tolist()
        if stream_id not in result:
            result[stream_id] = {
                "id": stream_id,
                "timestamp": msg["timestamp"],
                "stream_config": stream_config,
                "boundaries": boundaries,
                "include_classes": include_classes,
                "resized_image_size": resized_image.shape[:2],
                "original_images": [],
                "masked_and_resized_images": [],
                "resized_images": [],
            }
        result[stream_id]["original_images"].append(original_image)
        result[stream_id]["resized_images"].append(resized_image)
        result[stream_id]["masked_and_resized_images"].append(masked_and_resized_image)
    if len(result.keys()) > 0:
        return result
    return None


def model_process_target(model_process_in_queue: torch_mp.Queue,
                         model_process_out_queue: torch_mp.Queue,
                         model: YOLO,
                         device=torch.device("cpu"),
                         cpu_count=1,
                         num_processes=1,
                         seed=1,
                         process_number=2,
                         shared_data_for_stream_config: Optional[SharedMemoryDict] = None
                         ):
    torch.manual_seed(seed + process_number)

    #### define the num threads used in current sub-processes
    torch.set_num_threads(floor(cpu_count / num_processes))
    stream_filters_dict: Dict[int, Any] = {}
    run_loop = 1
    model = model.to(device=device)
    while run_loop:
        messages = get_batch(model_process_in_queue, 5, shared_data_for_stream_config, stream_filters_dict)
        if messages is None:
            continue
        for stream_id, item in messages.items():
            stream_config = item["stream_config"]
            include_classes = item["include_classes"]
            resized_image_size = item["resized_image_size"]
            masked_and_resized_images = torch.tensor(item["masked_and_resized_images"],
                                                     dtype=torch.float32,
                                                     device=device)
            # print(masked_and_resized_images.shape)
            results: Results = model.track(masked_and_resized_images,
                                           classes=include_classes,
                                           conf=stream_config["confidence_threshold"] or 0.25,
                                           iou=stream_config["iou_threshold"] or 0.75,
                                           imgsz=resized_image_size,
                                           device=device,
                                           save=False,
                                           stream=False,
                                           max_det=50,
                                           persist=False,
                                           tracker="bytetrack.yaml",
                                           verbose=False)

            final_result = dict()
            for result in results:
                if result.boxes.shape[0] != 0:
                    for box in result.boxes.cpu():
                        item_class = int(box.cls.item()) if box.cls is not None else None
                        item_id = int(box.id.item()) if box.id is not None else None
                        if item_class is not None and item_id is not None:
                            if item_class not in final_result:
                                final_result[item_class] = {}
                            final_result[item_class][item_id] = {
                                "id": item_id,
                                "class": item_class,
                                "label": result.names[item_class],
                                # "cords": box.xyxy.tolist(),
                                "confidence": round(box.conf.item(), 2),
                            }

            obj = {
                **item,
                "process_number": process_number,
                "results": final_result,
            }
            model_process_out_queue.put(obj)
