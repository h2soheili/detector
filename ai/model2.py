from math import floor
from typing import Dict, Any, Optional

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
        msg = model_process_in_queue.get()
        if msg is None:
            continue
        stream_id = msg["id"]
        original_image = msg["image"]  # BGR
        original_image = np.array(original_image, dtype=np.float32)

        stream_config = shared_data_for_stream_config[str(stream_id)]
        if stream_id not in stream_filters_dict:
            stream_filters_dict[stream_id] = get_detection_filters(stream_config)
        boundaries, include_classes = stream_filters_dict[stream_id]
        masked_and_resized_image, resized_image = resize_and_mask_img(original_image,
                                                                      boundary=boundaries,
                                                                      img_size=stream_config["img_size"])
        # image_size = img.shape[:2]
        resized_image_size = resized_image.shape[:2]
        # rgb_resized_image = resized_image[..., -1] # BGR to RGB
        results: Results = model.track(masked_and_resized_image,
                                       classes=include_classes,
                                       conf=stream_config["confidence_threshold"] or 0.25,
                                       iou=stream_config["iou_threshold"] or 0.75,
                                       imgsz=resized_image_size,
                                       device=device,
                                       save=False,
                                       stream=False,
                                       max_det=50,
                                       persist=True,
                                       tracker="bytetrack.yaml",
                                       verbose=False)

        final_result = dict()
        for result in results:
            if result.boxes.shape[0] != 0:
                for box in result.boxes.cpu():
                    """"
                    {"class": image_cls,
                                          "label": label,
                                          "cords": cords,
                                          "confidence": confidence}
                    """
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
            "process_number": process_number,
            "id": stream_id,
            "results": final_result,
            "original_image": original_image,
            "masked_and_resized_image": masked_and_resized_image.tolist(),
            "resized_image": resized_image.tolist(),
            "timestamp": msg["timestamp"]
        }
        model_process_out_queue.put(obj)
