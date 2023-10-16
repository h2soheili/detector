from math import floor
from typing import Dict, Any, Optional, List

import cv2
import numpy as np
import torch
import torch.multiprocessing as torch_mp
from shared_memory_dict import SharedMemoryDict
from ultralytics import YOLO
from ultralytics.engine.results import Results


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
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
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
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
    img_copy = img.copy()
    resized_img = letterbox(img_copy, img_size, stride=stride, auto=auto)[0]
    if apply_mask:
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
    run_loop = True
    model = model.to(device=device)
    i = 0
    while run_loop:
        # print("______rank ", rank)
        # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
        # print(results)
        # time.sleep(1)
        # get message
        # print(1111)
        msg = model_process_in_queue.get()
        # print(msg)
        if msg:
            # print('model process')
            # print("model_process_target", msg)
            stream_id = msg["id"]
            stream_config = shared_data_for_stream_config[str(stream_id)]
            if not stream_config:
                continue
            if not stream_id in stream_filters_dict:
                stream_filters_dict[stream_id] = get_detection_filters(stream_config)

            img = msg["data"]
            img = np.array(img, dtype=np.float32)
            hw = img.shape[:2]
            boundaries, include_classes = stream_filters_dict[stream_id]
            img, resized_img = resize_and_mask_img(img,
                                                   boundary=boundaries,
                                                   img_size=stream_config["img_size"])

            # print(resized_img.shape)
            # if i > 100:
            #     im = Image.fromarray(resized_img, 'RGB')  # RGB PIL image
            #     # im = Image.fromarray(resized_img.T, 'BGR')  # RGB PIL image
            #     im.show()
            #     im.save(str(time.time()) + 'results.png')  # save image
            #     break
            # i+=1
            # print(i)
            # continue
            # t0 = time.time()
            results: Results = model.track(img,
                                           classes=include_classes,
                                           conf=stream_config["confidence_threshold"] or 0.25,
                                           iou=stream_config["iou_threshold"] or 0.75,
                                           imgsz=img.shape[:2],
                                           device=device,
                                           save=False,
                                           stream=False,
                                           max_det=50,
                                           verbose=False)
            # t1 = time.time()
            # print(f"took {t1-t0}.2f")
            # Process results generator
            final_result = []
            for result in results:
                # im_array = result.plot()  # plot a BGR numpy array of predictions
                # im.show()  # show image
                # im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                # im.save('results.jpg')  # save image
                if result.boxes:
                    track_ids = result.boxes.id.int().cpu().tolist()
                    for box in result.boxes.cpu():
                        """"
                        {"class": image_cls,
                                              "label": label,
                                              "cords": cords,
                                              "confidence": confidence}
                        """
                        item_class = int(box.cls.item())
                        final_result.append({
                            "cords": box.xyxy.tolist(),
                            "class": item_class,
                            "label": result.names[item_class],
                            "confidence": box.conf.item(),
                        })

            if len(final_result) > 0:
                model_process_out_queue.put({
                    "process_number": process_number,
                    "id": stream_id,
                    "results": final_result,
                    "image": img.tolist(),
                    "resized_img": resized_img.tolist(),
                    "hw": hw,
                    "resized_hw": resized_img.shape[:2]
                })
