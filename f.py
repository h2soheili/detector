import os
import time
from collections import defaultdict
from datetime import datetime
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from ai.helpers import get_device, resize_and_mask_img
from ai.utils.dataloaders import LoadStreams2

if __name__ == '__main__':
    num_processes = 2
    seed = 1
    cpu_count = os.cpu_count()
    device = get_device("cpu")
    model = YOLO("yolov8m.pt").to(device=device)
    original_size = (1080, 1920)  # hw
    img_size = (640, 640)  # hw
    # img_size = (384, 640)  # hw
    dataset = LoadStreams2('0', img_size=img_size, stride=32, auto=True, vid_stride=1)
    # print(platform.system(), "platform.system()")
    points = None
    # points = [np.array([[20, 150], [600, 20], [600, 560], [500, 760], [20, 1060]], dtype=np.uint)]
    # new_points = [np.array(poly * (np.array(img_size[::-1]) / np.array(original_size[::-1])), dtype=np.uint) for poly in
    #               points]
    # print(img_size[::-1])
    # print(new_points)
    # re
    # exit()
    track_config = dict({
        "0": 0.1,
        "1": 0.3,
    })
    # track_history = defaultdict(lambda: {})
    track_history = defaultdict(lambda: [])

    data = {
        "images": [],
        "resized_images": [],
    }
    for path, images0 in dataset:
        now = datetime.now().timestamp()
        # print(2)
        # print(len(images0))
        image1, resized_img = resize_and_mask_img(images0[0], points, img_size=img_size, stride=32, auto=True)
        data["images"].append(image1)
        data["resized_images"].append(resized_img)
        # if len(data["images"]) < 5:
        #     continue
        # d = np.array(data["images"]).transpose(0, 3, 1, 2)
        # d = np.ascontiguousarray(d)
        results: Results = model.track(resized_img,
                                       stream=False,
                                       classes=None,
                                       conf=0.25,
                                       iou=0.75,
                                       imgsz=resized_img.shape[:2],
                                       device=device,
                                       max_det=20,
                                       persist=False,
                                       tracker="bytetrack.yaml",
                                       verbose=True)
        time.sleep(0.2)
        # Get the boxes and track IDs
        if results[0].boxes.shape[0] == 0:
            continue
        object_classes = results[0].boxes.cls.int().cpu().tolist() if results[0].boxes.cls is not None else []
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
        # for box in results[0].boxes
        if len(object_classes) != len(track_ids):
            print("track_ids", track_ids, " classes", object_classes)
            break
        print("track_ids", track_ids, " classes", object_classes)
            # Visualize the results on the frame
        annotated_frame = results[0].plot()
        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        # data["images"].clear()
        # data["resized_images"].clear()

        # print(1)
        # new_points = [
        #     np.array(poly * (np.array(resized_img.shape[:2][::-1]) / np.array(original_size[::-1])), dtype=np.uint)
        #     for
        #     poly in
        #     points] if points else None
        # print(new_points)
        # break
        # print("f")
        # annotated_frame = results[0].plot()
        # # Display the annotated frame
        # cv2.imshow("YOLOv8 Inference", annotated_frame)
        # continue
        # annotator = Annotator(resized_img, line_width=3, example="example")
        # # annotator = Annotator(image1, line_width=3, example="example")
        # entered = []
        # exited = []
        # for result in results:
        #     if result.boxes:
        #         print(444)
        #         track_ids = result.boxes.id.int().cpu().tolist()
        #         for box, track_id in zip(result.boxes.cpu(), track_ids):
        #             item_class = int(box.cls.item())
        #             item_class_str = str(item_class)
        #             label = f'{result.names[item_class]} {box.conf.item():.2f}'
        #             # color = (255, 122, 220)
        #             color = colors(item_class, True)
        #             cords = box.xyxy.cpu().numpy().flatten()
        #             annotator.box_label(cords, label, color=color)
        #             class_trackers = track_history[item_class_str]
        #             x, y, w, h = box
        #             track_id_str = str(track_id)
        #             if not track_id_str in class_trackers:
        #                 track_history[item_class_str][track_id_str] = {
        #                     "time": now,
        #                     "class": item_class,
        #                     "track_id": track_id,
        #                     "tracks": [(float(x), float(y))]
        #                 }
        #                 entered.append({
        #                     "class": item_class,
        #                     "label": result.names[item_class],
        #                     "confidence": box.conf.item(),
        #                     "cords": cords,
        #                 })
        #             else:
        #                 track_history[item_class_str][track_id_str]["tracks"].append((float(x), float(y)))
        #                 track_history[item_class_str][track_id_str]["time"] = now

        # for item_class_str in track_history.keys():
        #     for track_id_str in track_history[item_class_str].keys():
        #         last_state = track_history[item_class_str][track_id_str]
        #         conf = track_config[item_class_str]
        #         # if not conf:
        #         #     if (now - last_state["time"])>0.1:

        # print(1)
        # if platform.system() == 'Linux' and p not in windows:
        #     windows.append(p)
        #     cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
        #     cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        # im1 = annotator.result()
        # if new_points:
        #     cv2.polylines(im1, new_points, True, (255, 0, 0), 2)
        # cv2.imshow("webcam", im1)
        # cv2.waitKey(0)  # 1 millisecond
