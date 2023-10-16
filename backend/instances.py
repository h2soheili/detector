from multiprocessing import Queue

import torch.multiprocessing as torch_mp

from ai.detector import Detector
from backend.loop import get_loop
from backend.schemas import StreamInDB, DetectorConfigInDB, AfterDetectTriggerInDB, DetectionPolicyInDB
from backend.schemas import UserInDB

loop = get_loop()
detector = Detector()
from shared_memory_dict import SharedMemoryDict

shared_data_for_stream_config = SharedMemoryDict(name='stream_configs', size=1024)

# send message to child process from main process
model_process_in_queue = torch_mp.Queue()
# get message from child process
model_process_out_queue = torch_mp.Queue()

stream_process_out_queue = Queue()

from backend.stream_manager import StreamManager
from backend.socket_manager import SocketManager

stream_manager = StreamManager(model_process_in_queue, shared_data_for_stream_config)
socket_manager = SocketManager()

user = UserInDB(id=1, username="admin", password="pass")
trigger = AfterDetectTriggerInDB(**{
    "id": 1,
    "type": "alert",
    "config": {
        "file": "hello.mp3"
    },
    "target_classes": None
}, user=user)

config = DetectorConfigInDB(**{
    "id": 1,
    "include_classes": None,
    "exclude_classes": [0],
    "after_detect_triggers": [trigger.model_dump(mode='json')],
    # "boundary": None, # without bounding box
    # base on FHD monitor
    "boundary": [[20, 150], [600, 20], [600, 560], [500, 760], [20, 1060]],
}, user=user)

policy = DetectionPolicyInDB(**{
    "id": 1,
    "type": "debounce",
    "config": {"time": 0.001}
}, user=user)

stream = StreamInDB(**{
    "id": 1,
    "name": "webcam stream",
    # "source": "https://demo.unified-streaming.com/k8s/features/stable/video/tears-of-steel/tears-of-steel.ism/.m3u8",
    "source": "0",
    # "source": "http://192.168.1.101:8081/video",
    # "source": "rtsp://192.168.1.101:8554/live",
    "img_size": [640, 640],
    "stride": 32,
    "auto": True,
    "vid_stride": 1,
    "debounce_time": 0.0,
    "confidence_threshold": 0.25,
    "iou_threshold": 0.75,
    "configs": [],
    # "configs": [config.model_dump()],
    "detection_policy": policy.model_dump(mode='json')
}, user=user)
