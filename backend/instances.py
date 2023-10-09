from ai.detector import Detector
from backend.loop import get_loop
from backend.schemas import StreamInDB, DetectorConfigInDB, AfterDetectTriggerInDB, DetectionPolicyInDB
from backend.schemas import UserInDB

loop = get_loop()
detector = Detector()

from backend.stream_manager import StreamManager
from backend.socket_manager import SocketManager

stream_manager = StreamManager()
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
    "after_detect_triggers": [trigger.model_dump()],
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
    # "boundary": None, # without bounding box
    # base on FHD monitor
    "img_size": [480, 480],
    "stride": 32,
    "auto": True,
    "vid_stride": 1,
    "confidence_threshold": 0.55,
    "iou_threshold": 0.45,
    "configs": [config.model_dump()],
    "detection_policy": policy.model_dump()
}, user=user)
