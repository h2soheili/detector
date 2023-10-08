from ai.detector import Detector
from backend.loop import get_loop
from backend.schemas import StreamInDB
from backend.schemas import UserInDB

loop = get_loop()
detector = Detector()

from backend.stream_manager import StreamManager
from backend.socket_manager import SocketManager

stream_manager = StreamManager()
socket_manager = SocketManager()

user = UserInDB(id=1, username="admin", password="pass")

stream = StreamInDB(**{
    "id": 1,
    "name": "webcam stream",
    # "source": "https://demo.unified-streaming.com/k8s/features/stable/video/tears-of-steel/tears-of-steel.ism/.m3u8",
    "source": "0",
    # "boundary": None, # without bounding box
    # "boundary": [[[20, 20], [600, 20], [600, 1060], [20, 1060]]],
    "boundary": [[[500, 0], [500, 1080], [0, 0]],
                 [[1600, 20], [1700, 20], [1900, 1060], [1600, 1060]]],
    # base on FHD monitor
    "img_size": [480, 480],
    "stride": 32,
    "auto": True,
    "vid_stride": 1,
    "classes": None
}, user=user)

# stream = StreamInDB(**{
#     "id": 2,
#     "name": "online stream",
#     "source": "https://demo.unified-streaming.com/k8s/features/stable/video/tears-of-steel/tears-of-steel.ism/.m3u8",
#     "boundary": None,
#     "img_size": [
#         480,
#         480
#     ],
#     "stride": 32,
#     "auto": True,
#     "vid_stride": 1,
#     "classes": []
# }, user=user)
