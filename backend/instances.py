from ai.detector import Detector
from backend.loop import get_loop
from backend.schemas import UserInDB
from backend.stream_manager import StreamManager

loop = get_loop()

stream_manager = StreamManager()
detector = Detector(stream_manager=stream_manager)
user = UserInDB(id=1, username="admin", password="pass")
