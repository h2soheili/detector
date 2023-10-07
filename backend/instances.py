from ai.detector import Detector
from backend.loop import get_loop
from backend.schemas import UserInDB


loop = get_loop()
detector = Detector()

from backend.stream_manager import StreamManager

stream_manager = StreamManager()

# detector.process_stream = stream_manager.on_stream
user = UserInDB(id=1, username="admin", password="pass")
