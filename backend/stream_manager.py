from typing import Dict, Any

from backend.schemas import (StreamInDB)
from utils.dataloaders import LoadStreams


class StreamManager:
    def __init__(self):
        self.stream_objects: Dict[int, StreamInDB] = {}
        self.streams: Dict[int, LoadStreams] = {}

    def add_stream(self, stream: StreamInDB):
        if not stream.id in self.stream_objects:
            self.stream_objects[stream.id] = stream
            stream_dataset = LoadStreams(stream.source,
                                         img_size=stream.img_size,
                                         stride=stream.stride,
                                         auto=True,
                                         vid_stride=1)
            self.streams[stream.id] = stream_dataset

    def remove_stream(self, stream_id: Any):
        if id in self.stream_objects:
            try:
                stream = self.streams[stream_id]
                for t in stream.threads:
                    t.join()
            except Exception as e:
                print(e)
            finally:
                self.stream_objects.pop(stream_id)
                self.streams.pop(stream_id)
