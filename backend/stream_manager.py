from typing import Dict, Any

from ai.utils.dataloaders import LoadStreams2
from backend.instances import detector
from backend.schemas import (StreamInDB)


class StreamProcessor:
    # class StreamProcessor(Thread):
    def __init__(self, stream_id, stream, stream_object):
        # super().__init__(name=str(stream_id))
        self.stream = stream
        self.stream_object = stream_object
        self.stream_id = stream_id
        self.stop = False

    def run(self) -> None:
        stream = self.stream
        stream_object = self.stream_object
        for path, im0s in stream:
            # print("runrunrunrunrun 222", self.stop)
            if self.stop:
                break
            # print('on_stream')
            stream_data = (path, im0s)
            detector.detect(stream, stream_object, stream_data)


class StreamManager:
    def __init__(self):
        self.stream_objects: Dict[int, StreamInDB] = {}
        self.streams: Dict[int, LoadStreams2] = {}
        self.streams_threads: Dict[int, StreamProcessor] = {}

    def add_stream(self, stream: StreamInDB):
        if not stream.id in self.stream_objects:
            self.stream_objects[stream.id] = stream
            stream_dataset = LoadStreams2(stream.source,
                                          img_size=stream.img_size,
                                          stride=stream.stride,
                                          auto=True,
                                          vid_stride=1,
                                          debounce_time=0.0)
            self.streams[stream.id] = stream_dataset
            t = StreamProcessor(stream.id, stream_dataset, stream)
            self.streams_threads[stream.id] = t
            t.run()
            # t.start()

    def remove_stream(self, stream_id: Any):
        print('remove', stream_id, self.streams_threads.keys())
        if stream_id in self.stream_objects:
            print('remove', stream_id)
            self.stream_objects.pop(stream_id)
            self.streams.pop(stream_id)
            # self.streams_threads.get(stream_id).join(0)
            self.streams_threads.get(stream_id).stop = True
            self.streams_threads.pop(stream_id)
            print('remove', self.streams_threads.keys())
