from multiprocessing import Queue
from typing import Dict, Any
import torch.multiprocessing as torch_mp
from shared_memory_dict import SharedMemoryDict

from backend.schemas import (StreamInDB)
from backend.stream_process import StreamProcessor


class StreamManager:
    def __init__(self, model_process_in_queue: torch_mp.Queue, shared_data_for_stream_config: SharedMemoryDict):
        self.model_process_in_queue = model_process_in_queue
        self.shared_data_for_stream_config = shared_data_for_stream_config
        self.stream_objects: Dict[int, StreamInDB] = {}
        self.streams_processes: Dict[int, StreamProcessor] = {}

    def add_stream(self, stream_object: StreamInDB):
        if not stream_object.id in self.stream_objects:
            self.stream_objects[stream_object.id] = stream_object
            t = StreamProcessor(stream_object.id,
                                stream_object,
                                self.model_process_in_queue)
            self.streams_processes[stream_object.id] = t
            conf = stream_object.model_dump(mode="json")
            self.shared_data_for_stream_config[str(stream_object.id)] = conf
            # return
            # t.run()
            t.start()

    def remove_stream(self, stream_id: Any):
        print('remove', stream_id, " from processes ", self.streams_processes.keys())
        if stream_id in self.stream_objects:
            print('remove', stream_id)
            self.stream_objects.pop(stream_id)
            self.streams_processes.get(stream_id).stop = True
            self.streams_processes.get(stream_id).join(0)
            self.streams_processes.pop(stream_id)
            print('removed ... all processes are ..', self.streams_processes.keys())

    def update_stream(self, stream_object: StreamInDB):
        self.remove_stream(stream_object.id)
        self.add_stream(stream_object)

    def on_detect(self, msg):
        print(msg)
