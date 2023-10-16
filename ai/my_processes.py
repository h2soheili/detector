import multiprocessing as mp
import os
from multiprocessing.shared_memory import SharedMemory

import torch.multiprocessing as torch_mp
from shared_memory_dict import SharedMemoryDict
from ultralytics import YOLO

from ai.helpers import get_device
from ai.model import model_process_target


def serve_model(model_process_in_queue: torch_mp.Queue, model_process_out_queue: torch_mp.Queue, detector,
                shared_data_for_stream_config: SharedMemoryDict):
    num_processes = 6
    seed = 1
    cpu_count = os.cpu_count()
    device = get_device("cpu")
    model = YOLO("yolov8m.pt")
    # NOTE: this is required for the ``fork`` method to work
    # gradients are allocated lazily, so they are not shared here
    model.share_memory()
    processes = []
    for process_number in range(num_processes):
        p = torch_mp.Process(target=model_process_target,
                             args=(
                                 model_process_in_queue,
                                 model_process_out_queue,
                                 model,
                                 device,
                                 cpu_count,
                                 num_processes,
                                 seed, process_number, shared_data_for_stream_config))
        # We first train the model across `num_processes` processes
        processes.append(p)
    for p in processes:
        p.start()
    run_loop = True
    while run_loop:
        msg = model_process_out_queue.get()
        if msg:
            detector.on_detect(msg)

    for p in processes:
        p.join()

    print("________exit_________")


def send_stream_to_model_process(model_process_in_queue: torch_mp.Queue,
                                 stream_process_out_queue: mp.Queue):
    run_loop = True
    while run_loop:
        # print('1 process')
        msg = stream_process_out_queue.get()
        # print('1 process')
        if msg:
            model_process_in_queue.put(msg)
