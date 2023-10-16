import math
import multiprocessing as mp
import os
import sys
import time
from multiprocessing import Process, Queue
from typing import Any
from urllib.parse import urlparse
import torch.multiprocessing as torch_mp
import cv2
import numpy as np
from ultralytics.utils.checks import check_requirements

from backend.schemas import StreamInDB


def is_colab():
    # Is environment a Google Colab instance?
    return 'google.colab' in sys.modules


def is_kaggle():
    # Is environment a Kaggle Notebook?
    return os.environ.get('PWD') == '/kaggle/working' and os.environ.get('KAGGLE_URL_BASE') == 'https://www.kaggle.com'


# class StreamProcessor:
class StreamProcessor(Process):
    def __init__(self, stream_id: Any, stream_object: StreamInDB, model_process_in_queue: torch_mp.Queue, ):
        super().__init__(name=str(stream_id))
        self.stream_object = stream_object
        self.model_process_in_queue = model_process_in_queue
        self.stream_id = stream_id
        self.exit = mp.Event()
        self.images = []
        self.frames = float('inf')
        self.fps = 30

    def run(self) -> None:
        # Start thread to read frames from video stream
        source = self.stream_object.source
        if urlparse(source).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):  # if source is YouTube video
            # YouTube format i.e. 'https://www.youtube.com/watch?v=Zgi9g1ksQHc' or 'https://youtu.be/LNwODJXcvt4'
            check_requirements(('pafy', 'youtube_dl==2020.12.2'))
            import pafy
            source = pafy.new(source).getbest(preftype='mp4').url  # YouTube URL
        source = eval(source) if source.isnumeric() else source  # i.e. s = '0' local webcam
        if source == 0:
            assert not is_colab(), '--source 0 webcam unsupported on Colab. Rerun command in a local environment.'
            assert not is_kaggle(), '--source 0 webcam unsupported on Kaggle. Rerun command in a local environment.'
        print("---->>>>>> cv2.VideoCapture(s) ", source)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f'Failed to open {source}')
            cap.release()
            exit()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
        self.frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback
        self.fps = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

        _, self.images = cap.read()  # guarantee first frame

        n, f = 0, self.frames

        print(f'{source} Success ({self.frames} frames {w}x{h} at {self.fps:.2f} FPS)')
        while not self.exit.is_set() and cap.isOpened():
            n += 1
            grabbed = cap.grab()
            if not grabbed:
                continue
            if n % self.stream_object.vid_stride == 0:
                success, im = cap.retrieve()
                if success:
                    # print('stream process')
                    # self.images = im
                    self.model_process_in_queue.put({
                        "id": self.stream_id,
                        "data": im.tolist(),
                    })
                else:
                    print('WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.')
                    # self.images = np.zeros_like(self.images)
                    cap.open(source)  # re-open stream if signal was lost
            # if self.stream_object.debounce_time > 0:
            #     time.sleep(self.stream_object.debounce_time)  # wait time

        cap.release()
