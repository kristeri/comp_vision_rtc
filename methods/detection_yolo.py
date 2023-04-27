import cv2 as cv
import numpy as np
from time import time
import torch

# Set the device (GPU or CPU) used to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_SCORE = 0.85

class ObjectDetectionYOLO:
    def __init__(self):
        self.model = self.init_model()
        self.avg_fps = 0.00
        self.nof_frames = 0.00
        self.total_fps = 0.00
        self.fps_per_frame = []

    def init_model(self):
        # Initialize model with the best available weights
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)#.autoshape()
        return model

    def init_preprocess(self):
        # Initialize the inference transforms
        preprocess = self.weights.transforms()
        return preprocess

    def sync_cuda(self):
        if (DEVICE == "cuda"): torch.cuda.synchronize()

    def detect_objects_in_frame(self, frame):
        self.sync_cuda()
        start_time = time()

        result = self.model(frame)

        frame = np.squeeze(result.render())

        self.sync_cuda()
        end_time = time()
        print(f"Frame processing took: {end_time - start_time} seconds")

        fps = 1 / np.round(end_time - start_time, 3)
        arr = self.fps_per_frame
        arr.append(fps)
        self.fps_per_frame = arr
        fps_string = "FPS: {:.2f}".format(fps)
        self.nof_frames = self.nof_frames + 1
        self.total_fps += fps
        self.avg_fps = self.total_fps / self.nof_frames
        print(f"Average FPS: {self.avg_fps}")

        cv.putText(frame, fps_string, (0, 25), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        return frame
