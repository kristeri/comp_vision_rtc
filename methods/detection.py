from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from PIL import Image
import cv2 as cv
import numpy as np
from time import time
import torch
import math

# Set the device (GPU or CPU) used to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_SCORE = 0.85

class ObjectDetection:
    def __init__(self):
        self.weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        self.model = self.init_model()
        self.preprocess = self.init_preprocess()
        self.number_of_classes = len(self.weights.meta["categories"])
        self.colors = np.random.uniform(0, 255, size=(self.number_of_classes, 3))

    def init_model(self):
        # Initialize model with the best available weights
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=self.weights, box_score_thresh=MIN_SCORE)
        model.to(DEVICE)
        model.eval()
        for param in model.parameters():
            param.grad = None
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
        # Converting image to PIL format for computations
        pil_image = Image.fromarray(frame)
        
        # Apply inference preprocessing transforms
        img_transformed = self.preprocess(pil_image)

        # The transformed image is moved to device
        img_transformed = img_transformed.to(DEVICE)

        batch = [img_transformed]

        # Use the model and visualize the prediction
        prediction = self.model(batch)[0]
        labels = [self.weights.meta["categories"][i] for i in prediction["labels"]]

        # Scale factor for drawing
        scaled = math.ceil(len(frame) / 300)

        for i in range(0, len(prediction["boxes"])):
            # Get the confidence of the prediction
            confidence = prediction["scores"][i]
            if confidence > MIN_SCORE:
                # Get the index of the class label from the prediction
                idx = int(prediction["labels"][i])
                # Move to CPU
                box = prediction["boxes"][i].detach().cpu().numpy()
                # Get the coordinates of the object bounding box
                (x1, y1, x2, y2) = box.astype("int")

                text = "{}: {:.2f}%".format(labels[i], confidence * 100)
                # Draw the bounding box and label
                cv.rectangle(frame, (x1, y1), (x2, y2), self.colors[idx], 3)
                y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                cv.putText(frame, text, (x1, y), cv.FONT_HERSHEY_PLAIN, scaled, self.colors[idx], scaled)

        self.sync_cuda()
        end_time = time()
        print(f"Frame processing took: {end_time - start_time} seconds")

        fps = 1 / np.round(end_time - start_time, 3)
        fps_string = "FPS: {:.2f}".format(fps)
        cv.putText(frame, fps_string, (0, 25), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        return frame
