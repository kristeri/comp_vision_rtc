from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from PIL import Image
import cv2 as cv
import numpy as np
from time import time
import torch
import math

# Set the device (GPU or CPU) used to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_SCORE = 0.85

class InstanceSegmentation:
    def __init__(self):
        self.weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = self.init_model()
        self.preprocess = self.init_preprocess()
        self.number_of_classes = len(self.weights.meta["categories"])
        self.colors = np.random.uniform(0, 255, size=(self.number_of_classes, 3))
        self.avg_fps = 0.00
        self.nof_frames = 0.00
        self.total_fps = 0.00
        self.fps_per_frame = []

    def init_model(self):
        # Initialize model with the best available weights
        model = maskrcnn_resnet50_fpn_v2(weights=self.weights)
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

    def recognize_objects_in_frame(self, frame):
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

        for i in range(0, len(prediction["masks"])):
            confidence = prediction["scores"][i]
            if confidence > MIN_SCORE:
                idx = int(prediction["labels"][i])
                # Move to CPU
                mask = prediction['masks'][i, 0].detach().cpu().numpy()
                box = prediction['boxes'][i].detach().cpu().numpy()
                
                text = "{}: with confidence {:.2f}%".format(labels[i], confidence * 100)
                (x1, y1, x2, y2) = box.astype("int")

                text = "{}: {:.2f}%".format(labels[i], confidence * 100)
                #print("The object is: {}".format(text))
                scaled = 2

                r = np.zeros_like(mask).astype(np.uint8)
                g = np.zeros_like(mask).astype(np.uint8)
                b = np.zeros_like(mask).astype(np.uint8)
                r[mask > MIN_SCORE], g[mask > MIN_SCORE], b[mask > MIN_SCORE] = self.colors[idx]
                rgb_mask = np.stack([r, g, b], axis=2)

                frame = cv.addWeighted(frame, 1, rgb_mask, 1, 0)
                cv.rectangle(frame, (x1, y1), (x2, y2), self.colors[idx], scaled)
                y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                cv.putText(frame, text, (x1, y), cv.FONT_HERSHEY_PLAIN, scaled, self.colors[idx], scaled)

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
