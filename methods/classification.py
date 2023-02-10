from torchvision.models.convnext import convnext_large, ConvNeXt_Large_Weights
from PIL import Image
import cv2 as cv
import numpy as np
from time import time
import torch

# Set the device (GPU or CPU) used to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_SCORE = 0.85

class Classification:
    def __init__(self):
        self.weights = ConvNeXt_Large_Weights.DEFAULT
        self.model = self.init_model()
        self.preprocess = self.init_preprocess()
        self.number_of_classes = len(self.weights.meta["categories"])
        self.colors = np.random.uniform(0, 255, size=(self.number_of_classes, 3))

    def init_model(self):
        # Initialize model with the best available weights
        model = convnext_large(weights=self.weights)
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

    def classify_objects_in_frame(self, frame):
        self.sync_cuda()
        start_time = time()

        # Converting image to PIL format for computations
        pil_image = Image.fromarray(frame)
        
        # Apply inference preprocessing transforms
        img_transformed = self.preprocess(pil_image)
        # The transformed image is moved to device
        img_transformed = img_transformed.to(DEVICE)
        batch = torch.unsqueeze(img_transformed, 0)
        # Use the model and visualize the prediction
        prediction = self.model(batch)
        labels = self.weights.meta["categories"]
        probabilities = torch.nn.Softmax(dim=-1)(prediction)
        sorted = torch.argsort(probabilities, dim=-1, descending=True)

        for (i, idx) in enumerate(sorted[0, :5]):
            (label, probability) = (labels[idx.item()], probabilities.max().item())
            text = "{}: {:.2f}%".format(label, probability * 100)
            x = 0
            y = 50 + 20 * i
            idx = int(labels.index(label))
            cv.putText(frame, text, (x, y), cv.FONT_HERSHEY_PLAIN, 1.5, self.colors[idx], 2)

        self.sync_cuda()
        end_time = time()
        print(f"Frame processing took: {end_time - start_time} seconds")

        fps = 1 / np.round(end_time - start_time, 3)
        fps_string = "FPS: {:.2f}".format(fps)
        cv.putText(frame, fps_string, (0, 25), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        return frame
