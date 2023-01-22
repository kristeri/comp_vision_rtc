from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
import torchvision.transforms as transforms

from PIL import Image
import cv2 as cv
import numpy as np
from time import time
import torch

# Set the device (GPU or CPU) used to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_SCORE = 0.85

class SemanticSegmentation:
    def __init__(self):
        self.weights = FCN_ResNet50_Weights.DEFAULT
        self.model = self.init_model()
        self.preprocess = self.init_preprocess()
        self.number_of_classes = len(self.weights.meta["categories"])
        self.colors = np.concatenate([np.array([(0, 0, 0)]), np.random.uniform(0, 255, size=(self.number_of_classes - 1, 3))])

    def init_model(self):
        # Initialize model with the best available weights
        model = fcn_resnet50(weights=self.weights)
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

    def create_rgb_mask(self, image):
        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        for label in range(0, self.number_of_classes):
            mask = image == label
            r[mask], g[mask], b[mask] = self.colors[label]
        rgb_mask = np.stack([r, g, b], axis=2)
        return rgb_mask

    def recognize_objects_in_frame(self, frame):
        self.sync_cuda()
        start_time = time()

        # Converting image to PIL format for computations
        pil_image = Image.fromarray(frame)
        
        # Apply inference preprocessing transforms
        img_transformed = self.preprocess(pil_image)

        # The transformed image is moved to device
        img_transformed = img_transformed.to(DEVICE)

        batch = img_transformed.unsqueeze(0)

        # Use the model and visualize the prediction
        prediction = self.model(batch)['out']

        height_orginal, width_orginal, dim = frame.shape
        prediction = transforms.Resize((height_orginal, width_orginal))(prediction[0])
        normalized_masks = torch.nn.functional.softmax(prediction)
        mask = torch.argmax(normalized_masks, 0).cpu().detach().numpy() 
        rgb_mask = self.create_rgb_mask(mask)
        # Apply final mask to frame
        frame = cv.addWeighted(frame, 1, rgb_mask, 1, 0)

        self.sync_cuda()
        end_time = time()
        print(f"Frame processing took: {end_time - start_time} seconds")

        fps = 1 / np.round(end_time - start_time, 3)
        fps_string = "FPS: {:.2f}".format(fps)
        cv.putText(frame, fps_string, (0, 25), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        return frame
