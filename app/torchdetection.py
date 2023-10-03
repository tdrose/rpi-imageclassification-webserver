# Code for image processing and model predictions modified from: 
# https://pytorch.org/tutorials/intermediate/realtime_rpi.html

import time
import numpy as np

import cv2
from PIL import Image
from torchvision import transforms, models
import torch
torch.backends.quantized.engine = 'qnnpack'

from .imagenet_1000_classes import imagenet_1000_classes

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
cap.set(cv2.CAP_PROP_FPS, 10)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Image preprocessing
preprocess = transforms.Compose([
    # convert the frame to a CHW torch tensor for training
    transforms.ToTensor(),
    # normalize the colors to the range that mobilenet_v2/3 expect
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


net = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
net = torch.jit.script(net)

def get_imageprediction(net, cap, imagenet_1000_classes):
    with torch.no_grad():
        ret, image = cap.read()

        # convert opencv output from BGR to RGB
        image = image[:, :, [2, 1, 0]].copy()

        # preprocess
        input_tensor = preprocess(image)

        # create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0)

        # run model
        output = net(input_batch)
        # do something with output ...

        top = list(enumerate(output[0].softmax(dim=0)))
        top.sort(key=lambda x: x[1], reverse=True)
        idx, val = top[0]

        #print(f"{val.item()*100:.2f}% {imagenet_1000_classes[idx]}")

        return image, imagenet_1000_classes[idx], val.item()


if __name__ == '__main__':
    pass
