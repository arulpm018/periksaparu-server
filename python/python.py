#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v2


class CustomMobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.mnet = mobilenet_v2(weights=None)
        self.freeze()
        self.mnet.classifier = nn.Sequential(
            nn.Linear(1280, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,3),
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        return self.mnet(x)

    def freeze(self):
        for param in self.mnet.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.mnet.parameters():
            param.requires_grad = True



def predict_image(image_path):



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CustomMobileNetV2().to(device)

    weights = torch.load("./model/weights_best.pth", map_location="cpu")
    model.load_state_dict(weights)
    model = model.to(device)

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        model.eval()

        # Load image and apply transformations
        inputs = Image.open(image_path)
        inputs = train_transform(inputs)
        transformed_image = inputs.unsqueeze(0)

        # Make predictions
        output = model(transformed_image)
        preds = output.argmax(1).numpy()[0]

    # Return prediction as string label
        labels = {0: "normal", 1: "pneumonia", 2: "tuberkulosis"}
        return labels[preds]

import sys



data_to_pass_back = 'halo teman teman'
image_path = sys.argv[1]
prediction = data_to_pass_back
print(image_path)


sys.stdout.flush()





# %%
