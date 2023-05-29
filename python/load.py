#!/usr/bin/env python
# coding: utf-8

# In[1]:

import requests
import torch
from PIL import Image
from torch import nn, optim
from torchvision import transforms
from jcopdl.callback import Callback, set_config
from torchvision.models import mobilenet_v2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


mnet = mobilenet_v2(pretrained=False)


for param in mnet.parameters():
    param.requires_grad = False


class CustomMobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.mnet = mobilenet_v2(pretrained=True)
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
    

model = CustomMobileNetV2().to(device)
criterion = nn.NLLLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
callback = Callback(model, early_stop_patience=2,outdir="model")



weights = torch.load("./model/weights_best.pth", map_location="cpu")
model.load_state_dict(weights)
model = model.to(device);



train_transform = transforms.Compose([
    transforms.RandomRotation(5),
    transforms.Resize(256),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])




input = Image.open(f'../uploads/input.jpg')
input = train_transform(input)
input.shape
transformed_image = input.unsqueeze(0)
transformed_image.shape
hehe = {0:"normal",1:"pneumonia",2:"tuberkulosis"}

with torch.no_grad():
        model.eval()
        output = model(transformed_image)
        preds = output.argmax(1).numpy()[0]

hehe[preds]





# %%
