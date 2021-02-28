import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms,models,utils
from torchsummary import summary
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from PIL import Image

image_transform = {
    'train':transforms.Compose([
        transforms.RandomSizedCrop(size=300,scale=(0.8,1.1)),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(0.4,0.4,0.4),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=256),
        transforms.ToTensor(),
        transforms.Normalize([0.,485,0.456,0.406],\
                             [0.229,0.224,0.225])

    ]),
    'val': transforms.Compose([
        transforms.Resize(size=300),
        transforms.CenterCrop(size=256),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.4856,0.406],\
                             [0.229,0.224,0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=300),
        transforms.CenterCrop(size=256),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.4856,0.406],\
                             [0.229,0.224,0.225])
    ])
}
