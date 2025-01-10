import torch
from torch.nn import functional as F
from functools import  partial
import random

def cout_grad(grad, layer="", flatten=1):
    if flatten:
        print(f"Gradient of {layer}:", grad.flatten())
    else:
        print(f"Gradient of {layer}:", grad)


class CNN:
    def __init__(self):
        self.image = None
        self.out_conv1 = None
        self.out_relu1 = None
        self.out_pool1 = None
        self.out_conv2 = None
        self.out_relu2 = None
        self.out_pool2 = None
        self.out_fc1 = None
        self.out_fc2 = None
        self.out_ec = None

        # weight
        self.conv1_kernel = None

        self.conv2_kernel = None
        self.fc1_weight = None
        self.fc2_weight = None

        self.target = None

    def ini_param(self, image, conv1_kernel, conv2_kernel, fc1, fc2, target):
        self.image = torch.tensor(image, requires_grad=True)
        self.conv1_kernel = torch.tensor(conv1_kernel, requires_grad=True)
        self.conv2_kernel = torch.tensor(conv2_kernel, requires_grad=True)
        self.fc1_weight = torch.tensor(fc1, requires_grad=True)
        self.fc2_weight = torch.tensor(fc2, requires_grad=True)
        self.target = torch.zeros(10, requires_grad=True)
        self.target[target] = 1.0

    def train(self):
        self.out_conv1 = F.conv2d(self.image, self.conv1_kernel, padding=0)
        self.out_conv1.register_hook(partial(cout_grad, layer="conv1"))
        self.out_relu1 = F.relu(self.out_conv1)
        self.out_relu1.register_hook(partial(cout_grad, layer="relu1"))
        self.out_pool1 = F.max_pool2d(self.out_relu1)
        self.out_pool1.register_hook(partial(cout_grad, layer="pool1"))
        self.out_conv2 = F.conv2d(self.out_pool1, self.conv2_kernel, padding=0)




def main():
    pass