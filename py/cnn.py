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

        self.pd_res = None
        self.target = None

    def ini_param(self, image, conv1_kernel, conv2_kernel, fc1,fc1_bias, fc2,fc2_bias, target):
        self.image = torch.tensor(image, requires_grad=True)
        self.conv1_kernel = torch.tensor(conv1_kernel, requires_grad=True)
        self.conv2_kernel = torch.tensor(conv2_kernel, requires_grad=True)
        self.fc1_weight = torch.tensor(fc1, requires_grad=True)
        self.fc1_bias = torch.tensor(fc1_bias,requires_grad=True)
        self.fc2_weight = torch.tensor(fc2, requires_grad=True)
        self.fc2_bias = torch.tensor(fc2_bias, requires_grad=True)
        self.target = torch.zeros(10, requires_grad=True)
        self.target[target] = 1.0

    def train(self):
        self.out_conv1 = F.conv2d(self.image, self.conv1_kernel, padding=0)
        self.out_conv1.register_hook(partial(cout_grad, layer="conv1"))
        self.out_relu1 = F.relu(self.out_conv1)
        self.out_relu1.register_hook(partial(cout_grad, layer="relu1"))
        self.out_pool1 = F.max_pool2d(self.out_relu1,kernel_size=2,stride=2)
        self.out_pool1.register_hook(partial(cout_grad, layer="pool1"))
        self.out_conv2 = F.conv2d(self.out_pool1, self.conv2_kernel, padding=0)
        self.out_conv2.register_hook(partial(cout_grad, layer="conv2"))
        self.out_relu2 = F.relu(self.out_conv2)
        self.out_relu2.register_hook(partial(cout_grad, layer="relu2"))
        self.out_pool2 = F.max_pool2d(self.out_relu2, kernel_size=2, stride=2)
        self.out_pool2.register_hook(partial(cout_grad, layer="pool2"))
        self.fc_inpt = self.out_pool2.flatten()
        print(f"fc inpt size: {self.fc_inpt.shape}")
        self.out_fc1 = F.linear(self.fc_inpt,self.fc1_weight, self.fc1_bias)
        self.out_fc1.register_hook(partial(cout_grad, layer="fc1"))
        self.out_fc2 = F.linear(self.out_fc1, self.fc2_weight, self.fc2_bias)
        self.out_fc2.register_hook(partial(cout_grad, layer="fc2"))
        self.pd_res = F.softmax(self.out_fc2)
        loss = F.cross_entropy(self.pd_res, self.target)
        print(f"loss is {loss}")
        loss.backward()





def main():
    image = None
    conv1_kernel = None
    conv2_kernel = None
    fc1 = None
    fc1_bias = None
    fc2 = None
    fc2_bias = None
    target = None
