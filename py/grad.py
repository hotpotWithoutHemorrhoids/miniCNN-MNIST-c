import torch
from torch.nn import functional as F
from functools import  partial
import random
def cout_grad(grad, layer=""):
    print(f"Gradient of {layer}:", grad.flatten())

torch.manual_seed(4)

def conv(inp=None,kernel_weight=None, in_shape=(2,3,3), kernelsize=(2,2), out_channel=2, outsize=2):
    if kernel_weight is None:
        inp_x = torch.randn(1,*in_shape,requires_grad=True)
    else:
        inp_x = torch.tensor(inp, requires_grad=True).reshape(1,*in_shape)
    if kernel_weight is None:
        kernel_weight = torch.randn(out_channel,in_shape[0], *kernelsize, requires_grad=True)
    else:
        kernel_weight = torch.tensor(kernel_weight,requires_grad=True).reshape(out_channel,in_shape[0], *kernelsize)

    print(f"inp_x {inp_x.numel()}: {inp_x.shape}\n\n kernel {kernel_weight.numel()}: {kernel_weight.shape}")

    conv_out = F.conv2d(inp_x,kernel_weight,padding=0)
    conv_out.register_hook(partial(cout_grad,layer="conv"))
    print(f"conv_out {conv_out.numel()}: {conv_out.flatten()}")

    relu_conv = F.relu(conv_out)
    print(f"relu_conv {relu_conv.shape}: {relu_conv.flatten()}")

    conv_out_len = conv_out.size().numel()
    fc_weights = torch.randn(outsize, conv_out_len, requires_grad=True)
    fc_out = F.linear(relu_conv.flatten(), fc_weights)
    print(f"fc_out: {fc_out}")
    fc_out.register_hook(partial(cout_grad, layer="fc"))
    # s_out = F.softmax(fc_out,dim=0)
    # s_out.register_hook(partial(cout_grad, layer="softmax"))
    # print(f"s_out: {s_out}")
    target = torch.zeros(outsize)
    idx = random.randint(0,outsize-1)
    target[idx] = 1.0
    print(f"target: {target}")
    loss = F.cross_entropy(fc_out, target)
    print(f"cross_entropy: {loss}")
    loss.register_hook(partial(cout_grad, layer="cross_entropy"))
    loss.backward()
    print(f"diff fc weight: {fc_weights.grad.flatten()}")
    print(f"diff inp_x: {inp_x.grad.flatten()} \n\n diff conv weight: {kernel_weight.grad.flatten()}")

def pool(in_shape=(3,6,6), kernel_size=2, stride=2, output_size = 2):
    inp_x = torch.randn(1, *in_shape, requires_grad=True)
    print(f"in_shape: {inp_x.shape} inpx  : {inp_x.flatten()}")
    pool_out = F.max_pool2d(inp_x,kernel_size, stride)
    pool_out.register_hook(partial(cout_grad,layer="pool"))

    print(f"out_shape {pool_out.shape}, pool_out: {pool_out.flatten()}")

    pool_out_len = pool_out.size().numel()
    fc_weights = torch.randn(output_size, pool_out_len, requires_grad=True)
    print(f"fc_weights: {fc_weights}")
    fc_out = F.linear(pool_out.flatten(), fc_weights)
    print(f"fc_out: {fc_out}")

    target = torch.zeros(output_size)
    idx = random.randint(0, output_size-1)
    target[idx] = 1.0

    loss = F.cross_entropy(fc_out, target)
    print(f"target: {target}  cross_entropy: {loss}")
    loss.backward()
    # print(f"diff fc_out: {fc_out.grad}")
    # print(f"diff pool_ouut: {pool_out.grad}")
    print(f"diff inp: {inp_x.grad.flatten()}")




if __name__ == "__main__":
    # conv()
    inpx = [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.09,0.10,0.00,0.00,0.00,0.00,0.00,0.00,0.17,0.18,0.45,0.45,0.52,0.72,0.97,0.62,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.05,0.94,0.95,0.89,0.89,0.89,0.89,0.89,0.89,1.00,1.00,1.00,0.96,0.94,0.98,1.00,0.51,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.03,0.76,1.00,1.00,0.78,0.77,0.77,0.67,0.50,0.45,0.22,0.22,0.07,0.00,0.86,0.95,0.05,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.01,0.35,0.76,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.56,1.00,0.62,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.03,0.88,1.00,0.11,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.31,1.00,0.71,0.03,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.02,0.73,1.00,0.26,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.22,1.00,0.87,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.01,0.62,1.00,0.41,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.13,1.00,0.82,0.03,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.73,1.00,0.36,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.26,0.98,0.78,0.01,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.78,0.98,0.24,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.45,1.00,0.69,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.14,0.88,0.95,0.08,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.56,1.00,0.31,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.31,0.99,0.56,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.11,0.95,0.63,0.04,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.07,0.82,0.62,0.02,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.55,0.63,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00]
    kernel_weight = [-0.3113, -0.7130, -0.7291, -0.2992, -0.2529, -0.3602,  0.9394,  1.1614,
        -0.1706,  0.5119,  0.5962,  1.2911,  1.7541, -0.4149, -0.9922, -0.2986,
         0.6443, -0.2710, -0.1359,  2.5745, -0.5229,  0.9863,  0.2923,  1.0146,
         1.5558]
    conv(inpx,kernel_weight, (1,28,28),(5,5),1,10)