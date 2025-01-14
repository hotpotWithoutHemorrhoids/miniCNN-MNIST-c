import torch
from torch.nn import functional as F
from functools import  partial
import random
def cout_grad(grad, layer=""):
    print(f"Gradient of {layer}:", grad.flatten())

torch.manual_seed(4)

def conv(in_shape=(2,3,3), kernelsize=(2,2), out_channel=2, outsize=2):
    inp_x = torch.randn(1,*in_shape,requires_grad=True)
    kernel_weight = torch.randn(out_channel,in_shape[0], *kernelsize, requires_grad=True)
    print(f"inp_x {inp_x.numel()}: {inp_x.flatten()}\n\n kernel {kernel_weight.numel()}: {kernel_weight.flatten()}")

    conv_out = F.conv2d(inp_x,kernel_weight,padding=0)
    conv_out.register_hook(partial(cout_grad,layer="conv"))
    print(f"conv_out {conv_out.numel()}: {conv_out.flatten()}")

    relu_conv = F.relu(conv_out)
    print(f"relu_conv: {relu_conv.flatten()}")

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
    conv()