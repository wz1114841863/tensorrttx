import torch
from torch import nn
import torchvision
import sys
from torchsummary import summary

import init_paths
# print(sys.path)
from lib_py import save_weights

def main():
    print(f"cuda device count: {torch.cuda.device_count()}")
    net = torch.load("./googlenet.pth")
    net = net.eval()
    net = net.to("cuda:0")

    tmp = torch.ones(1, 3, 224, 224).to('cuda:0')
    out = net(tmp)
    print(f"out: {out}")

    summary(net, (3, 224, 224))
    save_weights(net, "./googlenet.wts")

if __name__ == "__main__":
    main()