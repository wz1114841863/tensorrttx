import torch
from torch import nn
from torch.nn import functional as F
import torchvision


def main():
    print('cuda device count:', torch.cuda.device_count())
    net = torchvision.models.alexnet(pretrained=True)
    net.eval()
    net = net.to('cuda:0')
    print(net)
    tmp = torch.ones(2, 3, 244, 244).to('cuda:0')
    out = net(tmp)
    print('alexnet out:', out.shape)
    torch.save(net, "alexnet.pth")

if __name__ == '__main__':
    main()