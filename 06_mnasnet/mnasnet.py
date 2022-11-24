import torch
from torch import nn
from torch.nn import functional as F
import torchvision

def main():
    print('cuda device count: ', torch.cuda.device_count())
    # net = torchvision.models.mnasnet0_5(pretrained=True)
    net = torchvision.models.mnasnet1_0(pretrained=True)
    net = net.eval()
    net = net.to('cuda:0')
    # print(net)
    tmp = torch.ones(2, 3, 224, 224).to('cuda:0')
    out = net(tmp)
    print('mnasnet out:', out)
    torch.save(net, "mnasnet1_0.pth")

if __name__ == '__main__':
    main()