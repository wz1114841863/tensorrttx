import torch
from torch import nn
from lenet5 import Lenet5
import os
import struct

def main():
    print('cuda device count: ', torch.cuda.device_count())
    net = torch.load('lenet5.pth')
    net = net.to('cuda:0')
    net.eval()

    tmp = torch.ones(1, 1, 32, 32).to('cuda:0')
    out = net(tmp)
    print('lenet out: ', out)

    with open("./lenet5.wts", 'w') as fp:
        fp.write(f"{len(net.state_dict().keys())}\n")
        for k, v in net.state_dict().items():
            # print('key: ', k)
            # print('value: ', v.shape)
            vr = v.reshape(-1).cpu().numpy()
            fp.write(f"{k} {len(vr)}")
            for vv in vr:
                fp.write(" ")
                fp.write(struct.pack(">f", float(vv)).hex())
            fp.write("\n")

if __name__ == "__main__":
    main()