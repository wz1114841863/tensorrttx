import torch
from torch import nn
import torchvision
import os
import struct
from torchsummary import summary

def main():
    print('cuda device count: ', torch.cuda.device_count())
    net = torch.load('mobilenetv3.pth')
    net = net.to('cuda:0')
    net = net.eval()
    print('model: ', net)
    #print('state dict: ', net.state_dict().keys())
    tmp = torch.ones(1, 3, 224, 224).to('cuda:0')
    # print('input: ', tmp)
    # out = net(tmp)
    # print('output:', out)
    # 14版本后才支持hard swish 和 hard sigmoid 算子
    torch.onnx.export(net, tmp, "./mobilenetv3.onnx", opset_version=14)

    # summary(net, (3, 224, 224))
    #return
    # f = open("mobilenetv3.wts", 'w')
    # f.write("{}\n".format(len(net.state_dict().keys())))
    # for k,v in net.state_dict().items():
    #     print('key: ', k)
    #     print('value: ', v.shape)
    #     vr = v.reshape(-1).cpu().numpy()
    #     f.write("{} {}".format(k, len(vr)))
    #     for vv in vr:
    #         f.write(" ")
    #         f.write(struct.pack(">f", float(vv)).hex())
    #     f.write("\n")

if __name__ == '__main__':
    main()