import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import struct
from torchsummary import summary


def main():
    print('cuda device count: ', torch.cuda.device_count())
    net = torchvision.models.vgg11(pretrained=False)
    net.load_state_dict(torch.load("./vggnet11.pth"))
    #net.fc = nn.Linear(512, 2)
    net = net.eval()
    net = net.to('cuda:0')
    # # print(net)
    tmp = torch.ones(1, 3, 224, 224).to('cuda:0')
    torch.onnx.export(net, tmp, "vggnet.onnx")
    # out = net(tmp)
    # print('vgg out:', out.shape)
    # torch.save(net, "vgg.pth")

    # summary(net, (3, 224, 224))
    # #return
    # f = open("vgg.wts", 'w')
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