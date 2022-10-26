import torch
from torch import nn
from torch.nn import functional as F
import torchvision


def main():
    print('cuda device count: ', torch.cuda.device_count())
    net = torchvision.models.resnet50(pretrained=True)
    net = net.to('cuda:0')
    net.eval()
    print(net)
    tmp = torch.randn((1, 3, 244, 244)).to("cuda:0")
    torch.onnx.export(net,  # model being run
                        tmp,  # model input (or a tuple for multiple inputs)
                        "./resnet50.onnx",  # where to save the model (can be a file or file-like object)
                        export_params=True,  # store the trained parameter weights inside the model file
                        opset_version=10,  # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names=['input'],  # the model's input names
                        output_names=['output']  # the model's output names)
    )
    tmp = torch.ones(2, 3, 244, 244).to('cuda:0')
    out = net(tmp)
    print('resnet50 out:', out.shape)
    torch.save(net, "./resnet50.pth")

if __name__ == '__main__':
    main()