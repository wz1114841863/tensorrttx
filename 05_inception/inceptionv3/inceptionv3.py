import torch
import torchvision

def main():
    print("cuda device cout: ", torch.cuda.device_count())
    net = torchvision.models.inception_v3(pretrained=True)
    net = net.eval()
    net = net.to('cuda:0')
    input = torch.ones(1, 3, 299, 299).to('cuda:0')
    torch.onnx.export(net, input, "./inceptionv3_gpu.onnx")
    tmp = torch.ones(2, 3, 299, 299).to('cuda:0')
    out = net(tmp)
    print('inception out:', out)
    torch.save(net, "inceptionv3.pth")

if __name__ == "__main__":
    main()