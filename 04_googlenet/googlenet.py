import torch
import torchvision

def main():
    print(f"cuda device count: {torch.cuda.device_count()}")
    net = torchvision.models.googlenet(pretrained=True)
    net = net.eval()
    net = net.to("cuda:0")

    # tmp = torch.ones(2, 3, 224, 224).to('cuda:0')
    # out = net(tmp)
    # print("googlenet out: ", out.shape)
    # torch.save(net, "./googlenet.pth")

    input = torch.randn((1, 3, 244, 244)).to("cuda:0")
    torch.onnx.export(net, input, "./googlenet_gpu.onnx")
if __name__ == "__main__":
    main()