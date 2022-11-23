import torch
from mlp import LinearRegressionModel 
import struct

def load_model(model_path=''):
    """
    Load saved model from file
    :param model_path: mlp.pth prepared using mlp.py
    :return net: loaded model
    """
    print("[INFO]: Loading saved model...")
    net = torch.load(model_path)
    net = net.to('cuda:0')
    net.eval()
    return net

def test_model(mlp_model):
    """
    Test model on custom input
    :param mlp_model: pre-trained model
    :return
    """
    print("[INFO]: Testing model on samplr input...")
    tmp = torch.ones(1, 1).to('cuda:0')
    out = mlp_model(tmp)
    print(f'[INFO]: Test Result is: ', out.detach().cpu().numpy())

def convert_to_wts(mlp_model):
    """
    Convert weights to .wts format for TensorRT Engine
    Weights are written in the following format:
        <total-weight-count>
        weight.name <weight-count> <weight-val1> <weight-val2> ...
        -- total-weight-count: is an integer
        -- weight.name: is used as key in TensorRT engine
        -- weight-count: no. of weights for current layer
        -- weight-valxx: float to c-bytes to hexadecimal
    """
    print("[INFO]: Writing weights to .wts ...")
    with open('./mlp.wts', 'w') as fp:
        fp.write(f'{len(mlp_model.state_dict().keys())}\n')
        for k, v in mlp_model.state_dict().items():
            vr = v.reshape(-1).cpu().numpy()
            fp.write(f'{k} {len(vr)}')
            for vv in vr:
                fp.write(" ")
                fp.write(struct.pack('>f', float(vv)).hex())
            fp.write('\n')
    print("[INFO]: Successfully, convered weights to WTS")

def main():
    mlp_model = load_model('mlp.pth')
    test_model(mlp_model)
    convert_to_wts(mlp_model)

if __name__ == '__main__':
    main()
