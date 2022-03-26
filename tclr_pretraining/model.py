import numpy as np 
import torch.nn as nn
import torch
import torchvision
from torchsummary import summary
from r3d import r3d_18
from r3d_prp import R3DNet
from mlp import mlp

def build_r3d_backbone(): #Official PyTorch R3D-18 model taken from https://github.com/pytorch/vision/blob/master/torchvision/models/video/resnet.py
    
    model = r3d_18(pretrained = False, progress = False)
    #Expanding temporal dimension of the final layer by replacing temporal stride with temporal dilated convolution, this doesn't cost any additional parameters!

    model.layer4[0].conv1[0] = nn.Conv3d(256, 512, kernel_size=(3, 3, 3),\
                                stride=(1, 2, 2), padding=(2, 1, 1),dilation = (2,1,1), bias=False)
    model.layer4[0].downsample[0] = nn.Conv3d(256, 512,\
                          kernel_size = (1, 1, 1), stride = (1, 2, 2), bias=False)
    return model

def build_r3d_mlp():
    f = build_r3d_backbone()
    g = mlp()
    model = nn.Sequential(f,g)
    return model
    
def load_r3d_mlp(saved_model_file):
    model = build_r3d_mlp()
    pretrained = torch.load(saved_model_file)
    pretrained_kvpair = pretrained['state_dict']
    model_kvpair = model.state_dict()

    for layer_name, weights in pretrained_kvpair.items():
        layer_name = layer_name.replace('module.','')
        model_kvpair[layer_name] = weights  

    model.load_state_dict(model_kvpair, strict=True)
    print(f'{saved_model_file} loaded successfully')
    
    return model 

def build_r3d_prp_backbone(): #A slightly different implementation of R3D-18 model used in PRP paper
    model=R3DNet((1,1,1,1),with_classifier=False)
    model.conv5.block1.downsampleconv = nn.Conv3d(256, 512, kernel_size = (1,1,1),
                                stride=(1,2,2), dilation = (2,1,1), bias=False)
    model.conv5.block1.conv1 = nn.Conv3d(256, 512, kernel_size = (3,3,3),
                                stride=(1,2,2), dilation = (2,1,1), padding=(2,1,1), bias=False)
    return model

def build_r3d_prp_mlp():
    f = build_r3d_prp_backbone()
    g = mlp()
    model = nn.Sequential(f,g)
    return model

if __name__ == '__main__':
    
    input = torch.rand(5, 3, 16, 112, 112).cuda()
    model = build_r3d_mlp()
    print(model)
    print()
    model.eval()
    model.cuda()
    
    output = model((input,'d'))

    print(output.shape)

    

