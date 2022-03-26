import numpy as np 
import torch.nn as nn
import torch
import torchvision
from r3d_classifier import r3d_18_classifier


# from r2p1d import r2plus1d_18, embedder
# from classifier_r2p1d import r2plus1d_18_classifier
# # from try1_model import r2plus1d_18_changed
# # from dilated_r2plus1d import r2plus1d_18
from torchvision.models.utils import load_state_dict_from_url

def build_r3d_classifier(num_classes = 102, kin_pretrained = False, self_pretrained = True, saved_model_file = None, linear = True):
    model = r3d_18_classifier(pretrained = kin_pretrained, progress = False)
    # if not kin_pretrained:
    model.layer4[0].conv1[0] = nn.Conv3d(256, 512, kernel_size=(3, 3, 3),\
                                stride=(1, 2, 2), padding=(2, 1, 1),dilation = (2,1,1), bias=False)
    model.layer4[0].downsample[0] = nn.Conv3d(256, 512,\
                        kernel_size = (1, 1, 1), stride = (1, 2, 2), bias=False)

    if self_pretrained == True:
        pretrained = torch.load(saved_model_file)
        if 'state_dict' in pretrained.keys():
            pretrained_kvpair = pretrained['state_dict']
        elif 'bb_state_dict' in pretrained.keys():
            pretrained_kvpair = pretrained['bb_state_dict'] 
        # print(pretrained_kvpair)
        # exit()
        model_kvpair = model.state_dict()

        # for layer_name, weights in pretrained_kvpair.items():
        #     if 'module.1.' in layer_name:
        #         continue
        #     elif '1.' == layer_name[:2]:
        #         continue
        #     print(layer_name)
        #     break
        # for layer_name, weights in model_kvpair.items():
        #     print(layer_name)
        #     break
        # exit()
        for layer_name, weights in pretrained_kvpair.items():
            if 'module.1.' in layer_name:
                continue              
            elif '1.' == layer_name[:2]:
                # print(layer_name)
                continue
            if 'module.0.' in layer_name:
                layer_name = layer_name.replace('module.0.','')
            if 'module.' in layer_name:
                layer_name = layer_name.replace('module.','')
            elif '0.' == layer_name[:2]:
                layer_name = layer_name[2:]
            if 'fc' in layer_name:
                continue
            # layer_name = layer_name.replace('module.0.','')
            model_kvpair[layer_name] = weights   
            # if linear == True:
            #     model.layer_name.requires_grad = False
        model.load_state_dict(model_kvpair, strict=True)
        print(f'model {saved_model_file} loaded successsfully!')
    # exit()

    model.fc = nn.Linear(512, num_classes)
    return model 

def load_r3d_classifier(num_classes = 102, saved_model_file = None):
    model = r3d_18_classifier(pretrained = False, progress = False)

    model.layer4[0].conv1[0] = nn.Conv3d(256, 512, kernel_size=(3, 3, 3),\
                                stride=(1, 2, 2), padding=(2, 1, 1),dilation = (2,1,1), bias=False)
    model.layer4[0].downsample[0] = nn.Conv3d(256, 512,\
                          kernel_size = (1, 1, 1), stride = (1, 2, 2), bias=False)

    model.fc = nn.Linear(512, num_classes)
    model_kvpair = model.state_dict()


    pretrained = torch.load(saved_model_file)
    pretrained_kvpair = pretrained['state_dict']
    # print(pretrained_kvpair)
    # exit()
        
    # exit()
    for layer_name, weights in pretrained_kvpair.items():
       
        model_kvpair[layer_name] = weights   
    model.load_state_dict(model_kvpair, strict=True)
    print(f'model {saved_model_file} loaded successsfully!')
    return model 


def build_r3d_backbone():
    model = r3d_18(pretrained = False, progress = False)
    
    model.layer4[0].conv1[0] = nn.Conv3d(256, 512, kernel_size=(3, 3, 3),\
                                stride=(1, 2, 2), padding=(2, 1, 1),dilation = (2,1,1), bias=False)
    model.layer4[0].downsample[0] = nn.Conv3d(256, 512,\
                          kernel_size = (1, 1, 1), stride = (1, 2, 2), bias=False)
    return model




if __name__ == '__main__':
    
    # summary(model, (16, 3, 112, 112))
    input = torch.rand(5, 3, 16, 112, 112).cuda()
    model = build_r3d_classifier(num_classes = 102, saved_model_file = '/home/c3-0/ishan/ss_saved_models/r3d82/model_best_e151_loss_0.9281.pth')

  
