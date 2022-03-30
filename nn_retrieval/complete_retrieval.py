#sbatch -a 0-3 multiple_complete_retrieval.slurm
#sbatch -a 0-5 multiple_complete_retrieval.slurm
import torch
from torch.autograd import Variable 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.optim.lr_scheduler import *  
import time
import os
import numpy as np
# from model import build_r3d_classifier, build_r3d_backbone, build_r3d_original
# from i3d import InceptionI3d
from model import *
import parameters_BL as params
import config as cfg
# from DL_ishanv3 import *
from dl_ret import *
import sys, traceback
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from tensorboardX import SummaryWriter
import cv2
from torch.utils.data import DataLoader
import math
import argparse
import itertools
import pickle
# from keras.utils import to_categorical

# if torch.cuda.is_available(): 
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')


def val_epoch(clips_per_vid, run_id, epoch,mode, skip, hflip, cropping_fac, pred_dict,label_dict, data_loader, model, criterion, writer, use_cuda):
    print(f'validation at epoch {epoch} - mode {mode} - skip {skip} - hflip {hflip} - cropping_fac {cropping_fac}')
    
    model.eval()

    losses = []
    predictions, ground_truth = [], []
    vid_paths = []

    for i, (inputs, label, vid_path,_) in enumerate(data_loader):
        vid_paths.extend(vid_path)
        ground_truth.extend(label)
        # inputs = inputs.permute(0,4,1,2,3)
        if len(inputs.shape) != 1:

            inputs = inputs.permute(0, 2, 1, 3, 4)

            # print(label)
            if use_cuda:
                inputs = inputs.cuda()
                label = torch.from_numpy(np.asarray(label)).cuda()
            # print(label)

        
            with torch.no_grad():

                output = model(inputs)
                # print(output.shape)
                # exit()
                output = output.squeeze(3)
                output = output.squeeze(3)

            predictions.extend(output.cpu())
            # print(len(predictions))


            if i+1 % 45 == 0:
                print(f'{i} batches are done')
                # print("Validation Epoch ", epoch , "mode", mode, "skip", skip, "hflip", hflip, " Batch ", i, "- Loss : ", np.mean(losses))
        
    del inputs, output, label
    # ground_truth = np.asarray(ground_truth)
    # pred_array = np.flip(np.argsort(predictions,axis=1),axis=1) # Prediction with the most confidence is the first element here
    # c_pred = pred_array[:,0] #np.argmax(predictions,axis=1).reshape(len(predictions))

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in pred_dict.keys():
            pred_dict[str(vid_paths[entry].split('/')[-1])] = []
            pred_dict[str(vid_paths[entry].split('/')[-1])] = (1/clips_per_vid)*predictions[entry].view(-1)

        else:
            # print('yes')
            pred_dict[str(vid_paths[entry].split('/')[-1])]+= (1/clips_per_vid)*predictions[entry].view(-1)

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in label_dict.keys():
            label_dict[str(vid_paths[entry].split('/')[-1])]= ground_truth[entry]

    print_pred_array = []

    # for entry in range(pred_array.shape[0]):
    #     temp = ''
    #     for i in range(5):
    #         temp += str(int(pred_array[entry][i]))+' '
    #     print_pred_array.append(temp)
    # print(f'check {print_pred_array[0]}')
    # results = open('Submission1.txt','w')
    # for entry in range(len(vid_paths)):
    #     content = str(vid_paths[entry].split('/')[-1] + ' ' + print_pred_array[entry])[:-1]+'\n'
    #     results.write(content)
    # results.close()
    
    # correct_count = np.sum(c_pred==ground_truth)
    # accuracy = float(correct_count)/len(c_pred)
    
    # print(f'Correct Count is {correct_count}')
    # print(f'Epoch {epoch}, mode {mode}, skip {skip}, hflip {hflip}, cropping_fac {cropping_fac}, Accuracy: {accuracy*100 :.3f}')
    return pred_dict, label_dict
    
def train_classifier(run_id, arch, m_file_name, num_modes):
    use_cuda = True
    best_score = 0
    writer = SummaryWriter(os.path.join(cfg.logs, str(run_id)))

    save_dir = os.path.join(cfg.saved_models_dir, run_id)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    # m_file_name = '/home/c3-0/ishan/ss_saved_models/r3d31_2c_full_e143_32f/model_221_bestAcc_0.7254_F1_0.7173.pth'
    # m_file_name = '/home/idave/ss_saved_models/r3d65/model_best_e234_loss_12.264.pth'
    # m_file_name = '/home/idave/ss_saved_models/r3d62/model_best_e104_loss_68.780.pth'
    # m_file_name = '/home/idave/ss_saved_models/r3d75/model_best_e157_loss_11.327.pth'
    if len(saved_model) ==0:
        print('It`s from scratch')
        model = build_r3d_encoder_ret(self_pretrained = False, num_classes = params.num_classes)

    else:
        if arch =='r3d18':
            model = build_r3d_encoder_ret(saved_model_file = m_file_name, num_classes = params.num_classes)
        if arch =='rpd18':
            model = build_rpd_encoder_ret(saved_model_file = m_file_name, num_classes = params.num_classes)
        

     
    criterion= nn.CrossEntropyLoss()

    if torch.cuda.device_count()>1:
        print(f'Multiple GPUS found!')
        model=nn.DataParallel(model)
        model.cuda()
        criterion.cuda()
    else:
        print('Only 1 GPU is available')
        model.cuda()
        criterion.cuda()
 
    if num_modes:
        # modes = num_modes
        modes = list(range(num_modes))

    else:
        modes = list(range(params.num_modes))
    print(f'Num modes {len(modes)}')

    skip = list(range(1,params.num_skips+1))
    hflip = params.hflip #list(range(2))
    cropping_fac1 = params.cropping_fac1
    print(f'Num skips {skip}')
    print(f'Cropping fac {cropping_fac1}')
    modes, skip,hflip, cropping_fac =  list(zip(*itertools.product(modes,skip,hflip,cropping_fac1)))
    


    print(f'There will be total {len(modes)} iterations')
    for epoch in range(1):
        
        print(f'Epoch {epoch} started')
        start=time.time()
        
        pred_dict = {}
        label_dict = {}
        val_losses =[]
        if not os.path.exists('./'+str(run_id) + '_retrieval/'+str(run_id)+ '_val_pred_dict.pkl'):

            for val_iter in range(len(modes)):
                # if (modes[val_iter] == 0 and skip[val_iter] ==1) or (modes[val_iter] == 2 and skip[val_iter] ==1):
                #     continue
                # try:
                    # break
                validation_dataset = multi_baseline_dataloader_val_strong(shuffle = True, data_percentage = params.data_percentage,\
                            mode = modes[val_iter], skip = skip[val_iter], hflip= hflip[val_iter], \
                            cropping_factor= cropping_fac[val_iter])
                validation_dataloader = DataLoader(validation_dataset, batch_size=params.v_batch_size, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn2)
                if val_iter ==0:
                    print(f'Validation dataset length: {len(validation_dataset)}')
                    print(f'Validation dataset steps per epoch: {len(validation_dataset)/params.v_batch_size}')    
                
                pred_dict, label_dict = val_epoch(len(modes), run_id, epoch,modes[val_iter],skip[val_iter],hflip[val_iter],cropping_fac[val_iter], \
                                                                    pred_dict, label_dict, validation_dataloader, model, criterion, writer, use_cuda)
            
            os.makedirs('./'+str(run_id) + '_retrieval')
            pickle.dump(pred_dict, open('./'+str(run_id) + '_retrieval/'+str(run_id)+ '_val_pred_dict.pkl','wb'))
            pickle.dump(label_dict, open('./'+str(run_id) + '_retrieval/'+str(run_id)+'_val_label_dict.pkl','wb'))
            val_features, val_labels = pred_dict, label_dict
        else:
            val_features = pickle.load(open('./'+str(run_id) + '_retrieval/'+str(run_id)+ '_val_pred_dict.pkl','rb'))
            val_labels = pickle.load(open('./'+str(run_id) + '_retrieval/'+str(run_id)+ '_val_label_dict.pkl','rb'))
        pred_dict, label_dict ={}, {}
        if not os.path.exists('./'+str(run_id) + '_retrieval/'+str(run_id)+ '_train_pred_dict.pkl'):

            for val_iter in range(len(modes)):
                # if (modes[val_iter] == 0 and skip[val_iter] ==1) or (modes[val_iter] == 2 and skip[val_iter] ==1):
                #     continue
                # try:
                train_dataset = multi_baseline_dataloader_train_strong(shuffle = True, data_percentage = params.data_percentage,\
                            mode = modes[val_iter], skip = skip[val_iter], hflip= hflip[val_iter], \
                            cropping_factor= cropping_fac[val_iter])
                train_dataloader = DataLoader(train_dataset, batch_size=params.v_batch_size, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn2)
                if val_iter ==0:
                    print(f'train dataset length: {len(train_dataset)}')
                    print(f'train dataset steps per epoch: {len(train_dataset)/params.v_batch_size}')    
                
                pred_dict, label_dict = val_epoch(len(modes), run_id, epoch,modes[val_iter],skip[val_iter],hflip[val_iter],cropping_fac[val_iter], \
                                                                    pred_dict, label_dict, train_dataloader, model, criterion, writer, use_cuda)
            pickle.dump(pred_dict, open('./'+str(run_id) + '_retrieval/'+str(run_id)+'_train_pred_dict.pkl','wb'))
            pickle.dump(label_dict, open('./'+str(run_id) + '_retrieval/'+str(run_id)+'_train_label_dict.pkl','wb'))
            train_features, train_labels = pred_dict, label_dict

        else:
            train_features = pickle.load(open('./'+str(run_id) + '_retrieval/'+str(run_id)+ '_train_pred_dict.pkl','rb'))
            train_labels = pickle.load(open('./'+str(run_id) + '_retrieval/'+str(run_id)+ '_train_label_dict.pkl','rb'))

        train_feat = []
        train_label = []
        for i in (list(train_features.keys())):
            train_feat.append(train_features[i])
            train_label.append(train_labels[i])

        val_feat = []
        val_label = []
        for i in (list(val_features.keys())):
            val_feat.append(val_features[i])
            val_label.append(val_labels[i])

        train_feat = torch.stack(train_feat,dim=0)    
        train_label = np.asarray(train_label)
        val_feat = torch.stack(val_feat,dim=0)
        val_label = np.asarray(val_label)

        # print(len(train_label))
        # print(len(val_label))
        # print(train_feat.shape)
        # print(val_feat.shape)

        #train features has size of 9537 x 4096
        #val feature has size of 3792 x 4096

        train_feat /= train_feat.norm(dim = 1)[:,None]
        val_feat /= val_feat.norm(dim = 1)[:,None]


        similarity_matrix = torch.mm(train_feat.cuda(), val_feat.cuda().T).cpu().numpy()
        # print(similarity_matrix.shape)
        # similarity matrix has shape of 9537 x 3783

        sorted_column_args = np.argsort(similarity_matrix,axis =0)
        top1_correct = 0
        top5_correct = 0
        top10_correct = 0
        top20_correct = 0

        for i in range(similarity_matrix.shape[1]):
            top_label = train_label[sorted_column_args[:,i]]
            if val_label[i] == top_label[-1]:
                top1_correct += 1
            if val_label[i] in top_label[-5:]:
                top5_correct += 1
            if val_label[i] in top_label[-10:]:
                top10_correct += 1
            if val_label[i] in top_label[-20:]:
                top20_correct += 1
        print(f'Top-1 correct is {top1_correct/similarity_matrix.shape[1]*100 :.2f}%')
        print(f'Top-5 correct is {top5_correct/similarity_matrix.shape[1]*100 :.2f}%')
        print(f'Top-10 correct is {top10_correct/similarity_matrix.shape[1]*100 :.2f}%')
        print(f'Top-20 correct is {top20_correct/similarity_matrix.shape[1]*100 :.2f}%')

        print(f'{top1_correct/similarity_matrix.shape[1]*100 :.2f}, {top5_correct/similarity_matrix.shape[1]*100 :.2f}, {top10_correct/similarity_matrix.shape[1]*100 :.2f}, {top20_correct/similarity_matrix.shape[1]*100 :.2f}')
        f = open('./'+str(run_id) + '_retrieval/'+str(run_id)+'_results.txt','w')
        f.writelines('Top-1, Top-5, Top-10, Top-20 \n')
        f.writelines(str(top1_correct/similarity_matrix.shape[1]*100)+', ' + str(top5_correct/similarity_matrix.shape[1]*100)+', ' +str(top10_correct/similarity_matrix.shape[1]*100)+', ' + str(top20_correct/similarity_matrix.shape[1]*100))
        f.close()
            #     val_losses.append(loss)

            #     predictions1 = np.zeros((len(list(pred_dict.keys())),102))
            #     ground_truth1 = []
            #     entry = 0
            #     for key in pred_dict.keys():
            #         predictions1[entry] = np.mean(pred_dict[key], axis =0)
            #         entry+=1

            #     for key in label_dict.keys():
            #         ground_truth1.append(label_dict[key])
                
            #     pred_array1 = np.flip(np.argsort(predictions1,axis=1),axis=1) # Prediction with the most confidence is the first element here
            #     c_pred1 = pred_array1[:,0]

            #     correct_count1 = np.sum(c_pred1==ground_truth1)
            #     accuracy11 = float(correct_count1)/len(c_pred1)

                
            #     print(f'Running Avg Accuracy is for epoch {epoch}, mode {modes[val_iter]}, skip {skip[val_iter]}, hflip {hflip[val_iter]}, cropping_fac {cropping_fac[val_iter]} is {accuracy11*100 :.3f}% ')  
            # except:
            #     print(f'Failed epoch {epoch}, mode {modes[val_iter]}, skip {skip[val_iter]}, hflip {hflip[val_iter]}, cropping_fac {cropping_fac[val_iter]} is {accuracy11*100 :.3f}% ')  


        # val_loss = np.mean(val_losses)
        # predictions = np.zeros((len(list(pred_dict.keys())),102))
        # ground_truth = []
        # entry = 0
        # for key in pred_dict.keys():
        #     predictions[entry] = np.mean(pred_dict[key], axis =0)
        #     entry+=1

        # for key in label_dict.keys():
        #     ground_truth.append(label_dict[key])
        
        # pred_array = np.flip(np.argsort(predictions,axis=1),axis=1) # Prediction with the most confidence is the first element here
        # c_pred = pred_array[:,0]

        # correct_count = np.sum(c_pred==ground_truth)
        # accuracy1 = float(correct_count)/len(c_pred)

        # writer.add_scalar('Validation Loss', np.mean(val_loss), epoch)
        # writer.add_scalar('Validation Accuracy', np.mean(accuracy1), epoch)
        
        # print(f'Overall Accuracy is for epoch {epoch} is {accuracy1*100 :.3f}% ')
        # file_name = f'RunID_{run_id}_Acc_{accuracy1*100 :.3f}_cf_{len(cropping_fac1)}_m_{params.num_modes}_s_{params.num_skips}.pkl'     
        # pickle.dump(pred_dict, open(file_name,'wb'))

        
        taken = time.time()-start
        print(f'Time taken for Epoch-{epoch} is {taken}')
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train baseline')

    parser.add_argument("--run_id", dest='run_id', type=str, required=False, default= "random",
                        help='run_id')
    parser.add_argument("--saved_model", dest='saved_model', type=str, required=False, default= "",
                        help='saved_model')
    parser.add_argument("--arch", dest='arch', type=str, required=False, default= "r3d18",
                        help='run_id')
    parser.add_argument("--modes", dest='modes', type=int, required=False, default= 0,
                        help='modes')
    args = parser.parse_args()
    run_id = args.run_id
    modes = args.modes
    arch = args.arch

    saved_model = args. saved_model
    train_classifier(str(run_id), arch, str(saved_model),modes)



        


