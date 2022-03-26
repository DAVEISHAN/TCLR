import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
import time
import os
import numpy as np
from model import build_r3d_classifier, load_r3d_classifier
import parameters_BL as params
import config as cfg
from dl_linear import *
import sys, traceback
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from tensorboardX import SummaryWriter
import cv2
from torch.utils.data import DataLoader
import math
import argparse
import itertools

from keras.utils import to_categorical

if torch.cuda.is_available(): 
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.benchmark = True

def train_epoch(run_id, learning_rate2,  epoch, data_loader, model, criterion, optimizer, writer, use_cuda):
    print('train at epoch {}'.format(epoch))
    for param_group in optimizer.param_groups:
        param_group['lr']=learning_rate2
        writer.add_scalar('Learning Rate', learning_rate2, epoch)  

        print("Learning rate is: {}".format(param_group['lr']))
  
  
    losses, weighted_losses = [], []
    loss_mini_batch = 0
    # optimizer.zero_grad()

    model.train()

    for i, (inputs, label, vid_path) in enumerate(data_loader):
        # print(f'label is {label}')
        # inputs = inputs.permute(0,4,1,2,3)
        # print(inputs.shape)
        optimizer.zero_grad()

        inputs = inputs.permute(0,2,1,3,4)
    
        
        if use_cuda:
            inputs = inputs.cuda()
            label = torch.from_numpy(np.asarray(label)).cuda()
        output = model(inputs)
        
    
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if i % 24 == 0: 
            print(f'Training Epoch {epoch}, Batch {i}, Loss: {np.mean(losses) :.5f}')
        
    print('Training Epoch: %d, Loss: %.4f' % (epoch,  np.mean(losses)))
    writer.add_scalar('Training Loss', np.mean(losses), epoch)

    del loss, inputs, output, label

    return model, np.mean(losses)

def val_epoch(run_id, epoch,mode,pred_dict,label_dict, data_loader, model, criterion, writer, use_cuda):
    print(f'validation at epoch {epoch} - mode {mode} ')
    
    model.eval()

    losses = []
    predictions, ground_truth = [], []
    vid_paths = []

    for i, (inputs, label, vid_path, _) in enumerate(data_loader):
        vid_paths.extend(vid_path)
        ground_truth.extend(label)
        if len(inputs.shape) != 1:

            inputs = inputs.permute(0, 2, 1, 3, 4)
            
            if use_cuda:
                inputs = inputs.cuda()
                label = torch.from_numpy(np.asarray(label)).cuda()

        
            with torch.no_grad():

                output = model(inputs)
                loss = criterion(output,label)

            losses.append(loss.item())


            predictions.extend(nn.functional.softmax(output, dim = 1).cpu().data.numpy())


            if i+1 % 45 == 0:
                print("Validation Epoch ", epoch , "mode", mode, " Batch ", i, "- Loss : ", np.mean(losses))
        
    del inputs, output, label, loss 

    ground_truth = np.asarray(ground_truth)
    pred_array = np.flip(np.argsort(predictions,axis=1),axis=1) 
    c_pred = pred_array[:,0] 

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in pred_dict.keys():
            pred_dict[str(vid_paths[entry].split('/')[-1])] = []
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions[entry])

        else:
            # print('yes')
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions[entry])

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in label_dict.keys():
            label_dict[str(vid_paths[entry].split('/')[-1])]= ground_truth[entry]

    print_pred_array = []

    
    
    correct_count = np.sum(c_pred==ground_truth)
    accuracy = float(correct_count)/len(c_pred)
    
    # print(f'Correct Count is {correct_count}')
    print(f'Epoch {epoch}, mode {mode}, Accuracy: {accuracy*100 :.3f}')
    return pred_dict, label_dict, accuracy, np.mean(losses)
    
def train_classifier(run_id, restart, saved_model, linear):
    use_cuda = True
    best_score = 0
    writer = SummaryWriter(os.path.join(cfg.logs, str(run_id)))

    save_dir = os.path.join(cfg.saved_models_dir, run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if restart:
        saved_model_file = save_dir + '/model_temp.pth'
        
        if os.path.exists(saved_model_file):
            model = load_r3d_classifier(saved_model_file= saved_model_file)
            epoch0 = torch.load(saved_model_file)['epoch']
        else:
            print(f'No such model exists: {saved_model_file} :(')
            if not (saved_model == None or len(saved_model) == 0 or saved_model =="kin400"):
    
                print(f'Trying to load {saved_model}')
                model = build_r3d_classifier(saved_model_file = saved_model, num_classes = 102)
            else:
                print(f'It`s a baseline experiment!')
                model = build_r3d_classifier(self_pretrained = False, saved_model_file = None, num_classes = 102) 
            epoch0 = 0

    else:
        if not (saved_model == None or len(saved_model) == 0 or saved_model =="kin400"):
            print(f'Trying to load {saved_model}')
            model = build_r3d_classifier(saved_model_file = saved_model, num_classes = 102)
        else:
            print(f'It`s a baseline experiment!')
            model = build_r3d_classifier(self_pretrained = False, saved_model_file = None, num_classes = 102) 
        epoch0 = 0
        '''if not (saved_model == None or saved_model ==''):
            epoch0 = 0
            model = build_r3d_classifier(saved_model_file = saved_model[31:], num_classes = 102)
        
        else:
            epoch0 = 0
            print(f'It`s a baseline experiment!')
            model = build_r3d_classifier(self_pretrained = False, saved_model_file = None, num_classes = 102) '''

    if linear:
        print('It`s a linear evaluation!')
        for m in list(model.parameters())[:-2]:
            m.requires_grad = False
    

    learning_rate1 = params.learning_rate
    best_score = 10000

    
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

    optimizer = optim.Adam(model.parameters(),lr=params.learning_rate)

    train_dataset = baseline_dataloader_train_strong(shuffle = False, data_percentage = params.data_percentage)

    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train)
    print(f'Train dataset length: {len(train_dataset)}')
    print(f'Train dataset steps per epoch: {len(train_dataset)/params.batch_size}')
    
    #We will do validation only at epochs mentioned in the following array
    if linear:
        val_array = [6]+[x for x in range(11,50)]
    

    modes = list(range(params.num_modes))
   
    print(f'Num modes {len(modes)}')
   
    accuracy = 0
    best_acc = 0 

    learning_rate2 = learning_rate1 
    scheduler_step = 1  
    scheduler_epoch = 0
       


    for epoch in range(epoch0, params.num_epochs):
        
        print(f'Epoch {epoch} started')
        start=time.time()

        try:

            if scheduler_epoch == params.scheduler_patience:
                print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
                print(f'Dropping learning rate to {learning_rate2/10} for epoch')
                print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
                learning_rate2 = learning_rate1/(10**scheduler_step)
                scheduler_epoch = 0
                scheduler_step += 1

            model, train_loss = train_epoch(run_id, learning_rate2,  epoch, train_dataloader, model, criterion, optimizer, writer, use_cuda)
            if train_loss > best_score:
                scheduler_epoch += 1
            else:
                best_score = train_loss
            if epoch in val_array:
                pred_dict = {}
                label_dict = {}
                val_losses =[]

                for val_iter in range(len(modes)):
                    try:
                        validation_dataset = multi_baseline_dataloader_val_strong(shuffle = True, data_percentage = params.data_percentage,\
                            mode = modes[val_iter])
                        validation_dataloader = DataLoader(validation_dataset, batch_size=params.v_batch_size, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn2)
                        if val_iter ==0:
                            print(f'Validation dataset length: {len(validation_dataset)}')
                            print(f'Validation dataset steps per epoch: {len(validation_dataset)/params.v_batch_size}')    
                        pred_dict, label_dict, accuracy, loss = val_epoch(run_id, epoch,modes[val_iter], pred_dict, label_dict, validation_dataloader, model, criterion, writer, use_cuda)
                        val_losses.append(loss)

                        predictions1 = np.zeros((len(list(pred_dict.keys())),102))
                        ground_truth1 = []
                        entry = 0
                        for key in pred_dict.keys():
                            predictions1[entry] = np.mean(pred_dict[key], axis =0)
                            entry+=1

                        for key in label_dict.keys():
                            ground_truth1.append(label_dict[key])
                        
                        pred_array1 = np.flip(np.argsort(predictions1,axis=1),axis=1) # Prediction with the most confidence is the first element here
                        c_pred1 = pred_array1[:,0]

                        correct_count1 = np.sum(c_pred1==ground_truth1)
                        accuracy11 = float(correct_count1)/len(c_pred1)

                        
                        print(f'Running Avg Accuracy is for epoch {epoch}, mode {modes[val_iter]}, is {accuracy11*100 :.3f}% ')  
                    except:
                        print(f'Failed epoch {epoch}, mode {modes[val_iter]}, is {accuracy11*100 :.3f}% ')  

                val_loss = np.mean(val_losses)
                predictions = np.zeros((len(list(pred_dict.keys())),102))
                ground_truth = []
                entry = 0
                for key in pred_dict.keys():
                    predictions[entry] = np.mean(pred_dict[key], axis =0)
                    entry+=1

                for key in label_dict.keys():
                    ground_truth.append(label_dict[key])
                
                pred_array = np.flip(np.argsort(predictions,axis=1),axis=1) # Prediction with the most confidence is the first element here
                c_pred = pred_array[:,0]

                correct_count = np.sum(c_pred==ground_truth)
                accuracy1 = float(correct_count)/len(c_pred)
                print(f'Val loss for epoch {epoch} is {np.mean(val_losses)}')
                print(f'Correct Count is {correct_count} out of {len(c_pred)}')
                writer.add_scalar('Validation Loss', np.mean(val_loss), epoch)
                writer.add_scalar('Validation Accuracy', np.mean(accuracy1), epoch)
                print(f'Overall Accuracy is for epoch {epoch} is {accuracy1*100 :.3f}% ')
                
                accuracy = accuracy1

            if accuracy > best_acc:
                print('++++++++++++++++++++++++++++++')
                print(f'Epoch {epoch} is the best model till now for {run_id}!')
                print('++++++++++++++++++++++++++++++')
                save_dir = os.path.join(cfg.saved_models_dir, run_id)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_file_path = os.path.join(save_dir, 'model_{}_bestAcc_{}.pth'.format(epoch, str(accuracy)[:6]))
                states = {
                    'epoch': epoch + 1,
                    # 'arch': params.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(states, save_file_path)
                best_acc = accuracy
            # else:
            save_dir = os.path.join(cfg.saved_models_dir, run_id)
            save_file_path = os.path.join(save_dir, 'model_temp.pth')
            states = {
                'epoch': epoch + 1,
                # 'arch': params.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)
            
        except:
            print("Epoch ", epoch, " failed")
            print('-'*60)
            traceback.print_exc(file=sys.stdout)
            print('-'*60)
            continue
        taken = time.time()-start
        print(f'Time taken for Epoch-{epoch} is {taken}')
        print()
        train_dataset = baseline_dataloader_train_strong(shuffle = False, data_percentage = params.data_percentage)
        train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train)
        print(f'Train dataset length: {len(train_dataset)}')
        print(f'Train dataset steps per epoch: {len(train_dataset)/params.batch_size}')
        if learning_rate2 < 1e-5:
            print(f'Learning rate is very low now, ending the process...s')
            exit()



        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to do linear evaluation ')

    parser.add_argument("--run_id", dest='run_id', type=str, required=False, default= "dummy_linear",
                        help='run_id')
    parser.add_argument("--restart", action='store_true')
    parser.add_argument("--saved_model", dest='saved_model', type=str, required=False, default= None,
                        help='run_id')
    parser.add_argument("--linear", action='store_true')
    # print()
    # print('Repeating r3d57, Optimizer grad inside each iteration')
    # print()

    args = parser.parse_args()
    print(f'Restart {args.restart}')

    run_id = args.run_id
    saved_model = args.saved_model
    linear = args.linear


    train_classifier(str(run_id), args.restart, saved_model, linear)


        


