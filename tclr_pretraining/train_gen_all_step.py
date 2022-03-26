import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
import time
import os
import numpy as np
from model import build_r3d_backbone, build_r3d_mlp, load_r3d_mlp
import parameters as params
import config as cfg
from dl_tclr import ss_dataset_gen1, collate_fn2
import sys, traceback
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from tensorboardX import SummaryWriter
import cv2
from torch.utils.data import DataLoader
import math
import argparse
from keras.utils import to_categorical
import random
from contrastive_loss.nt_xent_original import *
from contrastive_loss.global_local_temporal_contrastive import global_local_temporal_contrastive
from torch.cuda.amp import autocast, GradScaler

if torch.cuda.is_available(): 
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.benchmark = True


def train_epoch(scaler, run_id, learning_rate2, epoch, criterion, data_loader, model, optimizer, writer, use_cuda, criterion2):
    print('train at epoch {}'.format(epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate2
        writer.add_scalar('Learning Rate', learning_rate2, epoch)  

        print("Learning rate is: {}".format(param_group['lr']))
  
    losses = []
    losses_gsr_gdr, losses_ic2, losses_ic1, losses_local_local = [], [], [], []
    losses_global_local = []

    model.train()

    for i, (sparse_clip, dense_clip0, dense_clip1, dense_clip2, dense_clip3, a_sparse_clip, \
            a_dense_clip0, a_dense_clip1, a_dense_clip2, a_dense_clip3,_ ,_,_) in enumerate(data_loader):
        
        optimizer.zero_grad()

        sparse_clip = sparse_clip.permute(0,2,1,3,4) #aug_DL output is [120, 16, 3, 112, 112]], model expects [8, 3, 16, 112, 112]
        dense_clip0 = dense_clip0.permute(0,2,1,3,4)        
        dense_clip1 = dense_clip1.permute(0,2,1,3,4)
        dense_clip2 = dense_clip2.permute(0,2,1,3,4)
        dense_clip3 = dense_clip3.permute(0,2,1,3,4)

        # out_sparse will have output in this order: [sparse_clip[5], augmented_sparse_clip]
        # one element from the each of the list has 5 elements: see MLP file for details
        out_sparse = []
        # out_dense will have output in this order : [d0,d1,d2,d3,a_d0,...]
        out_dense = [[],[]]

        a_sparse_clip = a_sparse_clip.permute(0,2,1,3,4) #aug_DL output is [120, 16, 3, 112, 112]], model expects [8, 3, 16, 112, 112]
        a_dense_clip0 = a_dense_clip0.permute(0,2,1,3,4)        
        a_dense_clip1 = a_dense_clip1.permute(0,2,1,3,4)
        a_dense_clip2 = a_dense_clip2.permute(0,2,1,3,4)
        a_dense_clip3 = a_dense_clip3.permute(0,2,1,3,4)

        with autocast():

            out_sparse.append(model((sparse_clip.cuda(),'s')))
            out_sparse.append(model((a_sparse_clip.cuda(),'s')))

            out_dense[0].append(model((dense_clip0.cuda(),'d')))
            out_dense[0].append(model((dense_clip1.cuda(),'d')))
            out_dense[0].append(model((dense_clip2.cuda(),'d')))
            out_dense[0].append(model((dense_clip3.cuda(),'d')))

            out_dense[1].append(model((a_dense_clip0.cuda(),'d')))
            out_dense[1].append(model((a_dense_clip1.cuda(),'d')))
            out_dense[1].append(model((a_dense_clip2.cuda(),'d')))
            out_dense[1].append(model((a_dense_clip3.cuda(),'d')))


            criterion = NTXentLoss(device = 'cuda', batch_size=out_sparse[0][0].shape[0], temperature=params.temperature, use_cosine_similarity = False).cuda()
            criterion_local_local = NTXentLoss(device = 'cuda', batch_size=4, temperature=params.temperature, use_cosine_similarity = False).cuda()
            
            # Instance contrastive losses with the global clips (sparse clips)

            loss_ic2 = criterion(out_sparse[0][0], out_sparse[1][0])

            loss_ic1 = 0
            
            # Instance contrastive losses with the local clips (dense clips)
            for ii in range(2):
                for jj in range(2):
                    for chunk in range(1,5):
                        for chunk1 in range(1,5):
                            if (ii == jj and chunk == chunk1):
                                continue
                            loss_ic1 += criterion(out_dense[ii][chunk-1],out_dense[jj][chunk1-1])
            
            loss_ic1 /= 4 #scaling over ii and jj

            loss_local_local = 0
            # print(out_dense[0][0].shape) # this prints shape of [4,128]
            # print(torch.stack(out_dense[0],dim=1).shape) # this prints shape of [BS, 4, 128]
            # exit()
            for ii in range(out_dense[0][0].shape[0]): #for loop in the batch size
                loss_local_local += criterion_local_local(torch.stack(out_dense[0],dim=1)[ii], torch.stack(out_dense[1],dim=1)[ii])
            
            loss_global_local=0
            for ii in range(2):
                for jj in range(2):
                    loss_global_local += criterion2(torch.stack(out_sparse[ii][1:],dim=1), torch.stack(out_dense[jj],dim=1), params.temperature)

            loss = loss_ic2 + loss_ic1 + loss_local_local + loss_global_local

        loss_unw = loss_ic2.item()+ loss_ic1.item() + loss_local_local.item() + loss_global_local.item()
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss_unw)
        losses_local_local.append(loss_local_local.item())
        losses_global_local.append(loss_global_local.item())

        losses_ic1.append(loss_ic1.item())
        losses_ic2.append(loss_ic2.item())

        
        if (i+1) % 25 == 0: 
            print(f'Training Epoch {epoch}, Batch {i}, Loss: {np.mean(losses) :.5f}')
            print(f'Training Epoch {epoch}, Batch {i}, losses_local_local: {np.mean(losses_local_local) :.5f}')
            print(f'Training Epoch {epoch}, Batch {i}, losses_global_local: {np.mean(losses_global_local) :.5f}')
            print(f'Training Epoch {epoch}, Batch {i}, losses_ic2: {np.mean(losses_ic2) :.5f}')
            print(f'Training Epoch {epoch}, Batch {i}, losses_ic1: {np.mean(losses_ic1) :.5f}')

        # exit()
    print('Training Epoch: %d, Loss: %.4f' % (epoch,  np.mean(losses)))
    writer.add_scalar('Training Loss', np.mean(losses), epoch)
    writer.add_scalar('losses_local_local', np.mean(losses_local_local), epoch)
    writer.add_scalar('losses_global_local', np.mean(losses_global_local), epoch)
    writer.add_scalar('losses_ic2', np.mean(losses_ic2), epoch)
    writer.add_scalar('losses_ic1', np.mean(losses_ic1), epoch)
    
    del out_sparse, out_dense, loss, loss_ic2, loss_ic1, losses_local_local, loss_global_local

    return model, np.mean(losses), scaler
    
def train_classifier(run_id, restart):
    use_cuda = True
    writer = SummaryWriter(os.path.join(cfg.logs, str(run_id)))
    print(f'Temperature used for the nt_xent loss is {params.temperature}')
    print(f'Batch size {params.batch_size}')
    print(f'Weight decay {params.weight_decay}')


    save_dir = os.path.join(cfg.saved_models_dir, run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if restart:
        saved_model_file = save_dir + str(run_id) + '/model_temp.pth'
        
        try:
            model = load_r3d_mlp(saved_model_file= saved_model_file)
            epoch0 = torch.load(saved_model_file)['epoch']
            learning_rate1 = torch.load(saved_model_file)['learning_rate']
            best_score= torch.load(saved_model_file)['best_score'] 
            scheduler_epoch = torch.load(saved_model_file)['scheduler_epoch'] 
            scaler = torch.load(saved_model_file)['amp_scaler'] 

        except:
            print(f'No such model exists: {saved_model_file} :(')
            epoch0 = 0 
            model = build_r3d_mlp()
            scheduler_epoch = 0
            best_score = 10000
            learning_rate1 = params.learning_rate
            scaler = GradScaler()

    else:
        epoch0 = 0 
        model = build_r3d_mlp()
        scheduler_epoch = 0
        best_score = 10000
        scaler = GradScaler()

        learning_rate1 = params.learning_rate
    print(f'Starting learning rate {learning_rate1}')
    print(f'Scheduler_epoch {scheduler_epoch}')
    print(f'Best score till now is {best_score}')

    criterion = NTXentLoss(device = 'cuda', batch_size=params.batch_size, temperature=params.temperature, use_cosine_similarity = False)

    if torch.cuda.device_count()>1:
        print(f'Multiple GPUS found!')
        model=nn.DataParallel(model)
        model.cuda()
    else:
        print('Only 1 GPU is available')
        model.cuda()

    optimizer = optim.Adam(model.parameters(),lr=learning_rate1, weight_decay = params.weight_decay)
    train_dataset = ss_dataset_gen1(shuffle = True, data_percentage = params.data_percentage)
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn2)
    print(f'Train dataset length: {len(train_dataset)}')
    print(f'Train dataset steps per epoch: {len(train_dataset)/params.batch_size}')
   
    learning_rate2 = learning_rate1 
    scheduler_step = 1         

    for epoch in range(epoch0, params.num_epochs):
        print(f'Epoch {epoch} started')
        if epoch < params.warmup:
            learning_rate2 = params.warmup_array[epoch]*params.learning_rate

        if scheduler_epoch == params.scheduler_patience:
            print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
            print(f'Dropping learning rate to {learning_rate2/10} for epoch')
            print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
            learning_rate2 = learning_rate1/(10**scheduler_step)
            scheduler_epoch = 0
            scheduler_step += 1


        start=time.time()
        try:
            model, loss, scaler = train_epoch(scaler, run_id, learning_rate2, epoch, criterion, train_dataloader, model, optimizer, writer, use_cuda, criterion2 = global_local_temporal_contrastive)
                        
            if loss < best_score:
                print('++++++++++++++++++++++++++++++')
                print(f'Epoch {epoch} is the best model till now for {run_id}!')
                print('++++++++++++++++++++++++++++++')
                save_dir = os.path.join(cfg.saved_models_dir, run_id)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_file_path = os.path.join(save_dir, 'model_best_e{}_loss_{}.pth'.format(epoch, str(loss)[:6]))
                states = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'learning_rate': learning_rate2,
                    'amp_scaler': scaler,

                }
                torch.save(states, save_file_path)
                best_score = loss
                scheduler_epoch = 0
            elif epoch % 5 == 0:
                save_dir = os.path.join(cfg.saved_models_dir, run_id)
                save_file_path = os.path.join(save_dir, 'model_e{}_loss_{}.pth'.format(epoch, str(loss)[:6]))
                states = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'learning_rate': learning_rate2,
                    'amp_scaler': scaler,
                }
                torch.save(states, save_file_path)

            if loss > best_score:
                scheduler_epoch += 1
            
            save_dir = os.path.join(cfg.saved_models_dir, run_id)
            save_file_path = os.path.join(save_dir, 'model_temp.pth')
            states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate2,
                'best_score': best_score,
                'scheduler_epoch': scheduler_epoch,
                'amp_scaler': scaler,
            }
            torch.save(states, save_file_path)
        except:
            print("Epoch ", epoch, " failed")
            print('-'*60)
            traceback.print_exc(file=sys.stdout)
            print('-'*60)
            continue

        train_dataset = ss_dataset_gen1(shuffle = True, data_percentage = params.data_percentage)
        train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn2)
        print(f'Train dataset length: {len(train_dataset)}')
        print(f'Train dataset steps per epoch: {len(train_dataset)/params.batch_size}')
       
        taken = time.time()-start
        print(f'Time taken for Epoch-{epoch} is {taken}')
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train baseline')

    parser.add_argument("--run_id", dest='run_id', type=str, required=False, default= "dummy",
                        help='run_id')
    parser.add_argument("--restart", action='store_true')

    print()
    print('TCLR pretraining starts...')
    print()

    args = parser.parse_args()
    print(f'Restart {args.restart}')

    run_id = args.run_id
    print(f'Run_id {args.run_id}')

    train_classifier(str(run_id), args.restart)



        


