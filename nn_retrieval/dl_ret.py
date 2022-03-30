import os
import torch
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import config as cfg
import random
import pickle
import parameters_BL as params
import json
import math
import cv2
# from tqdm import tqdm
import time
import torchvision.transforms as trans
# from decord import VideoReader

class multi_baseline_dataloader_train_strong(Dataset):

    def __init__(self,  split = 1, shuffle = True, data_percentage = 1.0, mode = 0, skip = 1, \
                hflip=0, cropping_factor=1.0, split = 1):
        if split == 1:
            self.all_paths = open(os.path.join(cfg.path_folder, 'ucfTrainTestlist/trainlist01.txt'),'r').read().splitlines()
        elif split ==2: 
            self.all_paths = open(os.path.join(cfg.path_folder, 'ucfTrainTestlist/trainlist02.txt'),'r').read().splitlines()
        elif split ==3: 
            self.all_paths = open(os.path.join(cfg.path_folder, 'ucfTrainTestlist/trainlist03.txt'),'r').read().splitlines()
        else:
            print(f'Invalid split input: {split}')
        self.classes= json.load(open(cfg.class_mapping))['classes']
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.all_paths)
        
        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths)*self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]
        self.PIL = trans.ToPILImage()
        self.TENSOR = trans.ToTensor()
        self.mode = mode
        self.skip = skip
        self.hflip = hflip
        self.cropping_factor = cropping_factor
                       
    
    def __len__(self):
        return len(self.data)
            
    def __getitem__(self,index):        
        clip, label, vid_path, frame_list = self.process_data(index)
        return clip, label, vid_path, frame_list

    def process_data(self, idx):
    
        # label_building
        # vid_path = cfg.dataset_folder + self.data[idx].split(' ')[0]
        vid_path = cfg.path_folder + '/UCF-101/' + self.data[idx].split(' ')[0]
        

        # print(vid_path)
        # exit()
        label = self.classes[vid_path.split('/')[-2]] # This element should be activity name
        
        # clip_building
        clip, frame_list = self.build_clip(vid_path)

        return clip, label, vid_path, frame_list

    def build_clip(self, vid_path):
        try:
            cap = cv2.VideoCapture(vid_path)
            cap.set(1, 0)
            frame_count = cap.get(7)

            N = frame_count
            n = params.num_frames

          
            skip_frames_full = params.fix_skip 
            left_over = skip_frames_full*n
            F = N - left_over

            start_frame_full = 0 + int(np.linspace(0,F-10,params.num_modes)[self.mode])

            
            if start_frame_full< 0:
                start_frame_full = self.mode
                
            full_clip_frames = []

            
            full_clip_frames = start_frame_full + np.asarray(
                [int(int(skip_frames_full) * f) for f in range(params.num_frames)])

        

            count = -1
            # fail = 0
            full_clip = []
            list_full = []

            while (cap.isOpened()):
                count += 1
                ret, frame = cap.read()
                # print(frame.shape)
                if ((count not in full_clip_frames) and (ret == True)):
                    # ret, frame = cap.read()
                    continue
                # ret, frame = cap.read()
                if ret == True:
                    

                    if (count in full_clip_frames):
                        full_clip.append(self.augmentation(frame))
                        list_full.append(count)

                else:
                    break

            if len(full_clip) < params.num_frames and len(full_clip)>(params.num_frames/2):
                remaining_num_frames = params.num_frames - len(full_clip)
                full_clip = full_clip + full_clip[::-1][1:remaining_num_frames+1]
            # print(len(full_clip))
            assert (len(full_clip) == params.num_frames)

            return full_clip, list_full

        except:
            print(f'Clip {vid_path} Failed')
            return None, None

    def augmentation(self, image):
        image = self.PIL(image)
        

        if self.cropping_factor <= 1:
            image = trans.functional.center_crop(image,(int(params.ori_reso_h*self.cropping_factor),int(params.ori_reso_h*self.cropping_factor)))
        image = trans.functional.resize(image, (params.reso_h, params.reso_w))
        if self.hflip !=0:
            image = trans.functional.hflip(image)

        return trans.functional.to_tensor(image)

class multi_baseline_dataloader_val_strong(Dataset):

    def __init__(self,  split = 1, shuffle = True, data_percentage = 1.0, mode = 0, skip = 1, \
                hflip=0, cropping_factor=1.0,  split = 1):
        if split == 1:
            self.all_paths = open(os.path.join(cfg.path_folder, 'ucfTrainTestlist/testlist01.txt'),'r').read().splitlines()
        elif split ==2: 
            self.all_paths = open(os.path.join(cfg.path_folder, 'ucfTrainTestlist/testlist02.txt'),'r').read().splitlines()
        elif split ==3: 
            self.all_paths = open(os.path.join(cfg.path_folder, 'ucfTrainTestlist/testlist03.txt'),'r').read().splitlines()
        else:
            print(f'Invalid split input: {split}')
        self.classes= json.load(open(cfg.class_mapping))['classes']
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.all_paths)
        
        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths)*self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]
        self.PIL = trans.ToPILImage()
        self.TENSOR = trans.ToTensor()
        self.mode = mode
        self.skip = skip
        self.hflip = hflip
        self.cropping_factor = cropping_factor
                       
    
    def __len__(self):
        return len(self.data)
            
    def __getitem__(self,index):        
        clip, label, vid_path, frame_list = self.process_data(index)
        return clip, label, vid_path, frame_list

    def process_data(self, idx):
    
        # label_building
        vid_path = cfg.path_folder + '/UCF-101/' + self.data[idx].split(' ')[0]
        
        label = self.classes[vid_path.split('/')[-2]] # This element should be activity name
        
        # clip_building
        clip, frame_list = self.build_clip(vid_path)

        return clip, label, vid_path, frame_list

    def build_clip(self, vid_path):
        try:
            cap = cv2.VideoCapture(vid_path)
            cap.set(1, 0)
            frame_count = cap.get(7)

            N = frame_count
            n = params.num_frames

            # skip_max = int(N/n)
            # if (skip_max < 2) and (self.skip>1):
            #     # print(frame_count)
            #     # print('Skip_max =1 and already covered')
            #     return None


            skip_frames_full = params.fix_skip #int(skip_max/params.num_skips*self.skip)

            # if skip_frames_full<2:
            #     if self.skip==1:
            #         skip_frames_full = 1
            #     else:
            #         # print('Skip_rate already covered')
            #         return None
            
            left_over = skip_frames_full*n
            F = N - left_over

            start_frame_full = 0 + int(np.linspace(0,F-10,params.num_modes)[self.mode])

            # start_frame_full = int(F/(params.num_modes-1))*self.mode + 1 if params.num_modes>1 else 0
            if start_frame_full< 0:
                start_frame_full = self.mode
                # print()
                # print(f'oh no, {start_frame_full}')
                # return None, None

            full_clip_frames = []

            
            full_clip_frames = start_frame_full + np.asarray(
                [int(int(skip_frames_full) * f) for f in range(params.num_frames)])

        

            count = -1
            # fail = 0
            full_clip = []
            list_full = []

            while (cap.isOpened()):
                count += 1
                ret, frame = cap.read()
                # print(frame.shape)
                if ((count not in full_clip_frames) and (ret == True)):
                    # ret, frame = cap.read()
                    continue
                # ret, frame = cap.read()
                if ret == True:
                    # Resize - THIS CAN ALSO BE IMPROVED
                    # frame1 = cv2.resize(frame, (171, 128))
                    # print(f'new size{frame1.shape}')
                    # print()
                    # frame1 = self.augmentation(frame1[0:0 + 112, 29:29 + 112])

                    if (count in full_clip_frames):
                        full_clip.append(self.augmentation(frame))
                        list_full.append(count)

                else:
                    break

            if len(full_clip) < params.num_frames and len(full_clip)>(params.num_frames/2):
                # print()
                # print(f'Full clip has {len(full_clip)} frames')
                # print(f'Full video has {frame_count} frames')
                # print(f'Full video suppose to have these frames: {full_clip_frames}')
                # print(f'Actual video has           these frames: {list_full}')
                # print(f'final count value is {count}')
                # if params.num_frames - len(full_clip) >= 1:
                #     print(f'Clip {vid_path} is missing {params.num_frames - len(full_clip)} frames')
                # for remaining in range(params.num_frames - len(full_clip)):
                #     full_clip.append(frame1)
                remaining_num_frames = params.num_frames - len(full_clip)
                full_clip = full_clip + full_clip[::-1][1:remaining_num_frames+1]
            # print(len(full_clip))
            assert (len(full_clip) == params.num_frames)

            return full_clip, list_full

        except:
            print(f'Clip {vid_path} Failed')
            return None, None

    def augmentation(self, image):
        image = self.PIL(image)
        

        if self.cropping_factor <= 1:
            image = trans.functional.center_crop(image,(int(params.ori_reso_h*self.cropping_factor),int(params.ori_reso_h*self.cropping_factor)))
        image = trans.functional.resize(image, (params.reso_h, params.reso_w))
        if self.hflip !=0:
            image = trans.functional.hflip(image)

        return trans.functional.to_tensor(image)


def collate_fn1(batch):
    clip, label, vid_path = [], [], []
    for item in batch:
        if not (item[0] == None or item[1] == None or item[2] == None):
            # clip.append(torch.from_numpy(np.asarray(item[0],dtype='f')))
            clip.append(torch.stack(item[0],dim=0)) 

            label.append(item[1])
            vid_path.append(item[2])

    clip = torch.stack(clip, dim=0)

    return clip, label, vid_path

def collate_fn2(batch):

    f_clip, label, vid_path, frame_list = [], [], [], []
    # print(len(batch))
    for item in batch:
        if not (item[0] == None or item[1] == None or item[2] == None):
            f_clip.append(torch.stack(item[0],dim=0)) # I might need to convert this tensor to torch.float
            label.append(item[1])
            vid_path.append(item[2])
            frame_list.append(item[3])

        # else:
            # print('oh no2')
    # print(len(f_clip))
    f_clip = torch.stack(f_clip, dim=0)
    
    return f_clip, label, vid_path, frame_list 
            
def collate_fn_train(batch):

    f_clip, label, vid_path = [], [], [], 
    # print(len(batch))
    for item in batch:
        if not (item[0] == None or item[1] == None or item[2] == None):
            f_clip.append(torch.stack(item[0],dim=0)) # I might need to convert this tensor to torch.float
            label.append(item[1])
            vid_path.append(item[2])
        # else:
            # print('oh no2')
    # print(len(f_clip))
    f_clip = torch.stack(f_clip, dim=0)
    
    return f_clip, label, vid_path 

if __name__ == '__main__':

    '''train_dataset = multi_baseline_dataloader_train_strong(split =1 , shuffle = False, data_percentage = 1.0,  mode = 2)
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, \
        shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn2)

    print(f'Step involved: {len(train_dataset)/params.batch_size}')
    t=time.time()

    for i, (clip, label, vid_path,_) in enumerate(train_dataloader):
        if i%10 == 0:
            print()
            clip = clip.permute(0,1,3,4,2)
            print(f'Full_clip shape is {clip.shape}')
            print(f'Label is {label}')
            # pickle.dump(clip, open('f_clip.pkl','wb'))
            # pickle.dump(label, open('label.pkl','wb'))
            # exit()
    print(f'Time taken to load data is {time.time()-t}')'''

    train_dataset = multi_baseline_dataloader_val_strong(split =1 , shuffle = False, data_percentage = 1.0,  mode = 2)
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn2)

    print(f'Step involved: {len(train_dataset)/params.batch_size}')
    t=time.time()

    for i, (clip, label, vid_path, frame_list) in enumerate(train_dataloader):
        if i%25 == 0:
            print()
            # clip = clip.permute(0,1,3,4,2)
            print(f'Full_clip shape is {clip.shape}')
            print(f'Label is {label}')
            # print(f'Frame list is {frame_list}')
            
            # pickle.dump(clip, open('f_clip.pkl','wb'))
            # pickle.dump(label, open('label.pkl','wb'))
            # exit()
    print(f'Time taken to load data is {time.time()-t}')

