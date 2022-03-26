import numpy as np

num_frames = 16
batch_size = 40
learning_rate = 1e-3 #base learning rate
num_epochs = 1000 # training code will end after this num_of_epochs
data_percentage = 1.0 #just for debugging purpose, default = 1
temperature = 0.1 
weight_decay = 1e-9
sr_ratio = 4

ori_reso_h = 240
ori_reso_w = 320

reso_h = 112
reso_w = 112

warmup = 5 
warmup_array = [1/100,1/20,1/10,1/2,1]
scheduler_patience = 9


    


