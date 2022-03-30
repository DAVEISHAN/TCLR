import numpy as np
num_frames = 16#16
num_workers = 32
batch_size = 32
learning_rate = 1e-5
num_epochs = 300
data_percentage = 1.0
v_batch_size = 32

warmup_array = list(np.linspace(0,1, 10) + 1e-9)
warmup = len(warmup_array)
scheduler_patience = 7
fix_skip = 2
num_modes = 9#10
num_skips = 1
num_classes = 102
hflip = [0] #list(range(2))
cropping_fac1 = [0.8] #[0.7,0.85,0.8,0.75]

reso_h = 112
reso_w = 112

ori_reso_h = 240
ori_reso_w = 320


sr_ratio = 4