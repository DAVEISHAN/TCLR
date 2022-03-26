import numpy as np
num_frames = 16
num_workers = 8
batch_size = 32#16#24
learning_rate = 1e-2#1e-4#1e-5
num_epochs = 100
data_percentage = 1.0
v_batch_size = 48#80
warmup_array = list(np.linspace(0,1, 10) + 1e-9)
warmup = len(warmup_array)
scheduler_patience = 3
fix_skip = 2
num_modes = 10

reso_h = 112
reso_w = 112

ori_reso_h = 240
ori_reso_w = 320

sr_ratio = 4