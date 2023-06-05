
from easydict import EasyDict
import torch
import numpy as np
import math

cfg = EasyDict()

# set devices
cfg.factory_device = 0
cfg.devices = [0]

# directory
cfg.odometry_home = '/home/song/Dataset/odometry/'
cfg.proj_home = '/home/song/kew/kitti-scc-5dof/'
cfg.model = 'lccnet'

cfg.logdir = cfg.proj_home + 'checkpoint/' + cfg.model + '/'
cfg.data_subdir = 'data_odometry_color/dataset/sequences/'
cfg.traintxt = 'kitti-train.txt'
cfg.valtxt = 'kitti-val.txt'

cfg.inputw = 1088
cfg.inputh = 224
cfg.cutimg = 64
cfg.volume_in_channels= 128
cfg.qkv_channels = 512

# train
cfg.max_rot_err = 2.5
cfg.num_cpu = 8
cfg.batchsize = 32
cfg.learing_rate = 8e-4
cfg.maxepoch = 60
cfg.MultiStepLR_milstone = [30, 40]
cfg.MultiStepLR_gamma = 0.5

# val
cfg.number_of_frames = 8
cfg.test_num = 100