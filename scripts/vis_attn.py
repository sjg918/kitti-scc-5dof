
import os

import datetime
import random
import cv2
from PIL import Image
import time
import sys
import shutil
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from CFG.maincfg import cfg
from src.fen import FeatureExtractionNetwork_dualpath
from src.can import CrossAttentionNet_visattn

from src.kittifactory import DataFactory
from src.utils import *

number_of_frames = cfg.number_of_frames
log_folder = cfg.model


def val_kitti_temporal_filtering(vis_start, vis_end, random, angle):
    if os.path.exists(cfg.proj_home + '/' + log_folder + '/'):
        pass
    else:
        os.makedirs(cfg.proj_home + '/' + log_folder + '/')

    cent_ids = 'cuda:' + str(cfg.devices[0])

    back = FeatureExtractionNetwork_dualpath().to(cent_ids)
    Anet = CrossAttentionNet_visattn(cfg.volume_in_channels, cfg.qkv_channels).to(cent_ids)
        
    load_model(back, cfg.proj_home + 'checkpoint/ours-back.pth', cent_ids)
    load_model(Anet, cfg.proj_home + 'checkpoint/ours-Anet.pth', cent_ids)

    with open(cfg.proj_home + 'gendata/100miscalib-left.txt', 'r') as f:
        predefine_error_list_left = f.readlines()

    with open(cfg.proj_home + 'gendata/100miscalib-right.txt', 'r') as f:
        predefine_error_list_right = f.readlines()
        
    with open(cfg.proj_home + 'gendata/100number-kitti.txt', 'r') as f:
        predefine_frame_number = f.readlines()

    # define dataloader
    kitti_dataset = DataFactory(cfg, 'val')

    back.eval()
    Anet.eval()

    with torch.no_grad():
        for cnt_ in range(vis_start, vis_end):  
            print(cnt_, end='\r')
            maxrot = cfg.max_rot_err

            #rotx_left, roty_left, rotz_left = predefine_error_list_left[cnt_].strip().split(' ')
            if random:
                rotx_left = np.random.uniform(-maxrot, maxrot)
                roty_left = np.random.uniform(-maxrot, maxrot)
                rotz_left = np.random.uniform(-maxrot, maxrot)
            else:
                rotx_left, roty_left, rotz_left = angle[0, 0], angle[0, 1], angle[0, 2]
            rotx_left_, roty_left_, rotz_left_ = rotx_left, roty_left, rotz_left
            rotx_left = float(rotx_left)
            roty_left = float(roty_left) * (3.141592 / 180.0)
            rotz_left = float(rotz_left) * (3.141592 / 180.0)

            #rotx_right, roty_right, rotz_right = predefine_error_list_right[cnt_].strip().split(' ')
            if random:
                rotx_right = np.random.uniform(-maxrot, maxrot)
                roty_right = np.random.uniform(-maxrot, maxrot)
                rotz_right = np.random.uniform(-maxrot, maxrot)
            else:
                rotx_right, roty_right, rotz_right = angle[1, 0], angle[1, 1], angle[1, 2]
            rotx_right_, roty_right_, rotz_right_ = rotx_right, roty_right, rotz_right
            rotx_right = float(rotx_right)
            roty_right = float(roty_right) * (3.141592 / 180.0)
            rotz_right = float(rotz_right) * (3.141592 / 180.0)

            #rotx_mid = (rotx_left + rotx_right) / 2 * (np.pi / 180.0)
            rotx_left = rotx_left * (np.pi / 180.0)
            rotx_right = rotx_right * (np.pi / 180.0)

            rotmat_left = eulerAnglesToRotationMatrix([rotx_left, roty_left, rotz_left])
            misRTmat_left = np.zeros((4, 4), dtype=np.float32)
            misRTmat_left[:3, :3] = rotmat_left[:3,:3]
            misRTmat_left[0, 3] = 0
            misRTmat_left[1, 3] = 0
            misRTmat_left[2, 3] = 0
            misRTmat_left[3, 3] = 1

            rotmat_right = eulerAnglesToRotationMatrix([rotx_right, roty_right, rotz_right])
            misRTmat_right = np.zeros((4, 4), dtype=np.float32)
            misRTmat_right[:3, :3] = rotmat_right[:3,:3]
            misRTmat_right[0, 3] = 0
            misRTmat_right[1, 3] = 0
            misRTmat_right[2, 3] = 0
            misRTmat_right[3, 3] = 1
            
            framenumber = predefine_frame_number[cnt_].strip()
            framenumber = int(framenumber)
                            
            for cnt in range(framenumber, framenumber + number_of_frames):
                seq, imgnum = kitti_dataset.datalist[cnt].split(' ')
                imgnum = imgnum.strip()

                left_path = cfg.odometry_home + cfg.data_subdir + seq + '/image_2/' + imgnum + '.png'
                right_path = cfg.odometry_home + cfg.data_subdir + seq + '/image_3/' + imgnum + '.png'
                left_image, right_image = cv2.imread(left_path), cv2.imread(right_path)

                left_image = left_image[:370, :1226, :]
                right_image = right_image[:370, :1226, :]
                cv2.imwrite(cfg.proj_home + '/' + log_folder + '/left.png', left_image)
                cv2.imwrite(cfg.proj_home + '/' + log_folder + '/right.png', right_image)

                k_cam2 = kitti_dataset.calib_dict[seq]['K_cam2']
                k_cam3 = kitti_dataset.calib_dict[seq]['K_cam3']

                left_normalized_points = kitti_dataset.left_normcoord_dict[seq].copy()
                right_normalized_points = kitti_dataset.right_normcoord_dict[seq].copy()

                _, left_input_img = gen_error_on_gpu(left_image, k_cam2, left_normalized_points, cfg.factory_device, misRTmat_left)
                _, right_input_img = gen_error_on_gpu(right_image, k_cam3, right_normalized_points, cfg.factory_device, misRTmat_right)

                left_input_ = left_input_img.numpy()
                right_input_ = right_input_img.numpy()
                left_input_ = left_input_[cfg.cutimg:cfg.cutimg+cfg.inputh, cfg.cutimg:cfg.cutimg+cfg.inputw]
                right_input_ = right_input_[cfg.cutimg:cfg.cutimg+cfg.inputh, cfg.cutimg:cfg.cutimg+cfg.inputw]
                #cv2.imwrite(cfg.proj_home + '/left_input.png', left_input_)
                #cv2.imwrite(cfg.proj_home + '/right_input.png', right_input_)

                left_input_img = left_input_img.to(dtype=torch.float32).numpy() / 255.
                right_input_img = right_input_img.to(dtype=torch.float32).numpy() / 255.

                left_input_img = left_input_img[cfg.cutimg:cfg.cutimg+cfg.inputh, cfg.cutimg:cfg.cutimg+cfg.inputw, :]
                left_input_img = (left_input_img - kitti_dataset.mean) / kitti_dataset.std
                left_input_img = left_input_img.transpose(2, 0, 1)
                left_input_img = torch.from_numpy(left_input_img)
                left_normalized_points = left_normalized_points.reshape((370, 1226, 3)).transpose(2, 0, 1)
                left_normalized_points = left_normalized_points[:2, cfg.cutimg:cfg.cutimg+cfg.inputh, cfg.cutimg:cfg.cutimg+cfg.inputw]
                left_normalized_points = torch.from_numpy(left_normalized_points).to(dtype=torch.float32)
                left_input_img = torch.cat((left_input_img, left_normalized_points), dim=0)


                right_input_img = right_input_img[cfg.cutimg:cfg.cutimg+cfg.inputh, cfg.cutimg:cfg.cutimg+cfg.inputw, :]
                right_input_img = (right_input_img - kitti_dataset.mean) / kitti_dataset.std
                right_input_img = right_input_img.transpose(2, 0, 1)
                right_input_img = torch.from_numpy(right_input_img)
                right_normalized_points = right_normalized_points.reshape((370, 1226, 3)).transpose(2, 0, 1)
                right_normalized_points = right_normalized_points[:2, cfg.cutimg:cfg.cutimg+cfg.inputh, cfg.cutimg:cfg.cutimg+cfg.inputw]
                right_normalized_points = torch.from_numpy(right_normalized_points).to(dtype=torch.float32)
                right_input_img = torch.cat((right_input_img, right_normalized_points), dim=0)

                left_input = left_input_img.unsqueeze(0).to(device=cent_ids)
                right_input = right_input_img.unsqueeze(0).to(device=cent_ids)
                
                left_fea = back(left_input)
                right_fea = back(right_input)
                attnL, attnR = Anet(left_fea, right_fea)

                attnL = attnL.squeeze(0).cpu().numpy()
                attnR = attnR.squeeze(0).cpu().numpy()
                
                attnL_ = attnL[5]
                attnL_ = attnL_.reshape(14, 68)
                max_ = np.max(attnL_)
                min_ = np.min(attnL_)
                attnL_ = (attnL_ - min_) / (max_ - min_) * 255
                attnL_ = attnL_.astype(np.uint8)
                attnL_ = cv2.applyColorMap(attnL_, cv2.COLORMAP_JET)
                attnL_ = cv2.resize(attnL_, (1088, 224))

                left_input__ = (left_input_.astype(np.float32) / 2 + attnL_.astype(np.float32) / 2).astype(np.uint8)
                cv2.putText(left_input__, 'X:'+str(rotx_right_)[:4], (10, 20), 1, 1.5, (0, 255, 255), 1)
                cv2.putText(left_input__, 'Y:'+str(roty_right_)[:4], (120, 20), 1, 1.5, (0, 255, 255), 1)
                cv2.putText(left_input__, 'Z:'+str(rotz_right_)[:4], (230, 20), 1, 1.5, (0, 255, 255), 1)
                cv2.imwrite(cfg.proj_home + '/results/' + imgnum + 'left.png', left_input__)
                continue
            continue
    print('\n end')


if __name__ == '__main__':
    torch.manual_seed(677)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(677)
    np.random.seed(677)
    torch.multiprocessing.set_start_method('spawn')

    vis_start, vis_end = 33, 35
    angle = np.array([[1.0, 0.5, -0.5], [0, 0, -2.4]])
    
    val_kitti_temporal_filtering(vis_start, vis_end, False, angle)