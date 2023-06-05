
import torch.utils.data as data
from src.utils import *
import cv2

import numpy as np
import torch

class DataFactory(data.Dataset):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.mode = mode

        # train or val
        if mode == 'train':
            with open(cfg.proj_home + 'gendata/' + cfg.traintxt) as f:
                self.datalist = f.readlines()
        elif mode == 'val':
            with open(cfg.proj_home + 'gendata/' + cfg.valtxt) as f:
                self.datalist = f.readlines()
        else:
            print('datafactory init error !!!!')
            df=df

        self.calib_dict, self.left_normcoord_dict, self.right_normcoord_dict = init_dict(cfg)

        self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        # load
        seq, imgnum = self.datalist[idx].split(' ')
        imgnum = imgnum.strip()
        left_path = self.cfg.odometry_home + self.cfg.data_subdir + seq + '/image_2/' + imgnum + '.png'
        right_path = self.cfg.odometry_home + self.cfg.data_subdir + seq + '/image_3/' + imgnum + '.png'
        left_image, right_image = cv2.imread(left_path), cv2.imread(right_path)

        left_image = left_image[:370, :1226, :]
        right_image = right_image[:370, :1226, :]

        k_cam2 = self.calib_dict[seq]['K_cam2']
        k_cam3 = self.calib_dict[seq]['K_cam3']

        left_normalized_points = self.left_normcoord_dict[seq].copy()
        right_normalized_points = self.right_normcoord_dict[seq].copy()

        # miscalibration
        maxrot = self.cfg.max_rot_err

        left_rotx = np.random.uniform(-maxrot, maxrot)
        right_rotx = np.random.uniform(-maxrot, maxrot)
        rotx_mid = (left_rotx + right_rotx) / 2 * (np.pi / 180.0)
        
        left_rotx = left_rotx * (np.pi / 180.0)
        left_roty = np.random.uniform(-maxrot, maxrot) * (np.pi / 180.0)
        left_rotz = np.random.uniform(-maxrot, maxrot) * (np.pi / 180.0)
        right_rotx = right_rotx * (np.pi / 180.0)
        right_roty = np.random.uniform(-maxrot, maxrot) * (np.pi / 180.0)
        right_rotz = np.random.uniform(-maxrot, maxrot) * (np.pi / 180.0)
        
        left_miscalibrot = eulerAnglesToRotationMatrix([left_rotx, left_roty, left_rotz])
        left_miscalibRT = np.zeros((4, 4), dtype=np.float32)
        left_miscalibRT[:3, :3] = left_miscalibrot[:3,:3]
        left_miscalibRT[0, 3] = 0
        left_miscalibRT[1, 3] = 0
        left_miscalibRT[2, 3] = 0
        left_miscalibRT[3, 3] = 1

        right_miscalibrot = eulerAnglesToRotationMatrix([right_rotx, right_roty, right_rotz])
        right_miscalibRT = np.zeros((4, 4), dtype=np.float32)
        right_miscalibRT[:3, :3] = right_miscalibrot[:3,:3]
        right_miscalibRT[0, 3] = 0
        right_miscalibRT[1, 3] = 0
        right_miscalibRT[2, 3] = 0
        right_miscalibRT[3, 3] = 1

        _, left_input_img = gen_error_on_gpu(left_image, k_cam2, left_normalized_points, self.cfg.factory_device, left_miscalibRT)
        _, right_input_img = gen_error_on_gpu(right_image, k_cam3, right_normalized_points, self.cfg.factory_device, right_miscalibRT)

        # concatenate
        left_input_img = left_input_img.numpy() / 255.
        right_input_img = right_input_img.numpy() / 255.

        left_input_img = left_input_img[self.cfg.cutimg:self.cfg.cutimg+self.cfg.inputh, self.cfg.cutimg:self.cfg.cutimg+self.cfg.inputw, :]
        left_input_img = (left_input_img - self.mean) / self.std
        left_input_img = left_input_img.transpose(2, 0, 1)
        left_input_img = torch.from_numpy(left_input_img)
        left_normalized_points = left_normalized_points.reshape((370, 1226, 3)).transpose(2, 0, 1)
        left_normalized_points = left_normalized_points[:2, self.cfg.cutimg:self.cfg.cutimg+self.cfg.inputh, self.cfg.cutimg:self.cfg.cutimg+self.cfg.inputw]
        left_normalized_points = torch.from_numpy(left_normalized_points).to(dtype=torch.float32)
        left_input_img = torch.cat((left_input_img, left_normalized_points), dim=0)

        right_input_img = right_input_img[self.cfg.cutimg:self.cfg.cutimg+self.cfg.inputh, self.cfg.cutimg:self.cfg.cutimg+self.cfg.inputw, :]
        right_input_img = (right_input_img - self.mean) / self.std
        right_input_img = right_input_img.transpose(2, 0, 1)
        right_input_img = torch.from_numpy(right_input_img)
        right_normalized_points = right_normalized_points.reshape((370, 1226, 3)).transpose(2, 0, 1)
        right_normalized_points = right_normalized_points[:2, self.cfg.cutimg:self.cfg.cutimg+self.cfg.inputh, self.cfg.cutimg:self.cfg.cutimg+self.cfg.inputw]
        right_normalized_points = torch.from_numpy(right_normalized_points).to(dtype=torch.float32)
        right_input_img = torch.cat((right_input_img, right_normalized_points), dim=0)

        # gen target error
        left_miscalibrot = eulerAnglesToRotationMatrix([left_rotx - rotx_mid, left_roty, left_rotz])
        left_miscalibRT = np.zeros((4, 4), dtype=np.float32)
        left_miscalibRT[:3, :3] = left_miscalibrot[:3,:3]
        left_miscalibRT[0, 3] = 0
        left_miscalibRT[1, 3] = 0
        left_miscalibRT[2, 3] = 0
        left_miscalibRT[3, 3] = 1
        left_miscalibRT = np.linalg.inv(left_miscalibRT)

        right_miscalibrot = eulerAnglesToRotationMatrix([right_rotx - rotx_mid, right_roty, right_rotz])
        right_miscalibRT = np.zeros((4, 4), dtype=np.float32)
        right_miscalibRT[:3, :3] = right_miscalibrot[:3,:3]
        right_miscalibRT[0, 3] = 0
        right_miscalibRT[1, 3] = 0
        right_miscalibRT[2, 3] = 0
        right_miscalibRT[3, 3] = 1
        right_miscalibRT = np.linalg.inv(right_miscalibRT)

        left_targetQuaternion = quaternion_from_rotation_matrix(left_miscalibRT[:3, :3])
        left_targetQuaternion = torch.from_numpy(left_targetQuaternion).to(dtype=torch.float32)

        right_targetQuaternion = quaternion_from_rotation_matrix(right_miscalibRT[:3, :3])
        right_targetQuaternion = torch.from_numpy(right_targetQuaternion).to(dtype=torch.float32)

        if self.mode == 'val':
            return left_image, input_right_image, tgtRmat
            
        train_input_dict = {}
        train_input_dict['left'] = left_input_img
        train_input_dict['right'] = right_input_img
        train_input_dict['left_tgt_Q'] = left_targetQuaternion
        train_input_dict['right_tgt_Q'] = right_targetQuaternion

        return train_input_dict
    
    def collate_fn_cpu(self, batch):
        left_list = []
        right_list = []
        left_tgt_Q_list = []
        right_tgt_Q_list = []

        for train_input_dict in batch:
            left_list.append(train_input_dict['left'].unsqueeze(0))
            right_list.append(train_input_dict['right'].unsqueeze(0))
            left_tgt_Q_list.append(train_input_dict['left_tgt_Q'].unsqueeze(0))
            right_tgt_Q_list.append(train_input_dict['right_tgt_Q'].unsqueeze(0))
            continue
        left_list = torch.cat(left_list, dim=0)
        right_list = torch.cat(right_list, dim=0)
        left_tgt_Q_list = torch.cat(left_tgt_Q_list, dim=0)
        right_tgt_Q_list = torch.cat(right_tgt_Q_list, dim=0)

        train_input_dict = {
            'left': left_list,
            'right': right_list,
            'left_tgt_Q': left_tgt_Q_list,
            'right_tgt_Q': right_tgt_Q_list
        }
        return train_input_dict


# refer : https://github.com/utiasSTARS/pykitti
def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


# refer : https://github.com/utiasSTARS/pykitti
def _load_calib(calib_filepath):
    """Load and compute intrinsic and extrinsic calibration parameters."""
    # We'll build the calibration parameters as a dictionary, then
    # convert it to a namedtuple to prevent it from being modified later
    data = {}

    # Load the calibration file
    filedata = read_calib_file(calib_filepath)

    # Create 3x4 projection matrices
    P_rect_00 = np.reshape(filedata['P0'], (3, 4))
    P_rect_10 = np.reshape(filedata['P1'], (3, 4))
    P_rect_20 = np.reshape(filedata['P2'], (3, 4))
    P_rect_30 = np.reshape(filedata['P3'], (3, 4))

    data['P_rect_00'] = P_rect_00
    data['P_rect_10'] = P_rect_10
    data['P_rect_20'] = P_rect_20
    data['P_rect_30'] = P_rect_30

    # Compute the rectified extrinsics from cam0 to camN
    T1 = np.eye(4)
    T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
    T2 = np.eye(4)
    T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
    T3 = np.eye(4)
    T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

    # Compute the velodyne to rectified camera coordinate transforms
    data['T_cam0_velo'] = np.reshape(filedata['Tr'], (3, 4))
    data['T_cam0_velo'] = np.vstack([data['T_cam0_velo'], [0, 0, 0, 1]])
    data['T_cam1_velo'] = T1.dot(data['T_cam0_velo'])
    data['T_cam2_velo'] = T2.dot(data['T_cam0_velo'])
    data['T_cam3_velo'] = T3.dot(data['T_cam0_velo'])

    # Compute the camera intrinsics
    data['K_cam0'] = P_rect_00[0:3, 0:3]
    data['K_cam1'] = P_rect_10[0:3, 0:3]
    data['K_cam2'] = P_rect_20[0:3, 0:3]
    data['K_cam3'] = P_rect_30[0:3, 0:3]

    # Compute the stereo baselines in meters by projecting the origin of
    # each camera frame into the velodyne frame and computing the distances
    # between them
    p_cam = np.array([0, 0, 0, 1])
    p_velo0 = np.linalg.inv(data['T_cam0_velo']).dot(p_cam)
    p_velo1 = np.linalg.inv(data['T_cam1_velo']).dot(p_cam)
    p_velo2 = np.linalg.inv(data['T_cam2_velo']).dot(p_cam)
    p_velo3 = np.linalg.inv(data['T_cam3_velo']).dot(p_cam)

    data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
    data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)   # rgb baseline

    return data


def init_dict(cfg):    
    # calib dict
    caliblist = [_load_calib(cfg.odometry_home + cfg.data_subdir + str(i).zfill(2) + '/calib.txt') for i in range(1)]
    # calib_dict = {
    #     '00' : caliblist[0], '01' : caliblist[1], '02' : caliblist[2], '03' : caliblist[3], '04' : caliblist[4],
    #     '05' : caliblist[5], '06' : caliblist[6], '07' : caliblist[7], '08' : caliblist[8], '09' : caliblist[9],
    #     '10' : caliblist[10], '11' : caliblist[11], '12' : caliblist[12], '13' : caliblist[13], '14' : caliblist[14],
    #     '15' : caliblist[15], '16' : caliblist[16], '17' : caliblist[17], '18' : caliblist[18], '19' : caliblist[19],
    #     '20' : caliblist[20], '21' : caliblist[21]
    # }
    calib_dict = {
        '00' : caliblist[0]
    }

    # digital plane (u, v) to normalized coordinate (x, y, 1)
    left_normcoord_dict = {}
    right_normcoord_dict = {}
    for i in range(1):
        w, h = 1226, 370

        c, r = np.meshgrid(np.arange(w), np.arange(h))
        ones = np.ones((h, w), dtype=np.float32)
        points = np.stack([c, r, ones])
        points = points.reshape((3, -1))
        points = points.T

        k_cam3 = calib_dict[str(i).zfill(2)]['K_cam3']
        f_u = float(k_cam3[0, 0])
        f_v = float(k_cam3[1, 1])
        c_u = float(k_cam3[0, 2])
        c_v = float(k_cam3[1, 2])

        points[:, 0] = (points[:, 0] - c_u) / f_u
        points[:, 1] = (points[:, 1] - c_v) / f_v

        right_normcoord_dict[str(i).zfill(2)] = points

        k_cam2 = calib_dict[str(i).zfill(2)]['K_cam2']
        f_u = float(k_cam2[0, 0])
        f_v = float(k_cam2[1, 1])
        c_u = float(k_cam2[0, 2])
        c_v = float(k_cam2[1, 2])

        c, r = np.meshgrid(np.arange(w), np.arange(h))
        ones = np.ones((h, w), dtype=np.float32)
        points = np.stack([c, r, ones])
        points = points.reshape((3, -1))
        points = points.T

        points[:, 0] = (points[:, 0] - c_u) / f_u
        points[:, 1] = (points[:, 1] - c_v) / f_v

        left_normcoord_dict[str(i).zfill(2)] = points
        continue

    return calib_dict, left_normcoord_dict, right_normcoord_dict