
import os
import random
import cv2
import time
import sys

import torch
import torch.nn as nn
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from CFG.maincfg import cfg
from src.fen import FeatureExtractionNetwork_dualpath
from src.can import CrossAttentionNet_base

# from CFG.calibnet_cfg import cfg
# from src.calibnet import get_network, CalibNet_base

# from CFG.lccnet_cfg import cfg
# from src.lccnet import get_network, FlowNet_base

# from CFG.zhang_cfg import cfg
# from src.zhang import Zhang_ResNet, Zhang_base

from src.kittifactory import DataFactory
from src.utils import *
from src.PSMNet.stackhourglass import PSMNet

number_of_frames = cfg.number_of_frames


def load_depthmap(k_cam2, T_cam2_velo, velopoints):
    c_u = k_cam2[0, 2]
    c_v = k_cam2[1, 2]
    f_u = k_cam2[0, 0]
    f_v = k_cam2[1, 1]

    velopoints = rot_and_trs_points(velopoints, T_cam2_velo)

    u = f_u * velopoints[:, 0] / velopoints[:, 2] + c_u
    v = f_v * velopoints[:, 1] / velopoints[:, 2] + c_v
    z = velopoints[:, 2]

    mask = (u >= 0) * (u < 1226) * (v >= 0) * (v < 370) * (z > 0) * (z < 80)
    u = u[mask]
    v = v[mask]
    z = z[mask]
    proj_points = np.zeros((u.shape[0], 3), dtype=np.float32)
    proj_points[:, 0] = u
    proj_points[:, 1] = v
    proj_points[:, 2] = z
    
    u_ = np.clip(u.astype(np.int32), 0, 1226-1)
    v_ = np.clip(v.astype(np.int32), 0, 370-1)
    #z = cv2.applyColorMap(z.astype(np.uint8), cv2.COLORMAP_TURBO)
        
    depthmap = np.zeros((370, 1226), dtype=np.float32)
    depthmap_mask = np.zeros((370, 1226), dtype=np.bool_)
    for i in range(u.shape[0]):
        depthmap[v_[i], u_[i]] = z[i]
        depthmap_mask[v_[i], u_[i]] = True
    #     c1 = int(z[i, 0, 0])
    #     c2 = int(z[i, 0, 1])
    #     c3 = int(z[i, 0, 2])
    #     cv2.circle(left_image, (u[i], v[i]), 1, (c1,c2,c3), -1)
        continue
                
    # cv2.imwrite(path + 'lidar_proj.png', left_image)
    return proj_points, depthmap, depthmap_mask


def val_kitti_temporal_filtering(cent_ids, back, Anet, log_folder):
    if os.path.exists(cfg.proj_home + 'results/' + log_folder + '/'):
        pass
    else:
        os.makedirs(cfg.proj_home + 'results/' + log_folder + '/')


    with open(cfg.proj_home + 'gendata/100miscalib-left.txt', 'r') as f:
        predefine_error_list_left = f.readlines()

    with open(cfg.proj_home + 'gendata/100miscalib-right.txt', 'r') as f:
        predefine_error_list_right = f.readlines()
        
    with open(cfg.proj_home + 'gendata/100number-kitti.txt', 'r') as f:
        predefine_frame_number = f.readlines()

    # define dataloader
    kitti_dataset = DataFactory(cfg, 'val')
    
    pred_median_left_save = np.zeros((cfg.test_num, 3), dtype=np.float32)
    pred_median_right_save = np.zeros((cfg.test_num, 3), dtype=np.float32)

    back.eval()
    Anet.eval()

    timelist = []
    startflag = True

    with torch.no_grad():
        for cnt_ in range(cfg.test_num):
            
            pred_rotx_left = []
            pred_roty_left = []
            pred_rotz_left = []
            pred_rotx_right = []
            pred_roty_right = []
            pred_rotz_right = []
            
            print(cnt_, end='\r')

            rotx_left, roty_left, rotz_left = predefine_error_list_left[cnt_].strip().split(' ')
            rotx_left = float(rotx_left)
            roty_left = float(roty_left) * (3.141592 / 180.0)
            rotz_left = float(rotz_left) * (3.141592 / 180.0)

            rotx_right, roty_right, rotz_right = predefine_error_list_right[cnt_].strip().split(' ')
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

                k_cam2 = kitti_dataset.calib_dict[seq]['K_cam2']
                k_cam3 = kitti_dataset.calib_dict[seq]['K_cam3']

                left_normalized_points = kitti_dataset.left_normcoord_dict[seq].copy()
                right_normalized_points = kitti_dataset.right_normcoord_dict[seq].copy()

                _, left_input_img = gen_error_on_gpu(left_image, k_cam2, left_normalized_points, cfg.factory_device, misRTmat_left)

                _, right_input_img = gen_error_on_gpu(right_image, k_cam3, right_normalized_points, cfg.factory_device, misRTmat_right)

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

                #torch.cuda.synchronize()
                #starttime = time.time()
                
                left_fea = back(left_input)
                right_fea = back(right_input)
                left_quaternion, right_quaternion = Anet(left_fea, right_fea, None, None)
                # torch.cuda.synchronize()
                # if startflag:
                #     startflag = False
                #     pass
                # else:
                #     timelist.append(time.time() - starttime)

                left_quaternion = left_quaternion.squeeze(dim=0).cpu().numpy()
                pred_rot_matrix_left = rotation_matrix_from_quaternion(left_quaternion)

                predRTmat_left = np.zeros((4, 4), dtype=np.float32)
                predRTmat_left[:3, :3] = pred_rot_matrix_left[:3,:3]
                predRTmat_left[0, 3] = 0
                predRTmat_left[1, 3] = 0
                predRTmat_left[2, 3] = 0
                predRTmat_left[3, 3] = 1

                pred_Rot_left = rotationMatrixToEulerAngles(predRTmat_left[:3, :3]) * (180.0 / 3.141592)
                
                pred_rotx_left.append(pred_Rot_left[0])
                pred_roty_left.append(pred_Rot_left[1])
                pred_rotz_left.append(pred_Rot_left[2])

                right_quaternion = right_quaternion.squeeze(dim=0).cpu().numpy()
                pred_rot_matrix_right = rotation_matrix_from_quaternion(right_quaternion)

                predRTmat_right = np.zeros((4, 4), dtype=np.float32)
                predRTmat_right[:3, :3] = pred_rot_matrix_right[:3,:3]
                predRTmat_right[0, 3] = 0
                predRTmat_right[1, 3] = 0
                predRTmat_right[2, 3] = 0
                predRTmat_right[3, 3] = 1

                pred_Rot_right = rotationMatrixToEulerAngles(predRTmat_right[:3, :3]) * (180.0 / 3.141592)

                pred_rotx_right.append(pred_Rot_right[0])
                pred_roty_right.append(pred_Rot_right[1])
                pred_rotz_right.append(pred_Rot_right[2])
                
                continue
            
            starttime = time.time()

            pred_median_rotx_left = np.median(np.array(pred_rotx_left))
            pred_median_roty_left = np.median(np.array(pred_roty_left))
            pred_median_rotz_left = np.median(np.array(pred_rotz_left))
            pred_median_rotx_right = np.median(np.array(pred_rotx_right))
            pred_median_roty_right = np.median(np.array(pred_roty_right))
            pred_median_rotz_right = np.median(np.array(pred_rotz_right))

            if startflag:
                startflag = False
                pass
            else:
                timelist.append(time.time() - starttime)

            #ax = (pred_median_rotx_left - pred_median_rotx_right) / 2
            pred_median_left_save[cnt_, 0] = pred_median_rotx_left
            pred_median_left_save[cnt_, 1] = pred_median_roty_left
            pred_median_left_save[cnt_, 2] = pred_median_rotz_left
            pred_median_right_save[cnt_, 0] = pred_median_rotx_right
            pred_median_right_save[cnt_, 1] = pred_median_roty_right
            pred_median_right_save[cnt_, 2] = pred_median_rotz_right
            continue
    np.save(cfg.proj_home + 'results/' + log_folder + '/pred_median_left.npy', pred_median_left_save)
    np.save(cfg.proj_home + 'results/' + log_folder + '/pred_median_right.npy', pred_median_right_save)
    print('mean time : ', sum(timelist) / len(timelist))
    print('\n end')


def eval_100miscalib(log_folder):
    error_angle_left = []
    error_angle_right = []

    error_psm_ori_err = []
    error_psm_input_err = []
    error_psm_gt_err = []
    error_psm_pred_err = []
    error_psm_gt_pred_err = []
    error_psm_ssim = []

    with open(cfg.proj_home + 'gendata/100miscalib-left.txt', 'r') as f:
        predefine_error_list_left = f.readlines()

    with open(cfg.proj_home + 'gendata/100miscalib-right.txt', 'r') as f:
        predefine_error_list_right = f.readlines()

    with open(cfg.proj_home + 'gendata/100number-kitti.txt', 'r') as f:
        predefine_frame_number = f.readlines()

    pred_median_left = np.load(cfg.proj_home + 'results/' + log_folder + '/pred_median_left.npy')
    pred_median_right = np.load(cfg.proj_home + 'results/' + log_folder + '/pred_median_right.npy')

    kitti_dataset = DataFactory(cfg, 'val')
    k_cam2 = kitti_dataset.calib_dict['00']['K_cam2']
    k_cam3 = kitti_dataset.calib_dict['00']['K_cam3']
    left_normalized_points = kitti_dataset.left_normcoord_dict['00'].copy()
    right_normalized_points = kitti_dataset.right_normcoord_dict['00'].copy()
    T_cam2_velo = kitti_dataset.calib_dict['00']['T_cam2_velo']
    baseline = 0.54

    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(**normal_mean_var)])    

    cent_ids = 'cuda:' + str(cfg.factory_device)

    psmnet_model = PSMNet(192)
    psmnet_model = nn.DataParallel(psmnet_model, device_ids=[cfg.factory_device])
    state_dict = torch.load(cfg.proj_home + 'checkpoint/pretrained_model_KITTI2012.tar', map_location=cent_ids)
    psmnet_model.load_state_dict(state_dict['state_dict'])
    psmnet_model.cuda(cent_ids)
    psmnet_model.eval()

    for number in range(0, cfg.test_num):
        print(number)
        # ready RTs
        rotx_left, roty_left, rotz_left = predefine_error_list_left[number].strip().split(' ')
        rotx_left = float(rotx_left)
        roty_left = float(roty_left) * (3.141592 / 180.0)
        rotz_left = float(rotz_left) * (3.141592 / 180.0)
        rotx_right, roty_right, rotz_right = predefine_error_list_right[number].strip().split(' ')
        rotx_right = float(rotx_right)
        roty_right = float(roty_right) * (3.141592 / 180.0)
        rotz_right = float(rotz_right) * (3.141592 / 180.0)
        rotx_mid = (rotx_left + rotx_right) / 2 * (np.pi / 180.0)
        rotx_left = rotx_left * (np.pi / 180.0)
        rotx_right = rotx_right * (np.pi / 180.0)
        predx_left, predy_left, predz_left = pred_median_left[number] * (np.pi / 180.0)
        predx_right, predy_right, predz_right = pred_median_right[number] * (np.pi / 180.0)
        RTs = mkRTmatrices(
            rotx_left, roty_left, rotz_left, rotx_right, roty_right, rotz_right, rotx_mid,
            predx_left, predy_left, predz_left, predx_right, predy_right, predz_right
        )
        misRTmat_left, misRTmat_right, gtRTmat_left, gtRTmat_right, predRTmat_left, predRTmat_right, error_rot_left, error_rot_right = RTs
        error_angle_left.append(error_rot_left)
        error_angle_right.append(error_rot_right)
        error_angle_left.append(error_rot_left)
        error_angle_right.append(error_rot_right)

        # ready image
        imgnum = predefine_frame_number[number].strip()
        left_path = cfg.odometry_home + cfg.data_subdir + '00' + '/image_2/' + imgnum.zfill(6) + '.png'
        right_path = cfg.odometry_home + cfg.data_subdir + '00' + '/image_3/' + imgnum.zfill(6) + '.png'
        left_image, right_image = cv2.imread(left_path), cv2.imread(right_path)
        left_image = left_image[:370, :1226, :]
        right_image = right_image[:370, :1226, :]
        left_input_img, right_input_img, left_gt_img, right_gt_img, left_pred_img, right_pred_img = mkRotImages(
            left_image, right_image, k_cam2, k_cam3, left_normalized_points, right_normalized_points, cfg.factory_device,
            misRTmat_left, misRTmat_right, gtRTmat_left, gtRTmat_right, predRTmat_left, predRTmat_right, path=None
        )

        # ready LiDAR proj img
        velo_path = cfg.odometry_home + 'data_odometry_velodyne/dataset/sequences/00/velodyne/' + imgnum.zfill(6) + '.bin'
        velopoints = np.fromfile(velo_path, dtype=np.float32).reshape((-1, 4))[:, :3]
        proj_points, depthmap, depthmap_mask = load_depthmap(k_cam2, T_cam2_velo, velopoints)

        # start
        out =  mkPSMdispmap(
            left_image, right_image, left_input_img, right_input_img, left_gt_img, right_gt_img, left_pred_img, right_pred_img,
            psmnet_model, infer_transform, cent_ids, cfg.inputw, cfg.inputh, cfg.cutimg, k_cam2, path=None, proj_points=proj_points, depth_img=depthmap, depth_mask=depthmap_mask)
        ori_depth_err, input_depth_err, gt_depth_err, pred_depth_err, gt_pred_depth_err, ssim_dif = out
        error_psm_ori_err.append(ori_depth_err)
        error_psm_input_err.append(input_depth_err)
        error_psm_gt_err.append(gt_depth_err)
        error_psm_pred_err.append(pred_depth_err)
        error_psm_gt_pred_err.append(gt_pred_depth_err)
        error_psm_ssim.append(ssim_dif)

        # end

        continue
    
    error_angle_left = np.array(error_angle_left)
    print('left : ', np.mean(error_angle_left))
    error_angle_right = np.array(error_angle_right)
    print('right : ', np.mean(error_angle_right))

    psm_ori_err_mean = []
    psm_ori_err_std = []
    psm_input_err_mean = []
    psm_input_err_std = []
    psm_gt_err_mean = []
    psm_gt_err_std = []
    psm_pred_err_mean = []
    psm_pred_err_std = []
    psm_gt_pred_err_mean = []
    psm_gt_pred_err_std = []

    for i in range(cfg.test_num):
        psm_ori_err_mean.append(error_psm_ori_err[i].mean())
        psm_ori_err_std.append(error_psm_ori_err[i].std())
        psm_input_err_mean.append(error_psm_input_err[i].mean())
        psm_input_err_std.append(error_psm_input_err[i].std())
        psm_gt_err_mean.append(error_psm_gt_err[i].mean())
        psm_gt_err_std.append(error_psm_gt_err[i].std())
        psm_pred_err_mean.append(error_psm_pred_err[i].mean())
        psm_pred_err_std.append(error_psm_pred_err[i].std())
        psm_gt_pred_err_mean.append(error_psm_gt_pred_err[i].mean())
        psm_gt_pred_err_std.append(error_psm_gt_pred_err[i].std())
        continue

    psm_ori_err_mean = np.array(psm_ori_err_mean)
    psm_ori_err_std = np.array(psm_ori_err_std)
    psm_input_err_mean = np.array(psm_input_err_mean)
    psm_input_err_std = np.array(psm_input_err_std)
    psm_gt_err_mean = np.array(psm_gt_err_mean)
    psm_gt_err_std = np.array(psm_gt_err_std)
    psm_pred_err_mean = np.array(psm_pred_err_mean)
    psm_pred_err_std = np.array(psm_pred_err_std)
    psm_gt_pred_err_mean = np.array(psm_gt_pred_err_mean)
    psm_gt_pred_err_std = np.array(psm_gt_pred_err_std)

    print('psm ori mean : ', np.mean(psm_ori_err_mean), ' std : ',  np.mean(psm_ori_err_std))
    print('psm input mean : ', np.mean(psm_input_err_mean), ' std : ',  np.mean(psm_input_err_std))
    print('psm gt mean : ', np.mean(psm_gt_err_mean), ' std : ',  np.mean(psm_gt_err_std))
    print('psm pred mean : ', np.mean(psm_pred_err_mean), ' std : ',  np.mean(psm_pred_err_std))
    print('psm gt-pred mean : ', np.mean(psm_gt_pred_err_mean), ' std : ',  np.mean(psm_gt_pred_err_std))
    print('psm ssim : ', np.mean(error_psm_ssim))

    error_psm_ori_err = np.concatenate(error_psm_ori_err)
    error_psm_input_err = np.concatenate(error_psm_input_err)
    error_psm_gt_err = np.concatenate(error_psm_gt_err)
    error_psm_pred_err = np.concatenate(error_psm_pred_err)
    error_psm_gt_pred_err = np.concatenate(error_psm_gt_pred_err)

    np.save(cfg.proj_home + 'results/' + log_folder + '/error_psm_ori_err.npy', error_psm_ori_err)
    np.save(cfg.proj_home + 'results/' + log_folder + '/error_psm_input_err.npy', error_psm_input_err)
    np.save(cfg.proj_home + 'results/' + log_folder + '/error_psm_gt_err.npy', error_psm_gt_err)
    np.save(cfg.proj_home + 'results/' + log_folder + '/error_psm_pred_err.npy', error_psm_pred_err)
    np.save(cfg.proj_home + 'results/' + log_folder + '/error_psm_gt_pred_err.npy', error_psm_gt_pred_err)
    # end


if __name__ == '__main__':
    torch.manual_seed(677)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(677)
    np.random.seed(677)
    torch.multiprocessing.set_start_method('spawn')
    cent_ids = 'cuda:' + str(cfg.devices[0])

    back = FeatureExtractionNetwork_dualpath().to(cent_ids)
    Anet = CrossAttentionNet_base(cfg.volume_in_channels, cfg.qkv_channels).to(cent_ids)
    load_model(back, cfg.proj_home + 'checkpoint/ours-back.pth', cent_ids)
    load_model(Anet, cfg.proj_home + 'checkpoint/ours-Anet.pth', cent_ids)
    log_folder = cfg.model

    # back = get_network(18).to(cent_ids)
    # Anet = CalibNet_base().to(cent_ids)
    # load_model(back, cfg.proj_home + 'checkpoint/calibnet-back.pth', cent_ids)
    # load_model(Anet, cfg.proj_home + 'checkpoint/calibnet-Anet.pth', cent_ids)
    # log_folder = cfg.model

    # back = get_network(18).to(cent_ids)
    # Anet = FlowNet_base().to(cent_ids)
    # load_model(back, cfg.proj_home + 'checkpoint/lccnet-back.pth', cent_ids)
    # load_model(Anet, cfg.proj_home + 'checkpoint/lccnet-Anet.pth', cent_ids)
    # log_folder = cfg.model

    # back = Zhang_ResNet().to(cent_ids)
    # Anet = Zhang_base().to(cent_ids)
    # load_model(back, cfg.proj_home + 'checkpoint/zhang-back.pth', cent_ids)
    # load_model(Anet, cfg.proj_home + 'checkpoint/zhang-Anet.pth', cent_ids)
    # log_folder = cfg.model

    val_kitti_temporal_filtering(cent_ids, back, Anet, log_folder)
    eval_100miscalib(log_folder)
