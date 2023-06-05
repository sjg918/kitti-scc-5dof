
import numpy as np
import numba
import math
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

#refer : https://learnopencv.com/rotation-matrix-to-euler-angles/

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])


#refer : https://github.com/IIPCVLAB/LCCNet
def quaternion_from_rotation_matrix(matrix):
    if matrix.shape == (4, 4):
        R = matrix[:3, :3]
    elif matrix.shape == (3, 3):
        R = matrix
    else:
        raise TypeError("Not a valid rotation matrix")
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    q = np.zeros(4, dtype=np.float32)
    if tr > 0.:
        S = np.sqrt(tr+1.0) * 2
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q[0] = (R[2, 1] - R[1, 2]) / S
        q[1] = 0.25 * S
        q[2] = (R[0, 1] + R[1, 0]) / S
        q[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q[0] = (R[0, 2] - R[2, 0]) / S
        q[1] = (R[0, 1] + R[1, 0]) / S
        q[2] = 0.25 * S
        q[3] = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q[0] = (R[1, 0] - R[0, 1]) / S
        q[1] = (R[0, 2] + R[2, 0]) / S
        q[2] = (R[1, 2] + R[2, 1]) / S
        q[3] = 0.25 * S
    return q / np.linalg.norm(q)


def rotation_matrix_from_quaternion(q):
    mat = np.zeros((3,3), dtype=np.float32)

    mat[0, 0] = 1 - 2*q[2]**2 - 2*q[3]**2
    mat[0, 1] = 2*q[1]*q[2] - 2*q[3]*q[0]
    mat[0, 2] = 2*q[1]*q[3] + 2*q[2]*q[0]
    mat[1, 0] = 2*q[1]*q[2] + 2*q[3]*q[0]
    mat[1, 1] = 1 - 2*q[1]**2 - 2*q[3]**2
    mat[1, 2] = 2*q[2]*q[3] - 2*q[1]*q[0]
    mat[2, 0] = 2*q[1]*q[3] - 2*q[2]*q[0]
    mat[2, 1] = 2*q[2]*q[3] + 2*q[1]*q[0]
    mat[2, 2] = 1 - 2*q[1]**2 - 2*q[2]**2
    return mat


def rot_and_trs_points_cuda(points ,R):
    o = torch.ones((points.shape[0], 1), dtype=torch.float32, device=points.device)
    points = torch.cat((points, o), dim=-1)
    newpoints = R @ points.T
    #rot_and_trs_points_kernel(points.astype(np.float32), newpoints, R.astype(np.float32), points.shape[0])
    newpoints = (newpoints.T)[:, :3]
    return newpoints


def load_model(m, p, device):
    dict = torch.load(p, map_location=device)
    for i, k in zip(m.state_dict(), dict):
        weight = dict[k]
        m.state_dict()[i].copy_(weight)


def gen_error_on_gpu(tgtimg, K, normalized_points, cfg_device, miscalibRT):
    inputh = 370
    inputw = 1226
    
    normalized_points_cuda = torch.from_numpy(normalized_points).to(dtype=torch.float32).to(device=cfg_device)
    miscalibRT_cuda = torch.from_numpy(np.linalg.inv(miscalibRT)).to(dtype=torch.float32).to(device=cfg_device)
    tgtimg_cuda = torch.from_numpy(tgtimg).to(dtype=torch.float32, device=cfg_device).permute(2, 0, 1).contiguous().unsqueeze(0)

    normalized_points_cuda = rot_and_trs_points_cuda(normalized_points_cuda, miscalibRT_cuda)

    f_u = float(K[0, 0])
    f_v = float(K[1, 1])
    c_u = float(K[0, 2])
    c_v = float(K[1, 2])

    normalized_points_cuda[:, 0] = normalized_points_cuda[:, 0] * f_u / normalized_points_cuda[:, 2] + c_u
    normalized_points_cuda[:, 1] = normalized_points_cuda[:, 1] * f_v / normalized_points_cuda[:, 2] + c_v

    # corner1 = 0
    # corner2 = inputw - 1
    # corner3 = inputh * inputw - inputw
    # corner4 = inputh * inputw - 1
    # corner1x, corner1y = normalized_points_cuda[corner1, 0].item(), normalized_points_cuda[corner1, 1].item()
    # corner2x, corner2y = normalized_points_cuda[corner2, 0].item(), normalized_points_cuda[corner2, 1].item()
    # corner3x, corner3y = normalized_points_cuda[corner3, 0].item(), normalized_points_cuda[corner3, 1].item()
    # corner4x, corner4y = normalized_points_cuda[corner4, 0].item(), normalized_points_cuda[corner4, 1].item()
    # cornerlist = [corner1x, corner1y, corner2x, corner2y, corner3x, corner3y, corner4x, corner4y]

    #mask = (normalized_points_cuda[:, 0] > 0) * (normalized_points_cuda[:, 0] < inputw-1) * (normalized_points_cuda[:, 1] > 0) * (normalized_points_cuda[:, 1] < inputh-1)
    
    normalized_points_cuda[:, 0] = (normalized_points_cuda[:, 0] / inputw) * 2 - 1
    normalized_points_cuda[:, 1] = (normalized_points_cuda[:, 1] / inputh) * 2 - 1
    input_image = torch.zeros((3, inputh, inputw), dtype=torch.float32, device=cfg_device)
    grid = normalized_points_cuda[:, :2].view(inputh, inputw, 2).unsqueeze(0)
    input_image = F.grid_sample(tgtimg_cuda, grid, align_corners=True)

    return miscalibRT, input_image.squeeze(0).permute(1, 2, 0).contiguous().to(device='cpu')#, cornerlist


def mkRTmatrices(
    rotx_left, roty_left, rotz_left, rotx_right, roty_right, rotz_right, rotx_mid,
    predx_left, predy_left, predz_left, predx_right, predy_right, predz_right
    ):
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

    rotmat_left = eulerAnglesToRotationMatrix([rotx_left - rotx_mid, roty_left, rotz_left])
    gtRTmat_left = np.zeros((4, 4), dtype=np.float32)
    gtRTmat_left[:3, :3] = rotmat_left[:3,:3]
    gtRTmat_left[0, 3] = 0
    gtRTmat_left[1, 3] = 0
    gtRTmat_left[2, 3] = 0
    gtRTmat_left[3, 3] = 1
    gtRTmat_left = np.linalg.inv(gtRTmat_left)

    rotmat_right = eulerAnglesToRotationMatrix([rotx_right - rotx_mid, roty_right, rotz_right])
    gtRTmat_right = np.zeros((4, 4), dtype=np.float32)
    gtRTmat_right[:3, :3] = rotmat_right[:3,:3]
    gtRTmat_right[0, 3] = 0
    gtRTmat_right[1, 3] = 0
    gtRTmat_right[2, 3] = 0
    gtRTmat_right[3, 3] = 1
    gtRTmat_right = np.linalg.inv(gtRTmat_right)

    predrotmat_left = eulerAnglesToRotationMatrix([predx_left, predy_left, predz_left])
    predRTmat_left = np.zeros((4, 4), dtype=np.float32)
    predRTmat_left[:3, :3] = predrotmat_left[:3,:3]
    predRTmat_left[0, 3] = 0
    predRTmat_left[1, 3] = 0
    predRTmat_left[2, 3] = 0
    predRTmat_left[3, 3] = 1

    predrotmat_right = eulerAnglesToRotationMatrix([predx_right, predy_right, predz_right])
    predRTmat_right = np.zeros((4, 4), dtype=np.float32)
    predRTmat_right[:3, :3] = predrotmat_right[:3,:3]
    predRTmat_right[0, 3] = 0
    predRTmat_right[1, 3] = 0
    predRTmat_right[2, 3] = 0
    predRTmat_right[3, 3] = 1
  
    # cal angle err.
    error_MAT_left = predRTmat_left @ np.linalg.inv(gtRTmat_left)
    error_ROT_left = rotationMatrixToEulerAngles(error_MAT_left[:3, :3]) * (180.0 / np.pi)
    error_rot_left = np.mean(np.abs(error_ROT_left))

    error_MAT_right = predRTmat_right @ np.linalg.inv(gtRTmat_right)
    error_ROT_right = rotationMatrixToEulerAngles(error_MAT_right[:3, :3]) * (180.0 / np.pi)
    error_rot_right = np.mean(np.abs(error_ROT_right))

    return misRTmat_left, misRTmat_right, gtRTmat_left, gtRTmat_right, predRTmat_left, predRTmat_right, error_rot_left, error_rot_right


def mkRotImages(
        left, right, k2, k3, left_normalized_points, right_normalized_points, device_ids,
        misRTmat_left, misRTmat_right, gtRTmat_left, gtRTmat_right, predRTmat_left, predRTmat_right, path=None
    ):
    _, left_input_img = gen_error_on_gpu(left, k2, left_normalized_points, device_ids, misRTmat_left)
    left_input_img = left_input_img.to(torch.uint8).numpy()

    _, right_input_img = gen_error_on_gpu(right, k3, right_normalized_points, device_ids, misRTmat_right)
    right_input_img = right_input_img.to(torch.uint8).numpy()

    _, left_gt_img = gen_error_on_gpu(left_input_img, k2, left_normalized_points, device_ids, gtRTmat_left)
    left_gt_img = left_gt_img.to(torch.uint8).numpy()

    _, right_gt_img = gen_error_on_gpu(right_input_img, k3, right_normalized_points, device_ids, gtRTmat_right)
    right_gt_img = right_gt_img.to(torch.uint8).numpy()

    _, left_pred_img = gen_error_on_gpu(left_input_img, k2, left_normalized_points, device_ids, predRTmat_left)
    left_pred_img = left_pred_img.to(torch.uint8).numpy()

    _, right_pred_img = gen_error_on_gpu(right_input_img, k3, right_normalized_points, device_ids, predRTmat_right)
    right_pred_img = right_pred_img.to(torch.uint8).numpy()

    if path is not None:
        cv2.imwrite(path + 'L_ori.png', left)
        cv2.imwrite(path + 'R_ori.png', right)
        cv2.imwrite(path + 'L_input.png', left_input_img)
        cv2.imwrite(path + 'R_input.png', right_input_img)
        cv2.imwrite(path + 'L_gt.png', left_gt_img)
        cv2.imwrite(path + 'R_gt.png', right_gt_img)
        cv2.imwrite(path + 'L_pred.png', left_pred_img)
        cv2.imwrite(path + 'R_pred.png', right_pred_img)
    return left_input_img, right_input_img, left_gt_img, right_gt_img, left_pred_img, right_pred_img


@numba.jit(nopython=True)
def interpolate_dif(proj_points, depth_map, inputw, inputh, cutimg, N):
    dif = np.zeros((N), dtype=np.float32)
    for i in range(N):
        u = proj_points[i, 0]
        v = proj_points[i, 1]
        z = proj_points[i, 2]

        if (u < cutimg) or (u >= cutimg + inputw - 1) or (v < cutimg) or (v >= cutimg + inputh - 1):
            continue

        u = u - cutimg
        v = v - cutimg

        u1 = int(u)
        u2 = u1 + 1
        v1 = int(v)
        v2 = v1 + 1

        w1 = (u2 - u) * (v2 - v)
        w2 = (1 - u2 + u) * (v2 - v)
        w3 = (u2 - u) * (1 - v2 + v)
        w4 = (1 - u2 + u) * (1 - v2 + v)

        d1 = depth_map[v1, u1] * w1
        d2 = depth_map[v1, u2] * w2
        d3 = depth_map[v2, u1] * w3
        d4 = depth_map[v2, u2] * w4

        dif[i] = np.absolute(z - (d1 + d2 + d3 + d4))
        continue

    return dif



def mkPSMdispmap(
    left, right, left_input, right_input, left_gt, right_gt, left_pred, right_pred,
    psmnet_model, infer_transform, divice_ids, inputw, inputh, cutimg, k_cam2, path=None, proj_points=None, depth_img=None, depth_mask=None):
    left_PIL_img = Image.fromarray(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
    right_PIL_img = Image.fromarray(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))
    left_PIL_img = infer_transform(left_PIL_img)
    right_PIL_img = infer_transform(right_PIL_img) 
    left_input_PIL_img = Image.fromarray(cv2.cvtColor(left_input.astype(np.uint8), cv2.COLOR_BGR2RGB))
    right_input_PIL_img = Image.fromarray(cv2.cvtColor(right_input.astype(np.uint8), cv2.COLOR_BGR2RGB))
    left_input_PIL_img = infer_transform(left_input_PIL_img)
    right_input_PIL_img = infer_transform(right_input_PIL_img)
    left_gt_PIL_img = Image.fromarray(cv2.cvtColor(left_gt.astype(np.uint8), cv2.COLOR_BGR2RGB))
    right_gt_PIL_img = Image.fromarray(cv2.cvtColor(right_gt.astype(np.uint8), cv2.COLOR_BGR2RGB))
    left_gt_PIL_img = infer_transform(left_gt_PIL_img)
    right_gt_PIL_img = infer_transform(right_gt_PIL_img) 
    left_pred_PIL_img = Image.fromarray(cv2.cvtColor(left_pred.astype(np.uint8), cv2.COLOR_BGR2RGB))
    right_pred_PIL_img = Image.fromarray(cv2.cvtColor(right_pred.astype(np.uint8), cv2.COLOR_BGR2RGB))
    left_pred_PIL_img = infer_transform(left_pred_PIL_img)
    right_pred_PIL_img = infer_transform(right_pred_PIL_img) 

    # pad to width and hight to 16 times
    if left_PIL_img.shape[1] % 16 != 0:
        times = left_PIL_img.shape[1]//16       
        top_pad = (times+1)*16 -left_PIL_img.shape[1]
    else:
        top_pad = 0

    if left_PIL_img.shape[2] % 16 != 0:
        times = left_PIL_img.shape[2]//16                       
        right_pad = (times+1)*16-left_PIL_img.shape[2]
    else:
        right_pad = 0    

    left_PIL_img = F.pad(left_PIL_img,(0,right_pad, top_pad,0)).unsqueeze(0)
    right_PIL_img = F.pad(right_PIL_img,(0,right_pad, top_pad,0)).unsqueeze(0)
    left_input_PIL_img = F.pad(left_input_PIL_img,(0,right_pad, top_pad,0)).unsqueeze(0)
    right_input_PIL_img = F.pad(right_input_PIL_img,(0,right_pad, top_pad,0)).unsqueeze(0)
    left_gt_PIL_img = F.pad(left_gt_PIL_img,(0,right_pad, top_pad,0)).unsqueeze(0)
    right_gt_PIL_img = F.pad(right_gt_PIL_img,(0,right_pad, top_pad,0)).unsqueeze(0)
    left_pred_PIL_img = F.pad(left_pred_PIL_img,(0,right_pad, top_pad,0)).unsqueeze(0)
    right_pred_PIL_img = F.pad(right_pred_PIL_img,(0,right_pad, top_pad,0)).unsqueeze(0)

    left_PIL_img = left_PIL_img.cuda(divice_ids)
    right_PIL_img = right_PIL_img.cuda(divice_ids)
    left_input_PIL_img = left_input_PIL_img.cuda(divice_ids)
    right_input_PIL_img = right_input_PIL_img.cuda(divice_ids)
    left_gt_PIL_img = left_gt_PIL_img.cuda(divice_ids)
    right_gt_PIL_img = right_gt_PIL_img.cuda(divice_ids)
    left_pred_PIL_img = left_pred_PIL_img.cuda(divice_ids)
    right_pred_PIL_img = right_pred_PIL_img.cuda(divice_ids)

    with torch.no_grad():
        disp_ori = psmnet_model(left_PIL_img,right_PIL_img)
        disp_mis = psmnet_model(left_input_PIL_img,right_input_PIL_img)
        disp_gt = psmnet_model(left_gt_PIL_img,right_gt_PIL_img)
        disp_pred = psmnet_model(left_pred_PIL_img,right_pred_PIL_img)

    disp_ori = torch.squeeze(disp_ori)
    disp_gt = torch.squeeze(disp_gt)
    disp_mis = torch.squeeze(disp_mis)
    disp_pred = torch.squeeze(disp_pred)
    disp_ori = disp_ori.data.cpu().numpy()
    disp_gt = disp_gt.data.cpu().numpy()
    disp_mis = disp_mis.data.cpu().numpy()
    disp_pred = disp_pred.data.cpu().numpy()
    
    if top_pad !=0 and right_pad != 0:
        disp_ori = disp_ori[top_pad:,:-right_pad]
        disp_pred = disp_pred[top_pad:,:-right_pad]
        disp_mis = disp_mis[top_pad:,:-right_pad]
        disp_gt = disp_gt[top_pad:,:-right_pad]
    elif top_pad ==0 and right_pad != 0:
        disp_ori = disp_ori[:,:-right_pad]
        disp_pred = disp_pred[:,:-right_pad]
        disp_mis = disp_mis[:,:-right_pad]
        disp_gt = disp_gt[:,:-right_pad]
    elif top_pad !=0 and right_pad == 0:
        disp_ori = disp_ori[top_pad:,:]
        disp_pred = disp_pred[top_pad:,:]
        disp_mis = disp_mis[top_pad:,:]
        disp_gt = disp_gt[top_pad:,:]
    else:
        pass

    disp_ori = disp_ori[cutimg:cutimg+inputh, cutimg:cutimg+inputw]
    disp_mis = disp_mis[cutimg:cutimg+inputh, cutimg:cutimg+inputw]
    disp_gt = disp_gt[cutimg:cutimg+inputh, cutimg:cutimg+inputw]
    disp_pred = disp_pred[cutimg:cutimg+inputh, cutimg:cutimg+inputw]

    if path is not None:
        ori_disp_img = (disp_ori.astype(np.float32) / 192 * 255).astype(np.uint8)
        mis_disp_img = (disp_mis.astype(np.float32) / 192 * 255).astype(np.uint8)
        gt_disp_img = (disp_gt.astype(np.float32) / 192 * 255).astype(np.uint8)
        pred_disp_img = (disp_pred.astype(np.float32) / 192 * 255).astype(np.uint8)

        ori_disp_img = cv2.applyColorMap(ori_disp_img, cv2.COLORMAP_JET)
        mis_disp_img = cv2.applyColorMap(mis_disp_img, cv2.COLORMAP_JET)
        gt_disp_img = cv2.applyColorMap(gt_disp_img, cv2.COLORMAP_JET)
        pred_disp_img = cv2.applyColorMap(pred_disp_img, cv2.COLORMAP_JET)

        cv2.imwrite(path + 'ori_psm.png', ori_disp_img)
        cv2.imwrite(path + 'mis_psm.png', mis_disp_img)
        cv2.imwrite(path + 'gt_psm.png', gt_disp_img)
        cv2.imwrite(path + 'pred_psm.png', pred_disp_img)

    if depth_img is not None:
        depth_img = depth_img[cutimg:cutimg+inputh, cutimg:cutimg+inputw]
        depth_mask = depth_mask[cutimg:cutimg+inputh, cutimg:cutimg+inputw]
        f_u = k_cam2[0, 0]
        depth_left = f_u * 0.54 / disp_ori
        depth_left_input = f_u * 0.54 / disp_mis
        depth_left_gt = f_u * 0.54 / disp_gt
        depth_left_pred = f_u * 0.54 / disp_pred

        if path is not None:
            depth_mask_ = np.logical_not(depth_mask)
            depth_img_ = depth_img.copy()
            depth_img_[depth_mask_] = 0
            depth_mask_ = np.expand_dims(depth_mask_, 2).repeat(3).reshape(inputh, inputw, 3)

            ori_dif = np.absolute(depth_img_ - depth_left).astype(np.uint8)
            ori_dif = (ori_dif / 80 * 255).astype(np.uint8)
            ori_dif = cv2.applyColorMap(ori_dif, cv2.COLORMAP_JET)
            ori_dif[depth_mask_] = 0
            cv2.imwrite(path + 'psm_oridif.png', ori_dif)

            input_dif = np.absolute(depth_img_ - depth_left_input)
            input_dif = (input_dif / 80 * 255).astype(np.uint8)
            input_dif = cv2.applyColorMap(input_dif, cv2.COLORMAP_JET)
            input_dif[depth_mask_] = 0
            cv2.imwrite(path + 'psm_inputdif.png', input_dif)

            gt_dif = np.absolute(depth_img_ - depth_left_gt)
            gt_dif = (gt_dif / 80 * 255).astype(np.uint8)
            gt_dif = cv2.applyColorMap(gt_dif, cv2.COLORMAP_JET)
            gt_dif[depth_mask_] = 0
            cv2.imwrite(path + 'psm_gtdif.png', gt_dif)

            pred_dif = np.absolute(depth_img_ - depth_left_pred)
            pred_dif = (pred_dif / 80 * 255).astype(np.uint8)
            pred_dif = cv2.applyColorMap(pred_dif, cv2.COLORMAP_JET)
            pred_dif[depth_mask_] = 0
            cv2.imwrite(path + 'psm_preddif.png', pred_dif)

            gtpred_dif = np.absolute(depth_left_gt - depth_left_pred)
            gtpred_dif = (gtpred_dif / 80 * 255).astype(np.uint8)
            gtpred_dif = cv2.applyColorMap(gtpred_dif, cv2.COLORMAP_JET)
            gtpred_dif[depth_mask_] = 0
            cv2.imwrite(path + 'psm_gtpreddif.png', gtpred_dif)

            depth_img_ = (depth_img_ / 80 * 255).astype(np.uint8)
            depth_img_ = cv2.applyColorMap(depth_img_, cv2.COLORMAP_JET)
            depth_img_[depth_mask_] = 0
            cv2.imwrite(path + 'LiDAR_gt.png', depth_img_)

        num_proj_points = proj_points.shape[0]
        ori_depth_err = interpolate_dif(proj_points, depth_left.astype(dtype=np.float32), inputw, inputh, cutimg, num_proj_points)
        input_depth_err = interpolate_dif(proj_points, depth_left_input.astype(dtype=np.float32), inputw, inputh, cutimg, num_proj_points)
        gt_depth_err = interpolate_dif(proj_points, depth_left_gt.astype(dtype=np.float32), inputw, inputh, cutimg, num_proj_points)
        pred_depth_err = interpolate_dif(proj_points, depth_left_pred.astype(dtype=np.float32), inputw, inputh, cutimg, num_proj_points)

        gt_pred_depth_err = np.absolute(disp_gt.astype(dtype=np.float32) - disp_pred.astype(dtype=np.float32))
        ssim_dif = ssim(disp_gt, disp_pred)

        return ori_depth_err, input_depth_err, gt_depth_err, pred_depth_err, gt_pred_depth_err, ssim_dif
    
    # error_3px = disp_err / gt_disp_img.to(dtype=torch.float32)
    # error_3px = error_3px >= 0.05
    # error_3px = error_3px.sum() / error_3px.numel()
    return disp_err.mean(), ssim_dif



# refer : https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python
def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


