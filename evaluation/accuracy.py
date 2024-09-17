"""
@date: 2021/8/4
@description:
"""
import numpy as np
import cv2
import scipy

from evaluation.f1_score import f1_score_2d
from loss import GradLoss
from utils.boundary import corners2boundaries, layout2depth
from utils.conversion import depth2xyz, uv2xyz, get_u, depth2uv, xyz2uv, uv2pixel
from utils.height import calc_ceil_ratio
from evaluation.iou import calc_IoU, calc_Iou_height
from visualization.boundary import draw_boundaries
from visualization.floorplan import draw_iou_floorplan
from visualization.grad import show_grad

import torch
import matplotlib.pyplot as plt


def calc_accuracy(dt, gt, visualization=False, h=512, second_type=False, gt_label='origin', opening=False, branch_exchange=False):
    visb_iou_2ds = []
    visb_iou_3ds = []
    full_iou_2ds = []
    full_iou_3ds = []
    iou_heights = []

    visb_iou_floodplans = []
    full_iou_floodplans = []
    pano_bds = []
    opening_bds = []

    # if 'depth' not in dt.keys():
    #     dt['depth'] = gt['depth']

    # for both MP3D and ZInD-exchanged
    if gt_label == 'new':
        gt_depth = gt['new_depth']
        gt_corners = gt['new_corners']
    else:
        gt_depth = gt['depth']
        gt_corners = gt['corners']
    
    if second_type is True:
        dt_depth = dt['new_depth']
        new_dt_depth = dt['depth']  # for ambiguity detection to show two types of boundaries at the same time
    else:    
        dt_depth = dt['depth']
        new_dt_depth = dt['new_depth']


    for i in range(len(gt_depth)):
        # print(i)
        dt_xyz = dt['processed_xyz'][i] if 'processed_xyz' in dt else depth2xyz(np.abs(dt_depth[i]))

        visb_gt_xyz = depth2xyz(np.abs(gt_depth[i]))
        corners = gt_corners[i]
        full_gt_corners = corners[corners[..., 0] + corners[..., 1] != 0]  # Take effective corners
        full_gt_xyz = uv2xyz(full_gt_corners)


        dt_xz = dt_xyz[..., ::2]

        visb_gt_xz = visb_gt_xyz[..., ::2]
        full_gt_xz = full_gt_xyz[..., ::2]

        gt_ratio = gt['ratio'][i][0]
        if 'ratio' not in dt.keys():
            if 'boundary' in dt.keys():
                w = len(dt['boundary'][i])
                boundary = np.clip(dt['boundary'][i], 0.0001, 0.4999)
                depth = np.clip(dt_depth[i], 0.001, 9999)
                dt_ceil_boundary = np.concatenate([get_u(w, is_np=True)[..., None], boundary], axis=-1)
                dt_floor_boundary = depth2uv(depth)
                dt_ratio = calc_ceil_ratio(boundaries=[dt_ceil_boundary, dt_floor_boundary])
            else:
                dt_ratio = gt_ratio
        else:
            dt_ratio = dt['ratio'][i][0]

        visb_iou_2d, visb_iou_3d = calc_IoU(dt_xz, visb_gt_xz, dt_height=1 + dt_ratio, gt_height=1 + gt_ratio)
        full_iou_2d, full_iou_3d = calc_IoU(dt_xz, full_gt_xz, dt_height=1 + dt_ratio, gt_height=1 + gt_ratio)
        iou_height = calc_Iou_height(dt_height=1 + dt_ratio, gt_height=1 + gt_ratio)

        visb_iou_2ds.append(visb_iou_2d)
        visb_iou_3ds.append(visb_iou_3d)
        full_iou_2ds.append(full_iou_2d)
        full_iou_3ds.append(full_iou_3d)
        iou_heights.append(iou_height)

        if visualization:
            pano_img = cv2.resize(gt['image'][i].transpose(1, 2, 0), (h*2, h))
            # visb_iou_floodplans.append(draw_iou_floorplan(dt_xz, visb_gt_xz, iou_2d=visb_iou_2d, iou_3d=visb_iou_3d, side_l=h))
            # full_iou_floodplans.append(draw_iou_floorplan(dt_xz, full_gt_xz, iou_2d=full_iou_2d, iou_3d=full_iou_3d, side_l=h))
            visb_iou_floodplans.append(draw_iou_floorplan(dt_xz, visb_gt_xz, side_l=h))
            full_iou_floodplans.append(draw_iou_floorplan(dt_xz, full_gt_xz, side_l=h))
            gt_boundaries = corners2boundaries(gt_ratio, corners_xyz=full_gt_xyz, step=None, length=1024, visible=False)
            dt_boundaries = corners2boundaries(dt_ratio, corners_xyz=dt_xyz, step=None, visible=False,
                                               length=1024)#visb_gt_xyz.shape[0] if dt_xyz.shape[0] != visb_gt_xyz.shape[0] else None)

            pano_bd = draw_boundaries(pano_img, boundary_list=gt_boundaries, boundary_color=[0, 0, 1])  # Green
            pano_bd = draw_boundaries(pano_bd, boundary_list=dt_boundaries, boundary_color=[0, 1, 0])  # Blue

            pano_bds.append(pano_bd)

            # Ambiguity detection boundaries
            if opening:
                new_dt_xyz = dt['processed_xyz'][i] if 'processed_xyz' in dt else depth2xyz(np.abs(new_dt_depth[i]))
                new_dt_boundaries = corners2boundaries(gt_ratio, corners_xyz=new_dt_xyz, step=None, length=1024, visible=False)
                if branch_exchange: # i.e., zind
                    pano_bd = draw_boundaries(pano_img, boundary_list=dt_boundaries, boundary_color=[0, 1, 1])  # Cyan
                    pano_bd = draw_boundaries(pano_bd, boundary_list=new_dt_boundaries, boundary_color=[1, 0, 1])   # Magenta
                else:   # i.e., mp3d
                    pano_bd = draw_boundaries(pano_img, boundary_list=new_dt_boundaries, boundary_color=[0, 1, 1])
                    pano_bd = draw_boundaries(pano_bd, boundary_list=dt_boundaries, boundary_color=[1, 0, 1])
                opening_bds.append(pano_bd)

    visb_iou_2d = np.array(visb_iou_2ds).mean()
    visb_iou_3d = np.array(visb_iou_3ds).mean()
    full_iou_2d = np.array(full_iou_2ds).mean()
    full_iou_3d = np.array(full_iou_3ds).mean()
    iou_height = np.array(iou_heights).mean()

    if visualization:
        visb_iou_floodplans = np.array(visb_iou_floodplans).transpose(0, 3, 1, 2)  # NCHW
        full_iou_floodplans = np.array(full_iou_floodplans).transpose(0, 3, 1, 2)  # NCHW
        pano_bds = np.array(pano_bds).transpose(0, 3, 1, 2)
        if opening:
            opening_bds = np.array(opening_bds).transpose(0, 3, 1, 2)

    return [visb_iou_2d, visb_iou_3d, visb_iou_floodplans],\
           [full_iou_2d, full_iou_3d, full_iou_floodplans], iou_height, pano_bds, full_iou_2ds, full_iou_3ds, opening_bds


def calc_ce(dt, gt):
    w = 1024
    h = 512
    ce_s = []
    for i in range(len(gt['corners'])):
        floor_gt_corners = gt['corners'][i]
        # Take effective corners
        floor_gt_corners = floor_gt_corners[floor_gt_corners[..., 0] + floor_gt_corners[..., 1] != 0]
        floor_gt_corners = np.roll(floor_gt_corners, -np.argmin(floor_gt_corners[..., 0]), 0)
        gt_ratio = gt['ratio'][i][0]
        ceil_gt_corners = corners2boundaries(gt_ratio, corners_uv=floor_gt_corners, step=None)[1]
        gt_corners = np.concatenate((floor_gt_corners, ceil_gt_corners))
        gt_corners = uv2pixel(gt_corners, w, h)

        floor_dt_corners = xyz2uv(dt['processed_xyz'][i])
        floor_dt_corners = np.roll(floor_dt_corners, -np.argmin(floor_dt_corners[..., 0]), 0)
        dt_ratio = dt['ratio'][i][0]
        ceil_dt_corners = corners2boundaries(dt_ratio, corners_uv=floor_dt_corners, step=None)[1]
        dt_corners = np.concatenate((floor_dt_corners, ceil_dt_corners))
        dt_corners = uv2pixel(dt_corners, w, h)

        mse = np.sqrt(((gt_corners - dt_corners) ** 2).sum(1)).mean()
        ce = 100 * mse / np.sqrt(w ** 2 + h ** 2)
        ce_s.append(ce)

    return np.array(ce_s).mean()


def calc_pe(dt, gt):
    w = 1024
    h = 512
    pe_s = []
    for i in range(len(gt['corners'])):
        floor_gt_corners = gt['corners'][i]
        # Take effective corners
        floor_gt_corners = floor_gt_corners[floor_gt_corners[..., 0] + floor_gt_corners[..., 1] != 0]
        floor_gt_corners = np.roll(floor_gt_corners, -np.argmin(floor_gt_corners[..., 0]), 0)
        gt_ratio = gt['ratio'][i][0]
        gt_floor_boundary, gt_ceil_boundary = corners2boundaries(gt_ratio, corners_uv=floor_gt_corners, length=w)
        gt_floor_boundary = uv2pixel(gt_floor_boundary, w, h)
        gt_ceil_boundary = uv2pixel(gt_ceil_boundary, w, h)

        floor_dt_corners = xyz2uv(dt['processed_xyz'][i])
        floor_dt_corners = np.roll(floor_dt_corners, -np.argmin(floor_dt_corners[..., 0]), 0)
        dt_ratio = dt['ratio'][i][0]
        dt_floor_boundary, dt_ceil_boundary = corners2boundaries(dt_ratio, corners_uv=floor_dt_corners, length=w)
        dt_floor_boundary = uv2pixel(dt_floor_boundary, w, h)
        dt_ceil_boundary = uv2pixel(dt_ceil_boundary, w, h)

        gt_surface = np.zeros((h, w), dtype=np.int32)
        gt_surface[gt_ceil_boundary[..., 1], np.arange(w)] = 1
        gt_surface[gt_floor_boundary[..., 1], np.arange(w)] = 1
        gt_surface = np.cumsum(gt_surface, axis=0)

        dt_surface = np.zeros((h, w), dtype=np.int32)
        dt_surface[dt_ceil_boundary[..., 1], np.arange(w)] = 1
        dt_surface[dt_floor_boundary[..., 1], np.arange(w)] = 1
        dt_surface = np.cumsum(dt_surface, axis=0)

        pe = 100 * (dt_surface != gt_surface).sum() / (h * w)
        pe_s.append(pe)
    return np.array(pe_s).mean()


def calc_rmse_delta_1(dt, gt):
    rmse_s = []
    delta_1_s = []
    for i in range(len(gt['depth'])):
        gt_boundaries = corners2boundaries(gt['ratio'][i], corners_xyz=depth2xyz(gt['depth'][i]), step=None,
                                           visible=False)
        dt_xyz = dt['processed_xyz'][i] if 'processed_xyz' in dt else depth2xyz(np.abs(dt['depth'][i]))

        dt_boundaries = corners2boundaries(dt['ratio'][i], corners_xyz=dt_xyz, step=None,
                                           length=256 if 'processed_xyz' in dt else None,
                                           visible=True if 'processed_xyz' in dt else False)
        gt_layout_depth = layout2depth(gt_boundaries, show=False)
        dt_layout_depth = layout2depth(dt_boundaries, show=False)

        rmse = ((gt_layout_depth - dt_layout_depth) ** 2).mean() ** 0.5
        threshold = np.maximum(gt_layout_depth / dt_layout_depth, dt_layout_depth / gt_layout_depth)
        delta_1 = (threshold < 1.25).mean()
        rmse_s.append(rmse)
        delta_1_s.append(delta_1)
    return np.array(rmse_s).mean(), np.array(delta_1_s).mean()


def calc_f1_score(dt, gt, threshold=10):
    w = 1024
    h = 512
    f1_s = []
    precision_s = []
    recall_s = []
    for i in range(len(gt['corners'])):
        floor_gt_corners = gt['corners'][i]
        # Take effective corners
        floor_gt_corners = floor_gt_corners[floor_gt_corners[..., 0] + floor_gt_corners[..., 1] != 0]
        floor_gt_corners = np.roll(floor_gt_corners, -np.argmin(floor_gt_corners[..., 0]), 0)
        gt_ratio = gt['ratio'][i][0]
        ceil_gt_corners = corners2boundaries(gt_ratio, corners_uv=floor_gt_corners, step=None)[1]
        gt_corners = np.concatenate((floor_gt_corners, ceil_gt_corners))
        gt_corners = uv2pixel(gt_corners, w, h)

        floor_dt_corners = xyz2uv(dt['processed_xyz'][i])
        floor_dt_corners = np.roll(floor_dt_corners, -np.argmin(floor_dt_corners[..., 0]), 0)
        dt_ratio = dt['ratio'][i][0]
        ceil_dt_corners = corners2boundaries(dt_ratio, corners_uv=floor_dt_corners, step=None)[1]
        dt_corners = np.concatenate((floor_dt_corners, ceil_dt_corners))
        dt_corners = uv2pixel(dt_corners, w, h)

        Fs, Ps, Rs = f1_score_2d(gt_corners, dt_corners, [threshold])
        f1_s.append(Fs[0])
        precision_s.append(Ps[0])
        recall_s.append(Rs[0])

    return np.array(f1_s).mean(), np.array(precision_s).mean(), np.array(recall_s).mean()


def calc_precision_recall_f1(predicted_labels, true_labels):
    true_positives = np.sum(true_labels * predicted_labels).astype(np.float32)
    predicted_positives = np.sum(predicted_labels).astype(np.float32)
    actual_positives = np.sum(true_labels).astype(np.float32)

    precision = true_positives / (predicted_positives + 1e-10)  # TP / (TP + FP)
    recall = true_positives / (actual_positives + 1e-10)    # TP / (TP + FN)
    # f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

    # return precision, recall, f1_score
    return precision, recall


def calc_tp_fp_fn(predicted_confidence, true_label, threshold=3):
    prediction = np.where(np.array(predicted_confidence) >= threshold, 1, 0)
    tp, fp, fn = 0, 0, 0
    for i in range(len(true_label)):
        if true_label[i] == 1 and prediction[i] == 1:
            tp += 1
        elif true_label[i] == 0 and prediction[i] == 1:
            fp += 1
        elif true_label[i] == 1 and prediction[i] == 0:
            fn += 1

    return tp, fp, fn


def calc_ap(predicted_confidence, true_label, threshold=3):  # threshold: pixel difference, i.e., confidence
    # sort the confidence values in descending order
    sorted_indices = np.argsort(predicted_confidence, kind='heapsort')[::-1]    # get ordered indices
    true_label = np.array(true_label)[sorted_indices.astype(int)]
    predicted_confidence = np.array(predicted_confidence)[sorted_indices.astype(int)]

    prediction = np.where(predicted_confidence >= threshold, 1, 0)

    precision, recall = [], []
    true_positives = 0
    false_positives = 0
    total_positives = np.sum(true_label)

    for i in range(len(true_label)):    # every column
        if true_label[i] == 1 and prediction[i] == 1:
            true_positives += 1
        elif true_label[i] == 0 and prediction[i] == 1:
            false_positives += 1
        precision.append(true_positives / (true_positives + false_positives + 1e-10))
        recall.append(true_positives / (total_positives + 1e-10))

    area = precision_recall_curve(precision, recall, plot=False)

    final_precision = precision[-1] # whole testing set
    final_recall = recall[-1]
    print(f'AP: {area:.4f}, Precision: {final_precision:.4f}, Recall: {final_recall:.4f}')

    return area, final_precision, final_recall


def precision_recall_curve(precision: list, recall: list, plot=False):
    # Calculate the area under the precision-recall curve using the trapezoidal rule
    area = np.trapz(precision, recall)  # i.e., average precision

    if plot:
        # Plot the precision-recall curve
        plt.figure()
        plt.step(recall, precision, where='post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall Curve, AP={area:.4f}')
        plt.show()

    return area


def show_heat_map(dt, gt, vis_w=1024):
    dt_heat_map = dt['corner_heat_map'].detach().cpu().numpy()
    gt_heat_map = gt['corner_heat_map'].detach().cpu().numpy()
    dt_heat_map_imgs = []
    gt_heat_map_imgs = []
    for i in range(len(gt['depth'])):
        dt_heat_map_img = dt_heat_map[..., np.newaxis].repeat(3, axis=-1).repeat(20, axis=0)
        gt_heat_map_img = gt_heat_map[..., np.newaxis].repeat(3, axis=-1).repeat(20, axis=0)
        dt_heat_map_imgs.append(cv2.resize(dt_heat_map_img, (vis_w, dt_heat_map_img.shape[0])).transpose(2, 0, 1))
        gt_heat_map_imgs.append(cv2.resize(gt_heat_map_img, (vis_w, dt_heat_map_img.shape[0])).transpose(2, 0, 1))
    return dt_heat_map_imgs, gt_heat_map_imgs


def show_depth_normal_grad(dt, gt, device, vis_w=1024, second_type=False):
    grad_conv = GradLoss().to(device).grad_conv
    gt_grad_imgs = []
    dt_grad_imgs = []

    if second_type is True:
        gt['depth'] = gt['new_depth']
        dt['depth'] = dt['new_depth']

    if 'depth' not in dt.keys():
        dt['depth'] = gt['depth']

    if vis_w == 1024:
        h = 5
    else:
        h = int(vis_w / (12 * 10))

    for i in range(len(gt['depth'])):
        gt_grad_img = show_grad(gt['depth'][i], grad_conv, h)   # (5*3, 256, 3)
        dt_grad_img = show_grad(dt['depth'][i], grad_conv, h)
        vis_h = dt_grad_img.shape[0] * (vis_w // dt_grad_img.shape[1])  # 60, original_h: 5*3

        gt_grad_imgs.append(cv2.resize(gt_grad_img, (vis_w, vis_h), interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1))   # (3, 60, 1024)
        dt_grad_imgs.append(cv2.resize(dt_grad_img, (vis_w, vis_h), interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1))

    return gt_grad_imgs, dt_grad_imgs


def opening_util(value, h=5, is_gt=False):
    is_np = isinstance(value, np.ndarray)
    if is_np:
        value_stack = np.repeat(value.reshape(1, -1), int(h), axis=0)
    else:
        value_stack = np.repeat(value.cpu().numpy().reshape(1, -1), int(h), axis=0)

    w = value_stack.shape[1]    # value_stack: (h, 1024)
    image = np.zeros((h, w, 3), dtype=np.uint8)

    if is_gt:
        for i in range(w):
            pixel_color = (0, 0, 1) if value_stack[0][i] == 1 else (1, 1, 1)    # 1: blue, 0: white, later in main.py will *255
            image[:, i] = pixel_color
    else:
        colormap = plt.get_cmap('Greens')  # You can change 'viridis' to your desired colormap
        # Normalize the data to be between 0 and 1
        normalize = plt.Normalize(value_stack.min(), value_stack.max())
        # Create a colormap array by applying the colormap to your data
        image = colormap(normalize(value_stack))    # already 0~1
        image = image[..., :3]  # without alpha channel

    return image

def show_opening(dt, gt):
    gt_grad_imgs = []
    dt_grad_imgs = []
    for i in range(len(gt['depth'])):
        gt_opening_img = opening_util(gt['opening'][i], h=20, is_gt=True)
        dt_opening_img = opening_util(dt['opening'][i], h=20)
        gt_grad_imgs.append(gt_opening_img.transpose(2, 0, 1))
        dt_grad_imgs.append(dt_opening_img.transpose(2, 0, 1))

    return gt_grad_imgs, dt_grad_imgs
