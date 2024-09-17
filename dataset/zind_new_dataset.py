""" 
@Date: 2021/09/22
@description:
"""
import os
import json
import math
import numpy as np

from dataset.communal.read import read_image, read_label, read_zind
from dataset.communal.base_dataset import BaseDataset
from utils.logger import get_logger
from preprocessing.filter import filter_center, filter_boundary, filter_self_intersection
from utils.boundary import calc_rotation

from utils.boundary import visibility_corners
from dataset.zind.zind_two_head_preprocess import read_zind_subset, invalid_filter


class ZindNewDataset(BaseDataset):
    def __init__(self, root_dir, mode='train', model_type='normal', data_type='raw', simplicity='simple', primary='primary', shape=None, max_wall_num=0, 
                 aug=None, camera_height=1.6, logger=None, split_list=None, patch_num=256, keys=None, for_test_index=None, is_simple=True, 
                 is_ceiling_flat=False, vp_align=False):

        super().__init__(mode, shape, max_wall_num, aug, camera_height, patch_num, keys)
        if logger is None:
            logger = get_logger()
        self.patch_num = patch_num
        self.model_type = model_type

        # Preprocess zind data
        pano_list = read_zind_subset(root_dir, layout_type=data_type, simplicity=simplicity, primary=primary, mode=mode)
        if mode == 'train' or mode == 'val':   # invalid labels will make the training process fail
            self.data = invalid_filter(pano_list=pano_list, mode=mode, logger=logger, patch_num=patch_num)  # a data list contains suitable pano label for training
        else:
            # self.data = pano_list
            self.data = invalid_filter(pano_list=pano_list, mode=mode, logger=logger, patch_num=patch_num)

    def __getitem__(self, idx):
        # Original label
        label = self.data[idx]  # each pano
        rgb_path = label['img_path']
        image = read_image(rgb_path, self.shape)

        # New label
        if self.model_type == 'occlusion':
            new_label = {}
            new_label['new_corners'] = label['second_corners']  # new_label only needs 'new_corners' key

        # Output
        if self.model_type == 'normal':
            output = self.process_data(label, image, self.patch_num)
        elif self.model_type == 'occlusion':  # visible label
            output = self.process_data(label, image, self.patch_num, new_label)
        else:
            raise ValueError("Model type can only be normal or occlusion")

        return output