'''
Process ZInD dataset for Normal or Two-head LGT-Net.
Get suitable training data for:
1. Raw label
2. Visible label
3. Raw + Visible label
'''

import os
import json
import numpy as np

from utils.conversion import xyz2depth, xyz2uv, uv2xyz
from utils.boundary import corners2boundary, visibility_corners
from preprocessing.filter import filter_center, filter_boundary, filter_self_intersection

# ZIND_BAD_PREDICT_PATH = '/media/Pluto/frank/layout_ambiguity/zind'  # for subest evaluation

'''
Read selected subset:
1. layout type: [raw, visible, both]
2. simiplicity: [simple, complex, both]
3. primary: [primary, secondary, both]
4. mode: [train, val, test]
'''
def read_zind_subset(root_dir, layout_type='raw', simplicity='complex', primary='primary', mode='train'):
    
    partition_path = os.path.join(root_dir, "zind_partition.json")  # root_dir will be zind data directory
    simplicity_path = os.path.join(root_dir, "room_shape_simplicity_labels.json")

    with open(simplicity_path, 'r') as f:
        simple_tag = json.load(f)
        simple_panos = {}
        complex_panos = {}
        for k in simple_tag.keys():
            split = k.split('_')
            house_index = split[0]
            pano_index = '_'.join(split[-2:])
            if simple_tag[k]:   # simple
                simple_panos[f'{house_index}_{pano_index}'] = True
            elif not simple_tag[k]:   # complex
                complex_panos[f'{house_index}_{pano_index}'] = True

    pano_list = []
    with open(partition_path, 'r') as f1:
        house_list = json.load(f1)[mode]    # pick house with mode

    for house_index in house_list:
        with open(os.path.join(root_dir, house_index, f"zind_data.json"), 'r') as f2:
            data = json.load(f2)    # all data in the single house

        # parse each pano
        panos = []
        merger = data['merger']
        for floor in merger.values():
            for complete_room in floor.values():
                for partial_room in complete_room.values():
                    for pano_index in partial_room:
                        pano = partial_room[pano_index]
                        pano['index'] = pano_index
                        panos.append(pano)

        # for a single pano
        for pano in panos:
            # layout type
            if layout_type == 'raw':
                if 'layout_raw' not in pano:
                    continue    # filter out
                else:   # data we want
                    layout = pano['layout_raw'] # get layout data with specific type

            elif layout_type == 'visible':
                if 'layout_visible' not in pano:
                    continue
                else:
                    layout = pano['layout_visible']
        
            # TODO: when want to have gt opening, second layout in testing cannot be the same
            elif layout_type == 'both':
                if 'layout_raw' not in pano or 'layout_visible' not in pano:    # or the idx in __getitem__ will be wrong for two-head version
                    continue
                else:
                    layout = pano['layout_visible'] # always set visible as the first label, no matter train/val/test (_C.EVAL.EVAL_GT_MISMATCH will handle the problem)
                    second_layout = pano['layout_raw']
            else:
                raise ValueError("layout type choice: [raw, visible, both]")
            
            pano_index = pano['index']

            # Subset evaluation
            # with open(os.path.join(ZIND_BAD_PREDICT_PATH, f"final_subset.txt"), 'r') as f:
            #     split_list = ['_'.join(x.rstrip().split('_')[:3]) for x in f] # get the first three parts of the name and use '_' to connect them
            # if f'{house_index}_{pano_index}' not in split_list:
            #     continue

            # simplicity
            if simplicity == 'simple':
                if f'{house_index}_{pano_index}' not in simple_panos.keys():
                    continue
            elif simplicity == 'complex':
                if f'{house_index}_{pano_index}' not in complex_panos.keys():
                    continue
            elif simplicity == 'both':
                pass    # both are ok, no filter
            else:
                raise ValueError("simplicity choice: [simple, complex, both]")
            
            # primary
            if primary == 'primary':
                if not pano['is_primary']:
                    continue
            elif primary == 'secondary':
                if pano['is_primary']:
                    continue
            elif primary == 'both':
                pass    # both are ok, no filter
            else:
                raise ValueError("primary choice: [primary, secondary, both]")

            # ratio
            ratio = np.array([(pano['ceiling_height'] - pano['camera_height']) / pano['camera_height']], dtype=np.float32)

            # corners
            corner_xz = np.array(layout['vertices'])
            corner_xz[..., 0] = -corner_xz[..., 0]
            corner_xyz = np.insert(corner_xz, 1, pano['camera_height'], axis=1)
            corners = xyz2uv(corner_xyz).astype(np.float32)

            # openings
            openings = False
            if len(layout['openings']) != 0:
                openings = True

            # output
            pano_dict = {
                'id': f'{house_index}_{pano_index}',
                'img_path': os.path.join(root_dir, house_index, pano['image_path']),
                'corners': corners,
                'ratio': ratio,
                'openings': openings,   # True, False
                'is_inside': pano['is_inside'],
            }
            
            # bi-layout version
            if layout_type == 'both':
                # second corners
                second_corner_xz = np.array(second_layout['vertices'])
                second_corner_xz[..., 0] = -second_corner_xz[..., 0]
                second_corner_xyz = np.insert(second_corner_xz, 1, pano['camera_height'], axis=1)
                second_corners = xyz2uv(second_corner_xyz).astype(np.float32)

                # second openings
                second_openings = False
                if len(second_layout['openings']) != 0:
                    second_openings = True

                # add to output
                pano_dict['second_corners'] = second_corners
                pano_dict['second_openings'] = second_openings

            # output list
            pano_list.append(pano_dict)

    return pano_list


def invalid_filter(pano_list, mode, logger, patch_num=256):
    # if invalid, remove whole pano from the pano list
    # whole pano will include both layout and second layout (if it has), solve synchronization problem
    valid_pano_list= []
    invalid_num = 0
    new_invalid_num = 0
    for pano in pano_list:
        if not os.path.exists(pano['img_path']):
            logger.warning(f"{pano['img_path']} not exists")
            invalid_num += 1
            continue

        # visible_corners = visibility_corners(pano['corners'])
        # if len(visible_corners) < 3:
        #     # cannot form a polygon
        #     # there will be AssertionError of len(depth) == patch_num in process_data()
        #     # logger.warning(f"{pano['id']} visible corner < 3")
        #     invalid_num += 1
        #     continue

        if not filter_center(pano['corners']):
            # logger.warning(f"{pano['id']} camera center not in layout")
            # invalid_num += 1
            continue

        # if self.max_wall_num >= 10:
        #     if len(pano['corners']) < self.max_wall_num:
        #         invalid_num += 1
        #         continue
        # elif self.max_wall_num != 0 and len(pano['corners']) != self.max_wall_num:
        #     invalid_num += 1
        #     continue

        if not filter_boundary(pano['corners']):
            logger.warning(f"{pano['id']} boundary cross")
            invalid_num += 1
            continue

        if not filter_self_intersection(pano['corners']):
            logger.warning(f"{pano['id']} self_intersection")
            invalid_num += 1
            continue
        
        # # depth filter
        # depth = get_depth(visible_corners, length=patch_num, visible=False)
        # if len(depth) != patch_num:
        #     logger.warning(f"{pano['id']} len(depth) != patch_num")
        #     invalid_num += 1
        #     continue
        
        # same for second corner (if there is)
        if 'second_corners' in pano.keys(): # means it's two-head version data
            # second_visible_corners = visibility_corners(pano['second_corners'])
            # if len(second_visible_corners) < 3:
            #     # logger.warning(f"{pano['id']} second label visible corner < 3")
            #     new_invalid_num += 1
            #     continue

            if not filter_center(pano['second_corners']):
                # logger.warning(f"{pano['id']} camera center not in layout")
                # new_invalid_num += 1
                continue

            if not filter_boundary(pano['second_corners']):
                logger.warning(f"{pano['id']} boundary cross")
                new_invalid_num += 1
                continue

            if not filter_self_intersection(pano['second_corners']):
                logger.warning(f"{pano['id']} self_intersection")
                new_invalid_num += 1
                continue

            # # depth filter
            # second_depth = get_depth(second_visible_corners, length=patch_num, visible=False)
            # if len(second_depth) != patch_num:
            #     logger.warning(f"{pano['id']} second label len(depth) != patch_num")
            #     new_invalid_num += 1
            #     continue

        valid_pano_list.append(pano)

    logger.info(
        f"Build dataset mode: {mode}, valid: {len(valid_pano_list)} invalid: {invalid_num} new invalid: {new_invalid_num}")
    
    return valid_pano_list


def get_depth(corners, plan_y=1, length=256, visible=True):
    visible_floor_boundary = corners2boundary(corners, length=length, visible=visible)
    # The horizon-depth relative to plan_y
    visible_depth = xyz2depth(uv2xyz(visible_floor_boundary, plan_y), plan_y)
    return visible_depth