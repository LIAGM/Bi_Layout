import os
import re

ckpt_root = 'checkpoints/SWG_Transformer_LGT_Net'
result_path = 'results/test/head_origin_label_origin'

original_dir = 'yuju_single_original'   # change to the exp name for original label
new_dir = 'yuju_single_new_eval_original'  # change to the exp name for new label

original_path = os.path.join(ckpt_root, original_dir, result_path)
new_path = os.path.join(ckpt_root, new_dir, result_path)

original_files = [f for f in os.listdir(original_path)]
new_files = [f for f in os.listdir(new_path)]

oracle_2d_iou_list = []
oravle_3d_iou_list = []
from_original = 0
from_new = 0

for original_filename, new_filename in zip(original_files, new_files):
    original_match = re.findall(r'\d\.\d+', original_filename)
    new_match = re.findall(r'\d\.\d+', new_filename)

    if original_match:
        original_2d_iou = float(original_match[0])
        original_3d_iou = float(original_match[1])
        # print('original label:')
        # print(original_filename)
        # print(original_2d_iou)
        # print(original_3d_iou)
    else:
        print('No floating point number found in original filename')
    
    if new_match:
        new_2d_iou = float(new_match[0])
        new_3d_iou = float(new_match[1])
        # print('new label:')
        # print(new_filename)
        # print(new_2d_iou)
        # print(new_3d_iou)
    else:
        print('No floating point number found in new filename')

    if original_2d_iou >= new_2d_iou:
        # print('original label model is better')
        oracle_2d_iou_list.append(original_2d_iou)
        oravle_3d_iou_list.append(original_3d_iou)
        from_original += 1
    else:
        # print('new label model is better')
        if new_2d_iou - original_2d_iou > 0.1:
            print(f'{new_filename}, difference: {new_2d_iou - original_2d_iou}')
        oracle_2d_iou_list.append(new_2d_iou)
        oravle_3d_iou_list.append(new_3d_iou)
        from_new += 1

    # print('-'*3)

print('oracle 2d iou mean: ', sum(oracle_2d_iou_list) / len(oracle_2d_iou_list))
print('oracle 3d iou mean: ', sum(oravle_3d_iou_list) / len(oravle_3d_iou_list))
print('from original: ', from_original)
print('from new: ', from_new)