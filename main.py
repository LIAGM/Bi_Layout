"""
@Date: 2021/07/17
@description:
"""
import sys
import os
import shutil
import argparse
import numpy as np
import json
import torch
import torch.nn.parallel
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.cuda

from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from config.defaults import get_config, get_rank_config
from models.other.criterion import calc_criterion
from models.build import build_model
from models.other.init_env import init_env
from utils.logger import build_logger
from utils.misc import tensor2np_d, tensor2np
from dataset.build import build_loader
from evaluation.accuracy import calc_accuracy, show_heat_map, calc_ce, calc_pe, calc_rmse_delta_1, \
    show_depth_normal_grad, calc_f1_score, show_opening, calc_ap, calc_tp_fp_fn, precision_recall_curve
from postprocessing.post_process import post_process

# ambiguity
from utils.conversion import depth2xyz, uv2pixel
from utils.boundary import corners2boundaries

try:
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    debug = True if sys.gettrace() else False
    parser = argparse.ArgumentParser(description='Panorama Layout Transformer training and evaluation script')
    parser.add_argument('--cfg',
                        type=str,
                        default='/media/Pluto/frank/room_layout_project/src/config/zind.yaml',
                        metavar='FILE',
                        help='path to config file')

    parser.add_argument('--mode',
                        type=str,
                        default='train',
                        choices=['train', 'val', 'test'],
                        help='train/val/test mode')

    parser.add_argument('--val_name',
                        type=str,
                        choices=['val', 'test'],
                        help='val name')

    parser.add_argument('--bs', type=int,
                        help='batch size')

    parser.add_argument('--save_eval', action='store_true',
                        help='save eval result')

    parser.add_argument('--post_processing', type=str,
                        choices=['manhattan', 'atalanta', 'manhattan_old'],
                        help='type of postprocessing ')

    parser.add_argument('--need_cpe', action='store_true',
                        help='need to evaluate corner error and pixel error')

    parser.add_argument('--need_f1', action='store_true',
                        help='need to evaluate f1-score of corners')

    parser.add_argument('--need_rmse', action='store_true',
                        help='need to evaluate root mean squared error and delta error')

    parser.add_argument('--force_cube', action='store_true',
                        help='force cube shape when eval')

    parser.add_argument('--wall_num', type=int,
                        help='wall number')
    
    parser.add_argument('--ckpt_option', 
                        type=str,
                        default='best',
                        choices=['last', 'best', 'oracle', 'average'],
                        help='checkpoint options')
    
    parser.add_argument('--pure', action='store_true',
                        help='save pure result without depth')

    args = parser.parse_args()
    args.debug = debug
    print("arguments:")
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    print("-" * 50)
    return args


def main():
    args = parse_option()
    config = get_config(args)

    if config.TRAIN.SCRATCH and os.path.exists(config.CKPT.DIR) and config.MODE == 'train':
        print(f"Train from scratch, delete checkpoint dir: {config.CKPT.DIR}")
        f = [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(config.CKPT.DIR) if 'pkl' in f]
        if len(f) > 0:
            last_epoch = np.array(f).max()
            if last_epoch > 10:
                c = input(f"delete it (last_epoch: {last_epoch})?(Y/N)\n")
                if c != 'y' and c != 'Y':
                    exit(0)

        shutil.rmtree(config.CKPT.DIR, ignore_errors=True)

    os.makedirs(config.CKPT.DIR, exist_ok=True)
    os.makedirs(config.CKPT.RESULT_DIR, exist_ok=True)
    os.makedirs(config.LOGGER.DIR, exist_ok=True)

    if ':' in config.TRAIN.DEVICE:
        nprocs = len(config.TRAIN.DEVICE.split(':')[-1].split(','))
    if 'cuda' in config.TRAIN.DEVICE:
        if not torch.cuda.is_available():
            print(f"Cuda is not available(config is: {config.TRAIN.DEVICE}), will use cpu ...")
            config.defrost()
            config.TRAIN.DEVICE = "cpu"
            config.freeze()
            nprocs = 1

    if config.MODE == 'train':
        with open(os.path.join(config.CKPT.DIR, "config.yaml"), "w") as f:
            f.write(config.dump(allow_unicode=True))

    if config.TRAIN.DEVICE == 'cpu' or nprocs < 2:
        print(f"Use single process, device:{config.TRAIN.DEVICE}")
        main_worker(0, config, 1)
    else:
        print(f"Use {nprocs} processes ...")
        mp.spawn(main_worker, nprocs=nprocs, args=(config, nprocs), join=True)


def main_worker(local_rank, cfg, world_size):
    config = get_rank_config(cfg, local_rank, world_size)
    logger = build_logger(config)
    writer = SummaryWriter(config.CKPT.DIR)
    logger.info(f"Comment: {config.COMMENT}")
    cur_pid = os.getpid()
    logger.info(f"Current process id: {cur_pid}")
    torch.hub._hub_dir = config.CKPT.PYTORCH
    logger.info(f"Pytorch hub dir: {torch.hub._hub_dir}")
    init_env(config.SEED, config.TRAIN.DETERMINISTIC, config.DATA.NUM_WORKERS)

    # try to solve additional process when using ddp
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    print('local_rank: {}'.format(local_rank))

    model, optimizer, criterion, scheduler = build_model(config, logger)
    train_data_loader, val_data_loader = build_loader(config, logger)

    if 'cuda' in config.TRAIN.DEVICE:
        torch.cuda.set_device(config.TRAIN.DEVICE)

    if config.MODE == 'train':
        train(model, train_data_loader, val_data_loader, optimizer, criterion, config, logger, writer, scheduler)
    else:
        # iou_results, other_results = val_an_epoch(model, val_data_loader,
        #                                           criterion, config, logger, writer=None,
        #                                           epoch=config.TRAIN.START_EPOCH)
        iou_results, new_iou_results, oracle_iou_results = val_an_epoch(model, val_data_loader,
                                                  criterion, config, logger, writer=None,
                                                  epoch=config.TRAIN.START_EPOCH)
        results = dict(iou_results, **new_iou_results, **oracle_iou_results)
        if config.SAVE_EVAL:
            save_path = os.path.join(config.CKPT.RESULT_DIR, f"result.json")
            with open(save_path, 'w+') as f:
                json.dump(results, f, indent=4)


def save(model, optimizer, epoch, iou_d, new_iou_d, oracle_iou_d, logger, writer, config):
    # for save best checkpoint
    if config.MODEL.TYPE == 'occlusion':
        # save_3d_iou = (iou_d['full_3d'] + new_iou_d['new_full_3d'])/2
        save_3d_iou = iou_d['full_3d']
        model.save(optimizer, epoch, 
                   accuracy=save_3d_iou, 
                   logger=logger, 
                   acc_d=iou_d, 
                   acc_d_new=new_iou_d,
                   acc_d_oracle=oracle_iou_d,
                   config=config)
    else:
        save_3d_iou = iou_d['full_3d']
        model.save(optimizer, epoch, 
                   accuracy=save_3d_iou, 
                   logger=logger, 
                   acc_d=iou_d, 
                   config=config)
    for k in model.acc_d:
        writer.add_scalar(f"BestACC/{k}", model.acc_d[k]['acc'], epoch)


def train(model, train_data_loader, val_data_loader, optimizer, criterion, config, logger, writer, scheduler):
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        logger.info("=" * 200)
        train_an_epoch(model, train_data_loader, optimizer, criterion, config, logger, writer, epoch)
        if config.LOCAL_RANK == 0:
            # epoch_iou_d, _ = val_an_epoch(model, val_data_loader, criterion, config, logger, writer, epoch)
            epoch_iou_d, epoch_new_iou_d, oracle_iou_d = val_an_epoch(model, val_data_loader, criterion, config, logger, writer, epoch)
        else:
            val_an_epoch(model, val_data_loader, criterion, config, logger, writer, epoch)

        if config.LOCAL_RANK == 0:
            ddp = config.WORLD_SIZE > 1
            save(model.module if ddp else model, optimizer, epoch, epoch_iou_d, epoch_new_iou_d, oracle_iou_d, logger, writer, config)

        if scheduler is not None:
            if scheduler.min_lr is not None and optimizer.param_groups[0]['lr'] <= scheduler.min_lr:
                continue
            scheduler.step()
    writer.close()


def train_an_epoch(model, train_data_loader, optimizer, criterion, config, logger, writer, epoch=0):
    logger.info(f'Start Train Epoch {epoch}/{config.TRAIN.EPOCHS - 1}')
    model.train()

    if len(config.MODEL.FINE_TUNE) > 0:
        model.feature_extractor.eval()

    optimizer.zero_grad()

    data_len = len(train_data_loader)
    start_i = data_len * epoch * config.WORLD_SIZE
    bar = enumerate(train_data_loader)
    if config.LOCAL_RANK == 0 and config.SHOW_BAR:
        bar = tqdm(bar, total=data_len, ncols=100)  # ncols 200 -> 100

    device = config.TRAIN.DEVICE
    epoch_loss_d = {}
    for i, gt in bar:
        imgs = gt['image'].to(device, non_blocking=True)
        gt['depth'] = gt['depth'].to(device, non_blocking=True)
        gt['ratio'] = gt['ratio'].to(device, non_blocking=True)
        if config.MODEL.TYPE == 'occlusion':
            gt['new_depth'] = gt['new_depth'].to(device, non_blocking=True)
            gt['opening'] = gt['opening'].to(device, non_blocking=True).to(torch.float32)   # the target fot BCELoss shoud be torch.long, bug?
        if 'corner_heat_map' in gt:
            gt['corner_heat_map'] = gt['corner_heat_map'].to(device, non_blocking=True)
        if config.AMP_OPT_LEVEL != "O0" and 'cuda' in device:
            imgs = imgs.type(torch.float16)
            gt['depth'] = gt['depth'].type(torch.float16)
            gt['ratio'] = gt['ratio'].type(torch.float16)
        dt = model(imgs)
        loss, batch_loss_d, epoch_loss_d = calc_criterion(criterion, gt, dt, epoch_loss_d)
        # loss, batch_loss_d, epoch_loss_d, opening_loss = calc_criterion(criterion, gt, dt, epoch_loss_d)
        if config.LOCAL_RANK == 0 and config.SHOW_BAR:
            bar.set_postfix(batch_loss_d)

        # # Check all layers' name
        # for name, layer in model.named_modules():
        #     print(name)

        # Check weight for the specific layer
        # for name, param in model.named_parameters():
        #     if name == 'linear_depth_output.weight' or name == 'linear_opening_output.weight':
        #         print(name, param)
                # breakpoint()

        optimizer.zero_grad()
        if config.AMP_OPT_LEVEL != "O0" and 'cuda' in device:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
            # opening_loss.backward()
        optimizer.step()

        # Check gradients for the specific layer
        # for name, param in model.named_parameters():
        #     if name == 'linear_depth_output.weight' or name == 'linear_opening_output.weight':
        #         print(name, param.grad)
                # breakpoint()

        global_step = start_i + i * config.WORLD_SIZE + config.LOCAL_RANK
        for key, val in batch_loss_d.items():
            writer.add_scalar(f'TrainBatchLoss/{key}', val, global_step)

    if config.LOCAL_RANK != 0:
        return

    epoch_loss_d = dict(zip(epoch_loss_d.keys(), [np.array(epoch_loss_d[k]).mean() for k in epoch_loss_d.keys()]))
    s = 'TrainEpochLoss: '
    for key, val in epoch_loss_d.items():
        writer.add_scalar(f'TrainEpochLoss/{key}', val, epoch)
        s += f" {key}={val}"
    logger.info(s)
    writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
    logger.info(f"LearningRate: {optimizer.param_groups[0]['lr']}")


@torch.no_grad()
def val_an_epoch(model, val_data_loader, criterion, config, logger, writer, epoch=0):
    model.eval()
    logger.info(f'Start Validate Epoch {epoch}/{config.TRAIN.EPOCHS - 1}')
    data_len = len(val_data_loader)
    start_i = data_len * epoch * config.WORLD_SIZE
    bar = enumerate(val_data_loader)
    if config.LOCAL_RANK == 0 and config.SHOW_BAR:
        bar = tqdm(bar, total=data_len, ncols=100)
    device = config.TRAIN.DEVICE
    epoch_loss_d = {}
    # origin head to origin label
    epoch_iou_d = {
        'visible_2d': [],
        'visible_3d': [],
        'full_2d': [],
        'full_3d': [],
        'height': []
    }

    # new head to new label
    epoch_new_iou_d = {
        'new_visible_2d': [],
        'new_visible_3d': [],
        'new_full_2d': [],
        'new_full_3d': [],
        'new_height': [],
        # 'opening_acc': [],
        # 'opening_precision': [],
        # 'opening_recall': [],
        # 'opening_f1': [],
        'opening_ap': [],   # average precision, the area under the precision-recall curve
    }

    # new head to origin label
    epoch_n2o_iou_d = {
        'n2o_visible_2d': [],
        'n2o_visible_3d': [],
        'n2o_full_2d': [],
        'n2o_full_3d': [],
        'n2o_height': []
    }

    # origin head to new label
    epoch_o2n_iou_d = {
        'o2n_visible_2d': [],
        'o2n_visible_3d': [],
        'o2n_full_2d': [],
        'o2n_full_3d': [],
        'o2n_height': []
    }

    # better prediction from two head
    epoch_oracle_iou_d = {
        'oracle_full_2d': [],
        'oracle_full_3d': [],
    }
    from_original = 0
    from_new = 0
    original_better_id = []
    new_better_id = []

    # store large difference case
    original_better_large_id = []
    new_better_large_id = []

    # store bad prediction cases
    bad_pred_id = []

    epoch_other_d = {
        'ce': [],
        'pe': [],
        'f1': [],
        'precision': [],
        'recall': [],
        'rmse': [],
        'delta_1': []
    }

    show_index = np.random.randint(0, data_len)
    for i, gt in bar:
        imgs = gt['image'].to(device, non_blocking=True)
        gt['depth'] = gt['depth'].to(device, non_blocking=True)
        gt['ratio'] = gt['ratio'].to(device, non_blocking=True)
        if config.MODEL.TYPE == 'occlusion':
            gt['new_depth'] = gt['new_depth'].to(device, non_blocking=True)
            gt['opening'] = gt['opening'].to(device, non_blocking=True).to(torch.float32)   # the target fot BCELoss shoud be torch.long, bug?
        if 'corner_heat_map' in gt:
            gt['corner_heat_map'] = gt['corner_heat_map'].to(device, non_blocking=True)
        dt = model(imgs)

        # Only for ZInD dataset val/testing correct evaluation
        if config.EVAL.EVAL_GT_MISMATCH:
            # exchange label
            # corners
            gt_tmp_corners = gt['corners']
            gt_tmp_new_corners = gt['new_corners']
            gt['corners'] = gt_tmp_new_corners  # note: assign new object will not affect the original object outside of this function
            gt['new_corners'] = gt_tmp_corners
            # depth
            gt_tmp_depth = gt['depth']
            gt_tmp_new_depth = gt['new_depth']
            gt['depth'] = gt_tmp_new_depth
            gt['new_depth'] = gt_tmp_depth

            # exchange prediction
            dt_tmp_depth = dt['depth']
            dt_tmp_new_depth = dt['new_depth']
            dt['depth'] = dt_tmp_new_depth
            dt['new_depth'] = dt_tmp_depth

        # Ambiguity detection (openings detection)
        # Only when testing and bi_layout outputs and opening is needed
        if config.MODE == 'test' and config.MODEL.TYPE == 'occlusion' and config.EVAL.OPENING:
            opening = []
            for b in range(dt['depth'].shape[0]): # for each prediction in a batch
                depth_xyz = depth2xyz(dt['depth'][b].detach().cpu().numpy())
                new_depth_xyz = depth2xyz(dt['new_depth'][b].detach().cpu().numpy())
                floor_bd, _ = corners2boundaries(dt['ratio'][b].detach().cpu().numpy(), corners_xyz=depth_xyz, step=None, length=1024, visible=True)  # if set visible=False, then the boundary shape might exceed 1024
                new_floor_bd, _ = corners2boundaries(dt['ratio'][b].detach().cpu().numpy(), corners_xyz=new_depth_xyz, step=None, length=1024, visible=True)    
                
                # TODO: few cases boundary shape might < 1024, need to fix it (e.g., interpolation)
                # tmp solution: skip this ambiguity prediction
                if floor_bd.shape[0] != 1024 or new_floor_bd.shape[0] != 1024:
                    opening.append(np.zeros(1024))
                    continue

                floor_pixel_v = uv2pixel(floor_bd)[:, 1]
                new_floor_pixel_v = uv2pixel(new_floor_bd)[:, 1]

                pixel_diff = np.absolute(floor_pixel_v - new_floor_pixel_v)
                
                # opening_b = np.zeros(1024)    # all predict 0
                # opening_b = np.where(pixel_diff > 2, 1, 0)   # 1 if more than 2 pixels difference
                opening_b = np.where(pixel_diff > 2, pixel_diff, 0)   # confidence score

                opening.append(opening_b)

            opening = np.array(opening).reshape(-1, 1024)   # [b, 1024]
            dt['opening'] = opening

        vis_w = config.TRAIN.VIS_WEIGHT
        visualization = config.SAVE_EVAL
        # visualization = False  # (config.LOCAL_RANK == 0 and i == show_index) or config.SAVE_EVAL
        # visualization = True  # (config.LOCAL_RANK == 0 and i == show_index) or config.SAVE_EVAL

        loss, batch_loss_d, epoch_loss_d = calc_criterion(criterion, gt, dt, epoch_loss_d)

        if config.EVAL.POST_PROCESSING is not None:
            depth = tensor2np(dt['depth'])
            dt['processed_xyz'] = post_process(depth, type_name=config.EVAL.POST_PROCESSING,
                                               need_cube=config.EVAL.FORCE_CUBE)

            if config.EVAL.FORCE_CUBE and config.EVAL.NEED_CPE:
                ce = calc_ce(tensor2np_d(dt), tensor2np_d(gt))
                pe = calc_pe(tensor2np_d(dt), tensor2np_d(gt))

                epoch_other_d['ce'].append(ce)
                epoch_other_d['pe'].append(pe)

            if config.EVAL.NEED_F1:
                f1, precision, recall = calc_f1_score(tensor2np_d(dt), tensor2np_d(gt))
                epoch_other_d['f1'].append(f1)
                epoch_other_d['precision'].append(precision)
                epoch_other_d['recall'].append(recall)

        if config.EVAL.NEED_RMSE:
            rmse, delta_1 = calc_rmse_delta_1(tensor2np_d(dt), tensor2np_d(gt))
            epoch_other_d['rmse'].append(rmse)
            epoch_other_d['delta_1'].append(delta_1)

        visb_iou, full_iou, iou_height, pano_bds, full_iou_2ds, full_iou_3ds, _ = calc_accuracy(tensor2np_d(dt), tensor2np_d(gt),
                                                                               visualization, h=vis_w // 2)
        epoch_iou_d['visible_2d'].append(visb_iou[0])
        epoch_iou_d['visible_3d'].append(visb_iou[1])
        epoch_iou_d['full_2d'].append(full_iou[0])
        epoch_iou_d['full_3d'].append(full_iou[1])
        epoch_iou_d['height'].append(iou_height)

        # # select bad predictions
        # if config.MODE == 'test':
        #     for i in range(len(full_iou_2ds)):
        #         if full_iou_2ds[i] < 0.6:
        #             bad_pred_id.append(f'{gt["id"][i]}_{full_iou_2ds[i]:.5f}'+'\n')

        if config.MODEL.TYPE == 'occlusion':
            # new head prediction compare to new label
            new_visb_iou, new_full_iou, new_iou_height, new_pano_bds, new_full_iou_2ds, new_full_iou_3ds, opening_bds = calc_accuracy(tensor2np_d(dt), tensor2np_d(gt),
                                                                               visualization, h=vis_w // 2, second_type=True, gt_label='new', opening=config.EVAL.OPENING, branch_exchange=config.EVAL.EVAL_GT_MISMATCH)
            epoch_new_iou_d['new_visible_2d'].append(new_visb_iou[0])
            epoch_new_iou_d['new_visible_3d'].append(new_visb_iou[1])
            epoch_new_iou_d['new_full_2d'].append(new_full_iou[0])
            epoch_new_iou_d['new_full_3d'].append(new_full_iou[1])
            epoch_new_iou_d['new_height'].append(new_iou_height)
            # epoch_new_iou_d['opening_acc'].append(opening_metrics_batch[0])
            # epoch_new_iou_d['opening_precision'].append(opening_metrics_batch[1])
            # epoch_new_iou_d['opening_recall'].append(opening_metrics_batch[2])
            # epoch_new_iou_d['opening_f1'].append(opening_metrics_batch[3])

            
            # new head prediction compare to origin label
            n2o_visb_iou, n2o_full_iou, n2o_iou_height, n2o_pano_bds, n2o_full_iou_2ds, n2o_full_iou_3ds, _ = calc_accuracy(tensor2np_d(dt), tensor2np_d(gt),
                                                                               visualization, h=vis_w // 2, second_type=True, gt_label='origin')
            epoch_n2o_iou_d['n2o_visible_2d'].append(n2o_visb_iou[0])
            epoch_n2o_iou_d['n2o_visible_3d'].append(n2o_visb_iou[1])
            epoch_n2o_iou_d['n2o_full_2d'].append(n2o_full_iou[0])
            epoch_n2o_iou_d['n2o_full_3d'].append(n2o_full_iou[1])
            epoch_n2o_iou_d['n2o_height'].append(n2o_iou_height)
            
            # # origin head prediction compare to new label
            # o2n_visb_iou, o2n_full_iou, o2n_iou_height, o2n_pano_bds, o2n_full_iou_2ds, o2n_full_iou_3ds, _ = calc_accuracy(tensor2np_d(dt), tensor2np_d(gt),
            #                                                                    visualization, h=vis_w // 2, second_type=False, gt_label='new')
            # epoch_o2n_iou_d['o2n_visible_2d'].append(o2n_visb_iou[0])
            # epoch_o2n_iou_d['o2n_visible_3d'].append(o2n_visb_iou[1])
            # epoch_o2n_iou_d['o2n_full_2d'].append(o2n_full_iou[0])
            # epoch_o2n_iou_d['o2n_full_3d'].append(o2n_full_iou[1])
            # epoch_o2n_iou_d['o2n_height'].append(o2n_iou_height)

            # disambiguate metric
            # select better iou from two predictions-----------------------------------
            disambiguate_pano_bds = []  # diambiguate qualitative results
            disambiguate_floorplan = []
            for i in range(len(full_iou_2ds)):
                # original head better
                if full_iou_2ds[i] > n2o_full_iou_2ds[i]:
                    epoch_oracle_iou_d['oracle_full_2d'].append(full_iou_2ds[i])
                    epoch_oracle_iou_d['oracle_full_3d'].append(full_iou_3ds[i])
                    from_original = from_original + 1
                    original_better_id.append(f'{gt["id"][i]}_origin_{full_iou_2ds[i]:.5f}_new_{n2o_full_iou_2ds[i]:.5f}'+'\n')
                    if config.SAVE_EVAL:
                        disambiguate_pano_bds.append(pano_bds[i])
                        disambiguate_floorplan.append(full_iou[2][i])
                    if abs(full_iou_2ds[i] - n2o_full_iou_2ds[i]) > 0.05:
                        original_better_large_id.append(f'{gt["id"][i]}_origin_{full_iou_2ds[i]:.5f}_new_{n2o_full_iou_2ds[i]:.5f}'+'\n')
                # new head better
                else:
                    epoch_oracle_iou_d['oracle_full_2d'].append(n2o_full_iou_2ds[i])
                    epoch_oracle_iou_d['oracle_full_3d'].append(n2o_full_iou_3ds[i])
                    from_new = from_new + 1
                    new_better_id.append(f'{gt["id"][i]}_origin_{full_iou_2ds[i]:.5f}_new_{n2o_full_iou_2ds[i]:.5f}'+'\n')
                    if config.SAVE_EVAL:
                        disambiguate_pano_bds.append(n2o_pano_bds[i])
                        disambiguate_floorplan.append(n2o_full_iou[2][i])
                    if abs(full_iou_2ds[i] - n2o_full_iou_2ds[i]) > 0.05:
                        new_better_large_id.append(f'{gt["id"][i]}_origin_{full_iou_2ds[i]:.5f}_new_{n2o_full_iou_2ds[i]:.5f}'+'\n')

        if config.LOCAL_RANK == 0 and config.SHOW_BAR:
            bar.set_postfix(batch_loss_d)

        global_step = start_i + i * config.WORLD_SIZE + config.LOCAL_RANK

        if writer:
            for key, val in batch_loss_d.items():
                writer.add_scalar(f'ValBatchLoss/{key}', val, global_step)

        # visualization -----------------------------
        if not visualization:
            continue

        gt_grad_imgs, dt_grad_imgs = show_depth_normal_grad(dt, gt, device, vis_w)
        if config.MODEL.TYPE == 'occlusion':
            new_gt_grad_imgs, new_dt_grad_imgs = show_depth_normal_grad(dt, gt, device, vis_w, second_type=True)

        dt_heat_map_imgs = None
        gt_heat_map_imgs = None
        if 'corner_heat_map' in gt:
            dt_heat_map_imgs, gt_heat_map_imgs = show_heat_map(dt, gt, vis_w)

        if config.TRAIN.VIS_MERGE or config.SAVE_EVAL:
            imgs = []
            for j in range(len(pano_bds)):
                # floorplan = np.concatenate([visb_iou[2][j], full_iou[2][j]], axis=-1)
                floorplan = full_iou[2][j]
                # margin_w = int(floorplan.shape[-1] * (60/512))
                # floorplan = floorplan[:, :, margin_w:-margin_w]

                vis_merge = [
                    pano_bds[j],    # without grad
                ]

                img = np.concatenate(vis_merge, axis=-2)
                img = np.concatenate([img, floorplan], axis=2)
                img = np.concatenate([img, ], axis=-1)

                imgs.append(img)

            # new labels
            if config.MODEL.TYPE == 'occlusion':
                # new head prediction compare to new label--------------------------------
                new_imgs = []
                for j in range(len(new_pano_bds)):
                    # floorplan = np.concatenate([visb_iou[2][j], full_iou[2][j]], axis=-1)
                    floorplan = new_full_iou[2][j]
                    # margin_w = int(floorplan.shape[-1] * (60/512))
                    # floorplan = floorplan[:, :, margin_w:-margin_w]

                    vis_merge = [
                        new_pano_bds[j] # without grad
                    ]

                    img = np.concatenate(vis_merge, axis=-2)
                    img = np.concatenate([img, floorplan], axis=2)
                    img = np.concatenate([img, ], axis=-1)

                    new_imgs.append(img)

                # # new head prediction compare to origin label--------------------------------
                # n2o_imgs = []
                # for j in range(len(n2o_pano_bds)):
                #     # floorplan = np.concatenate([visb_iou[2][j], full_iou[2][j]], axis=-1)
                #     floorplan = n2o_full_iou[2][j]
                #     margin_w = int(floorplan.shape[-1] * (60/512))
                #     floorplan = floorplan[:, :, margin_w:-margin_w]

                #     new_grad_h = new_dt_grad_imgs[0].shape[1]
                #     vis_merge = [
                #         gt_grad_imgs[j],
                #         n2o_pano_bds[j][:, new_grad_h:-new_grad_h],
                #         new_dt_grad_imgs[j],
                #         # n2o_pano_bds[j],
                #     ]
                #     if 'corner_heat_map' in gt:
                #         vis_merge = [dt_heat_map_imgs[j], gt_heat_map_imgs[j]] + vis_merge
                #     img = np.concatenate(vis_merge, axis=-2)

                #     img = np.concatenate([img, floorplan], axis=2)

                #     img = np.concatenate([img, ], axis=-1)
                #     # img = gt_grad_imgs[j]
                #     n2o_imgs.append(img)

                # # origin head prediction compare to new label--------------------------------
                # o2n_imgs = []
                # for j in range(len(o2n_pano_bds)):
                #     # floorplan = np.concatenate([visb_iou[2][j], full_iou[2][j]], axis=-1)
                #     floorplan = o2n_full_iou[2][j]
                #     # margin_w = int(floorplan.shape[-1] * (60/512))
                #     # floorplan = floorplan[:, :, margin_w:-margin_w]

                #     grad_h = dt_grad_imgs[0].shape[1]
                #     vis_merge = [
                #         new_gt_grad_imgs[j],
                #         o2n_pano_bds[j][:, grad_h:-grad_h],
                #         dt_grad_imgs[j]
                #     ]
                #     if 'corner_heat_map' in gt:
                #         vis_merge = [dt_heat_map_imgs[j], gt_heat_map_imgs[j]] + vis_merge
                #     img = np.concatenate(vis_merge, axis=-2)

                #     img = np.concatenate([img, floorplan], axis=2)

                #     img = np.concatenate([img, ], axis=-1)
                #     # img = gt_grad_imgs[j]
                #     o2n_imgs.append(img)

                # Disambiguate qualitative results
                disambiguate_imgs = []
                for j in range(len(disambiguate_pano_bds)):
                    floorplan = disambiguate_floorplan[j]

                    vis_merge = [
                        disambiguate_pano_bds[j] # without grad
                    ]

                    img = np.concatenate(vis_merge, axis=-2)
                    img = np.concatenate([img, floorplan], axis=2)
                    img = np.concatenate([img, ], axis=-1)

                    disambiguate_imgs.append(img)

                # Ambiguity detection qualitative results
                if config.EVAL.OPENING:
                    opening_imgs = []
                    gt_opening_imgs, dt_opening_imgs = show_opening(dt, gt)
                    for j in range(len(opening_bds)):

                        opening_h = dt_opening_imgs[0].shape[1]
                        vis_merge = [
                            gt_opening_imgs[j],
                            opening_bds[j][:, opening_h:-opening_h],
                            dt_opening_imgs[j],
                        ]

                        img = np.concatenate(vis_merge, axis=-2)
                        img = np.concatenate([img, ], axis=-1)

                        opening_imgs.append(img)

            if writer:
                # writer.add_images('VIS/Merge', np.array(imgs), global_step)
                pass    # too many images to save, tensorboard event file will be too large
            
            if config.SAVE_EVAL:
                # if ZInD, exchange branch order back to visible/raw (the original model output) to follow extended/enclosed order for images saving
                if config.EVAL.EVAL_GT_MISMATCH:
                    imgs[:], new_imgs[:] = new_imgs[:], imgs[:]
                
                iou_first_path = os.path.join(config.CKPT.RESULT_DIR, 'extended_results')
                os.makedirs(iou_first_path, exist_ok=True)
                for k in range(len(imgs)):
                    img = imgs[k] * 255.0
                    save_path = os.path.join(iou_first_path, f"{gt['id'][k]}_{full_iou_2ds[k]:.5f}_{full_iou_3ds[k]:.5f}.png")  # 3d iou is for single head oracle test
                    Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8)).save(save_path)

                if config.MODEL.TYPE == 'occlusion':
                    # new head prediction compare to new label--------------------------------
                    iou_first_path_second = os.path.join(config.CKPT.RESULT_DIR, 'enclosed_results')
                    os.makedirs(iou_first_path_second, exist_ok=True)
                    for k in range(len(new_imgs)):
                        img = new_imgs[k] * 255.0
                        save_path = os.path.join(iou_first_path_second, f"{gt['id'][k]}_{new_full_iou_2ds[k]:.5f}.png")
                        Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8)).save(save_path)

                    # # new head prediction compare to origin label--------------------------------
                    # iou_first_path_second = os.path.join(config.CKPT.RESULT_DIR, 'head_new_label_origin')
                    # os.makedirs(iou_first_path_second, exist_ok=True)
                    # for k in range(len(n2o_imgs)):
                    #     img = n2o_imgs[k] * 255.0
                    #     save_path = os.path.join(iou_first_path_second, f"{gt['id'][k]}_{n2o_full_iou_2ds[k]:.5f}.png")
                    #     Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8)).save(save_path)

                    # # origin head prediction compare to new label--------------------------------
                    # iou_first_path_second = os.path.join(config.CKPT.RESULT_DIR, 'head_origin_label_new')
                    # os.makedirs(iou_first_path_second, exist_ok=True)
                    # for k in range(len(o2n_imgs)):
                    #     img = o2n_imgs[k] * 255.0
                    #     save_path = os.path.join(iou_first_path_second, f"{gt['id'][k]}_{o2n_full_iou_2ds[k]:.5f}.png")
                    #     Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8)).save(save_path)

                    # disambiguate qualitative results--------------------------------
                    iou_first_path_second = os.path.join(config.CKPT.RESULT_DIR, 'disambiguate_results')
                    os.makedirs(iou_first_path_second, exist_ok=True)
                    for k in range(len(disambiguate_imgs)):
                        img = disambiguate_imgs[k] * 255.0
                        save_path = os.path.join(iou_first_path_second, f"{gt['id'][k]}_{new_full_iou_2ds[k]:.5f}.png")
                        Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8)).save(save_path)

                    # ambiguity detection qualitative results
                    if config.EVAL.OPENING:
                        iou_first_path_second = os.path.join(config.CKPT.RESULT_DIR, 'ambiguity_detection')
                        os.makedirs(iou_first_path_second, exist_ok=True)
                        for k in range(len(opening_imgs)):
                            img = opening_imgs[k] * 255.0
                            save_path = os.path.join(iou_first_path_second, f"{gt['id'][k]}_{new_full_iou_2ds[k]:.5f}.png")
                            Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8)).save(save_path)

        elif writer:
            # writer.add_images('IoU/Visible_Floorplan', visb_iou[2], global_step)
            # writer.add_images('IoU/Full_Floorplan', full_iou[2], global_step)
            # writer.add_images('IoU/Boundary', pano_bds, global_step)
            # writer.add_images('Grad/gt', gt_grad_imgs, global_step)
            # writer.add_images('Grad/dt', dt_grad_imgs, global_step)
            pass    # too many images to save, tensorboard event file will be too large

    if config.LOCAL_RANK != 0:
        return
    
    # Calculate average precision for opening (whole dataset per column)
    # print('--- @ threshold 10 ---')
    # ap_at_10, precision_at_10, recall_at_10 = calc_ap(gt['opening'], dt['opening'], threshold=10)

    epoch_loss_d = dict(zip(epoch_loss_d.keys(), [np.array(epoch_loss_d[k]).mean() for k in epoch_loss_d.keys()]))
    s = 'ValEpochLoss: '
    for key, val in epoch_loss_d.items():
        if writer:
            writer.add_scalar(f'ValEpochLoss/{key}', val, epoch)
        s += f" {key}={val}"
    logger.info(s)

    epoch_iou_d = dict(zip(epoch_iou_d.keys(), [np.array(epoch_iou_d[k]).mean() for k in epoch_iou_d.keys()]))
    s = 'ValEpochIoU: '
    for key, val in epoch_iou_d.items():
        if writer:
            writer.add_scalar(f'ValEpochIoU/{key}', val, epoch)
        s += f" {key}={val}"
    logger.info(s)

    # # Write bad predictions image ids
    # if config.MODE == 'test':
    #     with open(config.CKPT.RESULT_DIR+'/bad_predictions.txt', 'w') as f:
    #         for file in bad_pred_id:
    #             f.write(file)

    if config.MODEL.TYPE == 'occlusion':
        # new head prediction compare to new label--------------------------------
        epoch_new_iou_d = dict(zip(epoch_new_iou_d.keys(), [np.array(epoch_new_iou_d[k]).mean() for k in epoch_new_iou_d.keys()]))
        s = 'New_ValEpochIoU: '
        for key, val in epoch_new_iou_d.items():
            if writer:
                writer.add_scalar(f'New_ValEpochIoU/{key}', val, epoch)
            s += f" {key}={val}"
        logger.info(s)

        # # new head prediction compare to origin label--------------------------------
        # epoch_n2o_iou_d = dict(zip(epoch_n2o_iou_d.keys(), [np.array(epoch_n2o_iou_d[k]).mean() for k in epoch_n2o_iou_d.keys()]))
        # s = 'N2O_ValEpochIoU: '
        # for key, val in epoch_n2o_iou_d.items():
        #     if writer:
        #         writer.add_scalar(f'N2O_ValEpochIoU/{key}', val, epoch)
        #     s += f" {key}={val}"
        # logger.info(s)

        # # origin head prediction compare to new label--------------------------------
        # epoch_o2n_iou_d = dict(zip(epoch_o2n_iou_d.keys(), [np.array(epoch_o2n_iou_d[k]).mean() for k in epoch_o2n_iou_d.keys()]))
        # s = 'O2N_ValEpochIoU: '
        # for key, val in epoch_o2n_iou_d.items():
        #     if writer:
        #         writer.add_scalar(f'O2N_ValEpochIoU/{key}', val, epoch)
        #     s += f" {key}={val}"
        # logger.info(s)

        # Disambiguate quantitative results
        # select better prediction from two heads--------------------------------
        epoch_oracle_iou_d = dict(zip(epoch_oracle_iou_d.keys(), [np.array(epoch_oracle_iou_d[k]).mean() for k in epoch_oracle_iou_d.keys()]))
        s = 'Oracle_ValEpochIoU: '
        for key, val in epoch_oracle_iou_d.items():
            if writer:
                writer.add_scalar(f'Oracle_ValEpochIoU/{key}', val, epoch)
            s += f" {key}={val}"
        s += f" from original head={from_original}"
        s += f" from new head={from_new}"
        logger.info(s)

        # Record better prediction id
        if config.MODE == 'test':
            # # write original head better image ids
            # with open(config.CKPT.RESULT_DIR+'/origin_better_id.txt', 'a') as f:
            #     for file in original_better_id:
            #         f.write(file)
            # # write new head better image ids
            # with open(config.CKPT.RESULT_DIR+'/new_better_id.txt', 'a') as f:
            #     for file in new_better_id:
            #         f.write(file) 

            # write original head better image ids larger case ----------------------------
            # for MP3D is original label, for ZInD is raw label
            with open(config.CKPT.RESULT_DIR+'/original_prediction_better_id.txt', 'a') as f:
                for file in original_better_large_id:
                    f.write(file)
            # write new head better image ids larger case
            with open(config.CKPT.RESULT_DIR+'/new_prediction_better_id.txt', 'a') as f:
                for file in new_better_large_id:
                    f.write(file)

    epoch_other_d = dict(zip(epoch_other_d.keys(),
                             [np.array(epoch_other_d[k]).mean() if len(epoch_other_d[k]) > 0 else 0 for k in
                              epoch_other_d.keys()]))

    logger.info(f'other acc: {epoch_other_d}')
    return epoch_iou_d, epoch_new_iou_d, epoch_oracle_iou_d
    # return epoch_iou_d, epoch_other_d


if __name__ == '__main__':
    main()
