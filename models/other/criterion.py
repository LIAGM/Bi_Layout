"""
@date: 2021/7/19
@description:
"""
import torch
import loss

from utils.misc import tensor2np


def build_criterion(config, logger):
    criterion = {}
    device = config.TRAIN.DEVICE

    for k in config.TRAIN.CRITERION.keys():
        sc = config.TRAIN.CRITERION[k]
        if sc.WEIGHT is None or float(sc.WEIGHT) == 0:
            continue
        if sc.NAME == 'opening':
            pos_weight = torch.tensor([1.0])
            criterion[sc.NAME] = {
                'loss': getattr(loss, sc.LOSS)(pos_weight=pos_weight),
                'weight': float(sc.WEIGHT),
                'sub_weights': sc.WEIGHTS,
                'need_all': sc.NEED_ALL
            }
        else:
            criterion[sc.NAME] = {
                'loss': getattr(loss, sc.LOSS)(),
                'weight': float(sc.WEIGHT),
                'sub_weights': sc.WEIGHTS,
                'need_all': sc.NEED_ALL
            }

        criterion[sc.NAME]['loss'] = criterion[sc.NAME]['loss'].to(device)
        if config.AMP_OPT_LEVEL != "O0" and 'cuda' in device:
            criterion[sc.NAME]['loss'] = criterion[sc.NAME]['loss'].type(torch.float16)

        # logger.info(f"Build criterion:{sc.WEIGHT}_{sc.NAME}_{sc.LOSS}_{sc.WEIGHTS}")
    return criterion


def calc_criterion(criterion, gt, dt, epoch_loss_d):
    loss = None
    opening_loss = None
    postfix_d = {}
    for k in criterion.keys():  # k: CRITERION.XX.NAME
        if criterion[k]['need_all'] and k == 'grad':
            single_loss = criterion[k]['loss'](gt, dt)
            ws_loss = None
            for i, sub_weight in enumerate(criterion[k]['sub_weights']):
                if sub_weight == 0:
                    continue
                if ws_loss is None:
                    ws_loss = single_loss[i] * sub_weight
                else:
                    ws_loss = ws_loss + single_loss[i] * sub_weight
            single_loss = ws_loss if ws_loss is not None else single_loss
        elif criterion[k]['need_all'] and k == 'new_grad':
            single_loss = criterion[k]['loss'](gt, dt, second_type=True)
            ws_loss = None
            for i, sub_weight in enumerate(criterion[k]['sub_weights']):
                if sub_weight == 0:
                    continue
                if ws_loss is None:
                    ws_loss = single_loss[i] * sub_weight
                else:
                    ws_loss = ws_loss + single_loss[i] * sub_weight
            single_loss = ws_loss if ws_loss is not None else single_loss
        elif k == 'opening':
            single_loss = criterion[k]['loss'](dt[k], gt[k])
        else:
            assert k in gt.keys(), "ground label is None:" + k
            assert k in dt.keys(), "detection key is None:" + k
            if k == 'ratio' and gt[k].shape[-1] != dt[k].shape[-1]:
                gt[k] = gt[k].repeat(1, dt[k].shape[-1])
            single_loss = criterion[k]['loss'](gt[k], dt[k])

        postfix_d[k] = tensor2np(single_loss)
        if k not in epoch_loss_d.keys():
            epoch_loss_d[k] = []
        epoch_loss_d[k].append(postfix_d[k])

        # if k == 'opening' or k == 'depth':
        #     breakpoint()

        single_loss = single_loss * criterion[k]['weight']
        if loss is None:
            loss = single_loss
        else:
            loss = loss + single_loss

    k = 'loss'
    postfix_d[k] = tensor2np(loss)
    if k not in epoch_loss_d.keys():
        epoch_loss_d[k] = []
    epoch_loss_d[k].append(postfix_d[k])
    return loss, postfix_d, epoch_loss_d
    # return loss, postfix_d, epoch_loss_d, opening_loss
