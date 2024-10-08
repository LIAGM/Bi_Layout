"""
@Date: 2021/07/18
@description:
"""
import numpy as np
import torch.utils.data
from dataset.mp3d_dataset import MP3DDataset
from dataset.zind_dataset import ZindDataset
from dataset.zind_new_dataset import ZindNewDataset    # new


def build_loader(config, logger):
    name = config.DATA.DATASET
    ddp = config.WORLD_SIZE > 1
    train_dataset = None
    train_data_loader = None
    if config.MODE == 'train':
        train_dataset = build_dataset(mode='train', config=config, logger=logger)

    val_dataset = build_dataset(mode=config.VAL_NAME if config.MODE != 'test' else 'test', config=config, logger=logger)

    train_sampler = None
    val_sampler = None
    self_shuffle = True
    # change this pin memory options to see the speed changes
    self_pin_memory = False
    if ddp:
        if train_dataset:
            train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.DistributedSampler(val_dataset, shuffle=False)
        self_shuffle = False
        self_pin_memory = False

    batch_size = config.DATA.BATCH_SIZE
    num_workers = 0 if config.DEBUG else config.DATA.NUM_WORKERS
    pin_memory = config.DATA.PIN_MEMORY
    logger.info(f'Real num workers: {num_workers}')
    if train_dataset:
        logger.info(f'Train data loader batch size: {batch_size}')
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset, sampler=train_sampler,
            batch_size=batch_size,
            # shuffle=True,
            shuffle=self_shuffle,
            num_workers=num_workers,
            # pin_memory=pin_memory,
            pin_memory=self_pin_memory,
            drop_last=True,
        )
    batch_size = batch_size - (len(val_dataset) % np.arange(batch_size, 0, -1)).tolist().index(0)
    logger.info(f'Val data loader batch size: {batch_size}')
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, sampler=val_sampler,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # pin_memory=pin_memory,
        pin_memory=self_pin_memory,
        drop_last=False
    )
    logger.info(f'Build data loader: num_workers:{num_workers} pin_memory:{self_pin_memory}')
    return train_data_loader, val_data_loader


def build_dataset(mode, config, logger):
    name = config.DATA.DATASET
    if name == 'mp3d':
        dataset = MP3DDataset(
            root_dir=config.DATA.DIR,
            mode=mode,
            type = config.MODEL.TYPE,
            shape=config.DATA.SHAPE,
            max_wall_num=config.DATA.WALL_NUM,
            aug=config.DATA.AUG if mode == 'train' else None,
            camera_height=config.DATA.CAMERA_HEIGHT,
            logger=logger,
            for_test_index=config.DATA.FOR_TEST_INDEX,
            keys=config.DATA.KEYS,
            # head_mismatch=config.MODEL.HEAD_MISMATCH,
        )
    elif name == 'zind' and config.MODEL.TYPE == 'origin':
        dataset = ZindDataset(
            root_dir=config.DATA.DIR,
            mode=mode,
            type = config.MODEL.TYPE,
            shape=config.DATA.SHAPE,
            max_wall_num=config.DATA.WALL_NUM,
            aug=config.DATA.AUG if mode == 'train' else None,
            camera_height=config.DATA.CAMERA_HEIGHT,
            logger=logger,
            for_test_index=config.DATA.FOR_TEST_INDEX,
            is_simple=True,
            is_ceiling_flat=False,
            keys=config.DATA.KEYS,
            vp_align=config.EVAL.POST_PROCESSING is not None and 'manhattan' in config.EVAL.POST_PROCESSING
        )
    elif name == 'zind':   # new
        dataset = ZindNewDataset(
            root_dir=config.DATA.DIR,
            mode=mode,
            model_type = config.MODEL.TYPE,   # for new label
            data_type=config.DATA.TYPE,
            simplicity=config.DATA.SIMPLICITY,
            primary=config.DATA.PRIMARY,
            shape=config.DATA.SHAPE,
            max_wall_num=config.DATA.WALL_NUM,
            aug=config.DATA.AUG if mode == 'train' else None,
            # aug=None,
            camera_height=config.DATA.CAMERA_HEIGHT,
            logger=logger,
            for_test_index=config.DATA.FOR_TEST_INDEX,
            is_simple=True,
            is_ceiling_flat=False,
            keys=config.DATA.KEYS,
            vp_align=config.EVAL.POST_PROCESSING is not None and 'manhattan' in config.EVAL.POST_PROCESSING
        )
    else:
        raise NotImplementedError(f"Unknown dataset: {name}")

    return dataset
