"""Script for multi-gpu training."""
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.utils.data
from sampling_argmax.models import builder
from sampling_argmax.args import cfg, logger, args
from sampling_argmax.trainer import train, validate, validate_gt, validate_gt_3d
from sampling_argmax.utils.env import init_dist
from sampling_argmax.utils.metrics import NullWriter
from sampling_argmax.utils.transforms import get_coord

num_gpu = torch.cuda.device_count()
valid_batch = 1 * num_gpu


def _init_fn(worker_id):
    np.random.seed(args.seed)
    random.seed(args.seed)


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    if args.seed is not None:
        setup_seed(args.seed)

    if args.launcher == 'slurm':
        main_worker(None, args, cfg)
    else:
        ngpus_per_node = torch.cuda.device_count()
        args.ngpus_per_node = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args, cfg))


def main_worker(gpu, args, cfg):
    if args.seed is not None:
        setup_seed(args.seed)

    if gpu is not None:
        args.gpu = gpu

    init_dist(args)

    if args.log:
        cfg_file_name = os.path.basename(args.cfg)
        filehandler = logging.FileHandler(
            './exp/{}-{}/training.log'.format(args.exp_id, cfg_file_name))
        streamhandler = logging.StreamHandler()

        logger.setLevel(logging.INFO)
        logger.addHandler(filehandler)
        logger.addHandler(streamhandler)
    else:
        null_writer = NullWriter()
        sys.stdout = null_writer

    logger.info('******************************')
    logger.info(args)
    logger.info('******************************')
    logger.info(cfg)
    logger.info('******************************')

    args.nThreads = int(args.nThreads / num_gpu)

    # Model Initialize
    m = preset_model(cfg)

    m.cuda(args.gpu)
    m = torch.nn.parallel.DistributedDataParallel(m, device_ids=[args.gpu])

    criterion = builder.build_loss(cfg.LOSS).cuda()

    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(m.parameters(), lr=cfg.TRAIN.LR)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_FACTOR)

    train_dataset = builder.build_dataset(cfg.DATASET.TRAIN, preset_cfg=cfg.DATA_PRESET, train=True, heatmap2coord=cfg.TEST.HEATMAP2COORD)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=args.rank)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=(train_sampler is None), num_workers=args.nThreads, sampler=train_sampler, worker_init_fn=_init_fn)

    output_3d = cfg.DATA_PRESET.get('OUT_3D', False)

    if isinstance(cfg.TEST.get('HEATMAP2COORD'), list):
        heatmap_to_coord = [get_coord(None, cfg.DATA_PRESET.HEATMAP_SIZE, output_3d, _type) for _type in cfg.TEST['HEATMAP2COORD']]
    else:
        heatmap_to_coord = get_coord(cfg, cfg.DATA_PRESET.HEATMAP_SIZE, output_3d)

    args.trainIters = 0
    best_err = 999

    for i in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        args.epoch = i
        train_sampler.set_epoch(i)
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        logger.info(f'############# Starting Epoch {args.epoch} | LR: {current_lr} #############')

        # Training
        loss, acc = train(args, cfg, train_loader, m, criterion, optimizer)
        logger.epochInfo('Train', args.epoch, loss, acc)

        lr_scheduler.step()

        if (i + 1) % args.snapshot == 0:
            # Save checkpoint
            if args.log:
                torch.save(m.module.state_dict(), './exp/{}-{}/model_{}.pth'.format(args.exp_id, cfg.FILE_NAME, args.epoch))
            # Prediction Test
            with torch.no_grad():
                if output_3d:
                    err = validate_gt_3d(m, args, cfg, heatmap_to_coord)
                    if args.log and err <= best_err:
                        best_err = err
                        torch.save(m.module.state_dict(), './exp/{}-{}/best_model.pth'.format(args.exp_id, cfg.FILE_NAME))

                    logger.info(f'##### Epoch {args.epoch} | gt results: {err}/{best_err} #####')
                else:
                    gt_AP = validate_gt(m, args, cfg, heatmap_to_coord)
                    det_AP = validate(m, args, cfg, heatmap_to_coord)
                    logger.info(f'##### Epoch {args.epoch} | gt mAP: {gt_AP} | det mAP: {det_AP} #####')

        torch.distributed.barrier()  # Sync

        # Time to add DPG
        if i == cfg.TRAIN.DPG_MILESTONE:
            torch.save(m.module.state_dict(), './exp/{}-{}/final.pth'.format(args.exp_id, cfg.FILE_NAME))
            # Adjust learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = cfg.TRAIN.LR
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.DPG_STEP, gamma=0.1)
            # Reset dataset
            train_dataset = builder.build_dataset(cfg.DATASET.TRAIN, preset_cfg=cfg.DATA_PRESET, train=True, dpg=True, heatmap2coord=cfg.TEST.HEATMAP2COORD)
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=args.world_size, rank=args.rank)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=(train_sampler is None), num_workers=args.nThreads, sampler=train_sampler, worker_init_fn=_init_fn)

    torch.save(m.module.state_dict(), './exp/{}-{}/final_DPG.pth'.format(args.exp_id, cfg.FILE_NAME))


def preset_model(cfg):
    model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    if cfg.MODEL.PRETRAINED:
        logger.info(f'Loading model from {cfg.MODEL.PRETRAINED}...')
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED))
    elif cfg.MODEL.TRY_LOAD:
        logger.info(f'Loading model from {cfg.MODEL.TRY_LOAD}...')
        pretrained_state = torch.load(cfg.MODEL.TRY_LOAD)
        model_state = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items()
                            if k in model_state and v.size() == model_state[k].size()}

        model_state.update(pretrained_state)
        model.load_state_dict(model_state)
    else:
        logger.info('Create new model')
        logger.info('=> init weights')
        model._initialize()

    return model


if __name__ == "__main__":
    main()
