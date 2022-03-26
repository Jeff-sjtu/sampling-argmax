import json
import os
import pickle as pk

import numpy as np
import torch
from tqdm import tqdm

from sampling_argmax.models import builder
from sampling_argmax.utils.metrics import DataLogger, calc_accuracy, calc_coord_accuracy, evaluate_mAP
from sampling_argmax.utils.nms import oks_pose_nms
from sampling_argmax.utils.transforms import flip, flip_output


def train(opt, cfg, train_loader, m, criterion, optimizer):
    loss_logger = DataLogger()
    acc_logger = DataLogger()
    m.train()
    hm_shape = cfg.DATA_PRESET.get('HEATMAP_SIZE')
    anneal = cfg.DATA_PRESET.get('ANNEAL', False)
    depth_dim = cfg.MODEL.get('DEPTH_DIM')
    output_3d = cfg.DATA_PRESET.get('OUT_3D', False)
    hm_shape = (hm_shape[1], hm_shape[0], depth_dim)

    if opt.log:
        train_loader = tqdm(train_loader, dynamic_ncols=True)

    for i, (inps, labels, _, bboxes) in enumerate(train_loader):
        inps = inps.cuda()

        for k, _ in labels.items():
            if k == 'type':
                continue

            labels[k] = labels[k].cuda(opt.gpu)

        if anneal:
            output = m(inps, opt.epoch)
        else:
            output = m(inps)

        loss = criterion(output, labels)
        if cfg.TEST.get('HEATMAP2COORD') == 'heatmap':
            acc = calc_accuracy(output, labels)
        elif cfg.TEST.get('HEATMAP2COORD') == 'coord':
            acc = calc_coord_accuracy(output, labels, hm_shape, output_3d=output_3d)

        if isinstance(inps, list):
            batch_size = inps[0].size(0)
        else:
            batch_size = inps.size(0)

        loss_logger.update(loss.item(), batch_size)
        acc_logger.update(acc, batch_size)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        opt.trainIters += 1

        if opt.log:
            # TQDM
            train_loader.set_description(
                'loss: {loss:.8f} | acc: {acc:.4f}'.format(
                    loss=loss_logger.avg,
                    acc=acc_logger.avg)
            )

    if opt.log:
        train_loader.close()

    return loss_logger.avg, acc_logger.avg


def validate(m, opt, cfg, heatmap_to_coord, batch_size=20, use_nms=False):
    det_dataset = builder.build_dataset(cfg.DATASET.TEST, preset_cfg=cfg.DATA_PRESET, train=False, opt=opt, heatmap2coord=cfg.TEST.HEATMAP2COORD)
    det_dataset_sampler = torch.utils.data.distributed.DistributedSampler(
        det_dataset, num_replicas=opt.world_size, rank=opt.rank)
    det_loader = torch.utils.data.DataLoader(
        det_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False, sampler=det_dataset_sampler)
    kpt_json = []

    m.eval()

    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE
    flip_shift = cfg.TEST.get('FLIP_SHIFT', True)

    if opt.log:
        det_loader = tqdm(det_loader, dynamic_ncols=True)

    for inps, crop_bboxes, bboxes, img_ids, scores, imghts, imgwds in det_loader:
        inps = inps.cuda()
        output = m(inps)

        if opt.flip_test:
            inps_flipped = flip(inps).cuda()
            output_flipped = flip_output(
                m(inps_flipped), det_dataset.joint_pairs,
                width_dim=hm_size[1], shift=flip_shift)
            for k in output.keys():
                if isinstance(output[k], list):
                    continue
                if output[k] is not None:
                    output[k] = (output[k] + output_flipped[k]) / 2

        for i in range(inps.shape[0]):
            bbox = crop_bboxes[i].tolist()
            pose_coords, pose_scores = heatmap_to_coord(
                output, bbox, idx=i)

            keypoints = np.concatenate((pose_coords[0], pose_scores[0]), axis=1)
            keypoints = keypoints.reshape(-1).tolist()

            data = dict()
            data['bbox'] = bboxes[i, 0].tolist()
            data['image_id'] = int(img_ids[i])
            data['score'] = float(scores[i] + np.mean(pose_scores) + np.max(pose_scores))
            data['category_id'] = 1
            data['keypoints'] = keypoints
            data['area'] = float((crop_bboxes[i][2] - crop_bboxes[i][0]) * (crop_bboxes[i][3] - crop_bboxes[i][1]))

            kpt_json.append(data)

    with open(os.path.join(opt.work_dir, f'test_kpt_rank_{opt.rank}.pkl'), 'wb') as fid:
        pk.dump(kpt_json, fid, pk.HIGHEST_PROTOCOL)

    torch.distributed.barrier()  # Make sure all JSON files are saved

    if opt.rank == 0:
        kpt_json_all = []
        for r in range(opt.world_size):
            with open(os.path.join(opt.work_dir, f'test_kpt_rank_{r}.pkl'), 'rb') as fid:
                kpt_pred = pk.load(fid)

            os.remove(os.path.join(opt.work_dir, f'test_kpt_rank_{r}.pkl'))
            kpt_json_all += kpt_pred

            if True:
                kpt_json_all = oks_pose_nms(kpt_json_all)

        with open(os.path.join(opt.work_dir, 'test_kpt.json'), 'w') as fid:
            json.dump(kpt_json_all, fid)
        res = evaluate_mAP(os.path.join(opt.work_dir, 'test_kpt.json'), ann_type='keypoints')
        return res['AP']
    else:
        return 0


def validate_gt(m, opt, cfg, heatmap_to_coord, batch_size=20):
    gt_val_dataset = builder.build_dataset(cfg.DATASET.VAL, preset_cfg=cfg.DATA_PRESET, train=False, heatmap2coord=cfg.TEST.HEATMAP2COORD)
    gt_val_sampler = torch.utils.data.distributed.DistributedSampler(
        gt_val_dataset, num_replicas=opt.world_size, rank=opt.rank)

    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_dataset, batch_size=batch_size, shuffle=False, num_workers=20, drop_last=False, sampler=gt_val_sampler)
    kpt_json = []
    m.eval()

    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE
    flip_shift = cfg.TEST.get('FLIP_SHIFT', True)

    if opt.log:
        gt_val_loader = tqdm(gt_val_loader, dynamic_ncols=True)

    for inps, labels, img_ids, bboxes in gt_val_loader:
        inps = inps.cuda()
        output = m(inps)

        if opt.flip_test:
            inps_flipped = flip(inps).cuda()
            output_flipped = flip_output(
                m(inps_flipped), gt_val_dataset.joint_pairs,
                width_dim=hm_size[1], shift=flip_shift)
            for k in output.keys():
                if isinstance(output[k], list):
                    continue
                if output[k] is not None:
                    output[k] = (output[k] + output_flipped[k]) / 2

        for i in range(inps.shape[0]):
            bbox = bboxes[i].tolist()
            pose_coords, pose_scores = heatmap_to_coord(
                output, bbox, idx=i)

            keypoints = np.concatenate((pose_coords[0], pose_scores[0]), axis=1)
            keypoints = keypoints.reshape(-1).tolist()

            data = dict()
            data['bbox'] = bboxes[i].tolist()
            data['image_id'] = int(img_ids[i])
            data['score'] = float(np.mean(pose_scores) + np.max(pose_scores))
            data['category_id'] = 1
            data['keypoints'] = keypoints

            kpt_json.append(data)

    with open(os.path.join(opt.work_dir, f'test_gt_kpt_rank_{opt.rank}.pkl'), 'wb') as fid:
        pk.dump(kpt_json, fid, pk.HIGHEST_PROTOCOL)

    torch.distributed.barrier()  # Make sure all JSON files are saved

    if opt.rank == 0:
        kpt_json_all = []
        for r in range(opt.world_size):
            with open(os.path.join(opt.work_dir, f'test_gt_kpt_rank_{r}.pkl'), 'rb') as fid:
                kpt_pred = pk.load(fid)

            os.remove(os.path.join(opt.work_dir, f'test_gt_kpt_rank_{r}.pkl'))
            kpt_json_all += kpt_pred

        with open(os.path.join(opt.work_dir, 'test_gt_kpt.json'), 'w') as fid:
            json.dump(kpt_json_all, fid)
        res = evaluate_mAP(os.path.join(opt.work_dir, 'test_gt_kpt.json'), ann_type='keypoints')
        return res['AP']
    else:
        return 0


def validate_gt_3d(m, opt, cfg, heatmap_to_coord, batch_size=20):
    gt_val_dataset = builder.build_dataset(cfg.DATASET.VAL, preset_cfg=cfg.DATA_PRESET, train=False, heatmap2coord=cfg.TEST.HEATMAP2COORD)
    gt_val_sampler = torch.utils.data.distributed.DistributedSampler(
        gt_val_dataset, num_replicas=opt.world_size, rank=opt.rank)

    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False, sampler=gt_val_sampler)
    kpt_pred = {}
    m.eval()

    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE

    if opt.log:
        gt_val_loader = tqdm(gt_val_loader, dynamic_ncols=True)

    for inps, labels, img_ids, bboxes in gt_val_loader:
        inps = inps.cuda()
        output = m(inps)

        if opt.flip_test:
            inps_flipped = flip(inps).cuda()

            output_flipped = flip_output(
                m(inps_flipped), gt_val_dataset.joint_pairs,
                width_dim=hm_size[1], shift=True)
            for k in output.keys():
                output[k] = (output[k] + output_flipped[k]) / 2

        for i in range(inps.shape[0]):
            bbox = bboxes[i].tolist()
            pose_coords, pose_scores = heatmap_to_coord(
                output, bbox, idx=i)
            assert pose_coords.shape[0] == 1

            kpt_pred[int(img_ids[i])] = {
                'uvd': pose_coords[0]
            }

    with open(os.path.join(opt.work_dir, f'test_gt_kpt_rank_{opt.rank}.pkl'), 'wb') as fid:
        pk.dump(kpt_pred, fid, pk.HIGHEST_PROTOCOL)

    torch.distributed.barrier()  # Make sure all JSON files are saved

    if opt.rank == 0:
        kpt_all_pred = {}
        for r in range(opt.world_size):
            with open(os.path.join(opt.work_dir, f'test_gt_kpt_rank_{r}.pkl'), 'rb') as fid:
                kpt_pred = pk.load(fid)

            os.remove(os.path.join(opt.work_dir, f'test_gt_kpt_rank_{r}.pkl'))
            kpt_all_pred.update(kpt_pred)

        tot_err = gt_val_dataset.evaluate(kpt_all_pred, os.path.join('exp', 'test_h36m_3d_kpt.json'))
        return tot_err
    else:
        return -1


def validate_gt_mtfl(m, opt, cfg, heatmap_to_coord, batch_size=20):
    gt_val_dataset = builder.build_dataset(
        cfg.DATASET.VAL, preset_cfg=cfg.DATA_PRESET, train=False, heatmap2coord=cfg.TEST.HEATMAP2COORD)
    gt_val_sampler = torch.utils.data.distributed.DistributedSampler(
        gt_val_dataset, num_replicas=opt.world_size, rank=opt.rank)

    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_dataset, batch_size=batch_size, shuffle=False, num_workers=5, drop_last=False, sampler=gt_val_sampler)
    kpt_json = {}
    m.eval()

    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE

    if opt.log:
        gt_val_loader = tqdm(gt_val_loader, dynamic_ncols=True)

    for inps, labels, img_ids, bboxes in gt_val_loader:
        inps = inps.cuda()
        output = m(inps)

        if opt.flip_test:
            inps_flipped = flip(inps).cuda()
            output_flipped = flip_output(
                m(inps_flipped), gt_val_dataset.joint_pairs,
                width_dim=hm_size[1], shift=True)
            for k in output.keys():
                output[k] = (output[k] + output_flipped[k]) / 2

        for i in range(inps.shape[0]):
            bbox = bboxes[i].tolist()
            pose_coords, pose_scores = heatmap_to_coord(
                output, bbox, idx=i)

            keypoints = pose_coords[0]
            kpt_json[img_ids[i]] = {
                'landmarks': keypoints
            }

    with open(os.path.join(opt.work_dir, f'test_gt_kpt_rank_{opt.rank}.pkl'), 'wb') as fid:
        pk.dump(kpt_json, fid, pk.HIGHEST_PROTOCOL)

    torch.distributed.barrier()  # Make sure all JSON files are saved

    if opt.rank == 0:
        kpt_json_all = {}
        for r in range(opt.world_size):
            with open(os.path.join(opt.work_dir, f'test_gt_kpt_rank_{r}.pkl'), 'rb') as fid:
                kpt_pred = pk.load(fid)

            os.remove(os.path.join(opt.work_dir, f'test_gt_kpt_rank_{r}.pkl'))
            kpt_json_all.update(kpt_pred)

        res = gt_val_dataset.evaluate(kpt_json_all)
        return res
    else:
        return {'abs_err': 0, 'rel_err': 0}
