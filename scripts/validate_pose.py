"""Validation script."""
import torch
import torch.multiprocessing as mp
from sampling_argmax.models import builder
from sampling_argmax.args import cfg, args
from sampling_argmax.trainer import validate, validate_gt, validate_gt_3d
from sampling_argmax.utils.env import init_dist
from sampling_argmax.utils.transforms import get_coord


def main():
    if args.launcher in ['none', 'slurm']:
        main_worker(None, args, cfg)
    else:
        ngpus_per_node = torch.cuda.device_count()
        args.ngpus_per_node = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args, cfg))


def main_worker(gpu, args, cfg):

    if gpu is not None:
        args.gpu = gpu

    init_dist(args)

    torch.backends.cudnn.benchmark = True

    m = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    print(f'Loading model from {args.checkpoint}...')
    m.load_state_dict(torch.load(args.checkpoint, map_location='cpu'), strict=False)

    m.cuda(args.gpu)
    m = torch.nn.parallel.DistributedDataParallel(m, device_ids=[args.gpu])

    output_3d = cfg.DATA_PRESET.get('OUT_3D', False)

    heatmap_to_coord = get_coord(cfg, cfg.DATA_PRESET.HEATMAP_SIZE, output_3d)

    with torch.no_grad():
        if output_3d:
            err = validate_gt_3d(m, args, cfg, heatmap_to_coord)
            if args.log:
                print(f'##### gt results: {err} #####')
        else:
            detbox_AP = validate(m, args, cfg, heatmap_to_coord, args.valid_batch)
            gt_AP = validate_gt(m, args, cfg, heatmap_to_coord, args.valid_batch)

            if args.log:
                print('##### gt box: {} mAP | det box: {} mAP #####'.format(gt_AP, detbox_AP))


if __name__ == "__main__":
    main()
