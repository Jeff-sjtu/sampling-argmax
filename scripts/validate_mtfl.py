"""Validation script."""
import torch
import torch.multiprocessing as mp
from sampling_argmax.models import builder
from sampling_argmax.args import cfg, args
from sampling_argmax.trainer import validate_gt_mtfl

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
        err = validate_gt_mtfl(m, args, cfg, heatmap_to_coord)
        abs_err = err['abs_err']
        rel_err = err['rel_err']
        print(f'##### Abs Err: {abs_err:2f} | Rel Err: {rel_err:2f} #####')


if __name__ == "__main__":
    main()
