# Localization with Sampling-Argmax

[[`Paper`](https://jeffli.site/sampling-argmax/resources/neurips2021-sampling-argmax.pdf)]
[[`arXiv`](https://arxiv.org/abs/2110.08825)]
[[`Project Page`](https://jeffli.site/sampling-argmax/)]

> [Localization with Sampling-Argmax]()  
> Jiefeng Li, Tong Chen, Ruiqi Shi, Yujing Lou, Yong-Lu Li, Cewu Lu  
> NeurIPS 2021  

<div align="center">
    <img src="asserts/sampling-argmax.jpg", width="400" alt><br>
    Differentiable Sampling
</div>

## Requirements
* Python 3.6+
* PyTorch >= 1.2
* torchvision >= 0.3.0

## Install
1. Install [PyTorch](https://pytorch.org/)
``` bash
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0
```

2. Install `sampling_argmax`
``` bash
python setup.py develop
```

## Fetch data

Please download data from [MSCOCO](http://cocodataset.org/#download), [Human3.6M](http://vision.imar.ro/human3.6m/) and [MTFL](http://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html). Download and extract them under `./data`, and make them look like this:
```
|-- exp
|-- sampling_argmax
|-- configs
|-- data
`-- |-- coco
        |-- annotations
        |-- train2017
        `-- val2017
    |-- h36m
        |-- annotations
        `-- images
    |-- mtfl
    `-- |-- AFLW
        |---net_7876
        |---lfw_5590
        |-- training.json
        `-- testing.json

```


## Train from scratch

``` bash
# COCO Keypoint
./scripts/train_pose.sh configs/coco/256x192_res50_lr1e-3_1x-simple-integral.yaml coco_samp
# Human3.6M
./scripts/train_pose.sh configs/h36m/256x192_adam_lr1e-3-simple_3d_base_1x_h36mmpii.yaml h36m_samp
# MTFL
./scripts/train_mtfl.sh configs/mtfl/256x192_res50_lr1e-3_1x-mtfl-simple-integral.yaml mtfl_samp
```

## Evaluation
``` bash
# COCO Keypoint
./scripts/validate_pose.sh configs/coco/256x192_res50_lr1e-3_1x-simple-integral.yaml ${CKPT}
# Human3.6M
./scripts/validate_pose.sh configs/h36m/256x192_adam_lr1e-3-simple_3d_base_1x_h36mmpii.yaml ${CKPT}
# MTFL
./scripts/validate_mtfl.sh configs/mtfl/256x192_res50_lr1e-3_1x-mtfl-simple-integral.yaml ${CKPT}
```

## Results
### COCO Keypoint
Results on COCO validation set:

<center>

| Method | AP @0.5:0.95 | AP @0.5 | AP @0.75 |
|:-------|:-----:|:-------:|:-------:|
| Samp. Uni. | 68.2 | 87.2 | 75.0 |
| Samp. Tri. | 69.8 | 87.9 | 76.2 |
| Samp. Gau. | 68.3 | 87.3 | 75.2 |

</center>

### Human3.6M
Results on S9 and S11:

<center>

| Method | MPJPE | PA-MPJPE |
|:-------|:-----:|:-------:|
| Samp. Uni. | 49.6 | 39.1 |
| Samp. Tri. | 49.5 | 39.1 |
| Samp. Gau. | 50.9 | 39.0 |

</center>

### MTFL
Results on MTFL:

<center>

| Method | Abs | Rel |
|:-------|:-----:|:-------:|
| Samp. Uni. | 3.00 | 6.86 |
| Samp. Tri. | 2.98 | 6.82 |
| Samp. Gau. | 2.94 | 6.96 |

</center>


If you find our code or paper useful, please consider citing
```bibtex
@inproceedings{li2021localization,
    title={Localization with Sampling-Argmax},
    author={Li, Jiefeng and Chen, Tong and Shi, Ruiqi and Lou, Yujing and Li, Yong-Lu and Lu, Cewu},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2021}
}
```