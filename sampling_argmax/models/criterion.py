import torch
import torch.nn as nn

from .builder import LOSS


def weighted_l1_loss(input, target, weights, size_average):
    out = torch.abs(input - target)
    out = out * weights
    if size_average and weights.sum() > 0:
        return out.sum() / len(input)
    else:
        return out.sum()


def weighted_l1_loss2(input, target, weights, size_average):
    input = input * 64
    target = target * 64
    out = torch.abs(input - target)
    out = out * weights
    if size_average and weights.sum() > 0:
        return out.sum() / weights.sum()
    else:
        return out.sum()


@LOSS.register_module
class MSELoss(nn.Module):
    ''' MSE Loss
    '''
    def __init__(self):
        super(MSELoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, output, labels):
        pred_hm = output['heatmap']
        gt_hm = labels['target_hm']
        gt_hm_weight = labels['target_hm_weight']
        loss = 0.5 * self.criterion(pred_hm.mul(gt_hm_weight), gt_hm.mul(gt_hm_weight))

        return loss


@LOSS.register_module
class L1JointRegression(nn.Module):
    ''' L1 Joint Regression Loss
    '''
    def __init__(self, OUTPUT_3D=False, size_average=True):
        super(L1JointRegression, self).__init__()
        self.size_average = size_average

    def forward(self, output, labels):
        pred_jts = output.pred_jts
        if pred_jts.dim() == 4:
            shape = pred_jts.shape
            gt_uv = labels['target_uv'].reshape(shape[0], shape[1], 1, shape[3])
            gt_uv_weight = labels['target_uv_weight'].reshape(shape[0], shape[1], 1, shape[3])
            return weighted_l1_loss(pred_jts, gt_uv, gt_uv_weight, self.size_average) / shape[2]
        else:
            gt_uv = labels['target_uv'].reshape(pred_jts.shape)
            gt_uv_weight = labels['target_uv_weight'].reshape(pred_jts.shape)

        return weighted_l1_loss(pred_jts, gt_uv, gt_uv_weight, self.size_average)


@LOSS.register_module
class L1LossDim(nn.Module):
    def __init__(self, size_average=True):
        super(L1LossDim, self).__init__()
        self.size_average = size_average

    def forward(self, output, labels):
        pred_jts = output.pred_jts
        if pred_jts.dim() == 4:
            shape = pred_jts.shape
            gt_jts = labels['target_uvd'].reshape(shape[0], shape[1], 1, shape[3])
            gt_jts_weight = labels['target_uvd_weight'].reshape(shape[0], shape[1], 1, shape[3])
            return weighted_l1_loss2(pred_jts, gt_jts, gt_jts_weight, self.size_average) / shape[2]

        else:
            gt_jts = labels['target_uvd'].reshape(pred_jts.shape)
            gt_jts_weight = labels['target_uvd_weight'].reshape(pred_jts.shape)

        return weighted_l1_loss2(pred_jts, gt_jts, gt_jts_weight, self.size_average)
