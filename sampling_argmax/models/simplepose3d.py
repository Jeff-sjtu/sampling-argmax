import torch
import torch.nn as nn
from easydict import EasyDict
from torch.nn import functional as F

from .builder import SPPE
from .layers.Resnet import ResNet


def uni2tri(eps):
    # eps U[0, 1]
    # PDF:
    # y = x + 1 (-1 < x < 0)
    # y = -x + 1 (0 < x < 1)
    # CDF:
    # y = x^2 / 2 + x + 1/2 (-1 < x < 0)
    # y = -x^2 / 2 + x + 1/2 (0 < x < 1)
    # invcdf:
    # x = sqrt(2y) - 1, y < 0.5
    # x = 1 - sqrt(2 - 2y), y > 0.5
    tri = torch.where(eps < 0.5, torch.sqrt(2 * eps) - 1, 1 - torch.sqrt(2 - 2 * eps))
    p = torch.where(tri < 0, tri + 1, - tri + 1)
    return tri, p


def retrive_p(hm, x):
    # hm: (B, K, W) or (B, K, S, W)
    # x:  (B, K, W) or (B, K, S, W)
    left_x = x.floor() + 1
    right_x = (x + 1).floor() + 1
    left_hm = F.pad(hm, (1, 1)).gather(-1, left_x.long())
    right_hm = F.pad(hm, (1, 1)).gather(-1, right_x.long())
    new_hm = left_hm + (right_hm - left_hm) * (x + 1 - left_x)

    return new_hm


def norm_heatmap(norm_type, heatmap, is_train, sample_num=1, tau=5):
    # Input tensor shape: [N,C,...]
    shape = heatmap.shape
    if norm_type == 'softmax':
        heatmap = heatmap.reshape(*shape[:2], -1)
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    elif norm_type == 'sampling':
        heatmap = heatmap.reshape(*shape[:2], 1, -1)

        if is_train:
            eps = torch.rand(*heatmap.shape[:2], sample_num, heatmap.shape[3], device=heatmap.device)
            log_eps = torch.log(-torch.log(eps))
            gumbel_heatmap = heatmap - log_eps / tau
            gumbel_heatmap = F.softmax(gumbel_heatmap, 3)
            return gumbel_heatmap.reshape(shape[0], shape[1], sample_num, shape[2])
        else:
            heatmap = F.softmax(heatmap, 3)
            return heatmap.reshape(*shape)
    else:
        raise NotImplementedError


@SPPE.register_module
class SimplePose3D(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **cfg):
        super(SimplePose3D, self).__init__()
        self._preset_cfg = cfg['PRESET']
        self.deconv_dim = cfg['NUM_DECONV_FILTERS']
        self.depth_dim = cfg['DEPTH_DIM']
        self.height_dim = self._preset_cfg['HEATMAP_SIZE'][0]
        self.width_dim = self._preset_cfg['HEATMAP_SIZE'][1]
        self.num_joints = self._preset_cfg['NUM_JOINTS']
        self.norm_type = self._preset_cfg['NORM_TYPE']

        self.num_sample = self._preset_cfg.get('NUM_SAMPLE', 10)
        self.basis_type = self._preset_cfg.get('BASIS', 'tri')

        self._norm_layer = norm_layer

        self.preact = ResNet(f"resnet{cfg['NUM_LAYERS']}")

        # Imagenet pretrain model
        import torchvision.models as tm  # noqa: F401,F403
        assert cfg['NUM_LAYERS'] in [18, 34, 50, 101, 152]
        x = eval(f"tm.resnet{cfg['NUM_LAYERS']}(pretrained=True)")

        self.feature_channel = {
            18: 512,
            34: 512,
            50: 2048,
            101: 2048
        }[cfg['NUM_LAYERS']]

        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                 if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)

        self.deconv_layers, out_channel = self._make_deconv_layer()
        self.final_layer = nn.Conv2d(
            out_channel, self.num_joints * self.depth_dim, kernel_size=1, stride=1, padding=0)

        self._tau = 2

    def _make_deconv_layer(self):
        deconv_layers = []
        num_deconv = len(self.deconv_dim)
        input_channel = self.feature_channel
        for i in range(num_deconv):
            if self.deconv_dim[i] > 0:
                deconv = nn.ConvTranspose2d(
                    input_channel, self.deconv_dim[i], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
                bn = self._norm_layer(self.deconv_dim[i])
                deconv_layers.append(deconv)
                deconv_layers.append(bn)
                deconv_layers.append(nn.ReLU(inplace=True))
                input_channel = self.deconv_dim[i]
            else:
                deconv_layers.append(nn.Identity())

        return nn.Sequential(*deconv_layers), input_channel

    def _initialize(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                # logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                # logger.info('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                # if self.deconv_with_bias:
                #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # logger.info('=> init {}.weight as 1'.format(name))
                # logger.info('=> init {}.bias as 0'.format(name))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                # logger.info('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, idx_epoch=None):
        out = self.preact(x)
        out = self.deconv_layers(out)
        out = self.final_layer(out)

        out = out.reshape((out.shape[0], self.num_joints, -1))
        out = norm_heatmap(self.norm_type, out, self.training, self.num_sample, self._tau)

        if idx_epoch is not None:
            self._tau = max(0.5, - idx_epoch / 50 + 5)

        if out.dim() == 3:
            maxvals, _ = torch.max(out.reshape((out.shape[0], self.num_joints, -1)), dim=2, keepdim=True)

            heatmaps = out / out.sum(dim=2, keepdim=True)

            heatmaps = heatmaps.reshape((heatmaps.shape[0], self.num_joints, self.depth_dim, self.height_dim, self.width_dim))

            hm_x = heatmaps.sum((2, 3))
            hm_y = heatmaps.sum((2, 4))
            hm_z = heatmaps.sum((3, 4))
        elif out.dim() == 4:
            maxvals, _ = torch.max(out.reshape((out.shape[0], self.num_joints, self.num_sample, -1)), dim=2, keepdim=True)

            heatmaps = out / out.sum(dim=3, keepdim=True)

            heatmaps = heatmaps.reshape((heatmaps.shape[0], self.num_joints, self.num_sample, self.depth_dim, self.height_dim, self.width_dim))

            hm_x = heatmaps.sum((3, 4))
            hm_y = heatmaps.sum((3, 5))
            hm_z = heatmaps.sum((4, 5))

        if self.training and self.basis_type == 'uni':
            w_x = torch.arange(
                hm_x.shape[-1], dtype=torch.float32, device=hm_x.device).expand_as(hm_x)
            w_y = torch.arange(
                hm_y.shape[-1], dtype=torch.float32, device=hm_y.device).expand_as(hm_y)
            w_z = torch.arange(
                hm_z.shape[-1], dtype=torch.float32, device=hm_z.device).expand_as(hm_z)
            eps_x = torch.rand_like(w_x) - 0.5
            eps_y = torch.rand_like(w_y) - 0.5
            eps_z = torch.rand_like(w_z) - 0.5

            w_x = w_x + eps_x
            w_y = w_y + eps_y
            w_z = w_z + eps_z
            out_hm_x = hm_x
            out_hm_y = hm_y
            out_hm_z = hm_z

            hm_x = hm_x * w_x
            hm_y = hm_y * w_y
            hm_z = hm_z * w_z
        elif self.training and self.basis_type == 'gaussian':
            w_x = torch.arange(hm_x.shape[-1], dtype=torch.float32, device=hm_x.device).expand_as(hm_x)
            w_y = torch.arange(hm_y.shape[-1], dtype=torch.float32, device=hm_y.device).expand_as(hm_y)
            w_z = torch.arange(hm_z.shape[-1], dtype=torch.float32, device=hm_z.device).expand_as(hm_z)

            eps_x = torch.randn_like(w_x)
            eps_y = torch.randn_like(w_y)
            eps_z = torch.randn_like(w_z)

            w_x = w_x + eps_x
            w_y = w_y + eps_y
            w_z = w_z + eps_z
            out_hm_x = hm_x
            out_hm_y = hm_y
            out_hm_z = hm_z

            hm_x = hm_x * w_x
            hm_y = hm_y * w_y
            hm_z = hm_z * w_z
        elif self.training and self.basis_type == 'tri':
            w_x = torch.arange(hm_x.shape[-1], dtype=torch.float32, device=hm_x.device).expand_as(hm_x)
            w_y = torch.arange(hm_y.shape[-1], dtype=torch.float32, device=hm_y.device).expand_as(hm_y)
            w_z = torch.arange(hm_z.shape[-1], dtype=torch.float32, device=hm_z.device).expand_as(hm_z)
            eps_x, _ = uni2tri(torch.rand_like(w_x))
            eps_y, _ = uni2tri(torch.rand_like(w_y))
            eps_z, _ = uni2tri(torch.rand_like(w_z))

            w_x = w_x + eps_x
            w_y = w_y + eps_y
            w_z = w_z + eps_z

            hm_x = retrive_p(hm_x, w_x)
            hm_y = retrive_p(hm_y, w_y)
            hm_z = retrive_p(hm_z, w_z)
            # recalculate the probability
            hm_x = hm_x / hm_x.sum(dim=-1, keepdim=True)
            hm_y = hm_y / hm_y.sum(dim=-1, keepdim=True)
            hm_z = hm_z / hm_z.sum(dim=-1, keepdim=True)
            out_hm_x = hm_x
            out_hm_y = hm_y
            out_hm_z = hm_z

            hm_x = hm_x * w_x
            hm_y = hm_y * w_y
            hm_z = hm_z * w_z
        else:
            out_hm_x = hm_x
            out_hm_y = hm_y
            out_hm_z = hm_z

            hm_x = hm_x * torch.arange(hm_x.shape[-1], dtype=torch.float32, device=hm_x.device).expand_as(hm_x)
            hm_y = hm_y * torch.arange(hm_y.shape[-1], dtype=torch.float32, device=hm_y.device).expand_as(hm_y)
            hm_z = hm_z * torch.arange(hm_z.shape[-1], dtype=torch.float32, device=hm_z.device).expand_as(hm_z)

        coord_x = hm_x.sum(dim=-1, keepdim=True)
        coord_y = hm_y.sum(dim=-1, keepdim=True)
        coord_z = hm_z.sum(dim=-1, keepdim=True)

        coord_x = coord_x / float(self.width_dim) - 0.5
        coord_y = coord_y / float(self.height_dim) - 0.5
        coord_z = coord_z / float(self.depth_dim) - 0.5

        pred_jts = torch.cat((coord_x, coord_y, coord_z), dim=-1)

        output = EasyDict(
            pred_jts=pred_jts,
            maxvals=maxvals.float(),
            hm_x=out_hm_x,
            hm_y=out_hm_y,
            hm_z=out_hm_z
        )
        return output
