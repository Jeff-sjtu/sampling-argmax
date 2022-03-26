import random

import cv2
import numpy as np
import torch

from ..bbox import _center_scale_to_box
from ..transforms import affine_transform, get_affine_transform, im_to_torch, flip_joints_3d


class SimpleFaceTransform(object):
    def __init__(self, dataset, scale_factor,
                 input_size, output_size, rot, sigma,
                 train, loss_type='heatmap'):
        self._joint_pairs = dataset.joint_pairs
        self._scale_factor = scale_factor
        self._rot = rot

        self._input_size = input_size
        self._heatmap_size = output_size

        self._sigma = sigma
        self._train = train
        self._loss_type = loss_type
        self._aspect_ratio = float(input_size[1]) / input_size[0]  # w / h
        self._feat_stride = np.array(input_size) / np.array(output_size)

        self.pixel_std = 1

    def _target_generator(self, joints_3d, num_joints):
        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_3d[:, 0, 1]
        target = np.zeros((num_joints, self._heatmap_size[0], self._heatmap_size[1]),
                          dtype=np.float32)
        tmp_size = self._sigma * 3

        for i in range(num_joints):
            mu_x = int(joints_3d[i, 0, 0] / self._feat_stride[0] + 0.5)
            mu_y = int(joints_3d[i, 1, 0] / self._feat_stride[1] + 0.5)
            # check if any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if (ul[0] >= self._heatmap_size[1] or ul[1] >= self._heatmap_size[0] or br[0] < 0 or br[1] < 0):
                # return image as is
                target_weight[i] = 0
                continue

            # generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # the gaussian is not normalized, we want the center value to be equal to 1
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (self._sigma ** 2)))

            # usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self._heatmap_size[1]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self._heatmap_size[0]) - ul[1]
            # image range
            img_x = max(0, ul[0]), min(br[0], self._heatmap_size[1])
            img_y = max(0, ul[1]), min(br[1], self._heatmap_size[0])

            v = target_weight[i]
            if v > 0.5:
                target[i, img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, np.expand_dims(target_weight, -1)

    def _integral_target_generator(self, joints_3d, num_joints, patch_height, patch_width):
        target_weight = np.ones((num_joints, 2), dtype=np.float32)
        target_weight[:, 0] = joints_3d[:, 0, 1]
        target_weight[:, 1] = joints_3d[:, 0, 1]

        target = np.zeros((num_joints, 2), dtype=np.float32)
        target[:, 0] = joints_3d[:, 0, 0] / patch_width - 0.5
        target[:, 1] = joints_3d[:, 1, 0] / patch_height - 0.5

        target_weight[target[:, 0] > 0.5] = 0
        target_weight[target[:, 0] < -0.5] = 0
        target_weight[target[:, 1] > 0.5] = 0
        target_weight[target[:, 1] < -0.5] = 0

        target = target.reshape((-1))
        target_weight = target_weight.reshape((-1))
        return target, target_weight

    def __call__(self, src, label):
        landmarks = np.stack(label['landmarks'])
        self.num_joints = landmarks.shape[0]
        gt_joints = np.zeros((self.num_joints, 2, 2), dtype=np.float32)
        for i in range(self.num_joints):
            gt_joints[i, 0, 0] = landmarks[i, 0]
            gt_joints[i, 1, 0] = landmarks[i, 1]
            gt_joints[i, :, 1] = 1

        input_size = self._input_size
        img_w = src.shape[1]
        img_h = src.shape[0]

        center = np.zeros(2, dtype=np.float32)
        center[0] = img_w / 2
        center[1] = img_h / 2
        scale = np.array(
            [img_w, img_h], dtype=np.float32
        )

        # rescale
        if self._train:
            sf = self._scale_factor
            scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        else:
            scale = scale * 1.0

        # rotation
        if self._train:
            rf = self._rot
            r = np.clip(np.random.randn() * rf, -rf * 2, rf *
                        2) if random.random() <= 0.6 else 0
        else:
            r = 0

        joints = gt_joints
        if random.random() > 0.5 and self._train:
            # src, fliped = random_flip_image(src, px=0.5, py=0)
            # if fliped[0]:
            assert src.shape[2] == 3
            src = src[:, ::-1, :]

            joints = flip_joints_3d(joints, img_w, self._joint_pairs)
            center[0] = img_w - center[0] - 1

        inp_h, inp_w = input_size
        trans = get_affine_transform(center, scale, r, [inp_w, inp_h])
        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)

        # deal with joints visibility
        for i in range(self.num_joints):
            if joints[i, 0, 1] > 0.0:
                joints[i, 0:2, 0] = affine_transform(joints[i, 0:2, 0], trans)

        # generate training targets
        target_hm, target_hm_weight = self._target_generator(joints.copy(), self.num_joints)
        target_uv, target_uv_weight = self._integral_target_generator(joints.copy(), self.num_joints, inp_h, inp_w)

        bbox = _center_scale_to_box(center, scale)

        img = im_to_torch(img)
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        output = {
            'image': img,
            'target_hm': torch.from_numpy(target_hm).float(),
            'target_hm_weight': torch.from_numpy(target_hm_weight).float(),
            'target_uv': torch.from_numpy(target_uv).float(),
            'target_uv_weight': torch.from_numpy(target_uv_weight).float(),
            'bbox': torch.Tensor(bbox),
        }

        return output
