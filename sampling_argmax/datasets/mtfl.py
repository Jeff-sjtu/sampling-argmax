import copy
import json
import os

import cv2
import numpy as np
import torch.utils.data as data
from sampling_argmax.models.builder import DATASET
from sampling_argmax.utils.presets import SimpleFaceTransform


@DATASET.register_module
class MTFL(data.Dataset):
    num_joints = 5
    joint_pairs = [[0, 1], [3, 4]]

    def __init__(self, train=True, **cfg):
        self._cfg = cfg
        self._preset_cfg = cfg['PRESET']
        self._root = cfg['ROOT']
        self._ann_file = os.path.join(self._root, cfg['ANN'])
        self._train = train

        if 'AUG' in cfg.keys():
            self._scale_factor = cfg['AUG']['SCALE_FACTOR']
            self._rot = cfg['AUG']['ROT_FACTOR']
        else:
            self._scale_factor = 0
            self._rot = 0

        self._input_size = self._preset_cfg['IMAGE_SIZE']
        self._output_size = self._preset_cfg['HEATMAP_SIZE']

        self._sigma = self._preset_cfg['SIGMA']

        self._loss_type = cfg['heatmap2coord']

        self.transformation = SimpleFaceTransform(
            self, scale_factor=self._scale_factor,
            input_size=self._input_size,
            output_size=self._output_size,
            rot=self._rot, sigma=self._sigma,
            train=self._train,
            loss_type=self._loss_type)

        self._items, self._labels = self._load_json()

    def __getitem__(self, idx):
        img_path = self._items[idx]
        label = copy.deepcopy(self._labels[idx])

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        target = self.transformation(img, label)

        img = target.pop('image')
        bbox = target.pop('bbox')
        return img, target, img_path, bbox

    def __len__(self):
        return len(self._items)

    def _load_json(self):
        items = []
        labels = []
        self.labels_dict = {}

        with open(self._ann_file, 'r') as fid:
            database = json.load(fid)

        for data_item in database:
            img_path = data_item['image_name']
            abs_path = os.path.join(self._root, img_path)

            items.append(abs_path)
            labels.append(data_item)
            self.labels_dict[abs_path] = np.stack(data_item['landmarks'])

        return items, labels

    def evaluate(self, pred_dict):
        abs_diff = []
        mean_error = []
        for key in self.labels_dict.keys():
            gt_landmarks = self.labels_dict[key]
            pred_landmarks = pred_dict[key]['landmarks']
            diff = (gt_landmarks - pred_landmarks) ** 2
            dist_eye = np.sqrt(np.sum((gt_landmarks[0] - gt_landmarks[1]) ** 2))
            if float(dist_eye) < 1e-3:
                continue
            relative_diff = np.sqrt(np.sum(diff, axis=1)) / float(dist_eye)

            diff = np.sqrt(np.sum(diff, axis=1)).mean()
            abs_diff.append(diff)
            relative_diff = relative_diff.mean()
            mean_error.append(relative_diff)

        abs_diff = np.mean(abs_diff)
        mean_error = np.mean(mean_error) * 100
        res = {
            'abs_err': abs_diff,
            'rel_err': mean_error
        }
        return res
