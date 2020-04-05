# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms
from utils import *

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class SingleDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 load_semantics = False,
                 seman_path = None,
                 load_hints = False,
                 hints_path = None,
                 load_unbnPred = False
                 ):
        super(SingleDataset, self).__init__()
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = '.png'

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1



        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()

        self.load_semantics = load_semantics
        self.seman_path = seman_path
        self.seman_resize = transforms.Resize((self.height, self.width), interpolation=pil.NEAREST)

        self.load_hints = load_hints
        self.hints_path = hints_path

        self.load_unbnPred = load_unbnPred
    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
                if i == 0 and im == 0:
                    inputs['border_morph_aug'] = self.to_tensor(self.color_aug2(f))
                    # torch.mean(torch.abs(inputs[(n + "_aug", im, i)] - inputs['border_morph_aug']))



    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5
        line = self.filenames[index].split()
        folder = line[0]

        org_K = self.get_K(folder).copy()

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None


        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = org_K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
            self.color_aug2 = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)
            self.color_aug2 = (lambda x: x)

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if self.load_semantics:
            seman_gt = self.get_seman(folder, frame_index, side, do_flip)
            inputs["seman_gt"] = torch.from_numpy(np.expand_dims(np.array(self.seman_resize(Image.fromarray(seman_gt))), 0).astype(np.int))
        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.check_cityscape_meta():
            inputs["cts_meta"] = self.get_cityscape_meta(folder)

        # baseline = self.get_baseLine(folder)
        rescale_fac = self.get_rescaleFac(folder)
        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1 * rescale_fac

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        camK, invcamK, realIn, realEx, velo = self.get_camK(folder, frame_index)

        inputs["camK"] = torch.from_numpy(camK).float() # Intrinsic by extrinsic
        inputs["invcamK"] = torch.from_numpy(invcamK).float() # inverse of Intrinsic by extrinsic
        inputs["realIn"] = torch.from_numpy(realIn).float() # Intrinsic
        inputs["realEx"] = torch.from_numpy(realEx).float() # Extrinsic, possibly edited to form in accordance with kitti

        # read the stereo
        if self.load_hints:
            depth_hint, depth_hint_mask = self.get_hints(folder, frame_index, side, do_flip)
            inputs["depth_hint"] = depth_hint
            inputs["depth_hint_mask"] = depth_hint_mask

        if velo is not None:
            inputs["velo"] = velo

        if self.load_unbnPred == True:
            inputs["depth_morphed"] = self.get_morphed_depths(folder, frame_index, side, do_flip)
        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_K(self, folder):
        raise NotImplementedError

    def get_rescaleFac(self, folder):
        raise NotImplementedError

    def check_cityscape_meta(self):
        raise NotImplementedError

    def get_cityscape_meta(self, folder):
        raise NotImplementedError

    def get_camK(self, folder, frame_index):
        raise NotImplementedError

    def get_morphed_depths(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_hints(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_seman(self, folder, frame_index, side, do_flip):
        raise NotImplementedError