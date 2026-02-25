# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import glob
import json

import torch
import numpy as np

from render import util

from dataset import Dataset

###############################################################################
# NERF image based dataset (synthetic)
###############################################################################

def _load_img(path):
    files = glob.glob(path + '.*')
    assert len(files) > 0, "Tried to find image file for: %s, but found 0 files" % (path)
    img = util.load_image_raw(files[0])
    if (img.dtype != np.float32): # LDR image
        img = torch.tensor(img.astype(np.float32) / np.iinfo(img.dtype).max, dtype=torch.float32)
        img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3])
    else:
        img = torch.tensor(img, dtype=torch.float32)
    return img

class DatasetNERF(Dataset):
    def __init__(self, cfg_path, FLAGS, num_samples=None):
        self.FLAGS = FLAGS
        self.num_samples = num_samples
        self.base_dir = os.path.dirname(cfg_path)

        # Load config / transforms
        self.cfg = json.load(open(cfg_path, 'r'))
        self.n_images = len(self.cfg['frames'])

        # Determine resolution & aspect ratio
        self.resolutions = _load_img(os.path.join(self.base_dir, self.cfg['frames'][0]['file_path'])).shape[0:2]
        self.aspect = self.resolutions[1] / self.resolutions[0]

        print("DatasetNERF: %d images with shape [%d, %d]" % (self.n_images, self.resolutions[0], self.resolutions[1]))

        # Pre-load from disc to avoid slow png parsing
        if self.FLAGS.pre_load:
            self.preloaded_data = []
            for i in range(self.n_images):
                self.preloaded_data += [self._parse_frame(self.cfg, i)]

    def _parse_frame(self, cfg, idx):
        # Config projection matrix (static, so could be precomputed)
        fovy   = util.fovx_to_fovy(cfg['camera_angle_x'], self.aspect)
        proj   = util.perspective(fovy, self.aspect, self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # Load image data and modelview matrix
        img    = _load_img(os.path.join(self.base_dir, cfg['frames'][idx]['file_path']))
        model_view_trnsfrm     = torch.linalg.inv(torch.tensor(cfg['frames'][idx]['transform_matrix'], dtype=torch.float32))
        model_view_trnsfrm     = model_view_trnsfrm @ util.rotate_x(-np.pi / 2)

        cam_posi = torch.linalg.inv(model_view_trnsfrm)[:3, 3]
        mvp_trnsfrm    = proj @ model_view_trnsfrm

        return img[None, ...], model_view_trnsfrm[None, ...], mvp_trnsfrm[None, ...], cam_posi[None, ...] # Add batch dimension

    def getMesh(self):
        return None # There is no mesh

    def __len__(self):
        return self.n_images if self.num_samples is None else self.num_samples

    def __getitem__(self, itr):
        img      = []
        # fovy     = util.fovx_to_fovy(self.cfg['camera_angle_x'], self.aspect)

        if self.FLAGS.pre_load:
            img, model_view_trnsfrm, mvp_trnsfrm, cam_posi = self.preloaded_data[itr % self.n_images]
        
        else:
            img, model_view_trnsfrm, mvp_trnsfrm, cam_posi = self._parse_frame(self.cfg, itr % self.n_images)

        return {
            'model_view_trnsfrm' : model_view_trnsfrm,
            'mvp_trnsfrm' : mvp_trnsfrm,
            'cam_posi' : cam_posi,
            'resolution' : self.FLAGS.train_res,
            'spp' : self.FLAGS.spp,
            'img' : img
        }
