# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import glob

import torch
import numpy as np

from render import util

from dataset import Dataset

def _load_mask(fn):
    img = torch.tensor(util.load_image(fn), dtype=torch.float32)
    if len(img.shape) == 2:
        img = img[..., None].repeat(1, 1, 3)
    return img

def _load_img(fn):
    img = util.load_image_raw(fn)
    if (img.dtype != np.float32): # LDR image
        img = torch.tensor(img.astype(np.float32) / np.iinfo(img.dtype).max, dtype=torch.float32)
        img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3])
    else:
        img = torch.tensor(img, dtype=torch.float32)
    return img

###############################################################################
# LLFF datasets (real world camera lightfields)
###############################################################################

class DatasetLLFF(Dataset):
    def __init__(self, base_dir, FLAGS, examples=None):
        self.FLAGS = FLAGS
        self.base_dir = base_dir
        self.examples = examples

        # Enumerate all image files and get resolution
        all_img = [f for f in sorted(glob.glob(os.path.join(self.base_dir, "images", "*"))) 
                   if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]

        self.resolution = _load_img(all_img[0]).shape[0:2]

        print("DatasetLLFF: %d images with shape [%d, %d]" % (len(all_img), self.resolution[0], self.resolution[1]))

        ## Load camera poses
        ## poses_bounds: (num_photos, 17), 3 x 3 Rotation Matrix, 3 x 1 Translation Vector, 1 x 4 Intrinsics/Additional Info
        ## [r00, r01, r02, t0, sensor_height, r10, r11, r12, t1, sensor_width, r20, r21, r22, t2, focal_length]
        poses_bounds = np.load(os.path.join(self.base_dir, 'poses_bounds.npy'))
        
        ## poses: (3, 5, num_photos)
        ## pose: [r00, r01, r02, t0, sensor_height;
        ##        r10, r11, r12, t1, sensor_width;
        ##        r20, r21, r22, t2, focal_length]
        poses        = poses_bounds[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
        ## switch (- Col 1) and Col 2
        poses        = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1) # Taken from nerf, swizzles from LLFF to expected coordinate system
        ## poses: (num_photos, 3, 5)
        poses        = np.moveaxis(poses, -1, 0).astype(np.float32)
        
        last_row = np.array([0,0,0,1], dtype=np.float32)[None, None, :].repeat(poses.shape[0], 0)
        self.cam_trnsfrm    = torch.tensor(np.concatenate((poses[:, :, 0 : 4], last_row), axis=1), dtype=torch.float32)
        self.aspect  = self.resolution[1] / self.resolution[0] # width / height
        self.fovy    = util.focal_length_to_fovy(poses[:, 2, 4], poses[:, 0, 4])

        # Recenter scene so lookat position is origin
        center                = util.lines_focal(self.cam_trnsfrm[..., 0 : 3, 3], -self.cam_trnsfrm[..., 0 : 3, 2])
        self.cam_trnsfrm[..., 0 : 3, 3] = self.cam_trnsfrm[..., 0 : 3, 3] - center[None, ...]
        print("DatasetLLFF: auto-centering at %s" % (center.cpu().numpy()))

        # Pre-load from disc to avoid slow png parsing
        if self.FLAGS.pre_load:
            self.preloaded_data = []
            for i in range(self.cam_trnsfrm.shape[0]):
                self.preloaded_data += [self._parse_frame(i)]

    def _parse_frame(self, idx):
        all_img_paths  = [f for f in sorted(glob.glob(os.path.join(self.base_dir, "images", "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
        all_mask_paths = [f for f in sorted(glob.glob(os.path.join(self.base_dir, "masks", "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
        assert (len(all_img_paths) == self.cam_trnsfrm.shape[0]) and (len(all_mask_paths) == self.cam_trnsfrm.shape[0])

        # Load image & mask data
        img  = _load_img(all_img_paths[idx])
        mask = _load_mask(all_mask_paths[idx])
        img  = torch.cat((img, mask[..., 0:1]), dim=-1)

        # Setup transforms
        # (4, 4)
        proj = util.perspective(self.fovy[idx, ...], self.aspect, self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])
        # (4, 4)
        model_view_trnsfrm = torch.linalg.inv(self.cam_trnsfrm[idx, ...])
        # (3, 1)
        cam_posi = torch.linalg.inv(model_view_trnsfrm)[:3, 3]
        # (4, 4)
        model_view_proj_trnsfrm = proj @ model_view_trnsfrm

        # Add batch dimension
        return img[None, ...], model_view_trnsfrm[None, ...], model_view_proj_trnsfrm[None, ...], cam_posi[None, ...]

    def getMesh(self):
        return None # There is no mesh

    def __len__(self):
        return self.cam_trnsfrm.shape[0] if self.examples is None else self.examples

    def __getitem__(self, itr):
        if self.FLAGS.pre_load:
            img, model_view_trnsfrm, model_view_proj_trnsfrm, cam_posi = self.preloaded_data[itr % self.cam_trnsfrm.shape[0]]
        
        else:
            img, model_view_trnsfrm, model_view_proj_trnsfrm, cam_posi = self._parse_frame(itr % self.cam_trnsfrm.shape[0])

        return {
            'model_view_trnsfrm' : model_view_trnsfrm,
            'mvp_trnsfrm' : model_view_proj_trnsfrm,
            'cam_posi' : cam_posi,
            'resolution' : self.resolution,
            'spp' : self.FLAGS.spp,
            'img' : img,
        }
