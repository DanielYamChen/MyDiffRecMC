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
from torch.utils.data.sampler import BatchSampler, SequentialSampler
import numpy as np
import matplotlib.pyplot as plt

from render import util
from render import light
from dataset import Dataset

###############################################################################
# Custom dataset
###############################################################################

def _load_img(path):

    img = util.load_image_raw(path)
    img = img.astype(np.float32) / np.iinfo(img.dtype).max
    # if (correct_expsr):
    #     if (path[-8 : -4] == "0512"):
    #         img[:, :, 0 : 3] = img[:, :, 0 : 3] / 2.0
    #     elif (path[-8 : -4] == "0128"):
    #         img[:, :, 0 : 3] = img[:, :, 0 : 3] * 2.0

    assert (len(img) > 0), "dataset_custom.py: Failed to find image for: %s" % (path)
    img = torch.tensor(img, dtype=torch.float32)
    
    return img


class InOrderBatchSampler(BatchSampler):
    '''
    Divide data in order into batches and shuffle order of batches for changing environment lights
    for different batches
    '''
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = list(range(len(self.dataset)))

    def __iter__(self):
        # Divide the dataset into ordered batches
        batches = [self.indices[i : (i + self.batch_size)] for i in range(0, len(self.dataset), self.batch_size)]
        
        # Randomly shuffle the order of the batches
        random_order = torch.randperm(len(batches)).tolist()

        # print(f"len(self.dataset)={len(self.dataset)}")
        # print(f"len(batches)={len(batches)}")
        # print(random_order)
        # assert (False)
        
        for idx in random_order:
            yield batches[idx]

    def __len__(self):
        return len(self.indices) // self.batch_size


class DatasetCustom(Dataset):
    def __init__(self, cfg_path, FLAGS, num_samples=None):
        self.FLAGS = FLAGS
        self.num_samples = num_samples
        self.base_dir = os.path.dirname(cfg_path)
        self.dataset_name = self.base_dir.split(os.sep)[-1]
        self.data_dir = FLAGS.gt_img_dir
        # self.data_dir = "train_imgs_wo_ground/" if "train" in cfg_path else "test_imgs/"
        # self.data_dir = "train_imgs_1024/" if "train" in cfg_path else "test_imgs/"
        # self.data_dir = "train_imgs/" if "train" in cfg_path else "test_imgs/"
        # self.data_dir = "train_imgs_part2/" if "train" in cfg_path else "test_imgs/"

        # Load config / transforms
        if (self.dataset_name == "experiment"):
            if ("ground_added" in self.data_dir):
                self.cfg = json.load(open(cfg_path[:-5] + "_ground_added.json", 'r'))
            elif ("wo_ground" in self.data_dir):
                self.cfg = json.load(open(cfg_path[:-5] + "_wo_ground.json", 'r'))
        else:
            self.cfg = json.load(open(cfg_path, 'r'))
        
        self.n_images = len(self.cfg['frames'])

        # Determine resolution & aspect ratio

        if (self.dataset_name == "custom"):
            ini_img_path = os.path.join(self.base_dir, self.data_dir, "img0000.png")
        elif (self.dataset_name == "experiment"):
            ini_img_path = os.path.join(self.base_dir, self.data_dir, "sun_180_polar_040_azi_000_U_middle_t_0128.png")
        elif (self.dataset_name == "sim_scenes"):
            # ini_img_path = os.path.join(self.base_dir, self.data_dir, "sun_000_polar_000_azi_000_U_middle_t_0128.png")
            paths = glob.glob(os.path.join(self.base_dir, self.data_dir, "*.png"))
            paths.sort()
            ini_img_path = paths[0]
        else:
            assert(False), f"dataset_custom.py: unknown dataset name: {self.dataset_name}"

        self.resolutions = _load_img(ini_img_path).shape[0:2]
        self.aspect = self.resolutions[1] / self.resolutions[0]

        print(f"DatasetCustom in {self.data_dir}: {self.n_images} images with shape {self.resolutions}")

        # Pre-load from disc to avoid slow png parsing
        if (self.FLAGS.pre_load):
            self.preloaded_data = []
            for i in range(self.n_images):
                self.preloaded_data += [self._parse_frame(self.cfg, i)]

    def _parse_frame(self, cfg, idx):
        # Config projection matrix (static, so could be precomputed)
        fovy = util.fovx_to_fovy(cfg['fov_x'], self.aspect)
        proj = util.perspective(fovy, self.aspect, self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # Load image and model-view matrix
        if (self.dataset_name == "custom"):
            img_path = os.path.join(self.base_dir, self.data_dir, "img" + str(idx).zfill(4) + ".png")
        elif (self.dataset_name in ["experiment", "sim_scenes"]):
            img_path = os.path.join(self.base_dir, self.data_dir, cfg['frames'][str(idx)]["img_name"])
        else:
            assert(False), f"dataset_custom.py: unknown dataset name: {self.dataset_name}"

        img = _load_img(img_path)
        cam_trnsfrm = torch.cat(
            (torch.tensor(cfg['frames'][str(idx)]["trnsfrm"], dtype=torch.float32),
             torch.tensor([[0., 0., 0., 1.]], dtype=torch.float32)),
            dim=0,
        )
        cam_trnsfrm = (util.rotate_x(np.pi / 2) @ cam_trnsfrm) @ util.rotate_y(np.pi)

        model_view_trnsfrm = torch.linalg.inv(cam_trnsfrm) # 4 x 4 matrix
        cam_posi = cam_trnsfrm[:3, 3]
        mvp_trnsfrm = proj @ model_view_trnsfrm

        cam_ctrl_params = torch.tensor(cfg['frames'][str(idx)]['cam_ctrl_params'], dtype=torch.float32)
        defocus_mtrx_name = cfg['frames'][str(idx)]['defocus_mtrx_name']
        envmap_name = cfg['frames'][str(idx)]['envmap_name']

        # Add batch dimension
        return img[None, ...], model_view_trnsfrm[None, ...], mvp_trnsfrm[None, ...], cam_posi[None, ...], cam_ctrl_params[None, ...], defocus_mtrx_name, envmap_name

    def getMesh(self):
        return None # There is no mesh

    def __len__(self):
        return self.n_images if self.num_samples is None else self.num_samples

    def __getitem__(self, itr):
        img      = []
        # fovy     = util.fovx_to_fovy(self.cfg['camera_angle_x'], self.aspect)

        if (self.FLAGS.pre_load):
            img, model_view_trnsfrm, mvp_trnsfrm, cam_posi, cam_ctrl_params, defocus_mtrx_name, envmap_name = self.preloaded_data[itr % self.n_images]
        
        else:
            img, model_view_trnsfrm, mvp_trnsfrm, cam_posi, cam_ctrl_params, defocus_mtrx_name, envmap_name = self._parse_frame(self.cfg, itr % self.n_images)

        return {
            'model_view_trnsfrm' : model_view_trnsfrm,
            'mvp_trnsfrm' : mvp_trnsfrm,
            'cam_posi' : cam_posi,
            'resolution' : self.FLAGS.train_res,
            'spp' : self.FLAGS.spp,
            'img' : img,
            'cam_ctrl_params': cam_ctrl_params,
            'defocus_mtrx_name': defocus_mtrx_name,
            'envmap_name': envmap_name,
        }
    
    def collate(self, batch):
        iter_res, iter_spp = batch[0]['resolution'], batch[0]['spp']

        ## check this batch has the same envmap_name
        for item_idx in range(1, len(batch)):
            assert (batch[0]['envmap_name'] == batch[item_idx]['envmap_name']), "Batch has different envmap_name ..."
        
        # print(batch[0]['envmap_name'])

        envmap_path = self.FLAGS.envlight + batch[0]['envmap_name'] + ".hdr"
        return {
            'model_view_trnsfrm' : torch.cat(list([item['model_view_trnsfrm'] for item in batch]), dim=0),
            'mvp_trnsfrm' : torch.cat(list([item['mvp_trnsfrm'] for item in batch]), dim=0),
            'cam_posi' : torch.cat(list([item['cam_posi'] for item in batch]), dim=0),
            'resolution' : iter_res,
            'spp' : iter_spp,
            'img' : torch.cat(list([item['img'] for item in batch]), dim=0),
            'cam_ctrl_params': torch.cat(list([item['cam_ctrl_params'] for item in batch]), dim=0),
            'defocus_mtrx_name': [item['defocus_mtrx_name'] for item in batch],
            'light': light.load_env(envmap_path, scale=self.FLAGS.env_scale) if (self.FLAGS.fix_envlight == False) else None,
            'envmap_name': batch[0]['envmap_name'], # debug
        }
