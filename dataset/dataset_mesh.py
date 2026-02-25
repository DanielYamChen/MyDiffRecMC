# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch

from render import util
from render import mesh
from render import render
from render import light
import render.optixutils as ou

from dataset import Dataset

###############################################################################
# Reference dataset using mesh & rendering
###############################################################################

class DatasetMesh(Dataset):
    
    def __init__(self, ref_mesh, glctx, cam_radius, FLAGS, validate=False, num_valid_frames=200, phys_cam=None):
        # Init 
        self.glctx            = glctx
        self.cam_radius       = cam_radius
        self.FLAGS            = FLAGS
        self.validate         = validate
        self.fovy             = np.deg2rad(50)
        self.aspect           = FLAGS.train_res[1] / FLAGS.train_res[0] # img_w / img_h
        self.num_valid_frames = num_valid_frames
        
        if FLAGS.add_phys_cam:
            assert (phys_cam is not None), "phys_cam should be added but None found ..."
            self.phys_cam = phys_cam


        print("DatasetMesh: ref mesh has %d triangles and %d vertices" % (ref_mesh.t_pos_idx.shape[0], ref_mesh.v_pos.shape[0]))

        print("Build Optix bvh")
        self.optix_ctx = ou.OptiXContext()
        ou.optix_build_bvh(self.optix_ctx, ref_mesh.v_pos, ref_mesh.t_pos_idx.int(), rebuild=1)
        print("Done building OptiX bvh")


        # Sanity test training texture resolution
        ref_texture_res = np.maximum(ref_mesh.material['kd'].getRes(), ref_mesh.material['ks'].getRes())
        if 'normal' in ref_mesh.material:
            ref_texture_res = np.maximum(ref_texture_res, ref_mesh.material['normal'].getRes())
        if FLAGS.texture_res[0] < ref_texture_res[0] or FLAGS.texture_res[1] < ref_texture_res[1]:
            print("---> WARNING: Picked a texture resolution lower than the reference mesh [%d, %d] < [%d, %d]" % (FLAGS.texture_res[0], FLAGS.texture_res[1], ref_texture_res[0], ref_texture_res[1]))

        # Pre-randomize a list with finite number of training samples
        if (hasattr(FLAGS, 'train_examples') and (FLAGS.train_examples is not None)):
            self.train_examples = [self._random_scene() for i in range(FLAGS.train_examples)]
       
        self.ref_mesh = mesh.compute_tangents(ref_mesh)
        self.envlight = light.load_env(FLAGS.envlight, scale=FLAGS.env_scale)

        if (FLAGS.add_phys_cam):
            #### ---- phys_cam parameter setting (Flir Blackfly S BFS-U3-31S4C) ---- ####
            ## default control parameters
            self.aperture_num = 4.0 # [1/1], for the 5mm-lens, (F1.6, F1.79, F2, F2.83, F4, F5.66, F8, F16)
            self.expsr_time = 0.256 # [sec], (0.032, 0.064, 0.128, 0.256, 0.512, 1.024, 2.048)
            self.ISO = 100 * np.power(10, 0/20) # [%]
            self.focal_length = 0.005 # [m]
            self.focus_dist = 1.0 * cam_radius # [m]

    def getMesh(self):
        return self.ref_mesh

    
    def _rotate_scene(self, itr):
        proj_mtx = util.perspective(self.fovy, self.FLAGS.display_res[1] / self.FLAGS.display_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # Smooth rotation for display
        ang    = (itr / self.num_valid_frames) * np.pi * 2 
        model_view_trnsfrm     = util.translate(0, 0, -self.cam_radius) @ (util.rotate_x(-0.4) @ util.rotate_y(ang))
        mvp_trnsfrm    = proj_mtx @ model_view_trnsfrm
        cam_posi = torch.linalg.inv(model_view_trnsfrm)[:3, 3]

        return model_view_trnsfrm[None, ...].cuda(), mvp_trnsfrm[None, ...].cuda(), cam_posi[None, ...].cuda(), self.envlight, self.FLAGS.display_res, self.FLAGS.spp

    
    def _random_scene(self):
        # ==============================================================================================
        #  Setup projection matrix
        # ==============================================================================================
        iter_res = self.FLAGS.train_res
        proj_mtx = util.perspective(self.fovy, iter_res[1] / iter_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # ==============================================================================================
        #  Random camera & light position
        # ==============================================================================================

        # Random rotation/translation matrix for optimization
        model_view_trnsfrm = util.translate(0, 0, -self.cam_radius) @ util.random_rotation_translation(0.25)

        mvp_trnsfrm = proj_mtx @ model_view_trnsfrm
        cam_posi    = torch.linalg.inv(model_view_trnsfrm)[:3, 3]

        return model_view_trnsfrm[None, ...].cuda(), mvp_trnsfrm[None, ...].cuda(), cam_posi[None, ...].cuda(), self.envlight, iter_res, self.FLAGS.spp # Add batch dimension

    
    def __len__(self):
        return self.num_valid_frames if self.validate else (self.FLAGS.iter + 0) * self.FLAGS.batch_size

    
    def __getitem__(self, itr):
        
        # ==============================================================================================
        #  Randomize scene parameters
        # ==============================================================================================
        if self.validate:
            model_view_trnsfrm, mvp_trnsfrm, cam_posi, lgt, iter_res, iter_spp = self._rotate_scene(itr)
        
        else:
            if hasattr(self, 'train_examples'):
                model_view_trnsfrm, mvp_trnsfrm, cam_posi, lgt, iter_res, iter_spp = self.train_examples[itr % len(self.train_examples)]
            
            else:
                model_view_trnsfrm, mvp_trnsfrm, cam_posi, lgt, iter_res, iter_spp = self._random_scene()

        # Post-mixing in background causes a small anti-aliasing error
        buffer = render.render_mesh(
            self.FLAGS,
            self.glctx,
            self.ref_mesh,
            mvp_trnsfrm,
            cam_posi,
            lgt,
            iter_res,
            None,
            torch.tensor([[self.aperture_num, self.expsr_time, self.ISO, self.focal_length, self.focus_dist]], dtype=torch.float32),
            spp=iter_spp,
            num_layers=self.FLAGS.layers,
            msaa=True,
            background=None,
            optix_ctx=self.optix_ctx,
            phys_cam=self.phys_cam,
        ) 
        ## buffer = {
        ##     'shaded': torch(1, img_h, img_w, 4),
        ##     'z_grad': torch(1, img_h, img_w, 4),
        ##     'normal': torch(1, img_h, img_w, 4),
        ##     'geometric_normal': torch(1, img_h, img_w, 4),
        ##     'kd': torch(1, img_h, img_w, 4),
        ##     'ks': torch(1, img_h, img_w, 4),
        ##     'kd_grad': torch(1, img_h, img_w, 4),
        ##     'ks_grad': torch(1, img_h, img_w, 4),
        ##     'normal_grad': torch(1, img_h, img_w, 4),
        ##     'diffuse_light': torch(1, img_h, img_w, 4),
        ##     'specular_light': torch(1, img_h, img_w, 4),
        ## }
        # print(buffer.keys())
        # for key in buffer:
        #     print(buffer[key].shape)

        img = buffer['shaded']

        return {
            'model_view_trnsfrm' : model_view_trnsfrm,
            'mvp_trnsfrm' : mvp_trnsfrm,
            'cam_posi' : cam_posi,
            'light' : lgt,
            'resolution' : iter_res,
            'spp' : iter_spp,
            'img' : img,
            'kerker': str(itr),
            'cam_ctrl_params': torch.tensor([[self.aperture_num, self.expsr_time, self.ISO, self.focal_length, self.focus_dist]], dtype=torch.float32),
        }

    
    def collate(self, batch):
        iter_res, iter_spp = batch[0]['resolution'], batch[0]['spp']
        out_batch = {
            'model_view_trnsfrm' : torch.cat(list([item['model_view_trnsfrm'] for item in batch]), dim=0),
            'mvp_trnsfrm' : torch.cat(list([item['mvp_trnsfrm'] for item in batch]), dim=0),
            'cam_posi' : torch.cat(list([item['cam_posi'] for item in batch]), dim=0),
            'resolution' : iter_res,
            'spp' : iter_spp,
            'light' : batch[0]['light'],
            'img' : torch.cat(list([item['img'] for item in batch]), dim=0),
            'kerker': [item['kerker'] for item in batch],
            'cam_ctrl_params': torch.cat(list([item['cam_ctrl_params'] for item in batch]), dim=0),
        }
        return out_batch

