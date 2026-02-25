# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import argparse
import json
import multiprocessing
import glob
import re

import numpy as np
from math import pi
import torch
import nvdiffrast.torch as dr
import xatlas
# import igl
import random
import multiprocessing
import shutil
import matplotlib.pyplot as plt

# Import data readers / generators
from dataset import DatasetMesh, DatasetNERF, DatasetLLFF, DatasetCustom, InOrderBatchSampler

# Import topology / geometry trainers
from geometry.dmtet import DMTetGeometry
from geometry.dlmesh import DLMesh

import render.renderutils as ru
from render import obj
from render import material
from render import util
from render import mesh
from render import texture
from render import mlptexture
from render import light
from render import render
from render.mesh import Mesh

from denoiser.denoiser import BilateralDenoiser
from models.cam_model import PhysDiffCamera
from models import networks

RADIUS = 3.0

# Enable to debug back-prop anomalies
#torch.autograd.set_detect_anomaly(True)

###############################################################################
# Loss setup
###############################################################################

@torch.no_grad()
def createLoss(FLAGS):
    if FLAGS.loss == "smape":
        return lambda img, ref: ru.image_loss(img, ref, loss='smape', tonemapper='none')
    elif FLAGS.loss == "mse":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='none')
    elif FLAGS.loss == "logl1":
        return lambda img, ref: ru.image_loss(img, ref, loss='l1', tonemapper='log_srgb')
    elif FLAGS.loss == "logl2":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='log_srgb')
    elif FLAGS.loss == "relativel2":
        return lambda img, ref: ru.image_loss(img, ref, loss='relmse', tonemapper='none')
    elif FLAGS.loss == "n2n":
        return lambda img, ref: ru.image_loss(img, ref, loss='n2n', tonemapper='none')
    else:
        assert False

###############################################################################
# Mix background into a dataset image
###############################################################################

@torch.no_grad()
def prepare_batch(target, train_res, bg_type):

    target['model_view_trnsfrm'] = target['model_view_trnsfrm'].cuda()
    target['mvp_trnsfrm'] = target['mvp_trnsfrm'].cuda()
    target['cam_posi'] = target['cam_posi'].cuda()
    target['img'] = target['img'].cuda()

    ## rescale batch of target images to align with train-resolutions
    if (train_res[0] != target['img'].shape[1] or train_res[1] != target['img'].shape[2]):
        target['img'] = util.scale_img_nhwc(target['img'], train_res)
        target['resolution'] = train_res

    assert (len(target['img'].shape) == 4), "Image shape should be [n, h, w, c]"
    if bg_type == 'checker':
        background = torch.tensor(util.checkerboard(target['img'].shape[1:3], 8), dtype=torch.float32, device='cuda')[None, ...]
    elif bg_type == 'black':
        background = torch.zeros(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'white':
        background = torch.ones(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'reference':
        background = target['img'][..., 0:3]
    elif bg_type == 'random':
        background = torch.rand(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    else:
        assert False, "Unknown background type %s" % bg_type

    target['background'] = background
    target['img'] = torch.cat((torch.lerp(background, target['img'][..., 0:3], target['img'][..., 3:4]), target['img'][..., 3:4]), dim=-1)

    return target

###############################################################################
# UV - map geometry & convert to a mesh
###############################################################################

@torch.no_grad()
def xatlas_uvmap(glctx, eval_mesh, indices, uvs, mat, FLAGS):
    # Convert to tensors
    indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
    
    uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
    faces = torch.tensor(indices_int64, dtype=torch.int64, device='cuda')

    new_mesh = mesh.Mesh(v_tex=uvs, t_tex_idx=faces, base=eval_mesh)

    mask, kd, ks = render.render_uv(glctx, new_mesh, FLAGS.texture_res, eval_mesh.material['kd_ks'])

    # Dilate all textures & use average color for background
    kd_avg = torch.sum(torch.sum(torch.sum(kd * mask, dim=0), dim=0), dim=0) / torch.sum(torch.sum(torch.sum(mask, dim=0), dim=0), dim=0)
    kd = util.dilate(kd, kd_avg[None, None, None, :], mask, 7)

    ks_avg = torch.sum(torch.sum(torch.sum(ks * mask, dim=0), dim=0), dim=0) / torch.sum(torch.sum(torch.sum(mask, dim=0), dim=0), dim=0)
    ks = util.dilate(ks, ks_avg[None, None, None, :], mask, 7)

    nrm_avg = torch.tensor([0, 0, 1], dtype=torch.float32, device="cuda")
    normal = nrm_avg[None, None, None, :].repeat(kd.shape[0], kd.shape[1], kd.shape[2], 1)
    
    new_mesh.material = mat.copy()
    del new_mesh.material['kd_ks']

    if FLAGS.transparency:
        kd = torch.cat((kd, torch.rand_like(kd[...,0:1])), dim=-1)
        print("kd shape", kd.shape)

    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')
    new_mesh.material.update({
        'kd'     : texture.Texture2D(kd.clone().detach().requires_grad_(True), min_max=[kd_min, kd_max]),
        'ks'     : texture.Texture2D(ks.clone().detach().requires_grad_(True), min_max=[ks_min, ks_max]),
        'normal' : texture.Texture2D(normal.clone().detach().requires_grad_(True), min_max=[nrm_min, nrm_max]),
    })

    return new_mesh

###############################################################################
# Utility functions for material
###############################################################################

def initial_guess_material(geometry, mlp, FLAGS, init_mat=None):
    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')
    if mlp:
        mlp_min = torch.cat((kd_min[0:3], ks_min), dim=0)
        mlp_max = torch.cat((kd_max[0:3], ks_max), dim=0)
        mlp_map_opt = mlptexture.MLPTexture3D(geometry.getAABB(), channels=6, min_max=[mlp_min, mlp_max])
        mat = {'kd_ks' : mlp_map_opt}
    else:
        # Setup Kd, Ks albedo, and specular textures
        if init_mat is None:
            num_channels = 4 if (FLAGS.layers > 1) else 3
            kd_init = torch.ones(size=FLAGS.texture_res + [num_channels], device='cuda') * (kd_max - kd_min)[None, None, 0:num_channels] + kd_min[None, None, 0:num_channels]
            kd_map_opt = texture.create_trainable(kd_init , FLAGS.texture_res, not FLAGS.custom_mip, [kd_min, kd_max])

            ksR = np.random.uniform(size=FLAGS.texture_res + [1], low=0.0, high=0.01)
            ksG = np.random.uniform(size=FLAGS.texture_res + [1], low=ks_min[1].cpu(), high=ks_max[1].cpu())
            ksB = np.random.uniform(size=FLAGS.texture_res + [1], low=ks_min[2].cpu(), high=ks_max[2].cpu())

            ks_map_opt = texture.create_trainable(np.concatenate((ksR, ksG, ksB), axis=2), FLAGS.texture_res, not FLAGS.custom_mip, [ks_min, ks_max])
        else:
            kd_map_opt = texture.create_trainable(init_mat['kd'], FLAGS.texture_res, not FLAGS.custom_mip, [kd_min, kd_max])
            ks_map_opt = texture.create_trainable(init_mat['ks'], FLAGS.texture_res, not FLAGS.custom_mip, [ks_min, ks_max])

        # Setup normal map
        if init_mat is None or 'normal' not in init_mat:
            normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])
        else:
            normal_map_opt = texture.create_trainable(init_mat['normal'], FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])

        mat = {
            'kd'     : kd_map_opt,
            'ks'     : ks_map_opt,
            'normal' : normal_map_opt
        }

    mat['bsdf'] = FLAGS.bsdf

    mat['no_perturbed_nrm'] = FLAGS.no_perturbed_nrm

    return mat

###############################################################################
# Validation & testing
###############################################################################

def validate_itr(glctx, batch_targets, ref_mesh, geometry, opt_material, lgt, FLAGS, denoiser, iter=0, phys_cam=None, defocus_net=None):
    result_dict = {}
    with torch.no_grad():
        opt_mesh = geometry.getMesh(opt_material)
        
        ## debug
        # print(f"\n opt_mesh={opt_mesh} \n")
        # print(f"\n batch_targets['cam_trnsfrm']={torch.linalg.inv(batch_targets['model_view_trnsfrm'][0])} \n")
        # print(f"\n batch_targets['cam_posi']={batch_targets['cam_posi']} \n")
        # print(f"\n batch_targets['envmap_name']={batch_targets['envmap_name']} \n")

        buffers = render.render_mesh(
            FLAGS,
            glctx,
            opt_mesh,
            batch_targets['mvp_trnsfrm'],
            batch_targets['cam_posi'],
            batch_targets['light'] if (lgt is None) else lgt,
            batch_targets['resolution'],
            batch_targets['defocus_mtrx_name'] if ('defocus_mtrx_name' in batch_targets) else None,
            batch_targets['cam_ctrl_params'],
            spp=batch_targets['spp'],
            num_layers=FLAGS.layers,
            background=batch_targets['background'],
            optix_ctx=geometry.optix_ctx,
            denoiser=denoiser,
            phys_cam=phys_cam,
            defocus_net=defocus_net,
        )
        ## debug
        # temp = buffers['shaded'].detach().cpu().numpy()
        # plt.imshow(temp[0, :, :, :3])
        # plt.show()
        # plt.imshow(temp[0, :, :, 3], cmap='gray')
        # plt.show()

        ## mask out rendered pixels outside of ROI
        buffers['shaded'][0, :, :, 0:4][buffers['shaded'][0, :, :, 3] < 0.99] = 0.
        # print(buffers['shaded'][0, :, :, 3] > 0.99)
        num_effect_pxs = np.sum((buffers['shaded'][0, :, :, 3] > 0.99).detach().cpu().numpy())

        result_dict['ref'] = util.rgb_to_srgb(batch_targets['img'][0, :, :, :])
        result_dict['opt'] = util.rgb_to_srgb(buffers['shaded'][0, :, :, :])
        result_image = torch.cat([result_dict['opt'], result_dict['ref']], axis=1)

        if (FLAGS.display is not None):
            white_bg = torch.ones_like(batch_targets['background'])
            for layer in FLAGS.display:
                if ('latlong' in layer and layer['latlong']):
                    result_dict['light_image'] = lgt.generate_image(FLAGS.display_res)
                    result_dict['light_image'] = util.rgb_to_srgb(result_dict['light_image'] / (1 + result_dict['light_image']))
                    result_image = torch.cat([result_image, result_dict['light_image']], axis=1)
                
                elif ('bsdf' in layer):
                    img = render.render_mesh(
                        FLAGS,
                        glctx,
                        opt_mesh,
                        batch_targets['mvp_trnsfrm'],
                        batch_targets['cam_posi'],
                        batch_targets['light'] if lgt is None else lgt,
                        batch_targets['resolution'],
                        batch_targets['defocus_mtrx_name'] if ('defocus_mtrx_name' in batch_targets) else None,
                        batch_targets['cam_ctrl_params'],
                        spp=batch_targets['spp'],
                        num_layers=FLAGS.layers,
                        background=white_bg,
                        bsdf=layer['bsdf'],
                        optix_ctx=geometry.optix_ctx,
                        phys_cam=phys_cam,
                        defocus_net=defocus_net,
                    )['shaded']
                    
                    if (layer['bsdf'] == 'kd'):
                        result_dict[layer['bsdf']] = util.rgb_to_srgb(img[..., 0:3])[0]
                    else:
                        result_dict[layer['bsdf']] = img[0, ..., 0:3]
                    
                    result_image = torch.cat([result_image, result_dict[layer['bsdf']]], axis=1)
                    
                    if (ref_mesh is not None):
                        img = render.render_mesh(
                            FLAGS,
                            glctx,
                            ref_mesh,
                            batch_targets['mvp_trnsfrm'],
                            batch_targets['cam_posi'],
                            batch_targets['light'],
                            batch_targets['resolution'],
                            batch_targets['defocus_mtrx_name'] if ('defocus_mtrx_name' in batch_targets) else None,
                            batch_targets['cam_ctrl_params'],
                            spp=batch_targets['spp'],
                            num_layers=FLAGS.layers,
                            background=white_bg,
                            bsdf=layer['bsdf'],
                            optix_ctx=geometry.optix_ctx,
                            phys_cam=phys_cam,
                            defocus_net=defocus_net,
                        )['shaded']

                        if (layer['bsdf'] == 'kd'):
                            result_dict[layer['bsdf'] + "_ref"] = util.rgb_to_srgb(img[..., 0:3])[0]
                        else:
                            result_dict[layer['bsdf'] + "_ref"] = img[0, ..., 0:3]
                        
                        result_image = torch.cat([result_image, result_dict[layer['bsdf'] + "_ref"]], axis=1)
                
                elif ('normals' in layer and not FLAGS.no_perturbed_nrm):
                    result_image = torch.cat([result_image, (buffers['perturbed_nrm'][0, ...,0:3] + 1.0) * 0.5], axis=1)
                
                elif ('diffuse_light' in layer):
                    result_image = torch.cat([result_image, util.rgb_to_srgb(buffers['diffuse_light'][..., 0:3])[0]], axis=1)
                
                elif ('specular_light' in layer):
                    result_image = torch.cat([result_image, util.rgb_to_srgb(buffers['specular_light'][..., 0:3])[0]], axis=1)


        return result_image, result_dict, num_effect_pxs


def validate(glctx, geometry, opt_material, lgt, dataset_valid, out_dir, FLAGS, denoiser, phys_cam=None, defocus_net=None):

    # ==============================================================================================
    #  Validation loop
    # ==============================================================================================
    img_cnt = 0
    mse_values = []
    psnr_values = []

    # Hack validation to use high sample count and no denoiser
    _n_samples = FLAGS.n_samples
    _denoiser = denoiser
    FLAGS.n_samples = 32
    denoiser = None

    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=1, collate_fn=dataset_valid.collate)

    os.makedirs(out_dir, exist_ok=True)
    lines = []
    print("Running validation")
    for it, target in enumerate(dataloader_valid):

        # Mix validation background
        target = prepare_batch(target, FLAGS.train_res, FLAGS.background)
        if (FLAGS.add_defocus_net):
            defocus_net.eval()
        result_image, result_dict, num_effect_pxs = validate_itr(
            glctx, target, dataset_valid.getMesh(), geometry, opt_material, lgt, FLAGS, denoiser,
            phys_cam=phys_cam, defocus_net=defocus_net,
        )
        if (FLAGS.add_defocus_net):
            defocus_net.train()
        
        # Compute metrics
        opt = torch.clamp(result_dict['opt'][:, :, 0:3], 0.0, 1.0) 
        ref = torch.clamp(result_dict['ref'][:, :, 0:3], 0.0, 1.0)

        mse = torch.nn.functional.mse_loss(opt, ref, size_average=None, reduce=None, reduction='sum').item()
        mse = mse / num_effect_pxs
        mse_values.append(float(mse))
        psnr = util.mse_to_psnr(mse)
        psnr_values.append(float(psnr))

        lines.append(str("%d, %1.8f, %1.8f \n" % (it, mse, psnr)))

        for k in result_dict.keys():
            if ((k == "ref") and (FLAGS.save_gt_test_imgs == False)):
                continue
            else:
                np_img = result_dict[k].detach().cpu().numpy()
                util.save_image(out_dir + '/' + ('val_%s_%06d.png' % (k, it)), np_img)

    avg_mse = np.mean(np.array(mse_values))
    avg_psnr = np.mean(np.array(psnr_values))
    lines.append(str("AVERAGES: %1.4f, %2.3f\n" % (avg_mse, avg_psnr)))
    
    print("MSE,      PSNR")
    print("%1.8f, %2.3f" % (avg_mse, avg_psnr))
    
    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as fout:
        fout.write('ID, MSE, PSNR\n')
        for line in lines:
            fout.write(line)

        

    # Restore sample count and denoiser
    FLAGS.n_samples = _n_samples
    denoiser = _denoiser

    return avg_psnr

###############################################################################
# Main shape fitter function / optimization loop
###############################################################################

def optimize_mesh(
    denoiser,
    glctx,
    glctx_display,
    geometry,
    opt_material,
    lgt,
    dataset_train,
    dataset_valid,
    FLAGS,
    warmup_iter=0,
    log_interval=20,
    pass_idx=0,
    pass_name="",
    optimize_light=True,
    optimize_geometry=True,
    phys_cam=None,
    defocus_net=None,
    ):

    # ==============================================================================================
    #  Setup torch optimizer
    # ==============================================================================================

    learning_rate = FLAGS.learning_rate[pass_idx] if isinstance(FLAGS.learning_rate, list) or isinstance(FLAGS.learning_rate, tuple) else FLAGS.learning_rate
    learning_rate_pos = learning_rate[0] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate
    learning_rate_mat = learning_rate[1] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate
    learning_rate_lgt = learning_rate[2] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate * 3.0

    def lr_schedule(iter, fraction):
        if (iter < warmup_iter):
            return iter / warmup_iter 
        else:
            return max(0.0, 10**(-(iter - warmup_iter)*0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs

    trainable_list = material.get_parameters(opt_material)

    if (optimize_light == True and FLAGS.fix_envlight == True):
        optimizer_light = torch.optim.Adam((lgt.parameters() if lgt is not None else []), lr=learning_rate_lgt)
        scheduler_light = torch.optim.lr_scheduler.LambdaLR(optimizer_light, lr_lambda=lambda x: lr_schedule(x, 0.9)) 

    if (optimize_geometry):
        optimizer_mesh = geometry.getOptimizer(learning_rate_pos)
        scheduler_mesh = torch.optim.lr_scheduler.LambdaLR(optimizer_mesh, lr_lambda=lambda x: lr_schedule(x, 0.9)) 

    if (FLAGS.add_defocus_net):
        optimizer_defocus = torch.optim.Adam(defocus_net.parameters(), lr=FLAGS.defocus_net_lr)
        scheduler_defocus = torch.optim.lr_scheduler.LambdaLR(optimizer_defocus, lr_lambda=lambda x: lr_schedule(x, 0.9)) 
        
    
    optimizer = torch.optim.Adam(trainable_list, lr=learning_rate_mat)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x, 0.9)) 

    # ==============================================================================================
    #  Image loss
    # ==============================================================================================
    image_loss_fn = createLoss(FLAGS)

    # ==============================================================================================
    #  Training loop
    # ==============================================================================================
    img_cnt = 0
    img_loss_vec = []
    reg_loss_vec = []
    iter_dur_vec = []

    if (FLAGS.learn_lighting == False and FLAGS.fix_envlight == False):
        dataloader_train = torch.utils.data.DataLoader(dataset_train, collate_fn=dataset_train.collate, batch_sampler=InOrderBatchSampler(dataset_train, FLAGS.batch_size))
        dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=1, collate_fn=dataset_train.collate, shuffle=True)
    else:
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=FLAGS.batch_size, collate_fn=dataset_train.collate, shuffle=True)
        dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=1, collate_fn=dataset_train.collate)

    def cycle(iterable):
        iterator = iter(iterable)
        while True:
            try:
                yield next(iterator)
            
            except StopIteration:
                iterator = iter(iterable)

    v_it = cycle(dataloader_valid)

    # Creates a GradScaler once at the beginning of training
    for it, batch_targets in enumerate(dataloader_train):

        # Mix randomized background into dataset image
        batch_targets = prepare_batch(batch_targets, FLAGS.train_res, 'random')

        # ==============================================================================================
        #  Display / save outputs. Do it before training so we get initial meshes
        # ==============================================================================================

        # Show/save image before training step (want to get correct rendering of input)
        display_image = FLAGS.display_interval and (it % FLAGS.display_interval == 0)
        save_image = FLAGS.save_interval and (it % FLAGS.save_interval == 0)
        if (display_image or save_image):
            if (defocus_net is not None):
                defocus_net.eval()

            result_image, result_dict, num_effect_pxs = validate_itr(
                glctx_display,
                prepare_batch(next(v_it), FLAGS.train_res, FLAGS.background),
                dataset_valid.getMesh(),
                geometry,
                opt_material,
                lgt,
                FLAGS,
                denoiser,
                iter=it,
                phys_cam=phys_cam,
                defocus_net=defocus_net,
            )
            if (defocus_net is not None):
                defocus_net.train()

            np_result_image = result_image.detach().cpu().numpy()
            if (display_image):
                util.display_image(np_result_image, title='%d / %d' % (it, FLAGS.iter))
            
            if (save_image):
                util.save_image(FLAGS.out_dir + '/' + ('img_%s_%06d.png' % (pass_name, img_cnt)), np_result_image)
                img_cnt = img_cnt + 1

        optimizer.zero_grad()
        if optimize_geometry:
            optimizer_mesh.zero_grad()
        
        if optimize_light:
            optimizer_light.zero_grad()

        if (defocus_net is not None):
            optimizer_defocus.zero_grad()

        # ==============================================================================================
        #  Initialize training
        # ==============================================================================================
        iter_start_time = time.time()

        # ==============================================================================================
        #  Geometry-specific training
        # ==============================================================================================
        if (optimize_light == True and FLAGS.fix_envlight == True):
            lgt.update_pdf()

        ## debug
        # print(target.keys())
        # for key in target:
        #     if (type(target[key]) == torch.Tensor):
        #         print(key, ":", target[key].shape)
        #     elif (type(target[key]) == list):
        #         print(key, ":", target[key])
        #     else:
        #         print(key, ":", target[key])
        #
        # assert (False)

        img_loss, reg_loss = geometry.tick(
            glctx, batch_targets, lgt, opt_material, image_loss_fn, it, FLAGS, denoiser,
            phys_cam=phys_cam, defocus_net=defocus_net,
        )

        # ==============================================================================================
        #  Final loss
        # ==============================================================================================
        total_loss = img_loss + reg_loss

        img_loss_vec.append(img_loss.item())
        reg_loss_vec.append(reg_loss.item())

        # ==============================================================================================
        #  Backpropagate
        # ==============================================================================================

        total_loss.backward()
        if (FLAGS.learn_lighting and hasattr(lgt, 'base') and lgt.base.grad is not None and optimize_light):
            lgt.base.grad *= 64

        if ('kd_ks' in opt_material):
            opt_material['kd_ks'].encoder.params.grad /= 8.0

        # Optionally clip gradients
        if (FLAGS.clip_max_norm > 0.0):
            if (optimize_geometry):
                torch.nn.utils.clip_grad_norm_(geometry.parameters() + trainable_list, FLAGS.clip_max_norm)
            else:
                torch.nn.utils.clip_grad_norm_(trainable_list, FLAGS.clip_max_norm)

        optimizer.step()
        scheduler.step()

        if (optimize_geometry):
            optimizer_mesh.step()
            scheduler_mesh.step()

        if (optimize_light):
            optimizer_light.step()
            scheduler_light.step()

        if (defocus_net is not None):
            # Clip the gradient norms for stable training.
            torch.nn.utils.clip_grad_norm_(defocus_net.parameters(), max_norm=10)
            optimizer_defocus.step()
            scheduler_defocus.step()

        # ==============================================================================================
        #  Clamp trainables to reasonable range
        # ==============================================================================================

        with torch.no_grad():
            if ('kd' in opt_material):
                opt_material['kd'].clamp_()
            if ('ks' in opt_material):
                opt_material['ks'].clamp_()
            if ('normal' in opt_material):
                opt_material['normal'].clamp_()
                opt_material['normal'].normalize_()
            if (lgt is not None):
                lgt.clamp_(min=0.01) # For some reason gradient dissapears if light becomes 0

        # ==============================================================================================
        #  Log & save outputs
        # ==============================================================================================
        torch.cuda.synchronize()
        iter_dur_vec.append(time.time() - iter_start_time)

        # Print/save log.
        if log_interval and (it % log_interval == 0):
            img_loss_avg = np.mean(np.asarray(img_loss_vec[-log_interval:]))
            reg_loss_avg = np.mean(np.asarray(reg_loss_vec[-log_interval:]))
            iter_dur_avg = np.mean(np.asarray(iter_dur_vec[-log_interval:]))
            
            remaining_time = (FLAGS.iter - it) * iter_dur_avg
            print("iter=%5d / %5d, img_loss=%.6f, reg_loss=%.6f, lr=%.5f, time=%.1f ms, rem=%s" % 
                (it, FLAGS.iter, img_loss_avg, reg_loss_avg, optimizer.param_groups[0]['lr'], iter_dur_avg*1000, util.time_to_text(remaining_time)))

    return geometry, opt_material



# def lambda_rule(epoch):
#     return (1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1))

#----------------------------------------------------------------------------
# Main function.
#----------------------------------------------------------------------------

# if __name__ == "__main__":
parser = argparse.ArgumentParser(description='nvdiffrecmc')
parser.add_argument('-i', '--iter', type=int, default=5000)
parser.add_argument('-b', '--batch', type=int, default=1)
parser.add_argument('-s', '--spp', type=int, default=1)
parser.add_argument('-l', '--layers', type=int, default=1)
parser.add_argument('-r', '--train-res', type=int, default=[512, 512])
parser.add_argument('-dr', '--display-res', type=int, default=None)
parser.add_argument('-tr', '--texture-res', nargs=2, type=int, default=[1024, 1024])
parser.add_argument('-di', '--display-interval', type=int, default=0)
parser.add_argument('-si', '--save-interval', type=int, default=1000)
parser.add_argument('-lr', '--learning-rate', type=float, default=0.01)
parser.add_argument('-mip', '--custom-mip', action='store_true', default=False)
parser.add_argument('-bg', '--background', default='checker', choices=['black', 'white', 'checker', 'reference'])
parser.add_argument('--loss', default='logl1', choices=['logl1', 'logl2', 'mse', 'smape', 'relativel2'])
parser.add_argument('-o', '--out-dir', type=str, default=None)
parser.add_argument('--config', type=str, default=None, help='Config file')
parser.add_argument('-rm', '--ref_mesh', type=str)
parser.add_argument('-bm', '--base-mesh', type=str, default=None)
parser.add_argument('--validate', type=bool, default=True)
# Render specific arguments
parser.add_argument('--n_samples', type=int, default=4)
parser.add_argument('--bsdf', type=str, default='pbr', choices=['pbr', 'diffuse', 'white'])
# Denoiser specific arguments
parser.add_argument('--denoiser', default='bilateral', choices=['none', 'bilateral'])
parser.add_argument('--denoiser_demodulate', type=bool, default=True)

FLAGS = parser.parse_args()
# FLAGS.config = "configs/bob_w_camera.json"
# FLAGS.config = "configs/custom.json"
# FLAGS.config = "configs/experiment.json"
# FLAGS.config = "configs/experiment_part.json"
# FLAGS.config = "configs/custom_mesh.json"
FLAGS.config = "configs/sim_scenes.json"
# FLAGS.config = "configs/" + FLAGS.config

FLAGS.mtl_override        = None        # Override material of model
FLAGS.dmtet_grid          = 64          # Resolution of initial tet grid. We provide 64 and 128 resolution grids. 
                                        #    Other resolutions can be generated with https://github.com/crawforddoran/quartet
                                        #    We include examples in data/tets/generate_tets.py
FLAGS.mesh_scale          = 2.1         # Scale of tet grid box. Adjust to cover the model
FLAGS.envlight            = None        # HDR environment probe
FLAGS.env_scale           = 1.0         # Env map intensity multiplier
FLAGS.probe_res           = 256         # Env map probe resolution
FLAGS.learn_lighting      = True        # Enable optimization of env lighting
FLAGS.display             = None        # Configure validation window/display. E.g. [{"bsdf" : "kd"}, {"bsdf" : "ks"}]
FLAGS.transparency        = False       # Enabled transparency through depth peeling
FLAGS.lock_light          = False       # Disable light optimization in the second pass
FLAGS.lock_pos            = False       # Disable vertex position optimization in the second pass
FLAGS.sdf_regularizer     = 0.2         # Weight for sdf regularizer.
FLAGS.laplace             = "relative"  # Mesh Laplacian ["absolute", "relative"]
FLAGS.laplace_scale       = 3000.0      # Weight for Laplace regularizer. Default is relative with large weight
FLAGS.pre_load            = True        # Pre-load entire dataset into memory for faster training
FLAGS.no_perturbed_nrm    = False       # Disable normal map
FLAGS.decorrelated        = False       # Use decorrelated sampling in forward and backward passes
FLAGS.kd_min              = [ 0.0,  0.0,  0.0,  0.0]
FLAGS.kd_max              = [ 1.0,  1.0,  1.0,  1.0]
FLAGS.ks_min              = [ 0.0,  0.08, 0.0]
FLAGS.ks_max              = [ 0.0,  1.0,  1.0]
FLAGS.nrm_min             = [-1.0, -1.0,  0.0]
FLAGS.nrm_max             = [ 1.0,  1.0,  1.0]
FLAGS.clip_max_norm       = 0.0
FLAGS.cam_near_far        = [0.001, 1000.0] # [m]

## regularization weights in loss function
FLAGS.lambda_kd           = 0.1 
FLAGS.lambda_ks           = 0.05
FLAGS.lambda_nrm          = 0.025
FLAGS.lambda_nrm2         = 0.25
FLAGS.lambda_chroma       = 0.0
FLAGS.lambda_diffuse      = 0.15
FLAGS.lambda_specular     = 0.0025

## default custom flags
FLAGS.fix_envlight        = True
FLAGS.add_phys_cam        = False
FLAGS.add_defocus_net     = False
FLAGS.save_gt_test_imgs   = True
FLAGS.defocus_mtrx_base_dir  = ""
FLAGS.log_interval = 20
FLAGS.rand_seed = 1234
FLAGS.mesh_offset = [0., 0., 0.]

if FLAGS.config is not None:
    data = json.load(open(FLAGS.config, 'r'))
    for key in data:
        FLAGS.__dict__[key] = data[key]

if FLAGS.display_res is None:
    FLAGS.display_res = FLAGS.train_res
if FLAGS.out_dir is None:
    FLAGS.out_dir = 'out/cube_%d' % (FLAGS.train_res)
else:
    FLAGS.out_dir = 'out/' + FLAGS.out_dir

print("Config / Flags:")
print("---------")
for key in FLAGS.__dict__.keys():
    print(f"{key} : {FLAGS.__dict__[key]}")
print("---------")

os.makedirs(FLAGS.out_dir, exist_ok=True)

## copy config to out_dir
shutil.copy(FLAGS.config, os.path.join(FLAGS.out_dir, "config.json"))

glctx         = dr.RasterizeGLContext() # Context for training
glctx_display = glctx if FLAGS.batch_size < 16 else dr.RasterizeGLContext() # Context for display

#### initialize the physics-based camera if added ####
assert not(FLAGS.add_phys_cam and FLAGS.add_defocus_net), "phys_cam and defocus_net cannot be added simultaneously"
if (FLAGS.add_phys_cam):

    phys_cam_model_params_path = "./sim_cam_model_params.json"
    phys_cam = PhysDiffCamera(FLAGS.train_res[0], FLAGS.train_res[1], 1234, 'cuda')
    with open(phys_cam_model_params_path) as json_file:
        cam_params = json.load(json_file)

    noise_amp = 0.40
    near_clip = 0.005 # [m]
    far_clip = 100 # [m]

    pixel_size = cam_params["pixel_size"] # [m]
    rgb_QEs = np.array(cam_params["rgb_QEs"], dtype=float)
    gain_params = cam_params["gain_params"]
    noise_params = cam_params["noise_params"]
    noise_params["noise_gains"] = noise_amp * np.array(noise_params["noise_gains"], dtype=float)
    noise_params["STD_reads"] = noise_amp * np.array(noise_params["STD_reads"], dtype=float)

    ## lens parameters (Arducam LN042 5mm lens)
    if (FLAGS.config == "configs/experiment.json"):
        print("customized cam model params for experiment ....")
        focal_length = (806.40536205223032 + 809.61061413615107)/2 * 2 / 0.300 / 1e6 # [m]
        hFOV = 2 * np.arctan(1024 / (2 * 809.61061413615107)) # [rad]
        max_scene_light = 10000 * 0.10 / np.sum(np.array([0.825, 0., 0.825])**2) # [lux = lumen/m^2]
        # max_scene_light = 1080 # [lux = lumen/m^2]

    else:        
        focal_length = cam_params["focal_length"] # [m]
        hFOV = cam_params["hFOV"] * pi / 180.0 # [rad]
        max_scene_light = 400
    
    sensor_width = 2 * focal_length * np.tan(hFOV / 2) # [m]

    phys_cam.SetModelParameters(sensor_width, pixel_size, max_scene_light, rgb_QEs, gain_params, noise_params)
    phys_cam.BuildVignetMask(sensor_width, focal_length) 
    phys_cam.artifact_switches = {
        "vignetting": False,
        "defocus_blur": True,
        "aggregate": False,
        "add_noise": False,
        "expsr2dv": False,
    }

    defocus_net = None

elif (FLAGS.add_defocus_net):
    defocus_net = networks.define_G(4, 3, 64, "unet_1024", gpu_ids=[0], use_dropout=True)
    phys_cam = None

else:
    phys_cam = None
    defocus_net = None

## log
if (FLAGS.add_phys_cam):
    print("phys_cam added:")
    for key in phys_cam.artifact_switches.keys():
        print(f"{key} : {phys_cam.artifact_switches[key]}")

#### set up random seed
random.seed(FLAGS.rand_seed)
np.random.seed(FLAGS.rand_seed)
torch.manual_seed(FLAGS.rand_seed)

# ==============================================================================================
#  Create data pipeline
# ==============================================================================================
if (os.path.splitext(FLAGS.ref_mesh)[1] == '.obj'):
    ref_mesh      = mesh.load_mesh(FLAGS.ref_mesh, FLAGS.mtl_override)
    dataset_train = DatasetMesh(ref_mesh, glctx, RADIUS, FLAGS, validate=False, phys_cam=phys_cam)
    dataset_valid = DatasetMesh(ref_mesh, glctx_display, RADIUS, FLAGS, validate=True, phys_cam=phys_cam)

elif os.path.isdir(FLAGS.ref_mesh):
    if os.path.isfile(os.path.join(FLAGS.ref_mesh, 'poses_bounds.npy')):
        dataset_train = DatasetLLFF(FLAGS.ref_mesh, FLAGS, examples=(FLAGS.iter + 1) * FLAGS.batch_size)
        dataset_valid = DatasetLLFF(FLAGS.ref_mesh, FLAGS)
    
    elif (FLAGS.ref_mesh.split(os.sep)[-2] in ['custom', 'experiment', 'sim_scenes']):
        dataset_train = DatasetCustom(os.path.join(FLAGS.ref_mesh, 'trnsfrms_and_configs_train.json'), FLAGS, num_samples=(FLAGS.iter + 1) * FLAGS.batch_size)
        dataset_valid = DatasetCustom(os.path.join(FLAGS.ref_mesh, 'trnsfrms_and_configs_test.json'), FLAGS, num_samples=FLAGS.validate_num)
    
    elif (os.path.isfile(os.path.join(FLAGS.ref_mesh, 'transforms_train.json'))  and not os.path.isfile(os.path.join(FLAGS.ref_mesh, 'intrinsics.txt'))):
        dataset_train = DatasetNERF(os.path.join(FLAGS.ref_mesh, 'transforms_train.json'), FLAGS, num_samples=(FLAGS.iter + 1) * FLAGS.batch_size)
        dataset_valid = DatasetNERF(os.path.join(FLAGS.ref_mesh, 'transforms_test.json'), FLAGS)
    
    else:
        assert False, "Invalid dataset format"

else:
    print("Invalid dataset format", FLAGS.ref_mesh)
    assert False, "Invalid dataset format"

# ==============================================================================================
#  Create trainable light
# ==============================================================================================
lgt = None
if (FLAGS.learn_lighting == True):
    lgt = light.create_trainable_env_rnd(FLAGS.probe_res, scale=0.0, bias=0.5)
elif (FLAGS.learn_lighting == False and FLAGS.fix_envlight == True):
    lgt = light.load_env(FLAGS.envlight, scale=FLAGS.env_scale, res=[FLAGS.probe_res, FLAGS.probe_res])

# ==============================================================================================
#  Setup denoiser
# ==============================================================================================

denoiser = None
if (FLAGS.denoiser == 'bilateral'):
    denoiser = BilateralDenoiser().cuda()
else:
    assert (FLAGS.denoiser == 'none'), "Invalid denoiser %s" % FLAGS.denoiser

if (FLAGS.base_mesh is None):
    # ==============================================================================================
    #  If no initial guess, use DMTet to create geometry
    # ==============================================================================================

    # Setup geometry for optimization
    geometry = DMTetGeometry(FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS.mesh_offset, FLAGS)

    # Setup textures, make initial guess from reference if possible
    mat = initial_guess_material(geometry, True, FLAGS)

    # Run optimization
    mat['no_perturbed_nrm'] = True
    geometry, mat = optimize_mesh(
        denoiser, glctx, glctx_display, geometry, mat, lgt, dataset_train, dataset_valid, FLAGS,
        log_interval=FLAGS.log_interval,
        pass_idx=0,
        pass_name="dmtet_pass1",
        optimize_light=FLAGS.learn_lighting,
        phys_cam=phys_cam,
        defocus_net=defocus_net,
    )
    
    # pass_idx = 1

    if (FLAGS.validate):
        validate(
            glctx_display, geometry, mat, lgt, dataset_valid, os.path.join(FLAGS.out_dir, "dmtet_validate"), FLAGS, denoiser,
            phys_cam=phys_cam, defocus_net=defocus_net,
        )
    print("finished validation")
    
    # Create initial guess mesh from result
    eval_mesh = geometry.getMesh(mat)
    
    # Create uvs with xatlas
    v_pos = eval_mesh.v_pos.detach().cpu().numpy()
    t_pos_idx = eval_mesh.t_pos_idx.detach().cpu().numpy()
    
    ## save, in case getting stuck
    os.makedirs(os.path.join(FLAGS.out_dir, "dmtet_mesh"), exist_ok=True)
    print(f"v_pos.shape={v_pos.shape}")
    print(f"t_pos_idx.shape={t_pos_idx.shape}")
    np.savez_compressed(os.path.join(FLAGS.out_dir, "dmtet_mesh/", "v_pos.npz"), val=v_pos)
    np.savez_compressed(os.path.join(FLAGS.out_dir, "dmtet_mesh/", "t_pos_idx.npz"), val=t_pos_idx)
    
    print("xatlas.parametrize starts working")
    t_start = time.time()
    """
    def parametrize_chunk(chunk):
        return xatlas.parametrize(chunk[0], chunk[1])

    rows_chunks = 2
    cols_chunks = 3
    num_chunks = rows_chunks * cols_chunks

    v_pos_chunks = num_chunks * [v_pos]
    # t_pos_idx_chunks = np.array_split(t_pos_idx, num_chunks)
    t_pos_idx_chunks = [
        t_pos_idx[np.where((v_pos[t_pos_idx[:, 0], 2]  < 0) & (v_pos[t_pos_idx[:, 0], 0] < -0.4))[0]],
        t_pos_idx[np.where((v_pos[t_pos_idx[:, 0], 2]  < 0) & (-0.4 <= v_pos[t_pos_idx[:, 0], 0]) & (v_pos[t_pos_idx[:, 0], 0] < 0.4))[0]],
        t_pos_idx[np.where((v_pos[t_pos_idx[:, 0], 2]  < 0) & ( 0.4 <= v_pos[t_pos_idx[:, 0], 0]))[0]],
        t_pos_idx[np.where((v_pos[t_pos_idx[:, 0], 2] >= 0) & (v_pos[t_pos_idx[:, 0], 0] < -0.4))[0]],
        t_pos_idx[np.where((v_pos[t_pos_idx[:, 0], 2] >= 0) & (-0.4 <= v_pos[t_pos_idx[:, 0], 0]) & (v_pos[t_pos_idx[:, 0], 0] < 0.4))[0]],
        t_pos_idx[np.where((v_pos[t_pos_idx[:, 0], 2] >= 0) & ( 0.4 <= v_pos[t_pos_idx[:, 0], 0]))[0]],
    ]
    for t_pos_idx_chunk_idx in range(len(t_pos_idx_chunks)):
        print(f"t_pos_idx_chunks[{t_pos_idx_chunk_idx}].shape={t_pos_idx_chunks[t_pos_idx_chunk_idx].shape}")

    with multiprocessing.Pool(num_chunks) as pool:
        results = pool.map(parametrize_chunk, zip(v_pos_chunks, t_pos_idx_chunks))
    
    vmapping, indices, uvs = [], [], []
    total_num_vertices = 0
    for result_idx in range(num_chunks):
        row_idx = result_idx // cols_chunks
        col_idx = result_idx % cols_chunks
        
        sub_vmapping, sub_indices, sub_uvs = results[result_idx]
        
        ## map sub_uvs to correct uvs on the big atlas
        sub_uvs[:, 0] = (row_idx + sub_uvs[:, 0]) / rows_chunks
        sub_uvs[:, 1] = (col_idx + sub_uvs[:, 1]) / cols_chunks
        
        ## shift sub_indices with aggregated number of vertices
        sub_indices += total_num_vertices
        
        ## aggregate total number of vertices
        total_num_vertices += len(sub_uvs)
        
        ## collect sub_vmapping, sub_indices, and sub_uvs
        vmapping.append(sub_vmapping)
        indices.append(sub_indices)
        uvs.append(sub_uvs)
    
    vmapping = np.hstack(vmapping)
    indices = np.vstack(indices)
    uvs = np.vstack(uvs)
    """
    
    vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)
    
    print(f"xatlas.parametrize took {(time.time() - t_start): .3f} sec")

    base_mesh = xatlas_uvmap(glctx_display, eval_mesh, indices, uvs, mat, FLAGS).clone()
    print("base_mesh built")
    base_mesh.v_pos = base_mesh.v_pos.clone().detach().requires_grad_(True)
    mat = material.create_trainable(base_mesh.material.copy())
    print("material created")
    
    geometry = DLMesh(base_mesh, FLAGS)

    # Dump mesh for debugging; save learnables
    obj.write_obj(os.path.join(FLAGS.out_dir, "dmtet_mesh/"), base_mesh)
    if (FLAGS.learn_lighting):
        light.save_env_map(os.path.join(FLAGS.out_dir, "dmtet_mesh/probe.hdr"), lgt)

    if (defocus_net is not None):
        torch.save(defocus_net.state_dict(), os.path.join(FLAGS.out_dir, "dmtet_mesh/defocus_net_pass1.ckpt"))

    # ==============================================================================================
    #  Pass 2: Train with fixed topology (mesh)
    # ==============================================================================================
    if ((FLAGS.pass2 is None) or (FLAGS.pass2 == True)):
        if (FLAGS.transparency):
            FLAGS.layers = 8

        mat['no_perturbed_nrm'] = False
        geometry, mat = optimize_mesh(
            denoiser, glctx, glctx_display, geometry, mat, lgt, dataset_train, dataset_valid, FLAGS,
            # pass_idx=pass_idx,
            pass_idx=1,
            pass_name="mesh_pass2",
            warmup_iter=100, # Use warmup to avoid nasty Adam spikes
            optimize_light=not FLAGS.lock_light,
            optimize_geometry=not FLAGS.lock_pos,
            log_interval=FLAGS.log_interval,
            phys_cam=phys_cam,
            defocus_net=defocus_net,
        )

        # ==============================================================================================
        #  Validate
        # ==============================================================================================
        if (FLAGS.validate):
            validate(
                glctx_display, geometry, mat, lgt, dataset_valid, os.path.join(FLAGS.out_dir, "validate"), FLAGS, denoiser,
                phys_cam=phys_cam, defocus_net=defocus_net,
            )

        # ==============================================================================================
        #  Dump output
        # ==============================================================================================
        final_mesh = geometry.getMesh(mat)

        os.makedirs(os.path.join(FLAGS.out_dir, "mesh"), exist_ok=True)
        obj.write_obj(os.path.join(FLAGS.out_dir, "mesh/"), final_mesh)
        if (lgt is not None):
            light.save_env_map(os.path.join(FLAGS.out_dir, "mesh/probe.hdr"), lgt) 

        if (defocus_net is not None):
            torch.save(defocus_net.state_dict(), os.path.join(FLAGS.out_dir, "dmtet_mesh/defocus_net_pass2.ckpt"))
else:
    # ==============================================================================================
    #  Train with fixed topology (mesh)
    # ==============================================================================================

    # Load initial guess mesh from file
    base_mesh = mesh.load_mesh(FLAGS.base_mesh)
    geometry = DLMesh(base_mesh, FLAGS)
    base_mesh.v_pos = base_mesh.v_pos.clone().detach().requires_grad_(True)

    mat = initial_guess_material(geometry, False, FLAGS, init_mat=base_mesh.material)

    geometry, mat = optimize_mesh(
        denoiser, glctx, glctx_display, geometry, mat, lgt, dataset_train, dataset_valid, FLAGS, 
        log_interval=FLAGS.log_interval,
        pass_idx=0,
        pass_name="mesh_pass",
        warmup_iter=0,
        optimize_light=not FLAGS.lock_light,
        optimize_geometry=not FLAGS.lock_pos,
        phys_cam=phys_cam,
        defocus_net=defocus_net,
    )

    # ==============================================================================================
    #  Validate
    # ==============================================================================================
    if (FLAGS.validate):
        validate(
            glctx_display, geometry, mat, lgt, dataset_valid, os.path.join(FLAGS.out_dir, "validate"), FLAGS, denoiser,
            phys_cam=phys_cam, defocus_net=defocus_net,
        )

    # ==============================================================================================
    #  Dump output
    # ==============================================================================================
    final_mesh = geometry.getMesh(mat)

    os.makedirs(os.path.join(FLAGS.out_dir, "mesh"), exist_ok=True)
    obj.write_obj(os.path.join(FLAGS.out_dir, "mesh/"), final_mesh)
    if (lgt is not None):
        light.save_env_map(os.path.join(FLAGS.out_dir, "mesh/probe.hdr"), lgt)

    if (defocus_net is not None):
        torch.save(defocus_net.state_dict(), os.path.join(FLAGS.out_dir, "dmtet_mesh/defocus_net_pass.ckpt"))
