#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate PSNR, SSIM, and LPIPS metrics

@author: Bo-Hsun Chen
"""

from skimage.metrics import structural_similarity as ssim
import lpips
import json
import cv2
import os
import numpy as np
import torch

#### arguments ####

scene_ID = 3
test_json_path = "./data/sim_scenes/trnsfrms_and_configs_test.json"
gt_imgs_dir = f"./data/sim_scenes/imgs_wo_ground_{str(scene_ID)}/" 
# test_json_path = "./data/experiment/trnsfrms_and_configs_test_wo_ground.json"
# gt_imgs_dir = f"./data/experiment/imgs_wo_ground/"

##############
#### MAIN ####
##############
LPIPS_loss_fn_vgg = lpips.LPIPS(net='vgg')
LPIPS_loss_fn_vgg.cuda()

with open(test_json_path) as json_file:
    test_json = json.load(json_file)

num_test_imgs = len(test_json["frames"])

conds = [
    f"sim_scenes_{scene_ID}",
    f"sim_scenes_{scene_ID}_wo_camera",
    f"sim_scenes_{scene_ID}_wo_defocus_blur",
    f"sim_scenes_{scene_ID}_wo_expsr_related",
    # f"experiment",
    # f"experiment_wo_camera",
    # f"experiment_wo_defocus_blur",
    # f"experiment_wo_expsr_related",
]

for cond in conds:
    
    print(cond)
    synth_imgs_dir = "./out/" + cond + "/validate/"
    PSNRs, LPIPSs, SSIMs = [], [], []
    
    for img_idx in range(num_test_imgs):
    # for img_idx in [5]: # debug
        img_name = test_json["frames"][str(img_idx)]["img_name"]
        
        synth_img = cv2.imread(os.path.join(synth_imgs_dir, f"val_opt_{str(img_idx).zfill(6)}.png"), cv2.IMREAD_UNCHANGED)
        synth_img = cv2.cvtColor(synth_img, cv2.COLOR_BGRA2RGBA)
        synth_img = synth_img.astype(float) / np.iinfo(synth_img.dtype).max
        non_effect_px = synth_img[:, :, 3] < 0.9
        synth_img[non_effect_px, 0 : 3] = 0.
        
        gt_img = cv2.imread(os.path.join(gt_imgs_dir, img_name), cv2.IMREAD_UNCHANGED)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGRA2RGBA)
        gt_img = gt_img.astype(float) / np.iinfo(gt_img.dtype).max
        non_effect_px = gt_img[:, :, 3] < 0.9
        gt_img[non_effect_px, 0 : 3] = 0.
        
        
        ### calculate PSNR
        num_effect_pxs = max(np.sum(synth_img[:, :, 3] > 0.99), np.sum(gt_img[:, :, 3] > 0.99))
        mse = np.sum((synth_img - gt_img)**2) / num_effect_pxs
        if (mse == 0.):
            psnr_value = np.inf
        else:
            psnr_value = -10. * np.log10(mse)
        
        PSNRs.append(psnr_value)
        
        ### calculate SSIM
        ssim_value = ssim(synth_img[:, :, 0 : 3], gt_img[:, :, 0 : 3], channel_axis=2, data_range=1.0)
        SSIMs.append(ssim_value)
        
        ### calculate LPIPS
        lpips_value = LPIPS_loss_fn_vgg(
            torch.Tensor((synth_img[:, :, 0 : 3] * 2 - 1.0)[:, :, :, np.newaxis].transpose((3, 2, 0, 1))).cuda(),
            torch.Tensor((gt_img[:, :, 0 : 3] * 2 - 1.0)[:, :, :, np.newaxis].transpose((3, 2, 0, 1))).cuda()
        ).item()
        LPIPSs.append(lpips_value)
        
        ### print log
        if ((img_idx + 1) % 20 == 0):
            print(f"Img [{img_idx + 1}/{num_test_imgs}] addressed")
        
    print(f"PSNR: {np.mean(PSNRs):.6f}")
    print(f"SSIM: {np.mean(SSIMs):.6f}")
    print(f"LPIPS: {np.mean(LPIPSs):.8f}")
    print()





    
    