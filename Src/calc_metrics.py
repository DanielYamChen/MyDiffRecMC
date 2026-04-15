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
import argparse

############################
######## Parameters ########
############################

#### Parse arguments ####

parser = argparse.ArgumentParser(description='get_calibtated_cam_output')
parser.add_argument(
    '--scene', required=True, type=str, default=None,
    help="for which scene: RealScene01, RealScene02, SimScene01, SimScene02, SimScene03",
)
parser.add_argument(
    '--cond', required=True, type=str, default=None,
    help="which condition of the scene: full, wo_expsr, wo_defocus, wo_camera",
)
parser.add_argument(
    '--defocus_type', type=str, default=None,
    help="which defocus type to run: gaussian, uniform"
)

args = parser.parse_args()
scene = args.scene
cond = args.cond
defocus_type = args.defocus_type

#### Derived parameters ####
if (scene[:9] == "RealScene"):
    test_json_path = f"../../DiffPhysCam_Data/NovelViewSynthesis_Data/{scene}/trnsfrms_and_configs_test_wo_ground.json"
    gt_imgs_dir = f"../../DiffPhysCam_Data/NovelViewSynthesis_Data/{scene}/imgs_wo_ground/" 

elif (scene[:8] == "SimScene"):
    if (scene[-2:] in ["01", "03"]):
        test_json_path = "../../DiffPhysCam_Data/NovelViewSynthesis_Data/SimScene_general/trnsfrms_and_configs_test_Scenes01and03.json"
    
    elif (scene[-2:] == "02"):
        test_json_path = "../../DiffPhysCam_Data/NovelViewSynthesis_Data/SimScene_general/trnsfrms_and_configs_test_Scene02.json"

    gt_imgs_dir = f"../../DiffPhysCam_Data/NovelViewSynthesis_Data/{scene}/imgs_wo_ground/" 

if ((scene[:9] == "RealScene") and (cond in ["full", "wo_expsr"])):
    synth_imgs_dir = f"../../DiffPhysCam_Data/NovelViewSynthesis_Output/{scene}_{cond}_defocus_{defocus_type}/validate/"
else:
    synth_imgs_dir = f"../../DiffPhysCam_Data/NovelViewSynthesis_Output/{scene}_{cond}/validate/"

######################
######## MAIN ########
######################

LPIPS_loss_fn_vgg = lpips.LPIPS(net='vgg')
LPIPS_loss_fn_vgg.cuda()

with open(test_json_path) as json_file:
    test_json = json.load(json_file)

num_test_imgs = len(test_json["frames"])


MSEs, PSNRs, LPIPSs, SSIMs = [], [], [], []

num_skipped_imgs = 0
for img_idx in range(num_test_imgs):
# for img_idx in [5]: # debug
    img_name = test_json["frames"][str(img_idx)]["img_name"]

    ## Skip some images in RealScene02 that are just for defocus-blur effect visualization
    if (scene == "RealScene02" and img_name.startswith("sun_No")):
        num_skipped_imgs += 1
        print(f"[{num_skipped_imgs}] Skipped {img_name}")
        continue
    
    else:
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
        
        MSEs.append(mse)
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
        
        ### Print log ###
        if ((img_idx + 1) % 20 == 0):
            print(f"Img [{img_idx + 1}/{num_test_imgs}] addressed")
    
print(f"MSE: {np.mean(MSEs):.6f}")
print(f"PSNR: {np.mean(PSNRs):.6f}")
print(f"SSIM: {np.mean(SSIMs):.6f}")
print(f"LPIPS: {np.mean(LPIPSs):.8f}")
print()





    
    