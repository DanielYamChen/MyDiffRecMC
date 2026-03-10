#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate defocus matrices from depth maps

@author: Bo-Hsun Chen
"""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import time
import cv2
cv2.setLogLevel(2)
import matplotlib.pyplot as plt
import importlib.util
import sys
import torch
from math import pi
import json
import argparse

############################
######## Parameters ########
############################

#### Parse arguments ####
parser = argparse.ArgumentParser(description='get_calibtated_cam_output')
parser.add_argument(
    '--scene', required=True, type=str, default="",
    help="for which scene: RealScene01, RealScene02",
)
parser.add_argument('--defocus_type', required=True, type=str, default="", help="which defocus type to run: gaussian or uniform")

args = parser.parse_args()
scene = args.scene
defocus_type = args.defocus_type

# scene = "RealScene01"
# defocus_type = "gaussian"
# cond = "wo_ground"

#### Paths ####
phys_cam_model_path = "../models/cam_model.py"
phys_cam_model_params_path = f"../../CameraCalibrateExp/phys_cam_model_params_defocus_{defocus_type}.json"
depth_map_dir = f"../../DiffPhysCam_Data/NovelViewSynthesis_Data/{scene}/depth_maps_wo_ground/"
dst_defocus_matrix_dir = f"../../DiffPhysCam_Data/NovelViewSynthesis_Data/{scene}/defocus_matrices_{defocus_type}_wo_ground_v2/"

#### General parameters ####
defocus_name_suffix_dict = {
    "wo_ground": "_separatedobjs_all",
    "ground_added": "_sugar",
}

img_w = 1024 # [px]
img_h = 768 # [px]

defocus_map_w = 1024 # [px]
defocus_map_h = 768 # [px]

torch_random_seed = 1234

max_scene_light = 1080 # [lux = lumen/m^2]
noise_amp = 0.40


#### Scene and camera setting parameters ####
sun_azis = [ # [deg]
    "180",
    "270",
]
polar_angles = [ # [deg]
    40,
    55,
    70,
    85,
]
num_polar_angles = len(polar_angles)

num_azi_angles_list = [
    24,
    18,
    18,
    18,
]

aperture_nums = [
    2.8,
    2.8,
    2.0,
    2.0
]

focus_dist_dicts = [ # [m]
    {"middle": 1.51135607},
    {"middle": 1.18595926},
    {"far": 1.67369843, "near": 0.51869843},
    {"far": 1.73961093, "near": 0.58461093},
]

def erode_depth_mask(depth, erosion_size=3, iterations=1):
    """
    depth: float32 depth map
    erosion_size: structuring element size
    iterations: erosion iterations

    return:
        eroded_mask
        boundary_band
    """

    # 1. 用 depth > 0 當 mask
    mask = (depth > 0).astype(np.uint8)

    # 2. 建立 erosion kernel
    kernel = np.ones((erosion_size, erosion_size), np.uint8)

    # 3. erosion
    eroded_mask = cv2.erode(mask, kernel, iterations=iterations)

    # 4. boundary band (被 erosion 吃掉的那圈)
    boundary_band = mask - eroded_mask

    return eroded_mask, boundary_band

######################
######## MAIN ########
######################

#### Construct phys_cam ####

### Import phys_cam model from MyDiffRecMC repo ###
spec = importlib.util.spec_from_file_location("phys_cam", phys_cam_model_path)
my_modules = importlib.util.module_from_spec(spec)
sys.modules["module.name"] = my_modules
spec.loader.exec_module(my_modules)
phys_cam = my_modules.PhysDiffCamera(img_h, img_w, torch_random_seed, 'cuda')

### Import calibrated model parameters ###
with open(phys_cam_model_params_path) as json_file:
    cam_params = json.load(json_file)

noise_params = cam_params["noise_params"]
noise_params["noise_gains"] = noise_amp * np.array(noise_params["noise_gains"], dtype=float)
noise_params["STD_reads"] = noise_amp * np.array(noise_params["STD_reads"], dtype=float)

## Lens parameters (Arducam LN042 5mm lens)
# focal_length = cam_params["focal_length"] # [m]
# hFOV = cam_params["hFOV"] * pi / 180.0 # [rad]
# obj_scale = 0.282051
focal_length = 0.00572951691782118 # [m]
hFOV = 1.1278099154119037 # [rad]
sensor_width = 2 * focal_length * np.tan(hFOV / 2) # [m]

phys_cam.BuildVignetMask(sensor_width, focal_length) 
phys_cam.artifact_switches = {
    "vignetting":   True,
    "defocus_blur": True,
    "aggregate":    True,
    "add_noise":    True,
    "expsr2dv":     True,
}

phys_cam.SetModelParameters(
    sensor_width,
    cam_params["pixel_size"], # [m],
    max_scene_light,
    np.array(cam_params["rgb_QEs"], dtype=float),
    cam_params["gain_params"],
    noise_params
)

assert (num_polar_angles == len(num_azi_angles_list)), "num_polar_angles != len(num_azi_angles_list)"
assert (num_polar_angles == len(aperture_nums)), "num_polar_angles != len(aperture_nums)"
assert (num_polar_angles == len(focus_dist_dicts)), "num_polar_angles != len(focus_dist_dicts)"

#### Generate defocus matrices over all sun_azis, polar_angles, azi_angles, and focus_dists ####
matrix_idx = 0
t_start = time.time()
for sun_azi in sun_azis:
    for polar_idx in range(num_polar_angles):
        polar_angle = polar_angles[polar_idx]
        for azi_idx in range(num_azi_angles_list[polar_idx]):
            azi_angle = 360 * (azi_idx) // num_azi_angles_list[polar_idx]
            for focus_dist_key, focus_dist in focus_dist_dicts[polar_idx].items():
                
                depth_map_name = f"depth_sun_{sun_azi}_polar_{polar_angle:03d}_azi_{azi_angle:03d}_U_{focus_dist_key}.exr"
                depth_map = cv2.imread(os.path.join(depth_map_dir, depth_map_name), cv2.IMREAD_UNCHANGED)
                if depth_map is None:
                    continue
                
                assert (depth_map.shape[0] == defocus_map_h and depth_map.shape[1] == defocus_map_w)
                eroded_mask, boundary_band = erode_depth_mask(depth_map, erosion_size=3, iterations=1)
                depth_map[eroded_mask == 0] = 0. # set depth to 0 for pixels in the boundary band (to avoid noisy depth values near object boundaries)
                defocus_matrix, defocus_D_map = phys_cam.GetSparseTensor(
                    depth_map,
                    {
                        "aperture_num": aperture_nums[polar_idx],
                        "expsr_time": 0.128, # [sec]
                        "ISO": 100.0, # [%]
                        "focal_length": focal_length,
                        "focus_dist": focus_dist,
                    },
                    kernel_type=defocus_type,
                )
                defocus_matrix = defocus_matrix.detach().cpu()
                defocus_D_map = defocus_D_map.astype(float) / defocus_D_map.max()
                defocus_D_map = cv2.resize(defocus_D_map, (img_w, img_h), cv2.INTER_LINEAR)
                
                defocus_matrix_path = os.path.join(dst_defocus_matrix_dir, depth_map_name[6:-4] + defocus_name_suffix_dict["wo_ground"] + ".npz")
                np.savez_compressed(
                    defocus_matrix_path,
                    indices=defocus_matrix.indices().numpy(),
                    values=defocus_matrix.values().numpy(),
                    shape=defocus_matrix.shape,
                )
                
                defocus_D_map_path = os.path.join(dst_defocus_matrix_dir, "defocus_" + depth_map_name[6:-4] + ".png")
                assert(cv2.imwrite(
                        defocus_D_map_path,
                        cv2.cvtColor((defocus_D_map * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    )), f"Failed to save {defocus_D_map_path}"
                
                ## print log ##
                matrix_idx += 1
                if (matrix_idx % 10 == 0):
                    print(f"{matrix_idx} defocus matrices created")
                
                pass

t_end = time.time()

print(f"takes {np.round(t_end - t_start, 4)} sec to build defocus matrices")







