#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:24:59 2024

Build terrain scene and take simulated photos for simulation

@author: Bo-Hsun Chen
"""

import drjit as dr
import mitsuba as mi
from models.cam_model import PhysDiffCamera
import torch.nn as nn
import torch
import cv2
import gzip

mi.set_variant('cuda_ad_rgb')
# mi.set_variant('scalar_rgb')
DRJIT_LIBLLVM_PATH="/usr/lib/llvm-15/lib/libLLVM.so"

from mitsuba import ScalarTransform4f as T
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import copy
import time
from math import pi
import math
import cv2
# from collections import defaultdict
import json


random.seed(1234)
np.random.seed(1234)
torch_random_seed = 1234

dr.flush_malloc_cache()
torch.cuda.empty_cache()
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

#############################
#### Function set starts ####
#############################
## return camera, angles of polar coordinates of camera, and camera's origin,
## given polar (theta) and azimuth (phi) angles
def GetCamera(phi, theta, hFOV, up_direction):
    '''
    return camera, angles of polar coordinates of camera, and camera's origin,
    given polar (theta) and azimuth (phi) angles
    '''
    camera_angles = [str(round(phi * 180/pi)).zfill(3), str(round(theta * 180/pi)).zfill(3)]
    camera_origin = [radius * np.cos(phi) * np.sin(theta), radius * np.sin(phi) * np.sin(theta), radius * np.cos(theta)]
    camera = mi.load_dict({
        'type': 'perspective',
        'fov': hFOV * 180.0/pi,
        'fov_axis': 'x',
        'to_world': T.look_at(target=[0., 0., 0.], origin=camera_origin, up=up_direction),
        'film': {
            'type': 'hdrfilm',
            'width': img_w, 'height': img_h,
            'filter': {'type': 'gaussian'},
            'sample_border': True,
        },
        'sampler': {
            'type': 'independent',
            'sample_count': num_spps
        },
    })
    
    return camera, camera_angles, camera_origin


## Converts polar and azimuth angles from the original coordinate system to the Mitsuba EnvMap coordinate system.
def CvtMiTsubaEnvAngles(old_azimuth, old_polar):
    """
    Converts polar and azimuth angles from the original coordinate system to the Mitsuba EnvMap coordinate system.

    Parameters:
    old_azimuth (float): Azimuth angle in radians (angle between the projected origin-to-point vector on the X-Y plane and the +X axis).
    old_polar (float): Polar angle in radians (angle between the origin-to-point vector and the +Z axis).
    
    Returns:
    tuple: (new_azimuth, new_polar) in radians.
        new_azimuth: Azimuth angle between the projected origin-to-point vector on the Z-X plane and the -Z axis.        
        new_polar: Polar angle between the origin-to-point vector and the +Y axis, ranging from 0 to pi radians.
    """

    # Corrected polar angle calculation
    new_polar = np.arccos(np.sin(old_polar) * np.sin(old_azimuth))

    # Corrected azimuth angle calculation
    new_azimuth = np.arctan2(np.sin(old_polar) * np.cos(old_azimuth), -np.cos(old_polar)) % (2 * pi)
    
    return new_azimuth, new_polar


## create and save a HDR environment map for the directional light
def CreateDirectLightEnvmap(azimuth, polar, light_color, amb_lgt_intensity, file_path):
    '''
    create and save a HDR environment map for the directional light
    '''
    # Initialize a 256x256 HDR image with zeros (black background)
    probe_res = 256
    envmap = amb_lgt_intensity * np.ones((probe_res, probe_res, 3), dtype=np.float32)
    
    # Convert spherical coordinates (azimuth, polar) to pixel coordinates
    # Azimuth is the horizontal angle (longitude)
    # Polar is the vertical angle (latitude)
    
    # Map polar angle (0 to pi) to vertical position (0 to resolution-1)
    assert (azimuth >= 0) and (polar >= 0), "azimuth and polar should both be positive when creating env map ..."
    y = int((polar / pi) * (probe_res - 1))
    
    # Map azimuth angle (0 to 2*pi) to horizontal position (0 to resolution-1)
    x = int((azimuth / (2 * pi)) * (probe_res - 1))
    
    # Add the light color to the corresponding pixel, you can use a small spot size for the light
    # Here, we are adding a Gaussian spot to simulate a light
    spot_half_size = 3  # half-size of the light spot in pixels
    for i in range(-spot_half_size, spot_half_size + 1):
        if (0 <= x + i and x + i < probe_res):
            for j in range(-spot_half_size, spot_half_size + 1):
                if (0 <= y + j and y + j < probe_res):
                    intensity = math.exp(-math.sqrt(i**2 + j**2))  # Gaussian falloff
                    envmap[y + j, x + i, :] += light_color * intensity
    
    # Save the environment map as an HDR image
    if (file_path is not None):
        assert(cv2.imwrite(
            file_path,
            cv2.cvtColor(envmap, cv2.COLOR_RGB2BGR)
        )), f"failed to save envmap at {file_path}"

    return envmap


###########################
#### Function set ends ####
###########################

mode = "train"

img_w = 512 * 2 # [px]
img_h = 384 * 2 # [px]
depth_map_w = 256 * 2
depth_map_h = 192 * 2

# dst_base_dir = "./data/custom/"
dst_base_dir = "./data/custom/" # for getting environment maps for experiment
dst_img_dir = mode + "_imgs/"
dst_envmap_dir = "envmaps/"
dst_defocus_mtrx_dir = "defocus_matrices/"
dst_depth_map_dir = "depth_maps_wo_ground/"
show_gt_imgs = False

## save options
save_envlgt = True
save_sparse_tensor = False
save_depth_map = False
save_gt_img = False
save_trnsfrms_and_configs = False

ignore_ground = True

## Load texture dataset ##
obj_base_dir = "../TextureMaterial/"
obj_cmpny_name = "TexturesCom"
obj_names_list = [
    "sand_rough_sliding_001", # ground
    "rock_boulder_003", # Object 1
    "rock_sandstone_001", # Object 2
    "rock_boulder_019", # Object 3
    "rubble_039", # Object 4
    "wood_log_010", # Object 5
]
obj_poses = [ # translation(X, Y ,Z) [m], axis angle(W (deg), X, Y, Z)
    [[0., 0., 0.], [0., 0., 0., 0.]], # ground
    [[ 0.960,  0.840, 0.030], [0., 0., 0., 0.]], # Object 1
    [[ 0.660, -0.630, 0.024], [0., 0., 0., 0.]], # Object 2
    [[-0.090,  0.000, 0.020], [0., 0., 0., 0.]], # Object 3
    [[-1.020,  0.420, 0.162], [180., 1., 0., 0.]], # Object 4
    [[-0.600, -0.960, 0.018], [150., 0., 0., 1.]], # Object 5
]
obj_scales = [
    [1.0, 1.0, 1.0], # ground
    [0.0042, 0.0042, 0.0042], # Object 1
    [0.0060, 0.0060, 0.0120], # Object 2
    [0.0042, 0.0042, 0.0042], # Object 3
    [0.0240, 0.0240, 0.0240], # Object 4
    [0.0060, 0.0060, 0.0060], # Object 5
]
obj_metallics = [
    0., # ground
    0., # Object 1
    0., # Object 2
    0., # Object 3
    0., # Object 4
    0., # Object 5
]
obj_speculars = [
    0.03, # ground
    0.05, # Object 1
    0.02, # Object 2
    0.03, # Object 3
    0.05, # Object 4
    0.05, # Object 5
]

# polar_range = 80 * (pi / 180.0) # [rad]
# num_polar_angles = 4
# num_azimuth_angles = 6
polar_angles = [ # [rad]
    # 30 * (pi/180.0),
    # 80 * (pi/180.0),
    
    # 40 * (pi/180.0),
    # 45 * (pi/180.0),
    # 50 * (pi/180.0),
    # 55 * (pi/180.0),
    # 60 * (pi/180.0),
    # 65 * (pi/180.0),
    # 70 * (pi/180.0),
    # 75 * (pi/180.0),
    # 80 * (pi/180.0),
    # 85 * (pi/180.0),
    # 90 * (pi/180.0),
    # 95 * (pi/180.0),
    # 100 * (pi/180.0),
    # 105 * (pi/180.0),
    # 110 * (pi/180.0),
    # 115 * (pi/180.0),
    # 120 * (pi/180.0),
    
    40 * (pi/180.0),
    55 * (pi/180.0),
    70 * (pi/180.0),
    85 * (pi/180.0),
    0 * (pi/180.0),
]
num_polar_angles = len(polar_angles)
num_azi_angles_list = [
    10, 
    15,
    15,
    20,
    1, # top view
]
# debug
# num_azi_angles_list = (len(polar_angles) - 1) * [10]
# num_azi_angles_list.append(2)
assert (num_polar_angles == len(num_azi_angles_list))

radius = 1.5 * 1.414 # [m]
focus_dists_list = [ # [m]
    [radius], # low polar angle
    [radius], # low polar angle
    [radius - 0.6 * 1.414, radius - 0.9 * 1.414], # high polar focusing at far / near sight
    [radius - 0.6 * 1.414, radius - 0.9 * 1.414], # high polar focusing at far / near sight
    [radius], # top view
    # [0., 0., 1.15],
    # [-0.5, 0., -0.6 - radius]
]
focus_dist_strs_list = [
    ['middle'], # low polar angle
    ['middle'], # low polar angle
    ['far', 'near'], # high polar angle
    ['far', 'near'], # high polar angle
    ['middle'], # top view
]

## debug
# focus_dists_list = num_polar_angles * [[radius]]
# focus_dist_strs_list = num_polar_angles * [['middle']]
assert (num_polar_angles == len(focus_dists_list))
assert (num_polar_angles == len(focus_dist_strs_list))

num_spps = 128 # samples per pixel
light_angle_list = [ # [deg], for simulations
    [0, 60],
    [90, 60],
    [180, 60],
    # [0, 0],
]
# light_angle_list = [ # [deg], for getting environment maps for experiment
#     [180, 45],
#     [270, 45],
# ]
expsr_times = [ # [sec]
    # 0.128,
    0.512,
    # 2.048
]

#### ---- phys_cam parameter setting (Flir Blackfly S BFS-U3-31S4C) ---- ####
## default control parameters
aperture_num = 4.0 # [1/1], for the 5mm-lens, (F1.6, F1.79, F2, F2.83, F4, F5.66, F8, F16)
expsr_time = 0.256 # [sec], (0.032, 0.064, 0.128, 0.256, 0.512, 1.024, 2.048)
ISO = 100 * np.power(10, 0/20) # [%]
focus_dist = 1.0 * radius # [m]

## camera model parameters
pixel_size = 3.45e-6 # [m]
rgb_QEs = [0.33707, 0.46310, 0.73207]
gain_params = {
    'defocus_gain': 5.5227,
    'defocus_bias': 0.,
    'vignet_gain': 0.483918,
    'aggregator_gain': 1.0e8, # proportional gain of illumination aggregation, [1/1]
    'expsr2dv_gains': [1., 1., 1.],
    'expsr2dv_biases': [0.036228, 0.055865, 0.111530],
    'expsr2dv_gamma': 0.,
    'expsr2dv_crf_type': 'linear',
    'max_CoC': 31,
}

noise_params = {
    'dark_currents': [0.000166311, 0.000341295, 0.000680946], # dark currents and hot-pixel noises, [electron/sec]
    'noise_gains': [0.20 * 0.00182512, 0.20 * 0.00215293, 0.20 * 0.00318984], # temporal noise gains, [1/1]
    'STD_reads': [0.20 * 2.56849e-05, 0.20 * 4.08999e-05, 0.20 * 8.33132e-05], # STDs of FPN and readout noises, [electron]
    'FPN_rng_seed': 1234, # seed of random number generator for readout and FPN noises
}

max_scene_light = 1080 # [lux = lumen/m^2]

## lens parameters (Arducam LN042 5mm lens)
focal_length = 0.005 # [m]
# hFOV = 63.7 * pi/180 # [rad]
hFOV = 63.7 * pi/180 # [rad]
sensor_width = 2 * focal_length * np.tan(hFOV / 2) # [m]
distort_params = [-0.141590, 0.162831, -0.108763] # (k1, k2, k3)

## estimate depth of field (DoF)
"""
cond = [["low_polar_midldle"], ["high_polar_far", "high_polar_near"], ["top_view_middle"]]
print("focused range")
for focus_dists_idx in range(len(focus_dists_list)):
    for focus_dist_idx in range(len(focus_dists_list[focus_dists_idx])):
        focus_dist = focus_dists_list[focus_dists_idx][focus_dist_idx]
        d_near = (focal_length**2 * focus_dist) / (focal_length**2 + aperture_num * pixel_size * (focus_dist - focal_length))
        d_far = (focal_length**2 * focus_dist) / (focal_length**2 - aperture_num * pixel_size * (focus_dist - focal_length))
        print(f"{cond[focus_dists_idx][focus_dist_idx]}: [{d_near: .4f}, {d_far: .4f}]")
"""

## Set up cameras ##
cameras_list = []
camera_angles_list = [] # [deg: %s]
camera_origins_list = []

for polar_idx in range(num_polar_angles):
    theta = polar_angles[polar_idx]
    cameras = []
    camera_angles = []
    camera_origins = []
    num_azi_angles = num_azi_angles_list[polar_idx]
    for azi_idx in range(num_azi_angles):
        phi = 1*pi/4 + 2 * pi * (azi_idx) / num_azi_angles
        camera, camera_angle, camera_origin = GetCamera(
            phi, theta, hFOV,
            [0., 0., 1.] if (polar_idx != num_polar_angles - 1) else [0., 1., 0.]
        )
        cameras.append(camera)
        camera_angles.append(camera_angle)
        camera_origins.append(camera_origin)
        
    cameras_list.append(cameras)
    camera_angles_list.append(camera_angles)
    camera_origins_list.append(camera_origins)


#### create/save/load HDR environment maps ####
envmap_dict = {}
for light_angle in light_angle_list:
    map_name = str(light_angle[0]).zfill(3) + "_" + str(light_angle[1]).zfill(3)
    map_path_prefix = os.path.join(dst_base_dir, dst_envmap_dir, "envmap_" + map_name)
    mi_azimuth, mi_polar = CvtMiTsubaEnvAngles(light_angle[0] * pi/180.0, light_angle[1] * pi/180.0)
    
    ## create env map for creating scene in Mitsuba
    envmap_dict[map_name] = CreateDirectLightEnvmap(
        mi_azimuth,
        mi_polar,
        2000.0 * np.array([1.0, 1.0, 1.0]), # for simulations
        0.5,
        map_path_prefix + "_mitsuba.hdr" if save_envlgt else None,
    )
    
    ## create and save env map for use in NVDiffRecMC
    _ = CreateDirectLightEnvmap(
        (270.0 - light_angle[0]) % 360.0 * pi/180.0, # convert to NVDiffRast EnvMap's azimuth angle
        light_angle[1] * pi/180.0,
        # 2000.0 * np.array([1.0, 1.0, 1.0]), # for simulations
        2000.0 * np.array([1.0, 1.0, 1.0]), # for experiments
        # 0.5, # for simulations
        1.0, # for experiments
        map_path_prefix + "_nvDiffRec.hdr" if save_envlgt else None,
    )

############################
#### Scene Construction ####
############################
scene_dict = {
    'type': 'scene',
    'light': {
        'type': 'envmap',
        # 'filename': map_path,
        'scale': 1.0,
        'bitmap': mi.Bitmap(envmap_dict[map_name]),
    },
    # 'light': {
    #     'type': 'directional',
    #     'direction': [0.0, -1.0, -1.0],
    #     'irradiance': {
    #         'type': 'rgb',
    #         'value': 5.0,
    #     },
    # },
}

## insert objects(rocks, wooden_logs, etc.)
obj_start_idx = 1 if ignore_ground else 0
# for obj_idx in [0, 1]:
for obj_idx in range(obj_start_idx, len(obj_names_list)):
    obj_name = obj_names_list[obj_idx]
    obj_path = obj_base_dir + obj_cmpny_name + "_" + obj_name + "/" + obj_name
    mesh_path = obj_path + ".obj"
    albedo_path = obj_path + "_albedo.png"
    roughness_path = obj_path + "_roughness.png"
    normal_path = obj_path + "_normal.png"
    scene_dict[obj_name] = {
        'type': 'obj',
        'filename': mesh_path,
        'to_world': T.translate(obj_poses[obj_idx][0]).rotate(axis=obj_poses[obj_idx][1][1:4], angle=obj_poses[obj_idx][1][0]).scale(obj_scales[obj_idx]),
        'bsdf_w_normal': {
            'type': 'normalmap',
            'normalmap': {
                'type': 'bitmap',
                'raw': True,
                'filename': normal_path,
            },
            'bsdf': {
                'type': 'principled',
                'base_color': {'type': 'bitmap', 'filename': albedo_path},
                'metallic': obj_metallics[obj_idx],
                'roughness': {'type': 'bitmap', 'raw': True, 'filename': roughness_path},
                'specular': obj_speculars[obj_idx]
            },
        },
    }

# if (ignore_ground):
#     obj_name = obj_names_list[0]
#     obj_path = obj_base_dir + obj_cmpny_name + "_" + obj_name + "/" + obj_name
#     mesh_path = obj_path + ".obj"
#     albedo_path = obj_path + "_albedo.png"
#     roughness_path = obj_path + "_roughness.png"
#     normal_path = obj_path + "_normal.png"
#     scene_dict[obj_name] = {
#         'type': 'obj',
#         'filename': mesh_path,
#         'to_world': T.translate(obj_poses[0][0]).rotate(axis=obj_poses[0][1][1:4], angle=obj_poses[0][1][0]).scale(obj_scales[0]),
#         'bsdf': {
#             'type': 'diffuse',
#             'reflectance': {
#                 'type': 'rgb',
#                 'value': [0.5, 0.5, 0.5],
#             },
#         },
#     }

scene = mi.load_dict(scene_dict)
scene_params = mi.traverse(scene)

#### build camera configuration table
camera_configs_list = []
for polar_angle_idx in range(len(polar_angles)):
    camera_configs = []
    for focus_dist_idx in range(len(focus_dists_list[polar_angle_idx])):
        focus_dist = focus_dists_list[polar_angle_idx][focus_dist_idx]
        focus_dist_str = focus_dist_strs_list[polar_angle_idx][focus_dist_idx]
        for expsr_time in expsr_times: # [sec]
            camera_configs.append({
                'aperture_num': 4.0, 'expsr_time': expsr_time, 'ISO': 100 * np.power(10, 0/20),
                'focal_length': focal_length, 'focus_dist': focus_dist, 'focus_dist_str': focus_dist_str,
            })
    
    camera_configs_list.append(camera_configs)

#### create the physic-based differentiable camera ####
device = 'cuda' if ('cuda' in mi.variant()) else 'cpu'

phys_camera = PhysDiffCamera(img_h, img_w, torch_random_seed, device)
# camera_config = {
#     'aperture_num': aperture_num,
#     'expsr_time': expsr_time,
#     'ISO': ISO,
#     'focal_length': focal_length,
#     'focus_dist': focus_dist
# }
# phys_camera.SetCtrlParameters(camera_config)
phys_camera.SetModelParameters(sensor_width, pixel_size, max_scene_light, rgb_QEs, gain_params, noise_params)
phys_camera.BuildVignetMask(sensor_width, focal_length)
phys_camera.artifact_switches = {
    "vignetting":   True,
    "defocus_blur": True,
    "aggregate":    True,
    "add_noise":    True,
    "expsr2dv":     True,
}

#### create g.t. image set ####
sparse_tensor_dict = {}
for polar_idx in range(num_polar_angles):
    sparse_tensor_dict[polar_idx] = {}
    for azi_idx in range(num_azi_angles_list[polar_idx]):
        sparse_tensor_dict[polar_idx][azi_idx] = {}

num_gt_imgs = len(light_angle_list) * np.sum([
    num_azi_angles_list[polar_idx] * len(camera_configs_list[polar_idx]) for polar_idx in range(num_polar_angles)
])

rgb_integrator = mi.load_dict({'type': 'path'})
depth_integrator = mi.load_dict({'type': 'depth'})

trnsfrms_and_configs = {}
trnsfrms_and_configs["fov_x"] = hFOV # [rad]
trnsfrms_and_configs["focal_length"] = focal_length # [m]
trnsfrms_and_configs["frames"] = {}

gt_imgs = []
depth_maps = [] # [m]
defocus_D_maps = []
img_idx = 0
t_start = time.time()
for light_angle_idx in range(len(light_angle_list)):
    light_angle = light_angle_list[light_angle_idx]
    envmap_key = str(light_angle[0]).zfill(3) + "_" + str(light_angle[1]).zfill(3)
    scene_params['light.data'] = envmap_dict[envmap_key]
    scene_params.update()
    gt_imgs.append([])
    depth_maps.append([])
    defocus_D_maps.append([])
    for polar_idx in range(num_polar_angles):
        gt_imgs[light_angle_idx].append([])
        depth_maps[light_angle_idx].append([])
        defocus_D_maps[light_angle_idx].append([])
        for azi_idx in range(num_azi_angles_list[polar_idx]):
            gt_imgs[light_angle_idx][polar_idx].append([])
            depth_maps[light_angle_idx][polar_idx].append([])
            defocus_D_maps[light_angle_idx][polar_idx].append([])
            for camera_config in camera_configs_list[polar_idx]:  
                cam_ctrl_params = [camera_config["aperture_num"],
                                   camera_config["expsr_time"],
                                   camera_config["ISO"],
                                   camera_config["focal_length"],
                                   camera_config["focus_dist"]]
                camera = cameras_list[polar_idx][azi_idx]
                radiance = mi.render(scene, scene_params, sensor=camera, integrator=rgb_integrator, spp=num_spps * 2, seed=1234)
                depth_map = mi.render(scene, scene_params, sensor=camera, integrator=depth_integrator, spp=num_spps * 2, seed=1234
                )
                depth_map = depth_map[:, :, 0].numpy()
                ROI_map = np.where(depth_map > 1e-6, 1.0, 0.).astype(np.float32)[..., np.newaxis]
                gt_img = radiance[:, :, 0:3].torch()
                
                if (save_depth_map):
                    depth_map_name = ("defocus_" + camera_angles_list[polar_idx][azi_idx][0]
                                        + "_" + camera_angles_list[polar_idx][azi_idx][1]
                                        + "_" + camera_config['focus_dist_str'] + ".npy")
                    np.save(
                        os.path.join(dst_base_dir, dst_depth_map_dir, depth_map_name),
                        depth_map,
                    )

                if (phys_camera.artifact_switches["defocus_blur"]):
                    try:
                        sparse_tensor = sparse_tensor_dict[polar_idx][azi_idx][camera_config['focus_dist_str']]
                        defocus_D_map = sparse_tensor_dict[polar_idx][azi_idx][camera_config['focus_dist_str'] + "_D_map"]
                    
                    except KeyError:
                        sparse_tensor, defocus_D_map = phys_camera.GetSparseTensor(
                            cv2.resize(depth_map, (depth_map_w, depth_map_h), interpolation=cv2.INTER_NEAREST),
                            camera_config
                        )
                        sparse_tensor_dict[polar_idx][azi_idx][camera_config['focus_dist_str']] = sparse_tensor
                        sparse_tensor_dict[polar_idx][azi_idx][camera_config['focus_dist_str'] + "_D_map"] = defocus_D_map
                        
                        
                        ## save sparse tensor
                        if (save_sparse_tensor):
                            tensor_name = ("defocus_" + camera_angles_list[polar_idx][azi_idx][0]
                                          + "_" + camera_angles_list[polar_idx][azi_idx][1]
                                          + "_" + camera_config['focus_dist_str'] + ".npz")
                            # with gzip.open(
                            #     os.path.join(dst_base_dir, dst_defocus_mtrx_dir, tensor_name),
                            #     mode="wb",
                            #     compresslevel=4,
                            # ) as file:
                            # torch.save(
                            #     sparse_tensor,
                            #     os.path.join(dst_base_dir, dst_defocus_mtrx_dir, tensor_name),
                            # )
                            # torch.save({
                            #     'indices': sparse_tensor.indices(),
                            #     'values': sparse_tensor.values(),
                            #     'shape': sparse_tensor.shape
                            # }, os.path.join(dst_base_dir, dst_defocus_mtrx_dir, tensor_name))
                            np.savez_compressed(
                                os.path.join(dst_base_dir, dst_defocus_mtrx_dir, tensor_name),
                                indices=sparse_tensor.indices().numpy(),
                                values=sparse_tensor.values().numpy(),
                                shape=sparse_tensor.shape,
                            )
                
                gt_img = phys_camera(
                    torch.tensor(radiance).unsqueeze(0).to(device),
                    [sparse_tensor.to(device)] if phys_camera.artifact_switches["defocus_blur"] else None,
                    torch.tensor([cam_ctrl_params], dtype=torch.float32),
                    depth_map_h, depth_map_w,
                )
                gt_img = gt_img.squeeze(0)
                
                # gt_imgs[light_angle_idx][camera_idx].append(torch.tensor(radiance).detach().cpu().numpy())
                gt_img = gt_img.detach().cpu().numpy()
                gt_img[:, :, 0][ROI_map[:, :, 0] < 0.5] = 0.
                gt_img[:, :, 1][ROI_map[:, :, 0] < 0.5] = 0.
                gt_img[:, :, 2][ROI_map[:, :, 0] < 0.5] = 0.
                gt_img = np.concatenate((gt_img, ROI_map), axis=2)
                
                gt_imgs[light_angle_idx][polar_idx][azi_idx].append(gt_img)
                depth_maps[light_angle_idx][polar_idx][azi_idx].append(depth_map)
                defocus_D_maps[light_angle_idx][polar_idx][azi_idx].append(
                    defocus_D_map if phys_camera.artifact_switches["defocus_blur"] else None
                )
                
                if (save_gt_img):
                    ## save gt_img to folder ##
                    assert(cv2.imwrite(
                        os.path.join(dst_base_dir, dst_img_dir, "img" + str(img_idx).zfill(4) + ".png"),
                        cv2.cvtColor((gt_img * 255).astype(np.uint8), cv2.COLOR_RGBA2BGRA),
                    ))
                    
                ## save cam_confing to folder ##
                transform = np.array(camera.world_transform().matrix)
                transform = np.squeeze(transform, axis=0)[0 : 3, :]
                transform = [[round(float(element), 8) for element in row] for row in transform]
                
                trnsfrms_and_configs["frames"][img_idx] = {
                    "trnsfrm": transform,
                    "cam_ctrl_params": [
                        round(float(camera_config["aperture_num"]), 8),
                        round(float(camera_config["expsr_time"]), 8),
                        round(float(camera_config["ISO"]), 8),
                        round(float(camera_config["focal_length"]), 8),
                        round(float(camera_config["focus_dist"]), 8),
                    ],
                    "defocus_mtrx_name": (
                        "defocus_" + camera_angles_list[polar_idx][azi_idx][0]
                        + "_" + camera_angles_list[polar_idx][azi_idx][1]
                        + "_" + camera_config['focus_dist_str']
                    ),
                    "envmap_name": "envmap_" + envmap_key,
                }
                
                ## print log ##
                if ((img_idx + 1) % 10 == 0 or img_idx == 0):
                    print(f"g.t. Img [{img_idx+1}/{num_gt_imgs}] created")
                
                img_idx += 1

if (save_trnsfrms_and_configs):
    with open(os.path.join(dst_base_dir, "trnsfrms_and_configs_" + mode + ".json"), "w") as outfile: 
        json.dump(trnsfrms_and_configs, outfile, indent=4)

t_end = time.time()

print(f"takes {np.round(t_end - t_start, 4)} sec to build g.t. set")            


## Show g.t. photos ##
if (show_gt_imgs):
    # PlotImages(gt_imgs, num_polar_angles, num_azimuth_angles)
    gt_img_idx = 0
    for light_idx in range(len(light_angle_list)):
        for polar_idx in range(num_polar_angles):
            for azi_idx in range(num_azi_angles_list[polar_idx]):
                for camera_config_idx in range(len(camera_configs_list[polar_idx])):
                    print(f"Img {gt_img_idx+1}:")
                    gt_img_idx += 1
                    
                    # plt.figure()
                    # plt.imshow(mi.util.convert_to_bitmap(gt_imgs[light_idx][camera_idx][camera_config_idx]))
                    title = f'phi={camera_angles_list[polar_idx][azi_idx][0]}, '
                    title += f'theta={camera_angles_list[polar_idx][azi_idx][1]}, '
                    title += f'light{light_angle_list[light_idx]}, '
                    title += f'expsr_t={round(1000*camera_configs_list[polar_idx][camera_config_idx]["expsr_time"])}, '
                    title += f'focus_dist={np.round(camera_configs_list[polar_idx][camera_config_idx]["focus_dist"], 3)}'
                    # plt.title(title)
                    
                    num_subplots = 2 + int(phys_camera.artifact_switches["defocus_blur"])
                    fig, axs = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 6))
                    # axs[0].imshow(mi.util.convert_to_bitmap(gt_imgs[light_idx][polar_idx][azi_idx][camera_config_idx]))
                    axs[0].imshow((gt_imgs[light_idx][polar_idx][azi_idx][camera_config_idx] * 255).astype(np.uint8))
                    axs[0].axis('off')
                    
                    depth_map = depth_maps[light_idx][polar_idx][azi_idx][camera_config_idx]
                    ax1 = axs[1].imshow(depth_map, cmap='plasma', vmin=depth_map.min(), vmax=depth_map.max())
                    plt.colorbar(ax1, ax=axs[1], orientation='horizontal', pad=0.01, shrink=0.90)
                    # plt.colorbar(ax1)
                    axs[1].axis('off')
                    
                    defocus_D_map = defocus_D_maps[light_idx][polar_idx][azi_idx][camera_config_idx]
                    # vmax = defocus_D_map.max()
                    vmax = 10
                    if (defocus_D_map is not None):
                        ax2 = axs[2].imshow(defocus_D_map, cmap='plasma', vmin=defocus_D_map.min(), vmax=vmax)
                        plt.colorbar(ax2, ax=axs[2], orientation='horizontal', pad=0.01, shrink=0.90)
                        axs[2].axis('off')
                    
                    plt.suptitle(title, y=0.8)
                    plt.show()
"""





