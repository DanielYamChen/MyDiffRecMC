#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrate environment light probe map in NVDiffRec

@author: Bo-Hsun Chen
"""

import numpy as np
import math
from math import pi
import torch
import nvdiffrast.torch as dr
import argparse
import matplotlib.pyplot as plt
import cv2
import os
import importlib.util
import sys
import json

from render import util
from render import mesh
from render import render
from render import light
import render.optixutils as ou

# cond = "simulation"
cond = "experiment"

save_envlgt = 1
viz = 1
fig_dpi = 200
num_rounds = 1

if (cond == "simulation"):
    obj_path = "/home/bohsun/UW_Madison/Research/TextureMaterial/calibrate_cube/calibrate_cube.obj"
    envlight_path = "/home/bohsun/UW_Madison/Research/MyDiffRecMC/data/custom/envmaps/envmap_090_045_nvDiffRec.hdr"
    # chrono_img_path = "/home/bohsun/UW_Madison/Research/CameraCalibrateExp/Data/Synth/ENVLIGHT_PNG/ambient_calibrate_Kd_0.5_rough_1.0_metallic_0.png"
    # chrono_img_path = "/home/bohsun/UW_Madison/Research/CameraCalibrateExp/Data/Synth/ENVLIGHT_PNG/point_calibrate_Kd_0.2_rough_0.4_metallic_0.png"
    chrono_img_path = "/home/bohsun/UW_Madison/Research/CameraCalibrateExp/Data/Synth/ENVLIGHT_PNG/all_light_calibrate.png"
    dst_envmap_dir = "./data/custom/envmaps/" # base directory for saving environment maps

    phys_cam = None
    cam_ctrl_params = [0., 0., 0., 0., 0.]
    point_light_color = [1.0, 1.0, 1.0]

elif (cond == "experiment"):
    # obj_path = "/home/bohsun/UW_Madison/Research/TextureMaterial/calibrate_paper/calibrate_paper.obj"
    obj_path = "/home/bohsun/UW_Madison/Research/MyDiffRecMC/out/experiment/mesh/mesh.obj"
    envlight_path = "/home/bohsun/UW_Madison/Research/MyDiffRecMC/data/experiment/envmaps/envmap_000_045_nvDiffRec_exp.hdr"
    
    # photo_path = "/home/bohsun/UW_Madison/Research/DiffCamExp/ExpData/Ambient_PNG/N_020_ISO_00_t_0064.png"
    # photo_path = "/home/bohsun/UW_Madison/Research/DiffCamExp/ExpData/Ambient_PNG/N_020_ISO_00_t_0128.png"
    # photo_path = "/home/bohsun/UW_Madison/Research/DiffCamExp/ExpData/Direct_PNG/N_020_ISO_00_t_0016.png"
    photo_path = "/home/bohsun/UW_Madison/Research/DiffCamExp/ExpData/Direct_PNG/N_020_ISO_00_t_0032.png"
    
    dst_envmap_dir = "./data/experiment/envmaps/" # base directory for saving environment maps
    
    # cam_ctrl_params = [2.0, int(photo_path[-8:-4]) / 1000.0, 100.0, 0.005, 0]
    cam_ctrl_params = [2.0, 0.128, 100.0, 0.005, 0]
    point_light_color = [255/255.0, 228/255.0, 206/255.0]

img_h = 768
img_w = 1024
aspect = img_w / img_h
fov_x = 64.61874824610845 * pi/180.0 # [rad]
# fov_x = 55.7 * pi/180.0 # [rad]
fov_y = util.fovx_to_fovy(fov_x, aspect)
# fov_y = pi/3
cam_near_far = [0.0001, 1e8] # [m]

parser = argparse.ArgumentParser(description='nvdiffrecmc')
flags = parser.parse_args()
flags.add_defocus_net = False
flags.decorrelated = False
flags.n_samples = 12

if (cond == "simulation"):
    flags.add_phys_cam = False

elif (cond == "experiment"):
    # Work Light 10,000 LM 5000K 1000W Halogen Equivalent led Work Light
    max_scene_light = 10000 * 0.10 / np.sum(np.array([0.825, 0., 0.825])**2) # [lux = lumen/m^2]
    flags.add_phys_cam = True
    flags.defocus_tensor_res = [0, 0]
    
    phys_cam_path = "./models/cam_model.py"
    phys_cam_model_params_path = "./sim_cam_model_params.json"    
    
    spec = importlib.util.spec_from_file_location("phys_cam", phys_cam_path)
    my_modules = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = my_modules
    spec.loader.exec_module(my_modules)

    phys_cam = my_modules.PhysDiffCamera(img_h, img_w, 1234, 'cuda')
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
    focal_length = cam_params["focal_length"] # [m]
    hFOV = cam_params["hFOV"] * pi / 180.0 # [rad]
    sensor_width = 2 * focal_length * np.tan(hFOV / 2) # [m]

    phys_cam.BuildVignetMask(sensor_width, focal_length) 
    phys_cam.artifact_switches = {
        "vignetting": True,
        "defocus_blur": False,
        "aggregate": True,
        "add_noise": False,
        "expsr2dv": True,
    }

    phys_cam.SetModelParameters(sensor_width, pixel_size, max_scene_light, rgb_QEs, gain_params, noise_params)

    
spp = 1
num_layers = 1

theta = np.arctan(0.560 / 1.200)
# cam_trnsfrm = torch.tensor([ # experiment ambient light, in Mitsuba format
#     [ 0.000,  np.sin(theta),  np.cos(theta),  -1.2 * 1.2],
#     [ 1.000,         0.000,         0.000,     0.06],
#     [ 0.000,  np.cos(theta), -np.sin(theta),  0.56 * 1.2],
#     [0., 0., 0., 1.0],
# ], dtype=torch.float32)

cam_trnsfrm = torch.tensor([ # experiment directional light, in Mitsuba format
    [ 0.000,  np.sin(theta),  np.cos(theta),  -1.2 * 1.2],
    [ 1.000,         0.000,         0.000,     0.],
    [ 0.000,  np.cos(theta), -np.sin(theta),  0.56 * 1.2],
    [0., 0., 0., 1.0],
], dtype=torch.float32)

# cam_trnsfrm = torch.tensor([ # in Mitsuba format
#     [ 0.000,  0.000,  1.000, -1.50],
#     [ 1.000,  0.000,  0.000,  0.],
#     [ 0.000,  1.000,  0.000,  0.],
#     [    0.,      0.,     0., 1.0],
# ], dtype=torch.float32)

light_angle_list = [ # (azimuth, polar), [deg]
    [0, 45], # for calibrating
    # [0, 45],
    # [90, 45],
    # [180, 45],
    # [45, 82],
    # [135, 82],
    # [315, 82],
    # [180, 45],
    # [270, 45],
]

#############################
#### Function set starts ####
#############################
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
ref_mesh = mesh.load_mesh(obj_path, None)

print("Build Optix bvh")
optix_ctx = ou.OptiXContext()
ou.optix_build_bvh(optix_ctx, ref_mesh.v_pos, ref_mesh.t_pos_idx.int(), rebuild=1)
print("Done building OptiX bvh")

glctx = dr.RasterizeGLContext() # Context for training

iter_res = [img_h, img_w]

proj = util.perspective(fov_y, aspect, cam_near_far[0], cam_near_far[1])

if (cond == "simulation"):
    cam_trnsfrm = (util.rotate_x(np.pi / 2) @ cam_trnsfrm) @ util.rotate_y(np.pi)
elif (cond == "experiment"):
    cam_trnsfrm = util.rotate_y(-np.pi/2) @ ((util.rotate_x(np.pi / 2) @ cam_trnsfrm) @ util.rotate_y(np.pi))

model_view_trnsfrm = np.linalg.inv(cam_trnsfrm) # 4 x 4 matrix
cam_posi = cam_trnsfrm[:3, 3]
mvp_trnsfrm = proj @ model_view_trnsfrm
mvp_trnsfrm = mvp_trnsfrm[None, ...].to('cuda')
cam_posi = cam_posi[None, ...].to('cuda')

# ambient_light_amount = 0.288 # for Scenes 1 and 3
# point_light_amount = 2375 # for Scenes 1 and 3

# ambient_light_amount = 0.288 / 3 # for Scene 2
# point_light_amount = 2375 / 3 # for Scene 2

# ambient_light_amount = 0.57 # for experiment, t = 0.064 sec
# ambient_light_amount = 0.64 # for experiment, t = 0.128 sec
ambient_light_amount = 0.61 # for experiment
# ambient_light_amount = 0.61 * 2.4 # for experiment
# ambient_light_amount = 1.0 # for experiment, heuristics

# point_light_amount = 5080 # for experiment, t = 0.016 sec
# point_light_amount = 5220 # for experiment, t = 0.032 sec
point_light_amount = 5150 # for experiment
# point_light_amount = 5150 * 0.57 # for experiment
# point_light_amount = 2000.0 # for experiment, heuristics

for round_idx in range(num_rounds):
    ## create and save env map for use in NVDiffRecMC
    if (save_envlgt):
        for light_angle in light_angle_list:
            map_name = str(light_angle[0]).zfill(3) + "_" + str(light_angle[1]).zfill(3)
            map_path_prefix = os.path.join(dst_envmap_dir, "envmap_" + map_name)
            _ = CreateDirectLightEnvmap(
                (270.0 - light_angle[0]) % 360.0 * pi/180.0, # convert to NVDiffRast EnvMap's azimuth angle
                light_angle[1] * pi/180.0,
                point_light_amount * np.array(point_light_color), # for simulations
                # 0. * np.array([1.0, 1.0, 1.0]), # for calibrating ambient light
                ambient_light_amount, # for simulations
                # 0., # 1for calibrating point light amount
                map_path_prefix + "_nvDiffRec_exp.hdr" if save_envlgt else None,
            )
    
    
    new_ref_mesh = mesh.compute_tangents(ref_mesh)
    envlight = light.load_env(envlight_path, scale=1.0)
    
    buffer = render.render_mesh(
        flags,
        glctx,
        new_ref_mesh,
        mvp_trnsfrm,
        cam_posi,
        envlight,
        iter_res,
        None,
        torch.tensor([cam_ctrl_params], dtype=torch.float32),
        spp=spp,
        num_layers=num_layers,
        msaa=True,
        background=None,
        optix_ctx=optix_ctx,
        phys_cam=phys_cam,
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
    
    print(f"Round {round_idx}:")
    
    nv_img = buffer['shaded'].cpu().numpy()
    del buffer
    torch.cuda.empty_cache()
    ROI_map = nv_img[0, :, :, 3]
    nv_img = nv_img[0, :, :, :3]
    if (viz):
        plt.figure(dpi=fig_dpi)
        plt.imshow(nv_img)
        plt.show()
    
    ## read the standard img from Chrono::Sensor
    if (cond == "simulation"):
        chrono_img = cv2.imread(chrono_img_path, cv2.IMREAD_UNCHANGED)
        chrono_img = cv2.cvtColor(chrono_img, cv2.COLOR_BGRA2RGB)
        chrono_img = chrono_img.astype(float) / np.iinfo(chrono_img.dtype).max
        plt.imshow(chrono_img)
        plt.show()
        
        # mean_chrono_img = np.mean(chrono_img[chrono_img > 1e-8])
        # mean_nv_img = np.mean(nv_img[ROI_map > 1e-8])
        # ambient_light_amount *= mean_chrono_img / mean_nv_img
        # print(f"new ambient_light_amount = {ambient_light_amount:.4f}")
        
        # mean_chrono_img = np.mean(chrono_img[chrono_img > 0.4])
        # mean_nv_img = np.mean(nv_img[nv_img > 0.4])
        # point_light_amount *= mean_chrono_img / mean_nv_img
        # print(f"new point_light_amount = {point_light_amount:.4f}")
    '''
    elif (cond == "experiment"):
        photo = cv2.imread(photo_path, cv2.IMREAD_UNCHANGED)
        photo = photo.astype(np.float32) / np.iinfo(photo.dtype).max
        photo = cv2.resize(photo, (img_w, img_h), cv2.INTER_AREA)
        photo[ROI_map < 0.5] = 0
        if (viz):
            plt.figure(dpi=fig_dpi)
            plt.imshow(photo)
            plt.show()
        
        mean_photo = np.mean(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)[ROI_map > 0.5])
        mean_nv_img = np.mean(cv2.cvtColor(nv_img, cv2.COLOR_RGB2GRAY)[ROI_map > 0.5])
        # ambient_light_amount *= mean_photo / mean_nv_img
        # print(f"new ambient_light_amount = {ambient_light_amount:.2f}")
        point_light_amount *= mean_photo / mean_nv_img
        print(f"new point_light_amount = {round(point_light_amount / 10) * 10:d}")
   '''
    