#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate camera pose table in JSON file and create folders for generating and saving

radiance of scene in Chrono

@author: Bo-Hsun
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
# DRJIT_LIBLLVM_PATH="/usr/lib/llvm-15/lib/libLLVM.so"

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


#############################
#### Function set starts ####
#############################
## return camera, angles of polar coordinates of camera, and camera's origin,
## given polar (theta) and azimuth (phi) angles
def GetCamera(phi, theta, radius, hFOV, up_direction):
    '''
    return camera, angles of polar coordinates of camera, and camera's origin,
    given polar (theta) and azimuth (phi) angles
    '''
    camera_angles = [str(round(theta * 180/pi)).zfill(3), str(round(phi * 180/pi)).zfill(3)]
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

###########################
#### Function set ends ####
###########################

create_folders = 0
save_trnsfrm_table = 1

dst_folder_dir = "/home/bohsun/UW_Madison/Research/build_my_chrono_wisc/bin/GetSceneRadiance/DEPTH_CAM/BIN/"
# dst_trnsfrm_table_path = "/home/bohsun/UW_Madison/Research/MyDiffRecMC/trnsfrms_and_configs_train.json"
dst_trnsfrm_table_path = "/home/bohsun/UW_Madison/Research/MyDiffRecMC/sim_cam_trnsfrm_table.json"

radius = 0.88 * 1.414 # [m]

hFOV = 64.61874824610845 * pi/180.0 # [rad]
img_w = 1024 # [px]
img_h = 768 # [px]
num_spps = 128 # samples per pixel

polar_angles = [ # [rad]
    40 * (pi/180.0),
    55 * (pi/180.0),
    70 * (pi/180.0),
    85 * (pi/180.0),
    0 * (pi/180.0),
]
num_polar_angles = len(polar_angles)
num_azi_angles_list = [
    20, 
    24,
    24,
    30,
    1, # top view
    # 4, # debug
    # 4, # debug
    # 4, # debug
    # 5, # debug
    # 1, # top view
]
assert (num_polar_angles == len(num_azi_angles_list))

## Set up cameras ##
cams = []
cam_angles = [] # [deg: %s]
cam_trnsfrm_dict = {}
# cam_trnsfrm_dict["fov_x"] = hFOV
# cam_trnsfrm_dict["frames"] = {}
cam_trnsfrm_idx = 0
for polar_idx in range(num_polar_angles):
    
    theta = polar_angles[polar_idx]
    num_azi_angles = num_azi_angles_list[polar_idx]
    
    for azi_idx in range(num_azi_angles):
        phi = 0 * pi/4 + 2 * pi * (azi_idx) / num_azi_angles
        cam, cam_angle, cam_origin = GetCamera(
            phi, theta, radius, hFOV,
            [0., 0., 1.] if (polar_idx != num_polar_angles - 1) else [0., 1., 0.]
        )
        
        cams.append(cam)
        cam_angles.append(cam_angle)
        
        transform = np.array(cam.world_transform().matrix)
        transform = np.squeeze(transform, axis=0)[0 : 3, :]
        transform = [[round(float(element), 8) for element in row] for row in transform]
        
        trnsfrm_name = f"polar_{cam_angle[0]}_azi_{cam_angle[1]}"
        if (create_folders):
            os.makedirs(os.path.join(dst_folder_dir, trnsfrm_name), exist_ok=True)
        
        cam_trnsfrm_dict[cam_trnsfrm_idx] = {
            "name": trnsfrm_name,
            "trnsfrm": transform,
        }
        
        # cam_trnsfrm_dict["frames"][cam_trnsfrm_idx] = {
        #     "img_name": "sun_000_" + trnsfrm_name + ".png",
        #     "trnsfrm": transform,
        #     "cam_ctrl_params": [0., 0., 0., 0., 0.],
        #     "defocus_mtrx_name": "",
        #     "envmap_name": "envmap_000_045_nvDiffRec",
        # }
        
        cam_trnsfrm_idx += 1
        
cam_trnsfrm_dict["length"] = cam_trnsfrm_idx


if (save_trnsfrm_table):
    with open(dst_trnsfrm_table_path, "w") as outfile: 
        json.dump(cam_trnsfrm_dict, outfile, indent=4)        
        
