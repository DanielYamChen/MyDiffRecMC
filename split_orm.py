#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split roughness and metallic maps from orm map and convert maps into 8-bit format

@author: Bo-Hsun Chen
"""

import numpy as np
import cv2
import os

#### arguments ####
mesh_dir = "../TextureMaterial/experiment_myst_village/"

#### parameters ####
src_orm_name = "texture_ks.png"
src_albedo_name = "texture_kd.png"
src_normal_name = "texture_n.png"

dst_albedo_name = "albedo.png"
dst_rough_name = "rough.png"
dst_metal_name = "metal.png"
dst_normal_name = "normal.png"

albedo = cv2.imread(os.path.join(mesh_dir, src_albedo_name), cv2.IMREAD_UNCHANGED)
assert(albedo is not None)
orm = cv2.imread(os.path.join(mesh_dir, src_orm_name), cv2.IMREAD_UNCHANGED)
assert(orm is not None)
normal = cv2.imread(os.path.join(mesh_dir, src_normal_name), cv2.IMREAD_UNCHANGED)
assert(normal is not None)

albedo = cv2.cvtColor(albedo, cv2.COLOR_BGRA2RGB)
orm = cv2.cvtColor(orm, cv2.COLOR_BGRA2RGB)
normal = cv2.cvtColor(normal, cv2.COLOR_BGRA2RGB)

albedo = albedo.astype(float) / np.iinfo(albedo.dtype).max
orm = orm.astype(float) / np.iinfo(orm.dtype).max
normal = normal.astype(float) / np.iinfo(normal.dtype).max

albedo = np.round(albedo * 255.0).astype(np.uint8)
rough = np.round(orm[:, :, 1] * 255.0).astype(np.uint8)
metal = np.round(orm[:, :, 2] * 255.0).astype(np.uint8)
normal = np.round(normal * 255.0).astype(np.uint8)

#### save files ####
assert(cv2.imwrite(
    os.path.join(mesh_dir, dst_albedo_name),
    cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR),
))
assert(cv2.imwrite(
    os.path.join(mesh_dir, dst_rough_name),
    rough,
))
assert(cv2.imwrite(
    os.path.join(mesh_dir, dst_metal_name),
    metal,
))
assert(cv2.imwrite(
    os.path.join(mesh_dir, dst_normal_name),
    cv2.cvtColor(normal, cv2.COLOR_RGB2BGR),
))


