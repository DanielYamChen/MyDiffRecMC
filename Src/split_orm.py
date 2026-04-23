#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split roughness and metallic maps from orm map and convert maps into 8-bit format

@author: Bo-Hsun Chen
"""

import numpy as np
import cv2
import os
import OpenEXR
import Imath

#### arguments ####
mesh_dir = "../../DiffPhysCam_Data/NovelViewSynthesis_Output/RealScene01_wo_defocus/mesh/"

#### parameters ####
src_orm_name = "texture_ks.png"
src_albedo_name = "texture_kd.png"
src_normal_name = "texture_n.png"

dst_albedo_name = "albedo.exr"
dst_rough_name = "rough.exr"
dst_metal_name = "metal.exr"
dst_normal_name = "normal.exr"

albedo = cv2.imread(os.path.join(mesh_dir, src_albedo_name), cv2.IMREAD_UNCHANGED)
assert(albedo is not None)
orm = cv2.imread(os.path.join(mesh_dir, src_orm_name), cv2.IMREAD_UNCHANGED)
assert(orm is not None)
normal = cv2.imread(os.path.join(mesh_dir, src_normal_name), cv2.IMREAD_UNCHANGED)
assert(normal is not None)

albedo = cv2.cvtColor(albedo, cv2.COLOR_BGRA2RGB)
orm = cv2.cvtColor(orm, cv2.COLOR_BGRA2RGB)
normal = cv2.cvtColor(normal, cv2.COLOR_BGRA2RGB)

albedo = albedo.astype(np.float32) / np.iinfo(albedo.dtype).max
orm = orm.astype(np.float32) / np.iinfo(orm.dtype).max
normal = normal.astype(np.float32) / np.iinfo(normal.dtype).max

# albedo = np.round(albedo * 255.0).astype(np.uint8)
# rough = np.round(orm[:, :, 1] * 255.0).astype(np.uint8)
# metal = np.round(orm[:, :, 2] * 255.0).astype(np.uint8)
# normal = np.round(normal * 255.0).astype(np.uint8)

#### Save files ####
FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

## Albedo
header_albedo = OpenEXR.Header(albedo.shape[1], albedo.shape[0])
header_albedo["channels"] = {
    "R": Imath.Channel(FLOAT),
    "G": Imath.Channel(FLOAT),
    "B": Imath.Channel(FLOAT),
}
exr_albedo = OpenEXR.OutputFile(os.path.join(mesh_dir, dst_albedo_name), header_albedo)
exr_albedo.writePixels({
    "R": np.ascontiguousarray(albedo[:, :, 0]).tobytes(),
    "G": np.ascontiguousarray(albedo[:, :, 1]).tobytes(),
    "B": np.ascontiguousarray(albedo[:, :, 2]).tobytes(),
})
exr_albedo.close()

## Roughness
header_rough = OpenEXR.Header(orm.shape[1], orm.shape[0])
header_rough["channels"] = {
    "R": Imath.Channel(FLOAT),
}
exr_rough = OpenEXR.OutputFile(os.path.join(mesh_dir, dst_rough_name), header_rough)
exr_rough.writePixels({
    "R": np.ascontiguousarray(orm[:, :, 1]).tobytes(),
})
exr_rough.close()

## Metallic
header_metal = OpenEXR.Header(orm.shape[1], orm.shape[0])
header_metal["channels"] = {
    "R": Imath.Channel(FLOAT),
}
exr_metal = OpenEXR.OutputFile(os.path.join(mesh_dir, dst_metal_name), header_metal)
exr_metal.writePixels({
    "R": np.ascontiguousarray(orm[:, :, 2]).tobytes(),
})
exr_metal.close()

## Normal
header_normal = OpenEXR.Header(normal.shape[1], normal.shape[0])
header_normal["channels"] = {
    "R": Imath.Channel(FLOAT),
    "G": Imath.Channel(FLOAT),
    "B": Imath.Channel(FLOAT),
}
exr_normal = OpenEXR.OutputFile(os.path.join(mesh_dir, dst_normal_name), header_normal)
exr_normal.writePixels({
    "R": np.ascontiguousarray(normal[:, :, 0]).tobytes(),
    "G": np.ascontiguousarray(normal[:, :, 1]).tobytes(),
    "B": np.ascontiguousarray(normal[:, :, 2]).tobytes(),
})
exr_normal.close()

