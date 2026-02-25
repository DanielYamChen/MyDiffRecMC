#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Develop parallelizing xatlas.parametrize when addressing large data

@author: Bo-Hsun Chen
"""

import numpy as np
import xatlas
import multiprocessing
import time
import matplotlib.pyplot as plt
import bpy
import igl

v_pos_path = "./v_pos_256.npz"
t_pos_idx_path = "./t_pos_idx_256.npz"

v_pos = np.load(v_pos_path)['val']
# t_pos_idx = np.load(t_pos_idx_path)['val'][:100000]
t_pos_idx = np.load(t_pos_idx_path)['val']

print(f"v_pos.shape={v_pos.shape}")
print(f"t_pos_idx.shape={t_pos_idx.shape}")

"""
# Create a new mesh in Blender
mesh = bpy.data.meshes.new(name="my_mesh")
mesh.from_pydata(v_pos.tolist(), [], t_pos_idx.tolist())
mesh.update()

# Create a new object and link to the scene
obj = bpy.data.objects.new("my_obj", mesh)
bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj

# Set the object mode to EDIT and unwrap the mesh
bpy.ops.object.mode_set(mode='EDIT')

# Add a UV map if it doesn't already exist
if len(mesh.uv_layers) == 0:
    bpy.ops.mesh.uv_texture_add()

# Use LSCM (Least Squares Conformal Mapping) UV unwrapping
t_start = time.time()
bpy.ops.uv.unwrap(method='CONFORMAL', margin=0.001)  # Conformal is LSCM in Blender
# bpy.ops.uv.smart_project()  # Use Blender's UV unwrapping
print(f"elapsed time: {(time.time() - t_start): .3f} sec")

# Switch back to OBJECT mode and get the UV map from the object
bpy.ops.object.mode_set(mode='OBJECT')
uv_layer = obj.data.uv_layers.active.data

# Convert UV coordinates to numpy array
uvs = np.array([[loop.uv.x, loop.uv.y] for loop in uv_layer])
vmapping = np.arange(len(v_pos))  # Identity mapping of vertices
indices = t_pos_idx               # Same indices as input
"""

"""
mode = "multi"
# mode = "single"

rows_chunks = 2
cols_chunks = 3
num_chunks = rows_chunks * cols_chunks

# Function to run parametrization on a chunk
def parametrize_chunk(chunk):
    return xatlas.parametrize(chunk[0], chunk[1])



t_start = time.time()

if (mode == "multi"):
    
    v_pos_chunks = num_chunks * [v_pos]
    # t_pos_idx_chunks = 6 * [t_pos_idx]
    # t_pos_idx_chunks = np.array_split(t_pos_idx, num_chunks)
    t_pos_idx_chunks = [
        t_pos_idx[np.where((v_pos[t_pos_idx[:, 0], 2] < 0) & (v_pos[t_pos_idx[:, 0], 0] < -0.4))[0]],
        t_pos_idx[np.where((v_pos[t_pos_idx[:, 0], 2] < 0) & (-0.4 <= v_pos[t_pos_idx[:, 0], 0]) & (v_pos[t_pos_idx[:, 0], 0] < 0.4))[0]],
        t_pos_idx[np.where((v_pos[t_pos_idx[:, 0], 2] < 0) & (0.4 <= v_pos[t_pos_idx[:, 0], 0]))[0]],
        t_pos_idx[np.where((v_pos[t_pos_idx[:, 0], 2] >= 0) & (v_pos[t_pos_idx[:, 0], 0] < -0.4))[0]],
        t_pos_idx[np.where((v_pos[t_pos_idx[:, 0], 2] >= 0) & (-0.4 <= v_pos[t_pos_idx[:, 0], 0]) & (v_pos[t_pos_idx[:, 0], 0] < 0.4))[0]],
        t_pos_idx[np.where((v_pos[t_pos_idx[:, 0], 2] >= 0) & (0.4 <= v_pos[t_pos_idx[:, 0], 0]))[0]],
    ]
    
    with multiprocessing.Pool(num_chunks) as pool:
        results = pool.map(parametrize_chunk, zip(v_pos_chunks, t_pos_idx_chunks))
    
    
    vmapping, indices, uvs = [], [], []
    total_num_vertices = 0
    for result_idx in range(num_chunks):
    # for result_idx in [3]:
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
    
elif (mode == "single"):
    vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)

print(f"elapsed time: {1000 * (time.time() - t_start): .3f} ms")
"""


# Compute the LSCM parametrization
bnd = igl.boundary_loop(t_pos_idx)  # Find the boundary loop (for fixed points)
# uv = np.zeros((v_pos.shape[0], 2), dtype=np.float32)  # Array to store the uv coordinates

## Map the boundary to a circle, preserving edge proportions
bnd_uv = igl.map_vertices_to_circle(v_pos, bnd)

## Harmonic parametrization for the internal vertices
uv = igl.harmonic(v_pos, t_pos_idx, bnd, bnd_uv, 1.0)

arap = igl.ARAP(v_pos, t_pos_idx, 2, np.zeros(0), with_dynamics=True)
uva = arap.solve(np.zeros((0, 0)), uv)

# Output
vmapping = np.arange(len(v_pos))  # Identity mapping of vertices
indices = t_pos_idx               # Same indices as input
uvs = uva                          # Resulting UV coordinates

print(f"vmapping.shape={vmapping.shape}")
print(f"indices.shape={indices.shape}")
print(f"uvs.shape={uvs.shape}")





