#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 17:35:17 2026

@author: bohsun
"""

import os
import json

# ===== USER CONFIG =====
json_path = "/home/bohsun/UW_Madison/Research/DiffPhysCam_Data/NovelViewSynthesis_Data/RealScene01/trnsfrms_and_configs_train_wo_ground.json"
base_dir = "/home/bohsun/UW_Madison/Research/chrono_wisc_bohsun/build_chrono_wisc_bohsun/bin/LuminanceCali_RealScene01/Cam/"  # change this to your desired base folder

# =======================

# Create base directory if not exists
os.makedirs(base_dir, exist_ok=True)

# Load JSON
with open(json_path, "r") as f:
    data = json.load(f)

frames = data["frames"]

# Use a set to avoid duplicate folder creation
folder_names = set()

for idx, frame in frames.items():
    img_name = frame["img_name"]
    
    # Remove "_t_XXXX.png"
    folder_name = img_name.split("_t_")[0]
    
    folder_names.add(folder_name)

# Create folders
for name in sorted(folder_names):
    folder_path = os.path.join(base_dir, name)
    os.makedirs(folder_path, exist_ok=True)

print(f"Created {len(folder_names)} folders in '{base_dir}'")