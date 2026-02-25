#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 17:05:42 2025

@author: bohsun
"""

import os
import shutil
import numpy as np
import cv2

def ReadImageFromFile(file_path):
    with open(file_path, 'rb') as file:
        # Read width, height, number of channels, and bit depth (each as uint16)
        width = int.from_bytes(file.read(2), byteorder='little')
        height = int.from_bytes(file.read(2), byteorder='little')
        num_ch = int.from_bytes(file.read(2), byteorder='little')
        bit_depth = int.from_bytes(file.read(2), byteorder='little')
        
        # Calculate the total number of pixels
        total_pixels = width * height * num_ch
        
        # Read the image data
        image_data = np.frombuffer(file.read(total_pixels * 2), dtype=np.uint16)
        
        # Reshape the data to match the image dimensions
        image = image_data.reshape((height, width, num_ch))
        image = cv2.flip(image, 0)
        
        return image


def ReadDepthFromFile(file_path):
    with open(file_path, 'rb') as file:
        # Read width, height, number of channels, and bit depth (each as uint16)
        width = int.from_bytes(file.read(2), byteorder='little')
        height = int.from_bytes(file.read(2), byteorder='little')
        num_chs = int.from_bytes(file.read(2), byteorder='little')
        bit_depth = int.from_bytes(file.read(2), byteorder='little')
        
        if (num_chs != 1):
            raise ValueError(f"Expected 1 channel, but got {num_chs}")
        
        # Calculate the total number of pixels
        total_pixels = width * height * num_chs
        
        # Read the map data
        depth_map = np.frombuffer(file.read(total_pixels * 4), dtype=np.float32)
        
        # Reshape the data to match the image dimensions
        depth_map = depth_map.reshape((height, width))
        depth_map = cv2.flip(depth_map, 0)
        
        return depth_map

src_img_folder = "/home/bohsun/UW_Madison/Research/build_my_chrono_wisc/bin/GetSceneRadiance/PHYS_CAM/BIN/"
dst_img_folder = "/home/bohsun/UW_Madison/Research/build_my_chrono_wisc/bin/GetSceneRadiance/PHYS_CAM/PNG/"
src_depth_folder = "/home/bohsun/UW_Madison/Research/build_my_chrono_wisc/bin/GetSceneRadiance/DEPTH_CAM/BIN/"
dst_depth_folder = "/home/bohsun/UW_Madison/Research/build_my_chrono_wisc/bin/GetSceneRadiance/DEPTH_CAM/FIG/"
file_name_ext = ".bin"


# Iterate through each sub-folder in the source folder
sub_folder_names = os.listdir(src_img_folder)
for sub_folder in sub_folder_names:
# for sub_folder in [sub_folder_names[5]]: # debug    
    sub_folder_path = os.path.join(src_img_folder, sub_folder)
    
    # Check if the path is a directory
    if os.path.isdir(sub_folder_path):
        # Get all frame files in the sub-folder
        frame_files = [f for f in os.listdir(sub_folder_path) if f.startswith("frame_") and f.endswith(file_name_ext)]
        
        # Sort the frame files based on the frame number
        frame_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        # Find the last second frame file
        if len(frame_files) >= 2:
            second_last_frame_file = frame_files[-3]
            
            # Define source and destination paths for the file
            src_img_path = os.path.join(src_img_folder, sub_folder, second_last_frame_file)
            src_depth_path = os.path.join(src_depth_folder, sub_folder, second_last_frame_file)
            
            img = ReadImageFromFile(src_img_path) # uint16 RGBA
            depth = ReadDepthFromFile(src_depth_path)
            
            img[depth > 100, 0:4] = 0
            
            dst_img_path = os.path.join(dst_img_folder, sub_folder + ".png")
            dst_depth_path = os.path.join(dst_depth_folder, sub_folder + ".bin")
            
            # save image
            cv2.imwrite(dst_img_path, cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
            
            # Copy and rename the file to the destination folder
            shutil.copy2(src_depth_path, dst_depth_path)
            # print(f"Copied and renamed: {src_file_path} to {dst_file_path}")
        
        else:
            print(f"Not enough frame files in {sub_folder_path} to perform the operation.")