#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:16:55 2024

Physics-based differentiable camera model implemented by using PyTorch

@author: Bo-Hsun Chen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2
import numpy as np
from numba import cuda, float32, int32
import math
from math import pi
import matplotlib.pyplot as plt
import time
torchvision.disable_beta_transforms_warning()

def Gaussian1D(x, kernel_size, sigma):
    return 1.0 / (math.sqrt(2.0 * pi) * sigma) * math.exp(-x**2 / (2.0 * sigma**2))


def GetGaussianKernel(size, sigma):
    """
    Generate a 2D Gaussian kernel.
    
    Parameters
    ----------
    size : int
        side length of the Gaussian kernel, must be an odd number, [px]
    sigma : float
        standard deviation of Gaussian function    
    """
    center = size // 2
    x = np.arange(0, size, dtype=np.float32) - center
    y = np.arange(0, size, dtype=np.float32) - center
    x, y = np.meshgrid(x, y, indexing='ij')
    kernel = np.exp(-0.5 * (x**2 + y**2) / sigma**2)
    kernel /= kernel.sum()
    
    return kernel


def GetGaussianMask(img_h, img_w, kernel_size, center):
    """
    Generate a mask containing a Gaussian blur kernel centered at the specified coordinates.
    
    Parameters
    ----------
    img_h : int
        image height, [px]
    img_w : int
        image width, [px]
    kernel_size : int
        side length of the Gaussian kernel, must be an odd number, [px]
    center : (int, int)
        center where the Gaussian kernel located, (row_idx, column_idx)
    """
    
    mask = torch.zeros((img_h, img_w), dtype=torch.float32)
    
    kernel = GetGaussianKernel(kernel_size, kernel_size / 6.0)
    half_k_size = kernel_size // 2
    y_c, x_c = center

    y_min = max(0, y_c - half_k_size)
    y_max = min(img_h, y_c + half_k_size + 1)
    x_min = max(0, x_c - half_k_size)
    x_max = min(img_w, x_c + half_k_size + 1)
    
    k_y_min = half_k_size - (y_c - y_min) # ??
    k_y_max = kernel_size - (half_k_size - (y_max - y_c - 1)) # ??
    k_x_min = half_k_size - (x_c - x_min) # ??
    k_x_max = kernel_size - (half_k_size - (x_max - x_c - 1)) # ??
    
    mask[y_min : y_max, x_min : x_max] = kernel[k_y_min : k_y_max, k_x_min : k_x_max]
    
    return mask
    

def GetDefocusWeights(img_h, img_w, kernel_size, px_idx):
    """
    return the weights and their indices in the defocus sparse matrix
    
    Parameters
    ----------
    img_h : int
        image height, [px]
    img_w : int
        image width, [px]
    kernel_size : int
        side length of the Gaussian kernel, must be an odd number, [px]
    px_idx : int
        index of the pixel
    """
    
    # mask = torch.zeros((img_h, img_w), dtype=torch.float32)
    
    kernel = GetGaussianKernel(kernel_size, kernel_size / 6.0)
    half_k_size = kernel_size // 2
    y_c = px_idx // img_w
    x_c = px_idx % img_w

    y_min = max(0, y_c - half_k_size)
    y_max = min(img_h, y_c + half_k_size + 1)
    x_min = max(0, x_c - half_k_size)
    x_max = min(img_w, x_c + half_k_size + 1)
    
    k_y_min = half_k_size - (y_c - y_min) # ??
    k_y_max = kernel_size - (half_k_size - (y_max - y_c - 1)) # ??
    k_x_min = half_k_size - (x_c - x_min) # ??
    k_x_max = kernel_size - (half_k_size - (x_max - x_c - 1)) # ??
    
    indices_in = img_w * np.arange(y_min, y_max) + np.arange(x_min, x_max)
    values = kernel[k_y_min : k_y_max, k_x_min : k_x_max].reshape(-1)
    
    return indices_in, values


@cuda.jit
def CudaGetDefocusWeightsKernel(img_h, img_w, kernel_size_flatten, indices_in, indices_out, values):
    
    ## version 2.0 ##
    num_px = img_w * img_h
    px_idx = cuda.grid(1) # index of input pixel
    if (px_idx < num_px):
        kernel_size = kernel_size_flatten[px_idx]
        if (kernel_size > 1):
            sigma = kernel_size / 6.0
            half_k_size = kernel_size // 2
            
            x_in = px_idx % img_w
            y_in = px_idx // img_w
            
            x_min = max(0, x_in - half_k_size)
            x_max = min(img_w - 1, x_in + half_k_size)
            
            y_min = max(0, y_in - half_k_size)
            y_max = min(img_h - 1, y_in + half_k_size)
            
            append_idx = 0
            for y_out in range(y_min, y_max + 1):
                for x_out in range(x_min, x_max + 1):
                    if (append_idx < indices_in.shape[1]):
                        indices_in[px_idx, append_idx] = px_idx
                        indices_out[px_idx, append_idx] = y_out * img_w + x_out
                        values[px_idx, append_idx] = np.float16(
                            math.exp(-0.5 * ((y_in - y_out)**2 + (x_in - x_out)**2) / sigma**2) / (2.0 * pi * sigma**2)
                        )
                        append_idx += 1
            
        else: # singularity value of the Gaussian blur formula when kernel_size = 1
            indices_in[px_idx, 0] = px_idx
            indices_out[px_idx, 0] = px_idx
            values[px_idx, 0] = 1.0
            
    
    ## version 1.0 ##
    """
    num_px = img_w * img_h
    px_idx = cuda.grid(1) # index of output buffer
    if (px_idx < num_px):
        
        kernel_size = kernel_size_flatten[px_idx]
        
        if (kernel_size > 1):
            sigma = kernel_size / 6.0
            half_k_size = kernel_size // 2
            
            x_out = px_idx % img_w
            y_out = px_idx // img_w
            
            x_min = max(0, x_out - half_k_size)
            x_max = min(img_w - 1, x_out + half_k_size)
            
            y_min = max(0, y_out - half_k_size)
            y_max = min(img_h - 1, y_out + half_k_size)
            
            append_idx = 0
            for y_in in range(y_min, y_max + 1):
                for x_in in range(x_min, x_max + 1):
                    indices_in[px_idx, append_idx] = img_w * y_in + x_in
                    indices_out[px_idx, append_idx] = px_idx
                    values[px_idx, append_idx] = math.exp(-0.5 * ((y_in - y_out)**2 + (x_in - x_out)**2) / sigma**2) / (2.0 * pi * sigma**2)
                    append_idx += 1
        
        else: # singularity value of the Gaussian blur formula
            indices_in[px_idx, 0] = px_idx
            indices_out[px_idx, 0] = px_idx
            values[px_idx, 0] = 1.0
    """


class PhysDiffCamera(nn.Module):
    
    def __init__(self, img_h, img_w, torch_random_seed, device):
        """
        model constructor

        Parameters
        ----------
        img_h : int
            image height, [px]
        img_w : int
            image width, [px]
        torch_random_seed : int
            torch random seed
        device : string
            device type, 'cpu' or 'cuda'
        """
        
        super(PhysDiffCamera, self).__init__()
        
        torchvision.disable_beta_transforms_warning()
        self.device = device
        self.require_grad = False
        
        # self.aperture_num = torch.tensor(0., dtype=torch.float32, requires_grad=self.require_grad).to(self.device) # aperture number = focal_length / aperture_diameter, [1/1]
        # self.expsr_time = torch.tensor(0., dtype=torch.float32, requires_grad=self.require_grad).to(self.device) # exposure time, [sec]
        # self.ISO = torch.tensor(0., dtype=torch.float32, requires_grad=self.require_grad).to(self.device) # ISO exposure gain, [1/1]
        # self.focal_length = torch.tensor(0., dtype=torch.float32, requires_grad=self.require_grad).to(self.device) # focal length, [m]
        # self.focus_dist = torch.tensor(0., dtype=torch.float32, requires_grad=self.require_grad).to(self.device) # focus distance, [m]
        
        self.sensor_width = torch.tensor(0., dtype=torch.float32, requires_grad=self.require_grad).to(self.device) # pixel size, [m]
        self.pixel_size = torch.tensor(0., dtype=torch.float32, requires_grad=self.require_grad).to(self.device) # pixel size, [m]
        self.max_scene_light = torch.tensor(0., dtype=torch.float32, requires_grad=self.require_grad).to(self.device) # maximum scene light amount, [lux = lm/m^2]
        self.rgb_QEs = torch.tensor([0., 0., 0.], dtype=torch.float32, requires_grad=self.require_grad).to(self.device) # RGB quantum efficiencies, [1/1]
        self.gain_params = {
            'defocus_gain': 0.,
            'defocus_bias': torch.tensor(0., dtype=torch.float32, requires_grad=self.require_grad).to(self.device),
            'vignet_gain': torch.tensor(0., dtype=torch.float32, requires_grad=self.require_grad).to(self.device),
            'aggregator_gain': torch.tensor(0., dtype=torch.float32, requires_grad=self.require_grad).to(self.device), # proportional gain of illumination aggregation, [1/1]
            'expsr2dv_gains': torch.tensor([0., 0., 0.], dtype=torch.float32, requires_grad=self.require_grad).to(self.device),
            'expsr2dv_biases': torch.tensor([0., 0., 0.], dtype=torch.float32, requires_grad=self.require_grad).to(self.device),
            'expsr2dv_gamma': torch.tensor(0., dtype=torch.float32, requires_grad=self.require_grad).to(self.device),
            'expsr2dv_crf_type': "",
            'max_CoC': 0
        }
        self.noise_params = {
            'dark_currents': torch.tensor([0., 0., 0.], dtype=torch.float32, requires_grad=self.require_grad).to(self.device), # dark currents and hot-pixel noises, [electron/sec]
            'noise_gains': torch.tensor([0., 0., 0.], dtype=torch.float32, requires_grad=self.require_grad).to(self.device), # temporal noise gains, [1/1]
            'STD_reads': torch.tensor([0., 0., 0.], dtype=torch.float32, requires_grad=self.require_grad).to(self.device), # STDs of FPN and readout noises, [electron]
            'FPN_rng_seed': 1234, # seed of random number generator for readout and FPN noises
        }
        self.artifact_switches = {
            "vignetting": True, "defocus_blur": True, "aggregate": True, "add_noise": True, "expsr2dv": True,
        }
        
        self.vignet_mask = None
        
        self.img_h = img_h
        self.img_w = img_w
        
        torch.manual_seed(torch_random_seed)
        self.readout_noise_base = torch.randn(img_h, img_w, 3).to(self.device)
        
        
    # def SetCtrlParameters(self, camera_config):
    #     """
    #     Set camera control parameters
        
    #     Parameters
    #     ----------
    #     camera_config['aperture_num'] : float
    #         F-number (or aperture number) = focal_length / aperture_diameter, [1/1] 
    #     camera_config['expsr_time'] : float
    #         exposure time, [sec]
    #     camera_config['ISO'] : float
    #         ISO exposure gain, [1/1]
    #     camera_config['focal_length'] : float
    #         focal length, [m]
    #     camera_config['focus_dist'] : float
    #         focus distance, [m]
    #     """
        
    #     self.aperture_num = torch.tensor(camera_config['aperture_num'], dtype=torch.float32, requires_grad=self.require_grad).to(self.device)
    #     self.expsr_time = torch.tensor(camera_config['expsr_time'], dtype=torch.float32, requires_grad=self.require_grad).to(self.device)
    #     self.ISO = torch.tensor(camera_config['ISO'], dtype=torch.float32, requires_grad=self.require_grad).to(self.device)
    #     self.focal_length = torch.tensor(camera_config['focal_length'], dtype=torch.float32, requires_grad=self.require_grad).to(self.device)
    #     self.focus_dist = torch.tensor(camera_config['focus_dist'], dtype=torch.float32, requires_grad=self.require_grad).to(self.device)
        
    
    def SetModelParameters(self, sensor_width, pixel_size, max_scene_light, rgb_QEs, gain_params, noise_params):
        """
        Set camera intrinsic model parameters

        Parameters
        ----------
        sensor_width : float
            equivalent width of the image sensor, [m]
        pixel_size : float
            length of a pixel, [m]
        max_scene_light_amount : float
            maximum light amount in the scene, consider distance-diminishing effect [lux = lumen/m^2]
        rgb_QEs : list[float]
            RGB quantum efficiencies, [1/1]
        gain_params : dict
            all gain-related parameters in camera model, [1/1]
        """
        
        self.sensor_width = torch.tensor(sensor_width, dtype=torch.float32, requires_grad=self.require_grad).to(self.device)
        self.pixel_size = torch.tensor(pixel_size, dtype=torch.float32, requires_grad=self.require_grad).to(self.device)
        self.max_scene_light = torch.tensor(max_scene_light, dtype=torch.float32, requires_grad=self.require_grad).to(self.device)
        self.rgb_QEs = torch.tensor(rgb_QEs, dtype=torch.float32, requires_grad=self.require_grad).to(self.device)
        
        self.gain_params['defocus_gain'] = gain_params['defocus_gain']
        self.gain_params['defocus_bias'] = torch.tensor(gain_params['defocus_bias'], dtype=torch.float32, requires_grad=self.require_grad).to(self.device)
        self.gain_params['vignet_gain'] = torch.tensor(gain_params['vignet_gain'], dtype=torch.float32, requires_grad=self.require_grad).to(self.device)
        self.gain_params['aggregator_gain'] = torch.tensor(gain_params['aggregator_gain'], dtype=torch.float32, requires_grad=self.require_grad).to(self.device)
        self.gain_params['expsr2dv_gains'] = torch.tensor(gain_params['expsr2dv_gains'], dtype=torch.float32, requires_grad=self.require_grad).to(self.device)
        self.gain_params['expsr2dv_biases'] = torch.tensor(gain_params['expsr2dv_biases'], dtype=torch.float32, requires_grad=self.require_grad).to(self.device)
        self.gain_params['expsr2dv_gamma'] = torch.tensor(gain_params['expsr2dv_gamma'], dtype=torch.float32, requires_grad=self.require_grad).to(self.device)
        self.gain_params['expsr2dv_crf_type'] = gain_params['expsr2dv_crf_type']
        self.gain_params['max_CoC'] = gain_params['max_CoC']
        
        self.noise_params['dark_currents'] = torch.tensor(noise_params['dark_currents'], dtype=torch.float32, requires_grad=self.require_grad).to(self.device)
        self.noise_params['noise_gains'] = torch.tensor(noise_params['noise_gains'], dtype=torch.float32, requires_grad=self.require_grad).to(self.device)
        self.noise_params['STD_reads'] = torch.tensor(noise_params['STD_reads'], dtype=torch.float32, requires_grad=self.require_grad).to(self.device)
        self.noise_params['FPN_rng_seed'] = noise_params['FPN_rng_seed']

        
    def BuildVignetMask(self, sensor_width, focal_length):
        """
        Build vignetting weight mask

        Parameters
        ----------
        sensor_width : float
            imaging sensor width, [m]
        focal_length : float
            focal length, [m]
        """
        
        x = (np.arange(self.img_w) - (self.img_w - 1)/2.0) * (sensor_width / self.img_w)
        y = (np.arange(self.img_h) - (self.img_h - 1)/2.0) * (sensor_width / self.img_w)
        theta_grid = np.arctan(np.sqrt(x**2 + y[..., np.newaxis]**2) / focal_length)
        
        
        self.vignet_mask = torch.from_numpy(1.0 - np.cos(theta_grid)**4).float()
        # self.vignet_mask = self.vignet_mask.unsqueeze(-1).repeat(1, 1, 3).to(self.device)
        self.vignet_mask = self.vignet_mask.unsqueeze(-1).expand(-1, -1, 3).to(self.device)
        self.vignet_mask = self.vignet_mask.unsqueeze(0)

        ## Convert the 2D numpy array mask to a 4D PyTorch tensor and register it as a buffer
        # self.register_buffer('vignet_mask', self.vignet_mask)
        
        self.vignet_mask.requires_grad = False

    
    def GetSparseTensor(self, depth_map, camera_config):
        """
        Build defocus-blur layer based on the depth map: part 1

        Parameters
        ----------
        depth_map : np.ndarray(float)
            depth map array with shape (img_height, img_width), [m]
        camera_config : {"aperture_num", "expsr_time", "ISO", "focal_length", "focus_dist"}, type=Dict
        """
        assert (self.gain_params['max_CoC'] > 0), "max circle-of-confusion should be larger than 0 ..."
        
        f = camera_config["focal_length"] # [m]
        U = camera_config["focus_dist"] # [m]
        N = camera_config["aperture_num"] # [1/1]
        C = self.pixel_size.item() # [m]
        a = self.gain_params['defocus_gain']
        b = self.gain_params['defocus_bias']
        img_h = depth_map.shape[0]
        img_w = depth_map.shape[1]
        num_px = img_h * img_w
        
        ## convert depth map to kernel-size map
        depth_flatten = torch.tensor(depth_map.reshape(-1), dtype=torch.float32, device=self.device)
        
        kernel_size_flatten = torch.zeros_like(depth_flatten, dtype=torch.float32, device=self.device)
        
        valid_px = (depth_flatten > 1e-2)
        kernel_size_flatten[valid_px] = f * f * torch.abs(depth_flatten[valid_px] - U) / (N * C * depth_flatten[valid_px] * (U - f))
        
        defocus_px = (kernel_size_flatten > b) # those pixels which are out-of-focus
        trans_px = ((1 < kernel_size_flatten) & (kernel_size_flatten <= b)) # those pixels within transition range
        kernel_size_flatten[defocus_px] = a * kernel_size_flatten[defocus_px]
        # kernel_size_flatten[trans_px] = (a * b - 1)/(b - 1) * kernel_size_flatten[trans_px] - b * (a - 1)/(b - 1)
        kernel_size_flatten[trans_px] = 1.0
        kernel_size_flatten = torch.ceil(kernel_size_flatten).int()

        # ensure all kernel-sizes are odd
        kernel_size_flatten = torch.where(kernel_size_flatten % 2 == 1, kernel_size_flatten, kernel_size_flatten + 1)
        kernel_size_flatten = torch.clip(kernel_size_flatten, 1, self.gain_params['max_CoC'])
        
        # print(f"take {np.round(1000 * (time.time() - t_start), 4)} ms to convert depth map to kernel-size map")
        
        ## Allocate arrays for indices and values
        kernel_size_max = max(1, kernel_size_flatten.max().item())
        indices_in = torch.zeros((num_px, kernel_size_max**2), dtype=torch.int32, device=self.device)
        indices_out = torch.zeros((num_px, kernel_size_max**2), dtype=torch.int32, device=self.device)
        values = torch.zeros((num_px, kernel_size_max**2), dtype=torch.float16, device=self.device)
        
        ## Transfer data to GPU
        dev_kernel_size_flatten = cuda.as_cuda_array(kernel_size_flatten)
        dev_indices_in = cuda.as_cuda_array(indices_in)
        dev_indices_out = cuda.as_cuda_array(indices_out)
        dev_values = cuda.as_cuda_array(values)
        
        # Set up kernel launch parameters
        threads_per_block = 512
        blocks_per_grid = (num_px + threads_per_block - 1) // threads_per_block
        
        # Launch the kernel
        # print(f"img_h={img_h}, img_w={img_w}, dev_kernel_size_flatten={kernel_size_flatten}")
        # assert False
        CudaGetDefocusWeightsKernel[blocks_per_grid, threads_per_block](
            img_h, img_w, dev_kernel_size_flatten, dev_indices_in, dev_indices_out, dev_values
        )
        cuda.synchronize()
               
        sparse_tensor = torch.sparse_coo_tensor(
            torch.stack([indices_out.view(-1), indices_in.view(-1)]),
            values.view(-1),
            size=(num_px, num_px), device='cpu', requires_grad=False
        ).coalesce()
        
        # print(f"take {np.round(1000 * (time.time() - t_start), 4)} ms to build sparse tensor")
        
        return sparse_tensor, kernel_size_flatten.reshape(img_h, img_w).cpu().numpy()
        
    
    def BuildDefocusBlurLayer(self, sparse_tensors, batch_img_in, depth_map_h, depth_map_w):
        """
        Build defocus-blur layer based on the depth map: part 1
    
        Parameters
        ----------
        sparse_tensor : tensor(float)
            ...
        batch_img_in : float
            input image tensor with shape (batch_size, img_height, img_width, num_channels), [px]
        """
        t_start = time.time()
        
        ## ...
        batch_img_out = []
        for img_idx in range(batch_img_in.shape[0]):
            img_out = []
            # print()
            # print(f"batch_img_in[img_idx, :, :, 0].view(-1, 1).shape={batch_img_in[img_idx, :, :, 0].view(-1, 1).shape}")
            # print(f"sparse_tensor.shape={sparse_tensor.shape}")
            # print()
            # print(batch_img_in[img_idx].shape)
            
            # batch_img_in[img_idx]: (h, w, ch)
            shrink_img = batch_img_in[img_idx].permute(2, 0, 1)
            if (depth_map_w < self.img_w):
                shrink_img = v2.Resize(
                    (depth_map_h, depth_map_w),
                    interpolation=v2.InterpolationMode.NEAREST,
                    antialias=True,
                )(shrink_img)
            
            # shrink_img: (ch, h, w)
            # print(shrink_img[:, :, 0].view(-1, 1).shape)
            for ch_idx in range(batch_img_in.shape[3]):
                
                img_out.append(torch.sparse.mm(
                    sparse_tensors[img_idx].float(), shrink_img[ch_idx, :, :].view(-1, 1)
                ).view(depth_map_h, depth_map_w))

            # (ch, h, w)
            img_out = torch.stack(img_out, dim=0)
            if (depth_map_w < self.img_w):
                img_out = v2.Resize(
                    (self.img_h, self.img_w),
                    interpolation=v2.InterpolationMode.BILINEAR,
                    antialias=True,
                )(img_out)
                
            batch_img_out.append(img_out.permute(1, 2, 0))
        
        batch_img_out = torch.stack(batch_img_out, dim=0)
        # print(f"take {np.round(1000 * (time.time() - t_start), 4)} ms to multiply sparse tensor and input image")
        
        return batch_img_out
   
    
    def forward(self, radiance, sparse_tensors, cam_ctrl_params, depth_map_h, depth_map_w):
    # def forward(self, radiance):
        # radiance, shape: (batch_size, img_height, img_width, num_channels), type: torch tensor
        # sparse_tensors: batch_size x [sparse_tensor], type: List(torch.tensor)
        # cam_ctrl_params:  batch_size x [aperture_num, expsr_time, ISO, focal_length, focus_dist], shape: (batch_size, 5), type: torch tensor
        #
        # Ensure the dimensions of the image and mask match
        # print(radiance.shape)
        # print(self.vignet_mask.shape)
        
        batch_size = radiance.shape[0]
        torch_shape = (1, 1, 1, 3)
        expand_size = (batch_size, -1, -1, -1)
        param_shape = (batch_size, 1, 1, 1)

        assert (radiance.shape == self.vignet_mask.expand(expand_size).shape), "Radiance and vignet-mask must have the same dimensions"
        
        cam_ctrl_params = cam_ctrl_params.to(self.device).requires_grad_(self.require_grad)
        aperture_num = cam_ctrl_params[:, 0].view(param_shape)
        expsr_time = cam_ctrl_params[:, 1].view(param_shape)
        ISO = cam_ctrl_params[:, 2].view(param_shape)
        focal_length = cam_ctrl_params[:, 3].view(param_shape)
        focus_dist = cam_ctrl_params[:, 4].view(param_shape)
        
        rgb_QEs = self.rgb_QEs.view(torch_shape).expand(expand_size)
        
        dark_currents = self.noise_params['dark_currents'].view(torch_shape).expand(expand_size)
        noise_gains = self.noise_params['noise_gains'].view(torch_shape).expand(expand_size)
        STD_reads = self.noise_params['STD_reads'].view(torch_shape).expand(expand_size)

        expsr2dv_gains = self.gain_params['expsr2dv_gains'].view(torch_shape).expand(expand_size)
        expsr2dv_biases = self.gain_params['expsr2dv_biases'].view(torch_shape).expand(expand_size)

        ## vignetting layer (element-wise multiplication)
        # print(radiance.device)
        # print(self.gain_params['vignet_gain'].device)
        # print(self.vignet_mask.device)
        if (self.artifact_switches["vignetting"]):
            img = radiance * (1.0 - self.gain_params['vignet_gain'] * self.vignet_mask.expand(expand_size))
        else:
            img = radiance
            
        ## defocus-blur layer ##
        if (self.artifact_switches["defocus_blur"] and (sparse_tensors is not None)):
            img = self.BuildDefocusBlurLayer(sparse_tensors, img, depth_map_h, depth_map_w)
        
        """
        f = self.focal_length.item() # [m]
        U = self.focus_dist.item() # [m]
        N = self.aperture_num.item() # [1/1]
        C = self.pixel_size.item() # [m]
        num_px = self.img_h * self.img_w
        
        ## convert depth map to kernel-size map
        depth_flatten = torch.tensor(depth_map.reshape(-1), dtype=torch.float32, device=self.device)
        
        kernel_size_flatten = torch.zeros_like(depth_flatten, dtype=torch.int, device=self.device)
        
        valid_mask = (depth_flatten > 1e-3)
        kernel_size_flatten[valid_mask] = torch.ceil(f * f * torch.abs(depth_flatten[valid_mask] - U) / (N * C * depth_flatten[valid_mask] * (U - f))).int()
        # print(kernel_size_flatten.cpu().numpy())
        
        defocus_mask = (kernel_size_flatten > 1) # those pixels which are out-of-focus
        kernel_size_flatten[defocus_mask] = (self.gain_params['defocus_gain'] * kernel_size_flatten[defocus_mask].float()).int()
        
        # ensure all kernel-sizes are odd and within range
        kernel_size_flatten = torch.where(kernel_size_flatten % 2 == 1, kernel_size_flatten, kernel_size_flatten + 1)
        kernel_size_flatten = torch.clip(kernel_size_flatten, 1, self.gain_params['max_CoC'])
        # print(kernel_size_flatten.cpu().numpy())
        
        ## Allocate arrays for indices and values
        kernel_size_max = kernel_size_flatten.max().item()
        indices_in = torch.zeros((num_px, kernel_size_max**2), dtype=torch.int32, device=self.device)
        indices_out = torch.zeros((num_px, kernel_size_max**2), dtype=torch.int32, device=self.device)
        values = torch.zeros((num_px, kernel_size_max**2), dtype=torch.float32, device=self.device)
        
        ## Transfer data to GPU
        dev_kernel_size_flatten = cuda.as_cuda_array(kernel_size_flatten)
        dev_indices_in = cuda.as_cuda_array(indices_in)
        dev_indices_out = cuda.as_cuda_array(indices_out)
        dev_values = cuda.as_cuda_array(values)
        # dev_kernel_size_flatten = cuda.to_device(kernel_size_flatten)
        # dev_indices_in = cuda.to_device(indices_in)
        # dev_indices_out = cuda.to_device(indices_out)
        # dev_values = cuda.to_device(values)
        
        ## Set up kernel launch parameters
        threads_per_block = 512
        blocks_per_grid = (num_px + threads_per_block - 1) // threads_per_block
        
        ## Launch the kernel
        CudaGetDefocusWeightsKernel[blocks_per_grid, threads_per_block](
            self.img_h, self.img_w, dev_kernel_size_flatten, dev_indices_in, dev_indices_out, dev_values
        )
        cuda.synchronize()
        # cuda.current_context().reset()
        
        
        # indices = np.vstack([indices_out, indices_in])
        sparse_tensor = torch.sparse_coo_tensor(
            torch.stack([indices_out.view(-1), indices_in.view(-1)]),
            values.view(-1),
            size=(num_px, num_px), device='cpu', requires_grad=False
        ).coalesce().to(self.device)
        
        # img2 = torch.sparse.mm(sparse_tensor, img[:, :, 0].view(-1, 1)).view(self.img_h, self.img_w).unsqueeze(-1).expand(-1, -1, 3)
        # img = torch.stack(
        #     [torch.sparse.mm(sparse_tensor, img[:, :, ch_idx].view(-1, 1)).view(self.img_h, self.img_w) for ch_idx in range(img.shape[2])],
        #     dim=2
        # )
        """
        
        ## aggregator layer
        # print(f"img.device={img.device}")
        # print(f"aperture_num.device={aperture_num.device}")
        if (self.artifact_switches["aggregate"]):
            img = img * self.gain_params['aggregator_gain'] * self.max_scene_light
            img = img / aperture_num.pow(2) * (self.pixel_size ** 2) * expsr_time
            img = img * rgb_QEs
            img = img + expsr_time * dark_currents
        
        ## noise-add layer
        if (self.artifact_switches["add_noise"]):
            img = img + torch.randn_like(img) * noise_gains * img.sqrt() 
            img = img + self.readout_noise_base * STD_reads
        
        ## expsr2dv layer
        if (self.artifact_switches["expsr2dv"]):
            if (self.gain_params['expsr2dv_crf_type'] == 'linear'):
                img = expsr2dv_gains * ISO * img + expsr2dv_biases
            
            elif (self.gain_params['expsr2dv_crf_type'] == 'gamma'):
                img = expsr2dv_gains * torch.pow(torch.log2(ISO * img), self.gain_params['expsr2dv_gamma']) + expsr2dv_biases
            
        img = torch.clamp(img, min=0., max=1.0)
        
        return img
    

    