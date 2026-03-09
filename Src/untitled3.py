#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 15:43:02 2025

@author: bohsun
"""

import numpy as np

ranges = np.array([
    [-0.94, -0.55, -1.8488, -1.64, -0.0614, 0.0703], # Rock 1 done
    [-1.54, -1.25, -1.086, -0.8814, -0.076, 0.06], # Rock 2 done
    [-1.555, -1.32, -0.0944, 0.064, -0.0704, 0.0269], # Rock 3 done
    [-1.1, -0.8025, 0.841, 1.219, -0.08, 0.07], # Rock 4 done
    [-0.453, -0.3048, -1.2264, -1.0759, -0.096, -0.012], # Rock 5 done
    [-0.4065, -0.2975, -0.3112, -0.2248, -0.0984, -0.0249], # Rock 6 done
    [-0.2652, -0.06, -0.1215, 0.04, -0.0904, 0.032], # Rock 7 done
    [-0.312, -0.132, 0.645, 0.885, -0.0666, 0.0719], # Rock 8 done
    [-0.1689, -0.0402, 0.905, 1.07, -0.056, 0.028], # Rock 9 done
    [0.76, 0.95, -1.187, -0.96, -0.09929, -0.0019], # Rock 10 done
    [0.3648, 0.5519, -0.567, -0.32, -0.0762, 0.088], # Rock 11 done
    [0.44, 0.516, 0.63, 0.7, -0.0506, 0.01], # Rock 12 done
    [0.8, 1.02, 0.86, 1.107, -0.069, 0.0525], # Rock 13 done
    [1.1762, 1.3894, -1.5092, -1.308, -0.1189, -0.0035], # Rock 14 done
    [1.451, 1.66, 0.510, 0.720, -0.050, 0.10] # Rock 15 done
])

posis = []
scales = []

for obj_idx in range(len(ranges)):
    posi_x = (ranges[obj_idx, 0] + ranges[obj_idx, 1]) / 2
    posi_y = (ranges[obj_idx, 2] + ranges[obj_idx, 3]) / 2
    posi_z = (ranges[obj_idx, 4] + ranges[obj_idx, 5]) / 2
    
    scale_x = ranges[obj_idx, 1] - ranges[obj_idx, 0]
    scale_y = ranges[obj_idx, 3] - ranges[obj_idx, 2]
    scale_z = ranges[obj_idx, 5] - ranges[obj_idx, 4]
    
    posis.append(np.array([posi_x, posi_y, posi_z]))
    scales.append(np.array([scale_x, scale_y, scale_z]))

posis = np.vstack(posis)
scales = np.vstack(scales)