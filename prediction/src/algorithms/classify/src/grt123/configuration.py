"""
Configuration file for the gtr123 team solution
part of the Winning algorithm for DSB2017

code was adopted from https://github.com/lfz/DSB2017
"""
import numpy as np


config = {
    'crop_size': [96, 96, 96],
    'scaleLim': [0.85, 1.15],
    'radiusLim': [6, 100],
    'stride': 4,
    'detect_th': 0.05,
    'conf_th': -1,
    'nms_th': 0.05,
    'filling_value': 160,
    'startepoch': 20,
    'lr_stage': np.array([50, 100, 140, 160]),
    'lr': [0.01, 0.001, 0.0001, 0.00001],
    'miss_ratio': 1,
    'miss_thresh': 0.03,
    'anchors': [10, 30, 60]
}
