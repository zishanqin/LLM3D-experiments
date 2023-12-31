# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


import bpy
import mathutils
import numpy as np
from numpy.random import uniform, normal, randint
from nodes.node_wrangler import Nodes, NodeWrangler
from nodes import node_utils
from nodes.color import color_category, hsv2rgba
from surfaces import surface

from util.math import FixedSeed
from util import blender as butil
from placement.factory import AssetFactory

from assets.fruits.general_fruit import FruitFactoryGeneralFruit

class FruitFactoryPineapple(FruitFactoryGeneralFruit):
    def __init__(self, factory_seed, scale=1.0, coarse=False, control=False, control_dict={}):
        super().__init__(factory_seed, scale=scale, coarse=coarse)
        self.name = 'pineapple'
        if 'shape' in control_dict:
            # print(control_dict)
            self.shape_params = control_dict['shape']
        else:
            self.shape_params = None

        if 'color' in control_dict:
            # an example: 
            # base color [ 0.87664568  0.17000981  0.          1.        ] 
            # alt color [ 0.83450949  0.18503807  0.          1.        ]
            # those are stored as keys in control_dict['color']

            self.base_color = control_dict['color']['base_color']
            self.alt_color = control_dict['color']['alt_color']
        else:
            self.base_color = None
            self.alt_color = None

    def sample_cross_section_params(self, surface_resolution=256):
        return {
            'cross_section_name': "circle_cross_section",
            'cross_section_func_args': {},
            'cross_section_input_args': {'random seed': uniform(-100, 100), 
                'radius': normal(1.2, 0.05),
                'Resolution': surface_resolution},
            'cross_section_output_args': {}
        }

    def sample_shape_params(self, surface_resolution=256):
        return {
            'shape_name': "shape_quadratic",
            'shape_func_args': {'radius_control_points': [(0.0, 0.1031), (0.1182, 0.5062), (uniform(0.3, 0.7), 0.5594), (0.8364, 0.425), (0.9864, 0.1406), (1.0, 0.0)]},
            'shape_input_args': {'Profile Curve': 'noderef-crosssection-Geometry', 
                'Start': (uniform(-0.1, 0.1), uniform(-0.1, 0.1), uniform(-0.8, -1.2)),
                'End': (0.0, 0.0, 1.0),
                'random seed pos': uniform(-100, 100),
                'noise scale pos': 0.5, 
                'noise amount pos': 0.4, 
                'Resolution': surface_resolution},
            'shape_output_args': {}
        }

    def sample_surface_params(self):
        if self.base_color is None:
            bottom_color = np.array((0.192, 0.898, 0.095))
            bottom_color[0] += np.random.normal(0.0, 0.025)
            bottom_color[1] += np.random.normal(0.0, 0.05)
            bottom_color[2] += np.random.normal(0.0, 0.05)
            bottom_color_rgba = hsv2rgba(bottom_color)

            mid_color = np.array((0.05, 0.96, 0.55))
            mid_color[0] += np.random.normal(0.0, 0.025)
            mid_color[1] += np.random.normal(0.0, 0.05)
            mid_color[2] += np.random.normal(0.0, 0.05)
            mid_color_rgba = hsv2rgba(mid_color)
            
            top_color = np.array((0.04, 0.99, 0.45))
            top_color[0] += np.random.normal(0.0, 0.025)
            top_color[1] += np.random.normal(0.0, 0.05)
            top_color[2] += np.random.normal(0.0, 0.05)
            top_color_rgba = hsv2rgba(top_color)

            center_color = np.array((0.07, 0.63, 0.84))
            center_color[0] += np.random.normal(0.0, 0.025)
            center_color[1] += np.random.normal(0.0, 0.05)
            center_color[2] += np.random.normal(0.0, 0.05)
            center_color_rgba = hsv2rgba(center_color)

            
        else:
            center_color_rgba = self.base_color
            if 'top' in self.alt_color:
                top_color_rgba = self.alt_color['top']
                mid_color_rgba = self.alt_color['mid']
                bottom_color_rgba = self.alt_color['bottom']
            else:
                top_color_rgba = self.alt_color
                mid_color_rgba = self.alt_color
                bottom_color_rgba = self.alt_color

        cell_distance = uniform(0.18, 0.22)

        return {
            'surface_name': "pineapple_surface",
            'surface_func_args': {'color_bottom': bottom_color_rgba,
                'color_mid': mid_color_rgba,
                'color_top': top_color_rgba,
                'color_center': center_color_rgba,},
            'surface_input_args': {'Geometry': 'noderef-shapequadratic-Mesh', 
                'spline parameter': 'noderef-shapequadratic-spline parameter',
                'point distance': cell_distance,
                'cell scale': cell_distance+0.02},
            'surface_output_args': {'radius': 'noderef-fruitsurface-spline parameter'},
            'surface_resolution': 64,
            'scale_multiplier': 1.8
        }

    def sample_stem_params(self):
        leaf_color = np.array((0.32, 0.79, 0.20))
        leaf_color[0] += np.random.normal(0.0, 0.025)
        leaf_color[1] += np.random.normal(0.0, 0.05)
        leaf_color[2] += np.random.normal(0.0, 0.05)
        leaf_color_rgba = hsv2rgba(leaf_color)
        
        return {
            'stem_name': "pineapple_stem",
            'stem_func_args': {'basic_color': leaf_color_rgba},
            'stem_input_args': {'rotation base': (-uniform(0.5, 0.55), 0.0, 0.0),
                'noise amount': 0.1, 
                'noise scale': uniform(10, 30),
                'number of leaves': randint(40, 80),
                'scale base': normal(0.5, 0.05),
                'scale z base': normal(0.15, 0.03),
                'scale z top': normal(0.62, 0.03),
                'rot z base': normal(-0.62, 0.03),
                'rot z top': normal(0.54, 0.03)},
            'stem_output_args': {}
        }

