# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import colorsys

import bpy
import numpy as np
from numpy.random import uniform

import util.blender as butil
from assets.monocot.growth import MonocotGrowthFactory
from assets.utils.decorate import add_distance_to_boundary, join_objects, displace_vertices
from assets.utils.draw import cut_plane, leaf
from assets.utils.misc import log_uniform
from surfaces.surface import shaderfunc_to_material
from util.blender import deep_clone_obj
from util.math import FixedSeed
from assets.utils.tag import tag_object, tag_nodegroup

class AgaveMonocotFactory(MonocotGrowthFactory):
    use_distance = True

    def __init__(self, factory_seed, coarse=False, control_dict={}):
        super(AgaveMonocotFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            if 'stem_offset' in control_dict:
                # 0, 0.25, 0.5
                self.stem_offset = control_dict['stem_offset']
            else:
                self.stem_offset = uniform(.0, .5)

            if 'angle' in control_dict:
                # np.pi/9, np.pi/6
                self.angle = control_dict['angle']
            else:
                self.angle = uniform(np.pi / 9, np.pi / 6)

            if 'z_drag' in control_dict:
                # 0.05, 0.06, 0.07, 0.08, 0.09, 0.1
                self.z_drag = control_dict['z_drag']
            else:
                self.z_drag = uniform(.05, .1)
    
            if 'min_y_angle' in control_dict:
                # 0.1pi, 0.15pi
                self.min_y_angle = control_dict['min_y_angle']
            else:
                self.min_y_angle = uniform(np.pi * .1, np.pi * .15)

            if 'max_y_angle' in control_dict:
                # 0.4pi, 0.5pi
                self.max_y_angle = control_dict['max_y_angle']
            else:
                self.max_y_angle = uniform(np.pi * .4, np.pi * .52)
            
            if 'count' in control_dict:
                # 4, 5, 6
                self.count = control_dict['count']
            else:
                self.count = int(log_uniform(32, 64))
            
            self.scale_curve = [(0, uniform(.8, 1.)), (.5, 1), (1, uniform(.6, 1.))]

            if 'bud_angle' in control_dict:
                # pi/8, pi/6, pi/4
                self.bud_angle = control_dict['bud_angle']
            else:
                self.bud_angle = uniform(np.pi / 8, np.pi / 4)
                
            self.cut_prob = 0 if uniform(0, 1) < .5 else uniform(.2, .4)

    @staticmethod
    def build_base_hue():
        return uniform(.12, .32)

    def build_leaf(self, face_size):
        x_anchors = 0, .2 * np.cos(self.bud_angle), uniform(1., 1.4), 1.5
        y_anchors = 0, .2 * np.sin(self.bud_angle), uniform(.1, .15), 0
        obj = leaf(x_anchors, y_anchors, face_size=face_size)
        distance = add_distance_to_boundary(obj)

        lower = deep_clone_obj(obj)
        z_offset = -log_uniform(.08, .16)
        z_ratio = uniform(1.5, 2.5)
        displace_vertices(lower, lambda x, y, z: (0, 0, (1 - (1 - distance) ** z_ratio) * z_offset))
        obj = join_objects([lower, obj])
        butil.modify_mesh(obj, "WELD", merge_threshold=2e-4)

        if uniform(0, 1) < self.cut_prob:
            angle = uniform(-np.pi / 3, np.pi / 3)
            cut_center = np.array([uniform(1., 1.4), 0, 0])
            cut_normal = np.array([np.cos(angle), np.sin(angle), 0])
            obj, cut = cut_plane(obj, cut_center, cut_normal)
            obj = join_objects([obj, cut])
            with butil.ViewportMode(obj, 'EDIT'), butil.Suppress():
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.region_to_loop()
                bpy.ops.mesh.remove_doubles(threshold=1e-2)

        self.decorate_leaf(obj)
        tag_object(obj, 'agave')
        return obj
