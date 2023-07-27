# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bpy
import numpy as np
from numpy.random import uniform

import util.blender as butil
from assets.creatures.animation.driver_repeated import repeated_driver
from assets.monocot.growth import MonocotGrowthFactory
from assets.utils.draw import bezier_curve, leaf
from assets.utils.decorate import assign_material, join_objects
from assets.utils.misc import log_uniform
from assets.utils.object import origin2leftmost
from nodes.node_wrangler import NodeWrangler
from placement.detail import remesh_with_attrs
from util.math import FixedSeed


class KelpMonocotFactory(MonocotGrowthFactory):
    max_leaf_length = 1.2
    align_angle = uniform(np.pi / 24, np.pi / 12)

    def __init__(self, factory_seed, coarse=False, control_dict={}):
        super(KelpMonocotFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.stem_offset = control_dict.get('stem_offset', 10.) # 0, 10
            self.angle = control_dict.get('angle', uniform(np.pi / 6, np.pi / 4)) # pi/6, pi/5, pi/4
            self.z_drag = control_dict.get('z_drag', uniform(.0, .2)) # 0, .1, .2
            self.min_y_angle = control_dict.get('min_y_angle', uniform(0, np.pi * .1)) # 0, .1pi 
            self.max_y_angle = control_dict.get('max_y_angle', self.min_y_angle) # 0, .1pi
            self.bend_angle = control_dict.get('bend_angle', uniform(0, np.pi / 6)) # 0, pi/6, pi
            self.twist_angle = control_dict.get('twist_angle', uniform(0, np.pi / 6)) # 0, pi/6, pi
            self.count = control_dict.get('count', 512) # 128, 256, 512
            self.leaf_prob = control_dict.get('leaf_prob', uniform(.6, .7)) # .6, .7
            self.align_angle = control_dict.get('align_angle', uniform(np.pi / 30, np.pi / 15)) # pi/30, pi/25, pi/20, pi/15
            self.radius = control_dict.get('radius', .02) # .02, .05, .1
            self.align_factor = self.make_align_factor()
            self.align_direction = self.make_align_direction()

            flow_angle = uniform(0, np.pi * 2)
            self.align_direction = (
                control_dict.get('align_direction_x', np.cos(flow_angle)),
                control_dict.get('align_direction_y', np.sin(flow_angle)),
                control_dict.get('align_direction_z', uniform(-.2, .2))
            )

            self.anim_freq = control_dict.get('anim_freq', 1 / log_uniform(100, 200)) # 1/100, 1/200
            self.anim_offset = control_dict.get('anim_offset', uniform(0, 1)) # 0, 0.5, 1
            self.anim_seed = control_dict.get('anim_seed', np.random.randint(1e5))


    def make_align_factor(self):
        def align_factor(nw: NodeWrangler):
            rand = nw.uniform(.7, .95)
            driver = rand.inputs[2].driver_add('default_value').driver
            driver.expression = repeated_driver(.7, .85, self.anim_freq, self.anim_offset, self.anim_seed)
            return nw.scalar_multiply(nw.bernoulli(.9), rand)

        return align_factor

    def make_align_direction(self):
        def align_direction(nw: NodeWrangler):
            direction = nw.combine(1, 0, 0)
            driver = direction.inputs[2].driver_add('default_value').driver
            driver.expression = repeated_driver(-.5, -.1, self.anim_freq, self.anim_offset, self.anim_seed)
            return direction

        return align_direction

    @staticmethod
    def build_base_hue():
        return uniform(.05, .25)

    def build_instance(self, i, face_size):
        x_anchors = np.array([0, -.02, -.04])
        y_anchors = np.array([0, uniform(.01, .02), 0])
        curves = []
        for angle in np.linspace(0, np.pi * 2, 6):
            anchors = [x_anchors, np.cos(angle) * y_anchors, np.sin(angle) * y_anchors]
            curves.append(bezier_curve(anchors))
        bud = butil.join_objects(curves)
        bud.location[0] += .02
        with butil.ViewportMode(bud, 'EDIT'):
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.convex_hull()
        remesh_with_attrs(bud, face_size)

        x_anchors = 0, uniform(.35, .65), uniform(.8, 1.2)
        y_anchors = 0, uniform(.06, .08), 0
        obj = leaf(x_anchors, y_anchors, face_size=face_size)
        obj = join_objects([obj, bud])
        self.decorate_leaf(obj, uniform(-2, 2), uniform(-np.pi / 4, np.pi / 4), uniform(-np.pi / 4, np.pi / 4))
        origin2leftmost(obj)
        return obj

    def create_asset(self, **params):
        obj = self.create_raw(apply=False)
        obj, mod = butil.modify_mesh(obj, 'SIMPLE_DEFORM', False, deform_method='TWIST', deform_axis='Z',
                                     return_mod=True)
        twist_driver = mod.driver_add('angle').driver
        extra_twist_angle = uniform(0, np.pi / 60)
        twist_driver.expression = repeated_driver(self.twist_angle - extra_twist_angle,
                                                  self.twist_angle + extra_twist_angle, self.anim_freq,
                                                  self.anim_offset, self.anim_seed)
        obj, mod = butil.modify_mesh(obj, 'SIMPLE_DEFORM', False, deform_method='BEND', deform_axis='Y',
                                     return_mod=True)
        bend_driver = mod.driver_add('angle').driver
        extra_bend_angle = uniform(0, np.pi / 60)
        bend_driver.expression = repeated_driver(self.bend_angle + extra_bend_angle,
                                                 self.bend_angle - extra_bend_angle, self.anim_freq,
                                                 self.anim_offset, self.anim_seed)
        obj.scale = uniform(.8, 1.2), uniform(.8, 1.2), self.z_scale
        assign_material(obj, self.material)
        return obj
