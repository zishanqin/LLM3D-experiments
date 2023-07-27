# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bmesh
import numpy as np
from numpy.random import uniform

from assets.utils.decorate import displace_vertices, join_objects, read_co
from assets.utils.draw import bezier_curve, leaf
from assets.utils.nodegroup import geo_radius
from assets.utils.object import origin2lowest
from surfaces import surface
from assets.monocot.growth import MonocotGrowthFactory
from assets.utils.misc import log_uniform
from util import blender as butil
from util.math import FixedSeed
from assets.utils.tag import tag_object, tag_nodegroup

class BananaMonocotFactory(MonocotGrowthFactory):

    def __init__(self, factory_seed, coarse=False):
        super(BananaMonocotFactory, self).__init__(factory_seed, coarse, control_dict={})
        with FixedSeed(factory_seed):
            if 'stem_offset' in control_dict:
                # 0.5, 0.6, 0.7, 0.8, 0.9, 1
                self.stem_offset = control_dict['stem_offset']
            else:
                self.stem_offset = uniform(.6, 1.)

            if 'angle' in control_dict:
                # np.pi/4, np.pi/3
                self.angle = control_dict['angle']
            else:
                self.angle = uniform(np.pi / 4, np.pi / 3)

            if 'z_scale' in control_dict:
                # 0.1, 0.2, 0.3, 0.4, 0.5
                self.z_scale = control_dict['z_scale']
            else:
                self.z_scale = uniform(1, 1.5)

            if 'z_drag' in control_dict:
                # 0.1, 0.15, 0.2
                self.z_drag = control_dict['z_drag']
            else:
                self.z_drag = uniform(.1, .2)
    
            if 'min_y_angle' in control_dict:
                # 0.05pi, 0.1pi
                self.min_y_angle = control_dict['min_y_angle']
            else:
                self.min_y_angle = uniform(np.pi * .05, np.pi * .1)

            if 'max_y_angle' in control_dict:
                # 0.25pi, 0.3pi, 0.35pi, 0.4pi, 0.45pi
                self.max_y_angle = control_dict['max_y_angle']
            else:
                self.max_y_angle = uniform(np.pi * .25, np.pi * .45)
            
            if 'min_leaf_range' in control_dict:
                # 0.5, 0.6, 0.7
                self.leaf_range = control_dict['min_leaf_range'], 1
            else:
                self.leaf_range = uniform(.5, .7), 1

            if 'count' in control_dict:
                # 4, 5
                self.count = control_dict['count']
            else:
                self.count = int(log_uniform(16, 24))
            
            self.scale_curve = [(0, uniform(.8, 1.)), (.5, 1), (1, uniform(.6, 1.))]

            if 'bud_angle' in control_dict:
                # pi/8, pi/6, pi/4
                self.bud_angle = control_dict['bud_angle']
            else:
                self.bud_angle = uniform(np.pi / 8, np.pi / 6)
            
            if 'cut_angle' in control_dict:
                self.cut_angle = control_dict['cut_angle']
            else:
                self.cut_angle = self.bud_angle + uniform(np.pi / 20, np.pi / 12)

            if 'scale_curve' in control_dict:
                # [(0, 0.4, 0.6)], where the second and third elements can be chosen to increase by 0.1 each time
                self.scale_curve = control_dict['scale_curve']
            else:
                self.scale_curve = [(0, uniform(.4, 1.)), (1, uniform(.6, 1.))]

            if 'radius' in control_dict:
                # .04, .05, .06
                self.radius = control_dict['radius']
            else:
                self.radius = uniform(.04, .06)
            
            if 'freq' in control_dict:
                # 6, 7, 8
                self.freq = control_dict['freq']
            else:
                self.freq = log_uniform(100, 300)

            self.n_cuts = np.random.randint(6, 10) if uniform(0, 1) < .8 else 0

    @staticmethod
    def build_base_hue():
        return uniform(.15, .35)

    def cut_leaf(self, obj):
        coords = read_co(obj)
        x, y, z = coords.T
        coords = coords[(np.abs(y) < .08) & (np.abs(y) > .01)]
        positive_coords = coords[coords.T[1] > 0]
        positive_coords = positive_coords[np.argsort(positive_coords[:, 0])]
        negative_coords = coords[coords.T[1] < 0]
        negative_coords = negative_coords[np.argsort(negative_coords[:, 0])]
        positive_coords = positive_coords[np.random.choice(len(positive_coords), self.n_cuts, replace=False)]
        negative_coords = negative_coords[np.random.choice(len(negative_coords), self.n_cuts, replace=False)]

        for (x1, y1, _), (x2, y2, _) in zip(np.concatenate([positive_coords[:-1], negative_coords[:-1]], 0),
                                            np.concatenate([positive_coords[1:], negative_coords[1:]], 0)):
            coeff = 1 if y1 > 0 else -1
            ratio = uniform(-2., .4)
            exponent = uniform(1.2, 1.6)

            def cut(x, y, z):
                m1 = x1 * np.sin(self.cut_angle) - y1 * np.cos(self.cut_angle) * coeff
                m2 = x2 * np.sin(self.cut_angle) - y2 * np.cos(self.cut_angle) * coeff
                m = x * np.sin(self.cut_angle) - y * np.cos(self.cut_angle) * coeff
                dist = ((x - x1) * (y1 - y2) + (y - y1) * (x1 - x2)) / np.sqrt(
                    (x1 - x2) ** 2 + (y1 - y2) ** 2 + .1)
                return 0, 0, np.where((m1 < m) & (m < m2) & (dist * coeff < 0),
                                      ratio * np.abs(dist) ** exponent, 0)

            displace_vertices(obj, cut)
        with butil.ViewportMode(obj, 'EDIT'):
            bm = bmesh.from_edit_mesh(obj.data)
            geom = [e for e in bm.edges if e.calc_length() > .02]
            bmesh.ops.delete(bm, geom=geom, context='EDGES')
            bmesh.update_edit_mesh(obj.data)

    def build_leaf(self, face_size):
        x_anchors = 0, .2 * np.cos(self.bud_angle), uniform(.8, 1.2), 2.
        y_anchors = 0, .2 * np.sin(self.bud_angle), uniform(.2, .25), 0
        obj = leaf(x_anchors, y_anchors, face_size=face_size)
        self.cut_leaf(obj)
        self.displace_veins(obj)
        self.decorate_leaf(obj)
        tag_object(obj, 'banana')
        return obj

    def displace_veins(self, obj):
        vg = obj.vertex_groups.new(name='distance')
        x, y, z = read_co(obj).T
        branch = np.cos(
            (np.abs(y) * np.cos(self.cut_angle) - x * np.sin(self.cut_angle)) * self.freq) > uniform(.85, .9,
                                                                                                     len(x))
        leaf = np.abs(y) < uniform(.002, .008, len(x))
        weights = branch | leaf
        for i, l in enumerate(weights):
            vg.add([i], l, 'REPLACE')
        butil.modify_mesh(obj, 'DISPLACE', strength=-uniform(5e-3, 8e-3), mid_level=0, vertex_group='distance')


class TaroMonocotFactory(BananaMonocotFactory):
    def __init__(self, factory_seed, coarse=False, control_dict={}):
        super(TaroMonocotFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            # Experiments
            self.stem_offset = control_dict.get('stem_offset', uniform(.05, .1)) # .05, .06, .07, .08, .09, .1
            self.radius = control_dict.get('radius', uniform(.02, .04)) # .02, .03, .04
            self.z_drag = control_dict.get('z_drag', uniform(.2, .3)) # .2, .25, .3
            self.bud_angle = control_dict.get('bud_angle', uniform(np.pi * .6, np.pi * .7)) # .5pi, .6pi, .7pi
            self.freq = control_dict.get('freq', log_uniform(10, 20)) # 3, 4, 5
            self.count = control_dict.get('count', int(log_uniform(12, 16))) # 2, 3
            self.n_cuts = control_dict.get('n_cuts', np.random.randint(1, 2) if uniform(0, 1) < .5 else 0) # 0, 1, 2
            self.min_y_angle = control_dict.get('min_y_angle', uniform(-np.pi * .25, -np.pi * .05)) # -.25pi, -.2pi, -.15pi, -.1pi, -.05pi 
            self.max_y_angle = control_dict.get('max_y_angle', uniform(-np.pi * .05, 0)) # -pi/2, -pi/3, -pi/4, 0

    def displace_veins(self, obj):
        vg = obj.vertex_groups.new(name='distance')
        x, y, z = read_co(obj).T
        branch = np.cos(uniform(0, np.pi * 2) + np.arctan2(y - np.where(y > 0, -1, 1) * uniform(.1, .2),
                                                           x - uniform(.1, .4)) * self.freq) > uniform(.98, .99,
                                                                                                       len(x))
        leaf = np.abs(y) < uniform(.002, .008, len(x))
        weights = branch | leaf
        for i, l in enumerate(weights):
            vg.add([i], l, 'REPLACE')
        butil.modify_mesh(obj, 'DISPLACE', strength=-uniform(5e-3, 8e-3), mid_level=0, vertex_group='distance')

    def build_leaf(self, face_size):
        x_anchors = 0, .2 * np.cos(self.bud_angle), uniform(.4, 1.), uniform(.8, 1.)
        y_anchors = 0, .2 * np.sin(self.bud_angle), uniform(.25, .3), 0
        obj = leaf(x_anchors, y_anchors, face_size=face_size)
        self.cut_leaf(obj)
        self.displace_veins(obj)
        self.decorate_leaf(obj, 2, leftmost=False)
        bezier = self.build_branch()
        obj = join_objects([obj, bezier])
        origin2lowest(obj)
        tag_object(obj, 'taro')
        return obj

    def build_branch(self):
        offset = uniform(.2, .3)
        length = uniform(1, 2)
        x_anchors = 0, -.05, - offset - uniform(.01, .02), -offset
        z_anchors = 0, 0, - length + .1, -length
        bezier = bezier_curve([x_anchors, 0, z_anchors])
        surface.add_geomod(bezier, geo_radius, apply=True, input_args=[uniform(.02, .03), 32])
        return bezier

    def build_instance(self, i, face_size):
        return self.build_leaf(face_size)
