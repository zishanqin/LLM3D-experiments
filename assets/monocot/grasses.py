# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import colorsys

import bpy
import numpy as np
from numpy.random import uniform

from assets.monocot.growth import MonocotGrowthFactory
from assets.utils.decorate import assign_material, join_objects, remove_vertices, write_attribute, \
    write_material_index
from assets.utils.draw import bezier_curve, leaf, spin
from assets.utils.mesh import polygon_angles
from assets.utils.misc import log_uniform

from nodes.node_info import Nodes
from nodes.node_wrangler import NodeWrangler

from placement.factory import AssetFactory, make_asset_collection
from placement.detail import remesh_with_attrs
from surfaces import surface
from surfaces.surface import read_attr_data, shaderfunc_to_material
from util import blender as butil
from util.math import FixedSeed
from assets.utils.tag import tag_object, tag_nodegroup

class GrassesMonocotFactory(MonocotGrowthFactory):

    def __init__(self, factory_seed, coarse=False):
        super(GrassesMonocotFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.stem_offset = uniform(1.5, 2.)
            self.angle = uniform(np.pi / 6, np.pi / 3)
            self.z_drag = uniform(.0, .2)
            self.min_y_angle = uniform(np.pi * .35, np.pi * .45)
            self.max_y_angle = uniform(np.pi * .45, np.pi * .5)
            self.count = int(log_uniform(16, 64))
            self.scale_curve = [(0, 1.), (1, .2)]
            self.bend_angle = np.pi / 2

    @staticmethod
    def build_base_hue():
        if uniform(0, 1) < .6:
            return uniform(.08, .12)
        else:
            return uniform(.2, .25)

    def build_leaf(self, face_size):
        x_anchors = np.array([0, uniform(.1, .2), uniform(.5, .7), 1.])
        y_anchors = np.array([0, uniform(.02, .03), uniform(.02, .03), 0])
        obj = leaf(x_anchors, y_anchors, face_size=face_size)

        cut_prob = .4
        if uniform(0, 1) < cut_prob:
            x_cutoff = uniform(.5, 1.)
            angle = uniform(-np.pi / 3, np.pi / 3)
            remove_vertices(obj, lambda x, y, z: (x - x_cutoff) * np.cos(angle) + y * np.sin(angle) > 0)
        self.decorate_leaf(obj)
        tag_object(obj, 'grasses')
        return obj

    @property
    def is_grass(self):
        return True


class WheatEarMonocotFactory(MonocotGrowthFactory):

    def __init__(self, factory_seed, coarse=False):
        super(WheatEarMonocotFactory, self).__init__(factory_seed, coarse, control_dict={})
        with FixedSeed(factory_seed):
            if 'stem_offset' in control_dict:
                # 0.4, 0.45, 0.5
                self.stem_offset = control_dict['stem_offset']
            else:
                self.stem_offset = uniform(.4, .5)

            if 'angle' in control_dict:
                # np.pi/6, np.pi/5, np.pi/4
                self.angle = control_dict['angle']
            else:
                self.angle = uniform(np.pi / 6, np.pi / 4)

    
            if 'min_y_angle' in control_dict:
                # pi/4, pi/3
                self.min_y_angle = control_dict['min_y_angle']
            else:
                self.min_y_angle = uniform(np.pi /4, np.pi /3)

            if 'max_y_angle' in control_dict:
                # pi/3, pi/2, pi
                self.max_y_angle = control_dict['max_y_angle']
            else:
                self.max_y_angle = np.pi/2
            
            if 'leaf_prob' in control_dict:
                # .5, .7, .9, 1
                self.leaf_prob = control_dict['leaf_prob']
            else:
                self.leaf_prob = uniform(.9, 1)

            if 'count' in control_dict:
                # 6, 7
                self.count = control_dict['count']
            else:
                self.count = int(log_uniform(96, 128))
            
            self.scale_curve = [(0, uniform(.8, 1.)), (.5, 1), (1, uniform(.6, 1.))]

            if 'bend_angle' in control_dict:
                # pi/2, pi
                self.bend_angle = control_dict['bend_angle']
            else:
                self.bend_angle = np.pi
            

    @staticmethod
    def build_base_hue():
        return uniform(.12, .28)

    def build_leaf(self, face_size):
        x_anchors = np.array([0, .05, .1])
        y_anchors = np.array([0, uniform(.01, .015), 0])
        curves = []
        for angle in polygon_angles(np.random.randint(4, 6)):
            anchors = [x_anchors, np.cos(angle) * y_anchors, np.sin(angle) * y_anchors]
            curves.append(bezier_curve(anchors))
        obj = butil.join_objects(curves)
        with butil.ViewportMode(obj, 'EDIT'):
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.convex_hull()
        remesh_with_attrs(obj, face_size / 2)
        tag_object(obj, 'wheat_ear')
        return obj


class WheatMonocotFactory(GrassesMonocotFactory):

    def __init__(self, factory_seed, coarse=False, control_dict={}):
        super(WheatMonocotFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.ear_factory = WheatEarMonocotFactory(factory_seed, coarse)
            if 'scale_curve' in control_dict:
                self.scale_curve = control_dict['scale_curve']
            else:
                self.scale_curve = [(0, 1.), (1, .6)]
            
            if 'leaf_range' in control_dict:
                self.leaf_range = control_dict['leaf_range']
            else:
                self.leaf_range = .1, .7

    @staticmethod
    def build_base_hue():
        return uniform(.08, .12)

    def create_asset(self, **params):
        obj = super().create_raw(**params)
        ear = self.ear_factory.create_asset(**params)
        butil.modify_mesh(ear, 'SIMPLE_DEFORM', deform_method='BEND',
                          angle=uniform(0, self.ear_factory.bend_angle))
        ear.location[-1] = self.stem_offset - .02
        obj = join_objects([obj, ear])
        self.decorate_monocot(obj)
        tag_object(obj, 'wheat')
        return obj


class MaizeMonocotFactory(GrassesMonocotFactory):

    def __init__(self, factory_seed, coarse=False, control_dict={}):
        super(MaizeMonocotFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.stem_offset = control_dict.get('stem_offset', uniform(2., 2.5)) # 2, 2.1, 2.2, 2.3, 2.4, 2.5
            self.scale_curve = control_dict.get('scale_curve', [(0, 1.), (1, .6)]) # [(0, 1.), (1, .6)] increase then decrease; 1 can be replaced as .7, .8, .9
            self.leaf_range = control_dict.get('leaf_range', (0.1, 0.7)) # .1, .2, .3, .4, .5, .6, .7

    def build_leaf(self, face_size):
        super().build_leaf(face_size)
        x_anchors = np.array([0, uniform(.1, .2), uniform(.5, .7), 1.])
        y_anchors = np.array([0, uniform(.03, .06), uniform(.03, .06), 0])
        obj = leaf(x_anchors, y_anchors, face_size=face_size)
        self.decorate_leaf(obj)
        tag_object(obj, 'maize_leaf')
        return obj

    def build_husk(self):
        x_anchors = 0, uniform(.04, .05), uniform(.03, .03), 0
        z_anchors = 0, .01, uniform(.24, .3), uniform(.35, .4)
        anchors = x_anchors, 0, z_anchors
        husk = spin(anchors)
        texture = bpy.data.textures.new(name='husk', type='STUCCI')
        texture.noise_scale = .01
        butil.modify_mesh(husk, 'DISPLACE', strength=.02, texture=texture)
        husk.location[-1] = self.stem_offset - .02
        husk.rotation_euler[0] = uniform(0, np.pi * .2)
        tag_object(husk, 'maize_husk')
        return husk

    def create_asset(self, **params):
        obj = super().create_raw(**params)
        husk = self.build_husk()
        obj = join_objects([obj, husk])
        self.decorate_monocot(obj)
        tag_object(obj, 'maize')
        return obj


class ReedEarMonocotFactory(MonocotGrowthFactory):

    def __init__(self, factory_seed, coarse=False, control_dict={}):
        super(ReedEarMonocotFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.stem_offset = control_dict.get('stem_offset', uniform(.3, .4)) # .3, .35, .4
            self.min_y_angle = control_dict.get('min_y_angle', uniform(np.pi / 4, np.pi / 3)) # pi/4, pi/3
            self.max_y_angle = control_dict.get('max_y_angle', self.min_y_angle + np.pi / 12) 
            self.count = control_dict.get('count', int(log_uniform(48, 96))) # 5, 6, 7
            self.radius = control_dict.get('radius', .002) # .002, .1, .5, 1


    def build_leaf(self, face_size):
        x_anchors = np.array([0, uniform(.02, .03), .05])
        y_anchors = np.array([0, uniform(.005, .01), 0])
        obj = leaf(x_anchors, y_anchors, face_size=face_size)
        return obj

    def create_asset(self, **params):
        obj = super(ReedEarMonocotFactory, self).create_asset(**params)
        write_attribute(obj, 1, 'ear', 'FACE')
        tag_object(obj, 'reed_ear')
        return obj


class ReedBranchMonocotFactory(MonocotGrowthFactory):
    max_branches = 6

    def __init__(self, factory_seed, coarse=False):
        super(ReedBranchMonocotFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.stem_offset = control_dict.get('stem_offset', uniform(.6, .8)) #.6, .7, .8
            self.branch_factory = ReedEarMonocotFactory(self.factory_seed)
            self.scale_curve = control_dict.get('scale_curve', [(0, 1), (.5, .6), (1, .1)]) # [(0, 1), (.5, 0.8), (1, 0.5)], [(0, 0.8), (.5, 0.7), (1, 0.4)]
            self.min_y_angle = control_dict.get('min_y_angle', uniform(-np.pi / 10, -np.pi / 8)) # -pi/10, -pi/9, -pi/8
            self.max_y_angle = control_dict.get('max_y_angle', uniform(-np.pi / 6, -np.pi / 8)) # -pi/6, -pi/7, -pi/8
            self.angle = control_dict.get('angle', 0) # 0, 15, 30, 45, 60
            self.radius = control_dict.get('radius', .005) # 0, .005, .01


    def make_collection(self, face_size):
        return make_asset_collection(self.branch_factory.create_raw, 2, 'leaves', verbose=False,
                                     face_size=face_size)


class ReedMonocotFactory(GrassesMonocotFactory):
    def __init__(self, factory_seed, coarse=False):
        super(ReedMonocotFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.stem_offset = control_dict.get('stem_offset', uniform(3., 4.)) # 3, 3.5, 4
            self.scale_curve = control_dict.get('scale_curve', [(0, 1.2), (1, .8)]) 
            self.branch_factory = ReedBranchMonocotFactory(factory_seed, coarse)
            self.branch_material = shaderfunc_to_material(self.shader_ear)

    @staticmethod
    def build_base_hue():
        return uniform(.08, .12)

    def create_asset(self, **params):
        obj = super().create_raw(**params)
        branch = self.branch_factory.create_asset(**params)
        self.branch_factory.decorate_monocot(branch)
        branch.location[-1] = self.stem_offset - .02
        obj = join_objects([obj, branch])
        butil.modify_mesh(obj, 'WELD', merge_threshold=1e-3)
        self.decorate_monocot(obj)

        assign_material(obj, [self.material, self.branch_material])
        write_material_index(obj, surface.read_attr_data(obj, 'ear', 'FACE').astype(int)[:, 0])
        tag_object(obj, 'reed')
        return obj

    @staticmethod
    def shader_ear(nw: NodeWrangler):
        color = *colorsys.hsv_to_rgb(uniform(.06, .1), uniform(.2, .5), log_uniform(.2, .5)), 1
        specular = uniform(.0, .2)
        clearcoat = 0 if uniform(0, 1) < .8 else uniform(.2, .5)
        noise_texture = nw.new_node(Nodes.NoiseTexture, input_kwargs={'Scale': 50})
        roughness = nw.build_float_curve(noise_texture, [(0, .5), (1, .8)])
        bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={
            'Base Color': color,
            'Roughness': roughness,
            'Specular': specular,
            'Clearcoat': clearcoat,
            'Subsurface': .01,
            'Subsurface Radius': (.01, .01, .01),
        })
        return bsdf
