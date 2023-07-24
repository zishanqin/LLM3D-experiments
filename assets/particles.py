# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Hei Law, Alexander Raistrick


import bpy
import numpy as np
from numpy.random import normal as N
import mathutils

import gin

from placement.factory import AssetFactory
from surfaces import surface

from nodes.node_wrangler import Nodes, NodeWrangler
from nodes import node_utils
from util import blender as butil
from util.random import random_general
from assets.utils.tag import tag_object, tag_nodegroup

from surfaces.templates import dirt
from infinigen_gpl.surfaces import snow

def shader_raindrop(nw):
    glass_bsdf = nw.new_node(
        'ShaderNodeBsdfGlass',
        input_kwargs={
            'IOR': 1.33,
        },
    )
    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={
            'Surface': glass_bsdf,
        },
    )

def geo_raindrop(nw):
    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[(
            'NodeSocketGeometry',
            'Geometry',
            None,
        )],
    )

    position = nw.new_node(Nodes.InputPosition)

    vector_curves = nw.new_node(
        Nodes.VectorCurve,
        input_kwargs={
            'Vector': position,
        },
    )
    node_utils.assign_curve(
        vector_curves.mapping.curves[0],
        [(-1.0, -1.0), (1.0, 1.0)],
    )
    node_utils.assign_curve(
        vector_curves.mapping.curves[1],
        [(-1.0, -1.0), (1.0, 1.0)],
    )
    node_utils.assign_curve(
        vector_curves.mapping.curves[2],
        [(-1.0, -0.15 * N(1, 0.15)), (-0.6091, -0.0938), (1.0, 1.0)],
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            'Geometry': group_input.outputs["Geometry"],
            'Position': vector_curves,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            'Geometry': set_position,
        },
    )

class RaindropFactory(AssetFactory):
    control_properties = {
            'radius': 1,
            'subdivisions': 5,
            'location': (0, 0, 0),
            'scale': (1, 1, 1)
        }

    def create_asset(self, **kwargs):
        
        self.control({'radius': 2})

        bpy.ops.mesh.primitive_ico_sphere_add(
            radius=self.control_properties['radius'],
            enter_editmode=False,
            subdivisions=self.control_properties['subdivisions'],
            align='WORLD',
            location=self.control_properties['location'],
            scale=self.control_properties['scale'],
        )

        sphere = bpy.context.object

        surface.add_geomod(sphere, geo_raindrop, apply=True)
        tag_object(sphere, 'raindrop')
        return sphere
    
    def control(self, control_dict):
        properties = ['radius', 'subdivisions', 'location', 'scale']
        # experiment setup
        # radius: [1, 2, 3, 4, 5]
        # subdivisions: [1, 2, 3, 4, 5, 6]
        # location: (x, y, z) where all of x, y, z range from [-1, 0, 1]
        # scale: (x, y, z) where all of x, y, z range from [1, 2]
        for p in properties:
            if p in control_dict:
                self.control_properties[p] = control_dict[p]
    
    def finalize_assets(self, assets):
        surface.add_material(assets, shader_raindrop)
    
class DustMoteFactory(AssetFactory):
    control_properties = {
        'radius': 1,
        'subdivisions': 2,
        'location': (0, 0, 0),
        'scale': (1, 1, 1)
    }

    def create_asset(self, **kwargs):

        self.control({'radius': 2})
        bpy.ops.mesh.primitive_ico_sphere_add(
            radius=self.control_properties['radius'],
            enter_editmode=False,
            subdivisions=self.control_properties['subdivisions'],
            align='WORLD',
            location=self.control_properties['location'],
            scale=self.control_properties['scale'],
        )
        tag_object(bpy.context.object, 'dustmote')
        return bpy.context.object

    def control(self, control_dict):
        properties = ['radius', 'subdivisions', 'location', 'scale']
        # experiment setup
        # radius: [1, 2, 3, 4, 5]
        # subdivisions: [1, 2, 3, 4, 5, 6]
        # location: (x, y, z) where all of x, y, z range from [-1, 0, 1]
        # scale: (x, y, z) where all of x, y, z range from [1, 2]
        for p in properties:
            if p in control_dict:
                self.control_properties[p] = control_dict[p]
    
    def finalize_assets(self, assets):
        dirt.apply(assets)
    
class SnowflakeFactory(AssetFactory):
    control_properties = {
        'vertices': 6
    }

    def create_asset(self, **params) -> bpy.types.Object:
        bpy.ops.mesh.primitive_circle_add(
            vertices=self.control_properties['vertices'],
            fill_type='TRIFAN',
        )
        tag_object(bpy.context.object, 'snowflake')
        return bpy.context.object
    
    def control(self, control_dict):
        properties = ['vertices']
        # experiment setup
        # vertices: [2, 4, 6, 8]
        for p in properties:
            if p in control_dict:
                self.control_properties[p] = control_dict[p]
    
    def finalize_assets(self, assets):
        snow.apply(assets, subsurface=0)
    
@gin.configurable
def wind_effector(strength):
    bpy.ops.object.effector_add(type='WIND')
    wind = bpy.context.active_object

    yaw = np.random.uniform(0, 360)
    wind.rotation_euler = np.deg2rad((90, 0, yaw))

    wind.field.strength = random_general(strength)
    wind.field.flow = 0

    return wind
    
@gin.configurable
def turbulence_effector(strength, noise, size=1, flow=0):
    bpy.ops.object.effector_add(type='TURBULENCE')
    wind = bpy.context.active_object
    wind.field.strength = random_general(strength)
    wind.field.noise = random_general(noise)
    wind.field.flow = random_general(flow)
    wind.field.size = random_general(size)
