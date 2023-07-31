# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import bpy

import numpy as np
from numpy.random import uniform, normal

from assets.creatures.geometry.curve import Curve
from util.blender import deep_clone_obj

from surfaces.templates import grass_blade_texture

from placement.factory import AssetFactory

from util import blender as butil
from assets.utils.tag import tag_object, tag_nodegroup

class GrassTuftFactory(AssetFactory):

    def __init__(self, seed, control=False, control_dict={}):

        super(GrassTuftFactory, self).__init__(seed)
        
        with FixedSeed(seed):
            self.fac_info = {}
        
            # Parameters
            self.fac_info['n_seg'] = 4

            self.fac_info['length_mean'] = uniform(0.05, 0.15)
            self.fac_info['length_std'] = self.fac_info['length_mean'] * uniform(0.2, 0.5)

            self.fac_info['curl_mean'] = uniform(10, 70)
            self.fac_info['curl_std'] = self.fac_info['curl_mean'] * np.clip(normal(0.3, 0.1), 0.01, 0.6)
            self.fac_info['curl_power'] = normal(1.2, 0.3)

            self.fac_info['blade_width_pct_mean'] = uniform(0.01, 0.03)
            self.fac_info['blade_width_var'] = uniform(0, 0.05)

            self.fac_info['taper_var'] = uniform(0, 0.1)

            self.fac_info['base_spread'] = uniform(0, self.fac_info['length_mean'] / 4)
            self.fac_info['base_angle_var'] = uniform(0, 15)

            properties = [
                'n_seg', # 1, 2, 3, 4, 5
                'length_mean', # 0.05, 0.10, 0.15, 0.20
                'length_std', # 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1
                'curl_mean', # 10, 20, 30, 40, 50, 60, 70
                'curl_std', # 3, 4, 5, 6, 7
                'curl_power', # 0.8, 1, 1.2, 1.4, 1.6
                'blade_width_pct_mean', # 0.01, 0.02, 0.03
                'blade_width_var', # 0, 0.01, 0.02, 0.03, 0.04, 0.05
                'taper_var', # 0, 0.05, 0.1
                'base_spread', # 0, 0.01, 0.05, 0.1, 0.15, 0.2
                'base_angle_var' # 0, 5, 10, 15
                ]
            for p in properties:
                if p in control_dict:
                    self.fac_info[p] = control_dict[p]
                    if p == 'length_mean':
                        if 'length_std' not in control_dict:
                            self.fac_info['length_std'] = self.fac_info['length_mean'] * uniform(0.2, 0.5)
                        if 'base_spread' not in control_dict:
                            self.fac_info['base_spread'] = uniform(0, self.fac_info['length_mean'] / 4)
                    if p == 'curl_mean' and 'curl_std' not in control_dict:
                        self.fac_info['curl_std'] = self.fac_info['curl_mean'] * np.clip(normal(0.3, 0.1), 0.01, 0.6)
            
            taper_y = np.linspace(1, 0, self.fac_info['n_seg']) * normal(1, self.fac_info['taper_var'], self.fac_info['n_seg'])
            taper_x = np.linspace(0, 1, self.fac_info['n_seg'])
            self.fac_info['taper_points'] = np.stack([taper_x, taper_y], axis=-1)

        # self.n_seg = 4
        # self.length_mean = uniform(0.05, 0.15)
        # self.length_std = self.length_mean * uniform(0.2, 0.5)

        # self.curl_mean = uniform(10, 70)
        # self.curl_std = self.curl_mean * np.clip(normal(0.3, 0.1), 0.01, 0.6)
        # self.curl_power = normal(1.2, 0.3)

        # self.blade_width_pct_mean = uniform(0.01, 0.03)
        # self.blade_width_var = uniform(0, 0.05)

        # self.taper_var = uniform(0, 0.1)
        # self.taper_y = np.linspace(1, 0, self.n_seg) * normal(1, self.taper_var, self.n_seg) 
        # self.taper_x = np.linspace(0, 1, self.n_seg)
        # self.taper_points = np.stack([self.taper_x, self.taper_y], axis=-1)

        # self.base_spread = uniform(0, self.length_mean/4)
        # self.base_angle_var = uniform(0, 15)

    def create_asset(self, **params) -> bpy.types.Object:
        
        n_blades = np.random.randint(30, 60)
        
        blade_lengths = normal(self.fac_info['length_mean'], self.fac_info['length_std'], (n_blades, 1))
        seg_lens = (blade_lengths / self.fac_info['n_seg'])
        
        seg_curls = normal(self.fac_info['curl_mean'], self.fac_info['curl_std'], (n_blades, self.fac_info['n_seg'])) 
        seg_curls *= np.power(np.linspace(0, 1, self.fac_info['n_seg']).reshape(1, self.fac_info['n_seg']), self.fac_info['curl_power'])
        seg_curls = np.deg2rad(seg_curls)

        point_rads = np.arange(self.fac_info['n_seg']).reshape(1, self.fac_info['n_seg']) * seg_lens
        point_angles = np.cumsum(seg_curls, axis=-1)
        point_angles -= point_angles[:, [0]]

        points = np.empty((n_blades, self.fac_info['n_seg'], 2))
        points[..., 0] = np.cumsum(point_rads * np.cos(point_angles), axis=-1)
        points[..., 1] = np.cumsum(point_rads * np.sin(point_angles), axis=-1)

        taper = Curve(self.fac_info['taper_points']).to_curve_obj()

        widths = blade_lengths.reshape(-1) * normal(self.fac_info['blade_width_pct_mean'], self.fac_info['blade_width_var'], n_blades)
        objs = []
        for i in range(n_blades):
            obj = Curve(points[i], taper=taper).to_curve_obj(name=f'_blade_{i}', extrude=widths[i], resu=2)
            objs.append(obj)

        with butil.SelectObjects(objs):
            bpy.ops.object.convert(target='MESH')
        butil.delete(taper)

        # Randomly pose and arrange the blades in a circle-ish cluster
        base_angles = uniform(0, 2 * np.pi, n_blades)
        base_rads = uniform(0, self.fac_info['base_spread'], n_blades)
        facing_offsets = np.rad2deg(normal(0, self.fac_info['base_angle_var'], n_blades))
        for a, r, off, obj in zip(base_angles, base_rads, facing_offsets, objs):
            obj.location = (-r * np.cos(a), r * np.sin(a), -0.05 * self.fac_info['length_mean'])
            obj.rotation_euler = (np.pi/2, -np.pi/2, -a + off)

        with butil.SelectObjects(objs):
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        with butil.SelectObjects(objs):
            bpy.ops.object.join()
            bpy.ops.object.shade_flat()
            parent = objs[0]

        tag_object(parent, 'grass_tuft')
        
        return parent

    def finalize_assets(self, assets):
        grass_blade_texture.apply(assets)

if __name__ == '__main__':
    f = GrassTuftFactory(0)
    obj = f.create_asset()