# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bpy
import numpy as np
from numpy.random import uniform

from .veratrum import VeratrumMonocotFactory
from .banana import BananaMonocotFactory, TaroMonocotFactory
from .agave import AgaveMonocotFactory
from .grasses import GrassesMonocotFactory, MaizeMonocotFactory, WheatMonocotFactory
from .growth import MonocotGrowthFactory
from .tussock import TussockMonocotFactory
from placement.factory import AssetFactory
from util.math import FixedSeed
from ..utils.decorate import join_objects
from ..utils.mesh import polygon_angles
from assets.utils.tag import tag_object, tag_nodegroup

class MonocotFactory(AssetFactory):
    max_cluster = 10

    def create_asset(self, i, **params) -> bpy.types.Object:
        params['decorate'] = True
        if self.factory.is_grass:
            n = np.random.randint(1, 6)
            angles = polygon_angles(n, np.pi / 4, np.pi * 2)
            radius = uniform(.08, .16, n)
            monocots = [self.factory.create_asset(**params, i=j + i * self.max_cluster) for j in range(n)]
            for m, a, r in zip(monocots, angles, radius):
                m.location = r * np.cos(a), r * np.sin(a), 0
            obj = join_objects(monocots)
            tag_object(obj, 'monocot')
            return obj
        else:
            m = self.factory.create_asset(**params)
            tag_object(m, 'monocot')
            return m

    def __init__(self, factory_seed, coarse=False, control_dict={}):
        # factory_method=None, grass=None, 
        super(MonocotFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            grass_factory = [TussockMonocotFactory, GrassesMonocotFactory, WheatMonocotFactory,
                MaizeMonocotFactory]
            nongrass_factory = [AgaveMonocotFactory, BananaMonocotFactory, TaroMonocotFactory,
                VeratrumMonocotFactory]
            # noinspection PyTypeChecker

            self.factory_methods = grass_factory + nongrass_factory
            if 'mode' in control_dict:
                # grass_only, nongrass_only
                if control_dict['mode'] == 'grass_only':
                    self.factory_methods = grass_factory
                elif control_dict['mode'] == 'nongrass_only':
                    self.factory_methods = nongrass_factory

            weights = np.array([1] * len(self.factory_methods))
            self.weights = weights / weights.sum()
            # if factory_method is None:
            with FixedSeed(self.factory_seed):
                if 'factory_method' in control_dict:
                    # Tussock, Grasses, Wheat, Maize, Agave, Banana, Taro, Veratrum
                    name = control_dict['factory_method']
                    if name == 'Tussock':
                        factory_method = TussockMonocotFactory
                    elif name == 'Grasses':
                        factory_method = GrassesMonocotFactory
                    elif name == 'Wheat':
                        factory_method = WheatMonocotFactory
                    elif name == 'Maize':
                        factory_method = MaizeMonocotFactory
                    elif name == 'Agave':
                        factory_method = AgaveMonocotFactory
                    elif name == 'Banana':
                        factory_method = BananaMonocotFactory
                    elif name == 'Taro':
                        factory_method = TaroMonocotFactory
                    elif name == 'Veratrum':
                        factory_method = VeratrumMonocotFactory
                        
                    if 'specific_configs' in control_dict:
                    # a dict that stores detailed params in the factory -- factory_method must be specified in this case
                    # details can be seen from the specific class
                        configs = control_dict['specific_configs']
                else:
                    factory_method = np.random.choice(self.factory_methods, p=self.weights)
                    configs = {}
                self.factory: MonocotGrowthFactory = factory_method(factory_seed, coarse, configs)

