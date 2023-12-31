# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick, Yiming Zuo, Alejandro Newell


import pdb
import logging

import gin
import numpy as np
from numpy.random import uniform, normal

import bpy

from assets.trees import tree, treeconfigs
from assets.leaves import leaf, leaf_v2, leaf_pine, leaf_ginko, leaf_broadleaf, leaf_maple
from assets.fruits import apple, blackberry, coconutgreen, durian, starfruit, strawberry, compositional_fruit
from nodes.node_info import Nodes
from nodes.node_wrangler import NodeWrangler
from . import tree_flower

from util import blender as butil
from util.math import FixedSeed
from util.blender import deep_clone_obj
from util import camera as camera_util

from placement.factory import AssetFactory, make_asset_collection
from placement import detail
from placement.split_in_view import split_inview

from surfaces import surface
from surfaces.scatters import rocks, grass

from assets.cloud.generate import CloudFactory
from ..utils.decorate import write_attribute

from assets.utils.tag import tag_object, tag_nodegroup

logger = logging.getLogger('trees')

@gin.configurable
class GenericTreeFactory(AssetFactory):

    scale = 0.35 # trees are defined in weird units currently, need converting to meters

    def __init__(
        self, 
        factory_seed, 
        genome: tree.TreeParams, 
        child_col, 
        trunk_surface, 
        realize=False, 
        meshing_camera=None, 
        cam_meshing_max_dist=1e7,
        coarse_mesh_placeholder=False,
        adapt_mesh_method='remesh', 
        decimate_placeholder_levels=0, 
        min_dist=None,
        control=False, 
        control_dict={},
        coarse=False
    ):

        super(GenericTreeFactory, self).__init__(factory_seed, coarse=coarse)

        self.genome = genome
        self.child_col = child_col
        self.trunk_surface = trunk_surface
        self.realize = realize

        self.camera = meshing_camera
        self.cam_meshing_max_dist = cam_meshing_max_dist
        self.adapt_mesh_method = adapt_mesh_method
        self.decimate_placeholder_levels = decimate_placeholder_levels
        self.coarse_mesh_placeholder = coarse_mesh_placeholder

        self.min_dist = min_dist

    def create_placeholder(self, i, loc, rot):

        logger.debug(f'generating tree skeleton')
        skeleton_obj = tree.tree_skeleton(
            self.genome.skeleton, self.genome.trunk_spacecol, self.genome.roots_spacecol, init_pos=(0, 0, 0), scale=self.scale)
        
        if self.coarse_mesh_placeholder:
            pholder =  self._create_coarse_mesh(skeleton_obj)
        else:
            pholder = butil.spawn_cube(size=4)

        butil.parent_to(skeleton_obj, pholder, no_inverse=True)
        return pholder
            
    
    def _create_coarse_mesh(self, skeleton_obj):
        logger.debug('generating skinned mesh')
        coarse_mesh = deep_clone_obj(skeleton_obj)
        surface.add_geomod(coarse_mesh, tree.skin_tree, input_kwargs={'params': self.genome.skinning}, apply=True)

        if self.decimate_placeholder_levels > 0:
            butil.modify_mesh(coarse_mesh, 'DECIMATE', decimate_type='UNSUBDIV', iterations=self.decimate_placeholder_levels)

        return coarse_mesh

    def finalize_placeholders(self, placeholders):
        if not self.coarse_mesh_placeholder:
            return
        with FixedSeed(self.factory_seed):
            logger.debug(f'adding {self.trunk_surface} to {len(placeholders)=}')
            self.trunk_surface.apply(placeholders)

    def asset_parameters(self, distance: float, vis_distance: float) -> dict:
        if self.min_dist is not None and distance < self.min_dist:
            logger.warn(f'{self} recieved {distance=} which violates {self.min_dist=}. Ignoring')
            distance = self.min_dist
        return dict(face_size=detail.target_face_size(distance), distance=distance)

    def create_asset(self, placeholder, face_size, distance, **kwargs) -> bpy.types.Object:

        skeleton_obj = placeholder.children[0]

        if not self.coarse_mesh_placeholder:
            placeholder = self._create_coarse_mesh(skeleton_obj)
            self.trunk_surface.apply(placeholder)
            butil.parent_to(skeleton_obj, placeholder, no_inverse=True)
            placeholder.hide_render = True

        if self.child_col is not None:
            assert self.genome.child_placement is not None

            max_needed_child_fs = detail.target_face_size(self.min_dist, global_multiplier=1) if self.min_dist is not None else None

            logger.debug(f'adding tree children using {self.child_col=}')
            butil.select_none()
            surface.add_geomod(skeleton_obj, tree.add_tree_children, input_kwargs=dict(
                child_col=self.child_col, params=self.genome.child_placement, 
                realize=self.realize, merge_dist=max_needed_child_fs
            ))

        if self.camera is not None and distance < self.cam_meshing_max_dist:
            assert self.adapt_mesh_method != 'remesh'
            skin_obj, outofview, vert_dists, _ = split_inview(placeholder, cam=self.camera, vis_margin=0.15)
            butil.parent_to(outofview, skin_obj, no_inverse=True, no_transform=True)
            face_size = detail.target_face_size(vert_dists.min())
        else:
            skin_obj = deep_clone_obj(placeholder, keep_modifiers=True, keep_materials=True)

        skin_obj.hide_render = False

        if self.adapt_mesh_method == 'remesh':
            butil.modify_mesh(skin_obj, 'SUBSURF', levels=self.decimate_placeholder_levels + 1) # one extra level to smooth things out or remesh is jaggedy

        with butil.DisableModifiers(skin_obj):
            detail.adapt_mesh_resolution(skin_obj, face_size, method=self.adapt_mesh_method, apply=True)

        butil.parent_to(skin_obj, placeholder, no_inverse=True, no_transform=True)

        if self.realize:
            logger.debug(f'realizing tree children')
            butil.apply_modifiers(skin_obj)
            butil.apply_modifiers(skeleton_obj)
            with butil.SelectObjects([skin_obj, skeleton_obj], active=0):
                bpy.ops.object.join()
        else:
            butil.parent_to(skeleton_obj, skin_obj, no_inverse=True)

        tag_object(skin_obj, 'tree')
        return skin_obj
        

@gin.configurable
def random_season(weights=None):
    options = ['autumn', 'summer', 'spring', 'winter']
    
    if weights is not None:
        weights = np.array([weights[k] for k in options])
    else:
        weights = np.array([0.25, 0.3, 0.4, 0.1])
    return np.random.choice(options, p=weights/weights.sum())

@gin.configurable
def random_species(season='summer', pine_chance=0.):
    tree_species_code = np.random.rand(32)

    if season is None:
        season = random_season()

    if tree_species_code[-1] < pine_chance:
        return treeconfigs.pine_tree(), 'leaf_pine'
    # elif tree_species_code < 0.2:
    #     tree_args = treeconfigs.palm_tree()
    # elif tree_species_code < 0.3:
    #     tree_args = treeconfigs.baobab_tree()
    else:
        return treeconfigs.random_tree(tree_species_code, season), None

def random_tree_child_factory(seed, leaf_params, leaf_type, season, **kwargs):

    if season is None:
        season = random_season()

    fruit_scale = 0.2

    if leaf_type is None:
        return None, None
    elif leaf_type == 'leaf':
        return leaf.LeafFactory(seed, leaf_params, **kwargs), surface.registry('greenery')
    elif leaf_type == 'leaf_pine':
        return leaf_pine.LeafFactoryPine(seed, season, **kwargs), None
    elif leaf_type == 'leaf_ginko':
        return leaf_ginko.LeafFactoryGinko(seed, season, **kwargs), None
    elif leaf_type == 'leaf_maple':
        return leaf_maple.LeafFactoryMaple(seed, season, **kwargs), None
    elif leaf_type == 'leaf_broadleaf':
        return leaf_broadleaf.LeafFactoryBroadleaf(seed, season, **kwargs), None
    elif leaf_type == 'leaf_v2':
        return leaf_v2.LeafFactoryV2(seed, **kwargs), None
    elif leaf_type == 'berry':
        return leaf.BerryFactory(seed, leaf_params, **kwargs), None
    elif leaf_type == 'apple':
        return apple.FruitFactoryApple(seed, scale=fruit_scale, **kwargs), None
    elif leaf_type == 'blackberry':
        return blackberry.FruitFactoryBlackberry(seed, scale=fruit_scale, **kwargs), None
    elif leaf_type == 'coconutgreen':
        return coconutgreen.FruitFactoryCoconutgreen(seed, scale=fruit_scale, **kwargs), None
    elif leaf_type == 'durian':
        return durian.FruitFactoryDurian(seed, scale=fruit_scale, **kwargs), None
    elif leaf_type == 'starfruit':
        return starfruit.FruitFactoryStarfruit(seed, scale=fruit_scale, **kwargs), None
    elif leaf_type == 'strawberry':
        return strawberry.FruitFactoryStrawberry(seed, scale=fruit_scale, **kwargs), None
    elif leaf_type == 'compositional_fruit':
        return compositional_fruit.FruitFactoryCompositional(seed, scale=fruit_scale, **kwargs), None
    elif leaf_type == 'flower':
        return tree_flower.TreeFlowerFactory(seed, rad=uniform(0.15, 0.25), **kwargs), None
    elif leaf_type == 'cloud':
        return CloudFactory(seed), None
    elif leaf_type == 'grass':
        return grass.GrassTuftFactory(seed), surface.registry('greenery')
    elif leaf_type == 'rocks':
        return rocks.BlenderRockFactory(seed, detail=1), surface.registry('rock_collection')
    else:
        raise ValueError(f'Unrecognized {leaf_type=}')   

def make_leaf_collection(seed, 
        leaf_params, n_leaf, leaf_types, 
        fruit_types=None, 
        season=None, relative_fruit_density=0.05):

    logger.debug(f'Starting make_leaf_collection({seed=}, {n_leaf=}, {fruit_types=}...)')

    if season is None:
        season = random_season()

    weights = []

    if not isinstance(leaf_types, list):
        leaf_types = [leaf_types]

    if not isinstance(fruit_types, list):
        fruit_types = [fruit_types]

    child_factories = []
    for leaf_type in leaf_types:
        if leaf_type is not None:
            leaf_factory, _ = random_tree_child_factory(seed, leaf_params, leaf_type=leaf_type, season=season)
            child_factories.append(leaf_factory)
            weights.append(1.0-relative_fruit_density)

    for fruit_type in fruit_types:
        if fruit_type is not None:
            fruit_factory, _ = random_tree_child_factory(seed, leaf_params, leaf_type=fruit_type, season=season)
            child_factories.append(fruit_factory)
            weights.append(relative_fruit_density)

    weights = np.array(weights)
    weights /= np.sum(weights) # normalize to 1       

    col = make_asset_collection(child_factories, n_leaf, verbose=True, weights=weights)
    # if leaf_surface is not None:
    #     leaf_surface.apply(list(col.objects))
    for obj in col.objects:
        butil.modify_mesh(obj, 'DECIMATE', ratio=0.07, apply=True)
        butil.apply_transform(obj, rot=True, scale=True)
        butil.apply_modifiers(obj)
    return col

def random_leaf_collection(season, n=5):
    (_, _, leaf_params), leaf_type = random_species(season=season)
    return make_leaf_collection(np.random.randint(1e5), leaf_params, n_leaf=n, leaf_types=leaf_type or 'leaf_v2')

def make_twig_collection(
    seed, 
    twig_params, leaf_params, 
    trunk_surface, 
    n_leaf, n_twig, 
    leaf_types, fruit_type=None, 
    season=None, 
    twig_valid_dist=6
):


    logger.debug(f'Starting make_twig_collection({seed=}, {n_leaf=}, {n_twig=}...)')

    if season is None:
        season = random_season()

    if leaf_types is not None or fruit_type is not None:
        child_col = make_leaf_collection(seed, leaf_params, n_leaf, leaf_types, fruit_type, season=season)
    else:
        child_col = None


    twig_factory = GenericTreeFactory(seed, twig_params, child_col, trunk_surface=trunk_surface, realize=True)

    # print(twig_factory, n_twig, twig_valid_dist)
    # exit()
    n_twig = 2 if n_twig is None else n_twig

    col = make_asset_collection(twig_factory, n_twig, verbose=False, distance=twig_valid_dist)


    if child_col is not None:
        child_col.hide_viewport = False
        butil.delete(list(child_col.objects))
    return col

@gin.configurable
class TreeFactory(GenericTreeFactory):

    n_leaf = 10
    n_twig = 2

    

    def control(self,control_dict):

        # Leaves
        if 'n_leaf' in control_dict:
            # experiments default: 10
            # Leaf count/quantity/number/amount;
            # Primary photosynthetic organs of plants;
            # This number can vary greatly depending on the species, size, and stage of growth
            self.n_leaf = control_dict['n_leaf']
        else:
            # random leaves; don't care about number of leaves
            self.n_leaf = np.random.randint(0,100)

        leaf_types = ['leaf', 'leaf_v2', 'leaf_broadleaf', 'leaf_ginko', 'leaf_maple', 'flower', 'berry', None]
        if 'leaf_type' in control_dict:
            # experiments default: leaf_broadleaf
            # Options include 'leaf', 'leaf_v2', 'leaf_broadleaf', 'leaf_ginko', 'leaf_maple'
            # leaf: medium-sized leaves; darker green;
            # leaf_v2: yellow medium-sized leaves;
            # leaf_broadleaf: lighter green;
            # leaf_ginko: green;
            # leaf_maple: leaves of maple trees, which are a type of deciduous tree belonging to the genus Acer. These leaves are known for their distinctive shape and vibrant colors, making them iconic symbols of autumn in many regions. The shape of maple leaves is characterized by palmate venation, meaning that the veins radiate from a central point like fingers of a hand.
            # flower: tiny yellow flowers
            # berry: green balls (resembling intricate polygons)
            # None: no leaves
            # Leaf types can be determined by seasons
            self.leaf_type = control_dict['leaf_type']
        else:
            self.leaf_type = np.random.choice(leaf_types)

        if 'fruit_type' in control_dict:
            # apple: big yellow apples;
            # blackberry: small black balls;
            # coconutgreen: big green balls;
            # durian: dark yellow durians;
            # starfruit: yellow middle-sized fruit;
            # strawberry: small orange balls;
            # composition_fruit
            self.fruit_type = control_dict['fruit_type']
        else:
            self.fruit_type = self.get_fruit_type()

        if 'n_twig' in control_dict:
            # experiments default: 3
            # Count/quantity/cluster/pile of twigs, branches;
            # Usually small, thin branches or stems on trees/thrubs/plants;
            # Usually characterized by their slender shape
            # They provide support for leaves, flowers, and fruits, allowing them to be exposed to sunlight and air;
            # Also transport water, nutrients, and sugars between the plant's roots and other parts of the plant;
            # They can serve as a site for the formation of new buds, from which new growth can emerge.
            self.n_twig = control_dict['n_twig']
        else:
            # random twigs; don't care about number of twigs
            self.n_twig = np.random.randint(0,10)

        if 'season' in control_dict:
            # experiments default: summer
            # summer: green leaves;
            # autumn: yellow leaves;
            # spring: no leaves but with some fruits
            # winter: no leaves, no fruits
            self.season = control_dict['season']
        else:
            self.season = np.random.choice(['summer', 'winter', 'autumn', 'spring'])


        return n_leaf,leaf_type,n_twig,season,fruit_type


    def get_leaf_type(self,season):
        # return np.random.choice(['leaf', 'leaf_v2', 'flower', 'berry', 'leaf_ginko'], p=[0, 0.70, 0.15, 0, 0.15])
        # return
        # return 'leaf_maple'
        leaf_type = np.random.choice(['leaf', 'leaf_v2', 'leaf_broadleaf', 'leaf_ginko', 'leaf_maple'], p=[0, 0.0, 0.70, 0.15, 0.15])
        flower_type = np.random.choice(['flower', 'berry', None], p=[1.0, 0.0, 0.0])
        if season == "spring":
            return [flower_type]
        else:
            return [leaf_type]
        # return [leaf_type, flower_type]
        # return ['leaf_broadleaf', 'leaf_maple', 'leaf_ginko', 'flower']


    def get_fruit_type(self):
        # return np.random.choice(['leaf', 'leaf_v2', 'flower', 'berry', 'leaf_ginko'], p=[0, 0.70, 0.15, 0, 0.15])
        # return
        # return 'leaf_maple'
        fruit_type = np.random.choice(['apple', 'blackberry', 'coconutgreen',
            'durian', 'starfruit', 'strawberry', 'compositional_fruit'],
            p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4])

        return [fruit_type]

    def __init__(self, seed, season=None, coarse=False, fruit_chance=0.2,control = False, control_dict={}, **kwargs):


        if(control_dict !={}):
            # print('1', self.n_twig)
            # exit()
            self.n_leaf, self.leaf_type, self.n_twig, self.season, self.fruit_type = self.control(control_dict)
            (tree_params, twig_params, leaf_params), _ = random_species(season=season)
            trunk_surface = surface.registry('bark')
            # print('1', self.n_twig)
            # exit()
        else:
            # print('2', self.n_twig)
            # exit()
            with FixedSeed(seed):
                if season is None:
                    season = np.random.choice(['summer', 'winter', 'autumn', 'spring'])

            with FixedSeed(seed):
                (tree_params, twig_params, leaf_params), leaf_type = random_species(season)


                leaf_type = leaf_type or self.get_leaf_type(season)
                if not isinstance(leaf_type, list):
                    leaf_type = [leaf_type]
                
                self.n_twigs = 2

                trunk_surface = surface.registry('bark')

                if uniform() < fruit_chance:
                    fruit_type = self.get_fruit_type()
                else:
                    fruit_type = None
            # print('2', self.n_twig)
            # exit()
        

        

        super(TreeFactory, self).__init__(seed, tree_params, child_col=None, trunk_surface=trunk_surface, coarse=coarse, **kwargs)

        with FixedSeed(seed):
            colname = f'assets:{self}.twigs'
            use_cached = colname in bpy.data.collections
            if use_cached == coarse:
                logger.warning(f'In {self}, encountered {use_cached=} yet {coarse=}, unexpected since twigs are typically generated only in coarse')

            if colname not in bpy.data.collections:
                self.child_col = make_twig_collection(seed, twig_params, leaf_params, trunk_surface, self.n_leaf, self.n_twig, leaf_type, fruit_type, season=season)
                self.child_col.name = colname
                assert self.child_col.name == colname, f'Blender truncated {colname} to {self.child_col.name}'
            else:
                self.child_col = bpy.data.collections[colname]

@gin.configurable
class BushFactory(GenericTreeFactory):

    n_leaf = 3
    n_twig = 3
    max_distance = 50

    def __init__(self, seed, coarse=False, control=False, control_dict={}, **kwargs):
        

        if tree_params is None:
            with FixedSeed(seed):
                shrub_shape = np.random.randint(2)
        else:
            shrub_shape = control_dict['shrub_shape']
        trunk_surface = surface.registry('bark')
        tree_params, twig_params, leaf_params = treeconfigs.shrub(shrub_shape=shrub_shape)

        super(BushFactory, self).__init__(seed, tree_params, child_col=None, trunk_surface=trunk_surface, coarse=coarse,  **kwargs)
            
        if 'n_leaf' in control_dict:
            # range = [2, 3, 4]
            assert isinstance(control_dict['n_leaf'], int)
            self.n_leaf = control_dict['n_leaf']
        
        if 'n_twig' in control_dict:
            # range = [2, 3, 4]
            assert isinstance(control_dict['n_twig'], int)
            self.n_twig = control_dict['n_twig']

        if 'max_distance' in control_dict:
            # range = [20, 30, 40, 50, 60, 70]
            assert isinstance(control_dict['max_distance'], int)
            self.max_distance = control_dict['max_distance']


        with FixedSeed(seed):  
            if 'leaf_type' in control_dict:
                assert isinstance(control_dict['leaf_type'], str)
                leaf_type = control_dict['leaf_type']
            elif 'random_leaf_type' in control_dict:
                # return a dict specifying the ratio out of 1
                type_dict = control_dict['random_leaf_type']
                assert sum(type_dict.values()) == 1
                leaf_type = np.random.choice(type_dict.keys(), 
                p=type_dict.values())
            else: 
                leaf_type = np.random.choice(['leaf', 'leaf_v2', 'flower', 'berry'], p=[0.1, 0.4, 0.5, 0])
            
            colname = f'assets:{self}.twigs'
            use_cached = colname in bpy.data.collections
            if use_cached == coarse:
                logger.warning(f'In {self}, encountered {use_cached=} yet {coarse=}, unexpected since twigs are typically generated only in coarse')

            
            if colname not in bpy.data.collections:
                self.child_col = make_twig_collection(seed, twig_params, leaf_params, trunk_surface, self.n_leaf, self.n_twig, leaf_type) 
                self.child_col.name = colname
                assert self.child_col.name == colname, f'Blender truncated {colname} to {self.child_col.name}'
            else:
                self.child_col = bpy.data.collections[colname]

    #     else:
    #         with FixedSeed(seed):
    #             leaf_type = np.random.choice(['leaf', 'leaf_v2', 'flower', 'berry'], p=[0.1, 0.4, 0.5, 0])
    #             colname = f'assets:{self}.twigs'
    #             use_cached = colname in bpy.data.collections
    #             if use_cached == coarse:
    #                 logger.warning(f'In {self}, encountered {use_cached=} yet {coarse=}, unexpected since twigs are typically generated only in coarse')

    #             if colname not in bpy.data.collections:
    #                 self.child_col = make_twig_collection(seed, self.twig_params, self.leaf_params, trunk_surface, self.n_leaf, self.n_twig, leaf_type) 
    #                 self.child_col.name = colname
    #                 assert self.child_col.name == colname, f'Blender truncated {colname} to {self.child_col.name}'
    #             else:
    #                 self.child_col = bpy.data.collections[colname]
    

    # def control(self, control_dict):
    #     if 'n_leaf' in control_dict:
    #         self.n_leaf = control_dict['n_leaf']
        
    #     if 'n_twig' in control_dict:
    #         self.n_twig = control_dict['n_twig']

    #     if 'max_distance' in control_dict:
    #         self.max_distance = control_dict['max_distance']

    #     if 'shrub_shape' in control_dict:
    #         shrub_shape = control_dict['shrub_shape']
    #         trunk_surface = surface.registry('bark')
    #         self.tree_params, self.twig_params, self.leaf_params = treeconfigs.shrub(shrub_shape=shrub_shape)

    #     if 'leaf_type' in control_dict:
    #         # range: ['leaf', 'leaf_v2', 'flower', 'berry']
    #         self.leaf_type = control_dict['leaf_type']




        