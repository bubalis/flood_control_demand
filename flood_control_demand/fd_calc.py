# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 10:51:06 2023

@author: Benjamin Dube
"""
#%%
from argparse import ArgumentParser
import warnings
import os
#%%

from util import floodDemandCalculator
import util

def parse_args():
    parser = ArgumentParser(
        description = 
        'A tool for calculating the relative value of different parts of a landscape in protecting downstream areas from flooding'
        )
    parser.add_argument( 'dem_path', type = str, required = True, 
                        help = 'path of the digital elevation model to use. Can be a relative or absolute path.' )
    parser.add_argument('flood_map_path', type = str, required = True, 
                        help = 'path of the flood polygon map. Can be a relative or absolute path.' )
    parser.add_argument('struct_map_path', type = str, required = True, 
                        help = 'path of the structure point map. Can be a relative or absolute path.' )
    parser.add_argument('-n', '--flood_map_not_SSurgo', action = 'store_true', 
                        help = 'Is the flood map not a ssurgo map?')
    parser.add_argument('-s', '--struct_weight_col', type = str, 
                        help = '-Optional. column for structure weights in the structure points map. For example, monetary value.',
                        nargs = '?', const = '')
    parser.add_argument('-a',  '--area_weight_col', type = str, 
                        help = 'Optional, Column for area weights in the flood areas polygon map. For instance, the annual inundation probability.',
                        nargs = '?', const = '')
    parser.add_argument('-w', '--wkdir', type = str, 
                        help = 'Optional, Working Directory for the routine.',
                        nargs = '?', const = '')
    
    parser.add_argument('-wp', '--watershed_path', type = str, 
                        help = 'Optional, Path to a watersheds polygon map, for running the routine piecewise.',
                        nargs = '?', const = '')
    parser.add_argument('-wi', '--watershed_id_col', type = str, 
                        help = 'Optional, column name for watershed names, for running the routine piecewise.')
    parser.add_argument('--flow_search_distance', type = int, nargs = '?', const = 50, 
            help = 'Search distance for highest flow area near a flood-prone polygon, which is where each polygons "flood watershed" is dileneated from. Default is 50 meters.')
   
    parser.add_argument('-t', '--test', action = 'store_true', help = 'Run the routine on test data?')
    
    parsed = parser.parse_args()
    
    
    
    if (parsed.watershed_shpfile or parsed.watershed_id_col) and not (parsed.watershed_shpfile and parsed.watershed_id_col):
        warnings.warn('If you pass a watershed map path or watershed_id_col, you must pass both to run piecewise. Running routine on whole area.')
    
    
    args = {**dict(
    wkdir = parsed.__dict__.get('wkdir', os.getcwd()),
    flood_map_is_ssurgo = not parsed.flood_map_not_SSurgo
    ), **{parsed.dict.get(x, None) for x in 
              ['dem_path', 'flood_map_path', 'struct_map_path', 'struct_weight_col', 
               'area_weight_col', 'watershed_path', 'watershed_id_col']
              
              }}
          
    args.update({key: os.path.join(args['wkdir'], value) for key, value in args.items() if 'path' in value and not os.path.isabs(value)})
           
    return args

def main():
    args = parse_args()
    
    if args.test:
        test()
    else:
        FD = floodDemandCalculator(args['wkdir'])
        
        kwargs = {k: v for k, v in args.items() if k in ['struct_map_path',
       'area_weight_col', 'struct_weight_col','flood_map_is_ssurgo', 'flow_search_distance']}
        if args['watershed_path'] and args['watershed_id_col']:
            FD.execute_piecewise(args['watershed_path'], args['watershed_id_col'],
                                 args['dem_path'], args['soil_map_path'], args['flow_search_distance']
                                 **kwargs)
        
        else:
            FD.execute( args['dem_path'], args['soil_map_path'], args['struct_map_path'],
             args['area_weight_col'],  args['struct_weight_col'], args['flood_map_is_ssurgo'])
    


def test():
    wkdir = os.path.join(os.path.split(util.__file__)[0], 'tests')
    data_dir = os.path.join(wkdir, 'test_data')
    Calc = floodDemandCalculator(wkdir)
    dem_path = os.path.join(data_dir, 'dem.tif')
    soil_map_path = os.path.join(data_dir, 'soil_map.shp')
    struct_map_path = os.path.join(data_dir, 'structure_map.shp')
    
    Calc.execute(dem_path, soil_map_path, struct_map_path)
    
    piecewise_wkdir = os.path.join(wkdir, 'piecewise')
    watersheds_map = os.path.join(data_dir, 'watershed_boundaries.shp')
    
    CalcP = floodDemandCalculator(piecewise_wkdir)
    
    CalcP.execute_piecewise(watersheds_map, 'mjr_wshed', dem_path, 
                            soil_map_path, struct_map_path)
    
