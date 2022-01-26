# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 10:32:01 2022

@author: benja
"""


import os

import rasterio as rio
import geopandas as gpd
import numpy as np

from shapely.geometry import Point, Polygon, MultiPolygon
from whitebox import WhiteboxTools
from rasterstats import zonal_stats
from contextlib import contextmanager
from rasterio.merge import merge



@contextmanager
def cwd(dir_):
    old_dir = os.getcwd()
    os.chdir(dir_)
    try:
        yield 
    finally:
        os.chdir(old_dir)

def max_rast_value_in_shp(geoms, raster_path, stat = 'max'):
    '''Get a list of points representing the location of the maximum values 
    of the underlying raster, in each polygon in the shapefile.
    
    Returns: a list of points, each representing the (or a) local maximum for 
    the same shape.
    
    The crs of the geometries.
    '''
    
    
    if isinstance(geoms, gpd.GeoDataFrame):
        geoms = geoms.geometry
    elif isinstance(geoms, gpd.GeoSeries):
        pass
    elif hasattr(geoms, '__iter__'):
        if isinstance(geoms[0], (Polygon, MultiPolygon)):
            geoms = gpd.GeoSeries(geoms)
    else:
        raise ValueError(
            f'Geometries are of type {type(geoms)} must be gdf or geoseries')
    
   
    
    with rio.open(raster_path) as src:
        data = src.read(1)
        meta = src.meta.copy()
        transform = src.transform
        
    #gdf.to_crs(meta['crs'], inplace= True)
    vals = zonal_stats(geoms, raster_path, stat = [stat])
    vals = [r[stat] for r in vals]
   
    out_geoms = []
    for v, shape in zip(vals, geoms):
        if not v:
            geoms.append(None)
            continue
        
        y, x = np.where(data == v)
        #calculate x and y as:
        #location of upper corner + cell_size * (cell_index + .5)
        y = (y+.5)*transform[4] + transform[5]
        x = (x +.5)*transform[0] + transform[2]
        
        #if there is more than one location: choose the first point
        #that fits in the polygon. 
        if x.shape[0] >1:
            for x,y in zip(x,y):
                point = Point(x,y)
                if shape.intersects(point):
                    break
        else:
            point = Point(x, y)
        out_geoms.append(point)
    return out_geoms, meta['crs']

def count_points_in(poly_gdf, pts_gdf, colname = 'all_structures'):
    '''Count the number of points from points df in each polygon
    in poly_gdf.
    

    Parameters
    ----------
    poly_gdf : geodataframe with polygon geometry.
        
    pts_gdf : geodataframe with points geometry
        
    colname : str 
    name of the column to add to poly_gdf
         The default is 'count'.

    Returns
    -------
    poly_gdf : TYPE
        DESCRIPTION.

    '''
    
    
    counts = gpd.sjoin(poly_gdf,  pts_gdf[['geometry']],  how = 'left',
              ).groupby(level = 0)[poly_gdf.columns[0]].count()
    
    poly_gdf[colname] = counts
    return poly_gdf


def numpy_replace(input_array, val_array, key_array = None):
    '''
    Replace the values in a numpy array based on the keys and values
    provided in val array. By default, the keys to match are the indicies of the
    val array. 
    
    example: 
        input_array = np.array([0, 0, 1, 1, 2])
        val_array = np.array([5,3, 11])
        
        numpy_replace(input_array, val_array)
        returns : np.array([5, 5, 3, 3, 11, 11])

    Parameters
    ----------
    input_array : numpy array
        Array to replace values in

    val_array: numpy array
        Array with the replacement values in it. 
        
    key_array: numpy array
        Array with the keys for the value array.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if not key_array:
        key_array = np.arange(0, val_array.shape[0])
    mapping_ar = np.zeros(key_array.max()+1,dtype=val_array.dtype)
    mapping_ar[key_array] = val_array
    return mapping_ar[input_array]
    


class floodDemandCalculator():
    
    def __init__(self, wkdir):
        self.wkdir = wkdir
        self.intermed_dir = os.path.join(wkdir, 'intermediate_data')
        self.unnested_dir = os.path.join(wkdir, 'watershed_rasters')
        self.results_dir = os.path.join(wkdir, 'results')
        
        for p in [self.wkdir, self.intermed_dir, self.unnested_dir]:
            if not os.path.exists(wkdir):
                os.makedirs(wkdir)
        
        self.fns = {k: os.path.join(self.intermed_dir, f'{k}.tif') 
               for k in ['pit_filled', 'd8_pointer', 'd8_flow_acc']
               }  
        
        self.fns['fld_area_pour_points'] = os.path.join(self.intermed_dir, 'pour_points.shp')
    
    def preprocess_rasters(self, dem_path):
        '''Process the digital elevation model to create needed rasters.
        Creates the following files in the directory "intermediate_rasters"
        sub-directory:
        pit_filled.tif - a pit-filled digital elevation model
        d8_pointer.tif- A Pointer raster designating flow direction for each pixel.
        d8_flow_acc.tif - A d8 flow accumulation raster.
        
        '''
        
        
        wbt = WhiteboxTools()
        wbt.flow_accumulation_full_workflow(dem_path, 
                                            self.fns['pit_filled'],
                                            self.fns['d8_pointer'],
                                            self.fns['d8_flow_acc']
                                       )
    
    def execute(self, dem_path, soil_map_path, struct_map_path, analysis_subsets = {'all_structures': None},
                flow_buffer = 50):
        '''Run all processes to create a set of flood control demand rasters.
        
        
        Parameters
        ----------
        dem_path : path
            path to the digital elevation model covering the whole study area
        soil_map_path : path
            path to the soil map covering the whole study area
        struct_map_path : path
            path to the structure map covering the whole study area.
        
        analysis_subsets : Dictionary of Dictionaries., optional
            Optional parameter for making multiple counts 
            of different structure types.
            
            To count a subset of the structures, 
            pass an additional line to the dictionary: 
            key: name of field for the counts
            value: dictionary, k - name of column to filter on
            
                
            for example, to get two sets of counts, 
            one for all structures, and another for just schools,
            you might pass:
                
            analysis_subsets = 
            {'all_structures': None # No filter given
            'schools': {'SITETYPE_M', ['SCHOOL K / 12', 'EDUCATIONAL']
                        }
                           }
        '''
        
        
        self.preprocess_rasters(dem_path)
        fld_zone = self.preprocess_flood_ares(soil_map_path, struct_map_path, analysis_subsets)
        self.delineate_watersheds(fld_zone, flow_buffer)
        self.sum_data(fld_zone, list(analysis_subsets.keys()))
        
        
    
    def execute_piecewise(self, wshed_shape, wshed_id_col, dem_path, 
                          soil_map_path, struct_map_path, 
                          analysis_subsets = {'all_structures': None}, 
                          flow_buffer = 50):
        '''
        Execute the flood-demand calculation routine piece by piece by major watershed,
        then merge into a final raster.
        Be sure that your delineated watersheds do not divide river-systems.
        

        Parameters
        ----------
        wshed_shape : A shapefile or path to one
           watershed delinations to be used in.
        wshed_id_col : str
            column name for the watershed id. This is used to name the sub-directories
        dem_path : path
            path to the digital elevation model covering the whole study area
        soil_map_path : path
            path to the soil map covering the whole study area
        struct_map_path : path
            path to the structure map covering the whole study area.
        
        analysis_subsets : Dictionary of Dictionaries., optional
            Optional parameter for making multiple counts 
            of different structure types.
            
            To count a subset of the structures, 
            pass an additional line to the dictionary: 
            key: name of field for the counts
            value: dictionary, k - name of column to filter on
            
                
            for example, to get two sets of counts, 
            one for all structures, and another for just schools,
            you might pass:
                
            analysis_subsets = 
            {'all_structures': None # No filter given
            'schools': {'SITETYPE_M', ['SCHOOL K / 12', 'EDUCATIONAL']
                        }
                           }
            
            
        flow_buffer : TYPE, optional
            DESCRIPTION. The default is 50.

        Returns
        -------
        None.

        '''
        wbt = WhiteboxTools()
        for i in wshed_shape.index:
            
            
            
            row = wshed_shape.loc[i]
            row['geometry'] = row.geometry.buffer(100)
            name = row[wshed_id_col]
            wkdir = os.path.join(self.wkdir, name)
            subFdCalc = floodDemandCalculator(
                wkdir)
            sub_paths = {
        k: os.path.join(subFdCalc.intermed_dir, fn) for 
        k, fn in [('bounds','watershed_bounds.shp'),
                   ('dem', 'dem.tif'),
                   ('soil_map', 'soil_map.shp'),
                   ('struct_map', 'struct_map.shp')]
                        
                                         }
            
            bounds_path = os.path.join(
                subFdCalc.intermed_dir, 'watershed_bounds.shp')
            
            
            gpd.GeoDataFrame(row).to_file(
                bounds_path)
            
            wbt.clip_raster_to_polygon(dem_path,
                                       bounds_path,
                                       sub_paths['dem'],
                                       )
            wbt.clip(soil_map_path, bounds_path, 
                                sub_paths['soil_map'])
            wbt.clip(struct_map_path, sub_paths['struct_map'])
            subFdCalc.execute(self, sub_paths['dem'], 
                              sub_paths['soil_map'],
                              sub_paths['struct_map'],
                              analysis_subsets,
                              flow_buffer
                              )
        
        
        for name in self.subset_names: #merge remaining rasters
            src_to_merge = [rio.open(os.path.join(self.wkdir, 
                                                  wshed_id,
                                         f'{name}.tif'))
                            for wshed_id in 
                            wshed_shape[wshed_id_col].unique()
                            ]
            src_to_merge = [src for src in src_to_merge if src]
                
            rast, out_transform = merge(src_to_merge, method = 'max')
            out_meta = src_to_merge[1].meta.copy()
            out_meta.update({
                             "height": rast.shape[1],
                            "width": rast.shape[2],
                             "transform": out_transform,
                            }
                           )
            for src in src_to_merge:
                src.close()
                
            with rio.open(os.path.join(self.results_dir, f'{name}.tif'), 
                          'w+', **out_meta ) as dst:
                
                dst.write(rast.astype(out_meta['dtype']))
                
                
    
    def preprocess_flood_areas(self, soil_map, struct_map, 
                               analysis_subsets = {'all_structures': None}):
        '''
        

        Parameters
        ----------
        soil_map : geodataframe or path to shapefile
            Contains polygons with 
            soil data, including flooding frequency
        struct_map : geodataframe or path to shapefile with points
            contains locations and types of structures. 
        analysis_subsets : dictionary, optional
            Optional parameter for making multiple counts 
            of different structure types.
            
            To count a subset of the structures, 
            pass an additional line to the dictionary: 
            key: name of field for the counts
            value: dictionary, k - name of column to filter on
            
                
            for example, to get two sets of counts, 
            one for all structures, and another for just schools,
            you might pass:
            analysis_subsets = 
            {'all_structures': None # No filter given
            'schools': {'SITETYPE_M', ['SCHOOL K / 12', 'EDUCATIONAL']
                        }
                           
                           }

        Returns
        -------
        geodataframe of flood zones that have at least one floodable structure.

        '''
        self.subset_names = list(analysis_subsets.keys())
        
        if isinstance(soil_map, str):
            soil_map = gpd.read_file(soil_map)
        if isinstance(struct_map, str):
            struct_map = gpd.read_file(struct_map)
            
            
        struct_map.to_crs(soil_map.crs, inplace = True)
        crs = soil_map.crs
        fld_zone = soil_map[soil_map['FLOOD'].isin(['water', 
                                               'rare',
                                               'occasional', 
                                               'frequent'])]
        del soil_map
        
        for name, filt in analysis_subsets.keys():
            if filt:
                if len(filt>1):
                    sub_structs = gpd.pd.concat(
                    struct_map[struct_map[k].isin(v)] for 
                    k, v in filt.items()).drop_duplicates()
                else:
                    sub_structs = struct_map[
                    struct_map[list(filt.keys())[0]].isin( 
                    list(filt.values())[0])]
            else:
                sub_structs = struct_map
                
                    
            fld_zone = count_points_in(fld_zone, struct_map, 
                                       colname = name)
            
        fld_zone = fld_zone[fld_zone[analysis_subsets.keys()].sum()>0]
        return fld_zone
        
    def delineate_watersheds(self, fld_zone, flow_buffer = 50):
        wbt = WhiteboxTools()
        fld_zone.geometry, crs = max_rast_value_in_shp(
            fld_zone.geometry.buffer(flow_buffer), 
         self.fns['d8_flow_acc'])
        
        fld_zone.crs = crs 
        
        fld_zone.to_file(self.fns['fld_area_pour_points'])
        wbt.unnest_basins(self.fns['d8_pointer'],  
                          self.fns['fld_area_pour_points'],
                          self.unnested_dir)
        
    
    def sum_data(self, fld_zone = None, subset_names = None):
        if not fld_zone:
            fld_zone = gpd.read_file(self.fns['fld_area_pour_points'])
            
        if not subset_names:
            subset_names = self.subset_names
        if not self.subset_names:
            subset_names = ['all_structures']
        
        fld_zone['wshed_area'] = 0
        for file in os.listdir(self.unnested_dir):
            data = rio.open(os.path.join(self.unnested_dir, file)).read(1)
            index, counts = np.unique(data, return_counts = True)
            drop_ind = np.where(index == -32768)[0]
            index = np.delete(index, drop_ind); 
            counts = np.delete(counts, drop_ind) 
            index = index -1
            fld_zone.loc[
            index, 'wshed_area'] += counts
        
        
        with rio.open(os.path.join(self.unnested_dir, file)) as src:
            out_meta = src.meta.copy()
        out_meta.update({'dtype': 'float64'})
        
        areas = fld_zone['wshed_area'].to_numpy()
        areas = np.append(areas, 1)
        
        index = fld_zone.index.to_numpy()
        index = np.append(index, -32768)
        
        for name in subset_names:
            self.sum_demand(name, fld_zone, areas, index, 
                            data.shape, out_meta)
            
                
    def sum_demand(self, name, fld_zone, areas, index, shape,
                   out_meta):
        struct_counts = fld_zone[name].to_numpy()
        struct_counts = np.append(struct_counts, 0)
        
        out_data = np.zeros(shape)
        for file in os.listdir(self.unnested_dir):
            data = rio.open(os.path.join(self.unnested_dir, 
                                         file)).read(1)     
            
            data = np.where(data == -32768, index.shape[0], data)-1 
            count_weights = numpy_replace(data, struct_counts)
            area_weights = numpy_replace(data, areas)
            
            out_data = out_data + count_weights / area_weights
            
        out_data = out_data * 100
        out_path = os.path.join(self.results_dir, f'{name}.tif')
        with rio.open(out_path, 'w+', **out_meta ) as dst:
            dst.write(np.array([out_data]).astype(out_meta['dtype']))
            
            
        