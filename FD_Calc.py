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
        
        
def load_if_path(shpfile):
    if isinstance(shpfile, str):
        return gpd.read_file(shpfile)
    else:
        return shpfile

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


def count_points_in(poly_gdf, pts_gdf, colname = 'count'):
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
    
    
    sindex = pts_gdf.sindex
    counts = []
    
    for other in poly_gdf.itertuples():
        bounds = other.geometry.bounds
        c = list(sindex.intersection(bounds))
        possible_matches = pts_gdf.iloc[c]
        count = possible_matches.intersects(
            other.geometry).sum()
        
        counts.append(count)
    poly_gdf[colname] = counts
    return poly_gdf



def count_points_in2(poly_gdf, pts_gdf, colname = 'all_struct'):
    '''
    This version is not working with the current
    
    Count the number of points from points df in each polygon
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
    
    
    joined = gpd.sjoin(poly_gdf, pts_gdf['geometry'],  how = 'left',
              )
    counts = joined[~joined['index_right'].isna()].groupby(level =0
                    )[joined.columns[0]].count()
    
    poly_gdf[colname] = 0
    poly_gdf.loc[counts.index, colname] = counts 
    
    return poly_gdf


def sum_pts_weights(poly_gdf, pts_gdf, colname = 'all_struct', 
                    pts_weight_col = 'value'):
    
    
    
    sindex = pts_gdf.sindex
    values = []
    
    for other in poly_gdf.itertuples():
        bounds = other.geometry.bounds
        c = list(sindex.intersection(bounds))
        possible_matches = pts_gdf.iloc[c]
        subset = possible_matches[possible_matches.intersects(
            other.geometry)]
        value = subset[pts_weight_col].sum()
        
        values.append(value)
        
    poly_gdf[colname] = values
    return poly_gdf
    

def sum_pts_weights2(poly_gdf, pts_gdf, colname = 'all_struct', 
                    pts_weight_col = 'value'):
    '''This weighting function is not working with my current
    version of geopandas.'''
    
    
    vals = gpd.sjoin(poly_gdf,  pts_gdf[['geometry', pts_weight_col]],  how = 'left',
              ).groupby(level = 0)[pts_weight_col].sum()
    
    poly_gdf[colname] = vals
    
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
        
        for p in [self.wkdir, self.intermed_dir, self.unnested_dir,
                  self.results_dir]:
            if not os.path.exists(p):
                os.makedirs(p)
        
        self.fns = {k: os.path.join(self.intermed_dir, f'{k}.tif') 
               for k in ['pit_filled', 'd8_pointer', 'd8_flow_acc']
               }  
        
        self.fns['fld_area_pour_points'] = os.path.join(self.intermed_dir, 'pour_points.shp')
        self.fns['fld_area_polys'] = os.path.join(self.intermed_dir, 'flood_polys.shp')
        
        self.subset_names = None
        self.crs = None
        
        
    def preprocess_rasters(self, dem_path):
        '''Process the digital elevation model to create needed rasters.
        Creates the following files in the directory "intermediate_rasters"
        sub-directory:
        pit_filled.tif - a pit-filled digital elevation model
        d8_pointer.tif- A Pointer raster designating flow direction for each pixel.
        d8_flow_acc.tif - A d8 flow accumulation raster.
        
        '''
        
        
        wbt = WhiteboxTools()
        self.crs = rio.open(dem_path).crs
        wbt.flow_accumulation_full_workflow(dem_path, 
                                            self.fns['pit_filled'],
                                            self.fns['d8_pointer'],
                                            self.fns['d8_flow_acc']
                                       )
        os.chdir(self.wkdir)
    
    def execute(self, dem_path, soil_map_path, struct_map_path, analysis_subsets = {'all_struct': None},
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
            {'all_struct': None # No filter given
            'schools': {'SITETYPE_M', ['SCHOOL K / 12', 'EDUCATIONAL']
                        }
                           }
        '''
        
        
        self.preprocess_rasters(dem_path)
        fld_zone = self.preprocess_flood_areas(soil_map_path, struct_map_path, analysis_subsets)
        self.delineate_watersheds(fld_zone, flow_buffer)
        self.sum_data(subset_names = list(analysis_subsets.keys()))
        
        
    
    def execute_piecewise(self, wshed_shape, wshed_id_col, dem_path, 
                          soil_map_path, struct_map_path, 
                          analysis_subsets = {'all_struct': None}, 
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
            column name for the watershed id. 
            This is used to name the sub-directories
        
        dem_path : path
        path to the digital elevation model 
        covering the whole study area
        
       soil_map_path : path
       path to the soil map 
       covering the whole study area
   
       struct_map_path : path
       path to the structure map 
       covering the whole study area.
        
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
        {'all_struct': None # No filter given
        'schools': {'SITETYPE_M', ['SCHOOL K / 12', 'EDUCATIONAL']
                    }
                       }
            
            
        flow_buffer : TYPE, optional
            DESCRIPTION. The default is 50.


        Returns
        -------
        None.

        '''
        subset_names = list(analysis_subsets.keys())
        
        wshed_shape = load_if_path(wshed_shape)
        
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
            
            
            gpd.GeoDataFrame([row]).to_file(
                bounds_path)
            
            wbt.clip_raster_to_polygon(dem_path,
                                       bounds_path,
                                       sub_paths['dem'],
                                       )
            wbt.clip(soil_map_path, bounds_path, 
                                sub_paths['soil_map'])
            wbt.clip(struct_map_path, bounds_path, sub_paths['struct_map'])
            
            subFdCalc.execute(sub_paths['dem'], 
                              sub_paths['soil_map'],
                              sub_paths['struct_map'],
                              analysis_subsets,
                              flow_buffer
                              )
        
        
        for name in subset_names: #merge remaining rasters
            src_to_merge = [rio.open(os.path.join(self.wkdir, 
                                                  wshed_id, 'results',
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
                               analysis_subsets = {'all_struct': None},
                               area_weights = None,
                               structure_weights = None
                               
                               ):
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
            {'all_struct': None # No filter given
            'schools': {'SITETYPE_M', ['SCHOOL K / 12', 'EDUCATIONAL']
                        }
                           
                           }
            
        area_weights: string, optional
        a name of a column in the soil_map gdf that indicates a relative weight for
        flood risk in each polygon.
        
        For instance, you could give an annual exceedence probability, to give
        higher weight to areas that flood more frequently. 
        
        
        structure_weights: string, optional
        A name of a column in the struct_map gdf that gives weights for valuing different structures.
        
        For instance, if the gdf has a column for the assessed value of buildings, 
        you could pass the name of this column.
        
        Alternately, you could create a column with other value weights to different types of structures.
        For instance, valuing municipal buildings > residential buildings > commericial buildings.

        Returns
        -------
        geodataframe of flood zones that have at least one floodable structure.

        '''
        self.subset_names = list(analysis_subsets.keys())
        
        soil_map = load_if_path(soil_map)
        struct_map = load_if_path(struct_map)
        
            
            
        struct_map.to_crs(soil_map.crs, inplace = True)
        crs = soil_map.crs
        fld_zone = soil_map[soil_map['FLOOD'].isin(['water', 
                                               'rare',
                                               'occasional', 
                                               'frequent'])]
        del soil_map
        
        for name, filt in analysis_subsets.items():
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
                
            if structure_weights:
                fld_zone = sum_pts_weights(fld_zone, sub_structs, 
                                           name, 
                                           structure_weights)
            else:
                fld_zone = count_points_in(fld_zone, sub_structs, 
                                       colname = name)
        
        #keep only flood areas with at least one relevant structure
        
        
        to_keep = fld_zone[list(analysis_subsets.keys())].sum(axis =1)>0
        
        fld_zone = fld_zone.loc[to_keep]
        
        if area_weights:
            fld_zone[list(analysis_subsets.keys())] *= fld_zone[area_weights]
        
        fld_zone.to_file(self.fns['fld_area_polys'])
        return fld_zone
        
    
    def find_pour_points(self, fld_zone = gpd.GeoDataFrame(), flow_buffer = 50):
        if fld_zone.empty:
            fld_zone = gpd.read_file(self.fns['fld_area_polys'])
            
        fld_zone.geometry, crs = max_rast_value_in_shp(
            fld_zone.geometry.buffer(flow_buffer), 
         self.fns['d8_flow_acc'])
        
        fld_zone.crs = crs 
        fld_zone.to_file(self.fns['fld_area_pour_points'])
        return fld_zone
    
        
    def delineate_watersheds(self, fld_zone = gpd.GeoDataFrame(), flow_buffer = 50):
        for file in os.listdir(self.unnested_dir):
            os.remove(os.path.join(self.unnested_dir, file))
        
        if fld_zone.empty:
            fld_zone = gpd.read_file(self.fns['fld_area_polys'])
            
        wbt = WhiteboxTools()
        fld_zone.geometry, crs = max_rast_value_in_shp(
            fld_zone.geometry.buffer(flow_buffer), 
         self.fns['d8_flow_acc'])
        
        fld_zone.crs = crs 
        
        fld_zone.to_file(self.fns['fld_area_pour_points'])
        wbt.unnest_basins(self.fns['d8_pointer'],  
                          self.fns['fld_area_pour_points'],
                          os.path.join(self.unnested_dir, 
                                       'unnested_wsheds.tif'))
        os.chdir(self.wkdir)
        
    
    def sum_data(self, fld_zone = gpd.GeoDataFrame(), subset_names = None,
                 crs = None):
        
        
        
        if self.crs:
            pass
        elif crs:
            self.crs = crs
        else:
            try:
                self.crs = rio.open(self.fns['d8_flow_acc']).crs
            except:
                print(
  '''Warning: Unable to determine the proper coordinate reference system for rasters.
  Results Rasters will have no CRS. To avoid this, pass the correct crs to this method.
  
  ''')
            
        if fld_zone.empty:
            fld_zone = gpd.read_file(self.fns['fld_area_pour_points'])
            
        if not subset_names:
            subset_names = self.subset_names
            
        if not self.subset_names:
            subset_names = ['all_struct']
        
        fld_zone['wshed_area'] = 0
        for file in os.listdir(self.unnested_dir):
            data = rio.open(os.path.join(self.unnested_dir, file)).read(1)
            index, counts = np.unique(data, return_counts = True)
            drop_ind = np.where(index == -32768)[0]
            index = np.delete(index, drop_ind); 
            counts = np.delete(counts, drop_ind) 
            index = index -1
            fld_zone.loc[
            list(index), 'wshed_area'] += counts
        
        
        with rio.open(os.path.join(self.unnested_dir, file)) as src:
            out_meta = src.meta.copy()
        out_meta.update({'dtype': 'float64'}, crs = crs)
        
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
            

if __name__ == '__main__':
    wkdir = os.path.join(os.getcwd(), 'tests')
    data_dir = os.path.join(wkdir, 'test_data')
    Calc = floodDemandCalculator(wkdir)
    dem_path = os.path.join(data_dir, 'dem.tif')
    soil_map_path = os.path.join(data_dir, 'soil_map.shp')
    struct_map_path = os.path.join(data_dir, 'structure_map.shp')
    
    #Calc.execute(dem_path, soil_map_path, struct_map_path)
    
    piecewise_wkdir = os.path.join(wkdir, 'piecewise')
    watersheds_map = os.path.join(data_dir, 'watershed_boundaries.shp')
    
    CalcP = floodDemandCalculator(piecewise_wkdir)
    
    CalcP.execute_piecewise(watersheds_map, 'mjr_wshed', dem_path, 
                            soil_map_path, struct_map_path)
    
    
        