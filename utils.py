# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 17:28:31 2022

@author: benja
"""

import os

import rasterio as rio
import geopandas as gpd
import numpy as np

from shapely.geometry import Point, Polygon, MultiPolygon

from rasterstats import zonal_stats
from contextlib import contextmanager

@contextmanager
def cwd(dir_):
    '''Context manager to change the directory to dir_ then change in back.'''
    
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
    '''
    Get a list of points representing the location of the maximum values 
    of the underlying raster, in each polygon in the shapefile.
    
    Returns: 
    
    The crs of the geometries.

    Parameters
    ----------
    geoms : a collection of shapes
    raster_path : path to the raster to find max values in
        DESCRIPTION.
    stat : Statistic to use, optional.
        The default is 'max'.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    out_geoms : TYPE
       a list of points, each representing the (or a) local maximum for 
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
    poly_gdf : the polygon geodataframe with a column added for the number of points.
    
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




def sum_pts_weights(poly_gdf, pts_gdf, colname = 'all_struct', 
                    pts_weight_col = 'value'):
    '''
    Sum the weighted value of all points in an area. 
    In this case, this might be the $ Value of the structures vulnerable to flooding.

    Parameters
    ----------
    Parameters
    ----------
    poly_gdf : geodataframe with polygon geometry.
        
    pts_gdf : geodataframe with points geometry
        
    colname : str 
    name of the column to add to poly_gdf
         The default is 'count'.

    pts_weight_col : str, optional
        name of the column in the points gdf containing weights. The default is 'value'.

    Returns
    -------
    poly_gdf : the polygon geodataframe with a column added for the 
    weighted number of points.

    '''
    
    
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
    the numpy array with the values replaced.

    '''
    if not key_array:
        key_array = np.arange(0, val_array.shape[0])
    mapping_ar = np.zeros(key_array.max()+1,dtype=val_array.dtype)
    mapping_ar[key_array] = val_array
    return mapping_ar[input_array]
    

def count_points_in2(poly_gdf, pts_gdf, colname = 'all_struct'):
    '''
    This version is not working with the current version of geopandas
    
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

def sum_pts_weights2(poly_gdf, pts_gdf, colname = 'all_struct', 
                    pts_weight_col = 'value'):
    '''This weighting function is not working with my current
    version of geopandas.'''
    
    
    vals = gpd.sjoin(poly_gdf,  pts_gdf[['geometry', pts_weight_col]],  how = 'left',
              ).groupby(level = 0)[pts_weight_col].sum()
    
    poly_gdf[colname] = vals
    
    return poly_gdf
