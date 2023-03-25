# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 10:42:55 2023

@author: Benjamin Dube
"""

from setuptools import setup







with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()




setup(
    name='flood_protection_demand',
    version= '1.0.1',

    url='https://github.com/bubalis/flood_protection_demand',
    author='Ben Dube',
    description = 'A tool for calculating the relative value of different locations in the landscape for protecting downstream areas from flooding.',
    author_email='benjamintdube@gmail.com',
    long_description_content_type="text/markdown",
    long_description = long_description,
    entry_points = {'console_scripts': 
    [
    'flood_demand_calc = flood_protection_demand.fd_calc:main',
                     ] },

    


   


    install_requires = ['affine==2.3.0',
     'attrs==21.4.0',
     'certifi==2021.10.8',
     'click==7.1.2',
     'click-plugins==1.1.1',
     'cligj==0.7.2',
     'Fiona==1.8.20',
     'geopandas==0.9.0',
     'munch==2.5.0',
     'numpy==1.20.3',
     'pandas==1.4.0',
     'pyparsing==3.0.7',
     'pyproj==3.3.0',
     'python-dateutil==2.8.2',
     'pytz==2021.3',
     'rasterio==1.2.1',
     'rasterstats==0.16.0',
     'Shapely==1.7.1',
     'simplejson==3.17.6',
     'six==1.16.0',
     'snuggs==1.4.7',
     'whitebox==2.0.3'],
    
    include_package_data=True,
    

)
