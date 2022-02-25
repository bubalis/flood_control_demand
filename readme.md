## Introduction
\
This module provides the tools to perform a simple "flood control ecosystem-service demand" calculation for a region. 

The resulting raster map gives the relative contribution of different parts of a landscape to mitigating flooding from storm run-off. 

The resulting raster is dimensionless, and the value for each pixel represents:

<img src="https://render.githubusercontent.com/render/math?math=\[ \sum_{a=1}^{n} B_a / W_a \]">

Where a is each flood-prone area downstream of the pixel, B is the value or number of buildings in the pixel (or some weight thereof) and W is the area of the upstream watershed of that pixel.  




## Data Needs  

To execute this method you will need:

A digital elevation model of your area of interest.

A shapefile containing the point locations of structures in your area. This may be obtained from enhanced 911 (E911) databases.

A shapefile of your area containing soils data from the NRCS Soil Survey Geographic Database (SSURGO).  




## Simple Example

The simplest syntax to use is:  
```
from FD_Calc import FloodDemandCalculator

FD = FloodDemandCalculator(wkdir)

FD.execute(dem_path, fld_map_path, structure_map_path)  
```

\
All paths passed as arguments to the function must be absolute paths.


## More complex useage:
\
### Running Piecewise:
\
There are several additional options for using this tool.

If you run into memory problems on your machine, you can try executing piecewise.
This lets you run your calculations one sub-watershed at a time. 
```
from FD_Calc import FloodDemandCalculator
import geopandas as gpd 

FD = FloodDemandCalculator(wkdir)

FD.execute_piecewise(watersheds_map_path, wshed_id_col, dem_path, 
                            fld_map_path, struct_map_path)
```
Make sure that your sub-watersheds aren't connected to one another in any way that is important for flood control. 
If one delineated sub-watershed is below another, the method won't return accurate results.

### Weighting Structures and Flood-Prone Areas.
\
If you have data in your shapefiles that describes relative risks that different areas will flood, or relative values of structures, you can pass that data to the arguments "area_weight_col" and "struct_weight_col" respectively.


### Non-SSURGO Flood Maps
\
By Default, this package delinates flood-prone areas using a SSURGO soil map. 
If you have a different map delineating areas at risk of flooding, you can use this, but you must pass "soil_map_is_SSURGO = FALSE" to the execute method.


## Installation
\
To install, download this repository and run  

```pip install -r requirements.txt ```

## Credits
\
This method was developed by Keri Bryan Watson and is described in Watson, et al (2019), but has not been made available as a single module.  

Most geoprocessing operations are carrie out using the WhiteboxTools open-source geoprocessing libary (Lindsay, 2014). https://github.com/jblindsay/whitebox-tools








## References:
Lindsay, J. B. (2014, April). The whitebox geospatial analysis tools project and open-access GIS. In Proceedings of the GIS Research UK 22nd Annual Conference, The University of Glasgow (pp. 16-18).  
\
Watson, K. B., Galford, G. L., Sonter, L. J., Koh, I., & Ricketts, T. H. (2019). Effects of human demand on conservation planning for biodiversity and ecosystem services. Conservation Biology, 33(4), 942–952.  








Copyright ⓒ 2022 Benjamin Dube.
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


