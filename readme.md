This module provides the tools to perform a simple "flood control ecosystem-service demand" calculation for a region. 

The resulting raster map gives the relative contribution of different parts of a landscape to mitigating flooding from storm run-off. 

The resulting raster is dimensionless, and the value for each pixel represents:

<img src="https://render.githubusercontent.com/render/math?math=\[ \sum_{a=1}^{n} B_a / W_a \]">

Where a is each flood-prone area downstream of the pixel, B is the number of buildings in the pixel (or some weight thereof) and W is the area of the upstream watershed of that pixel.


This method was developed by Keri Bryan Watson and is described in Watson, et al (2019), but has not been made available as a single module.


To execute this method you will need:

A digital elevation model of your area of interest.

A shapefile containing the point locations of structures in your area. This may be obtained from enhanced 911 (E911) databases.

A shapefile of your area containing soils data from the NRCS Soil Survey Geographic Database (SSURGO).










ⓒ Benjamin Dube.
This work is licensed under a Creative Commons Attribution 3.0 United States License.






References:
Watson, K. B., Galford, G. L., Sonter, L. J., Koh, I., & Ricketts, T. H. (2019). Effects of human demand on conservation planning for biodiversity and ecosystem services. Conservation Biology, 33(4), 942–952.
