# DART-state-space

## Introduction 

This is a set of python codes for evaluating output from the Data Assimilation Research Testbed ([DART](http://www.image.ucar.edu/DAReS/DART/)), using popular Python libraries like matplotlib and basemap.   

I developed these for my work using DART with the Community Atmosphere Model (CAM) and the Whole Atmosphere Community Climate Model (WACCM), so right now the codes are mostly written to accomodate those. 

## Experiment Dictionaries  

The foundation of these codes is that any experiment you do with DART plus some model is described in a Python dictionary. 
The way to create an experiment dictionary is with the function `basic_experiment_dict` in the `DART.py` module:  

	import DART as dart  
	E = dart.basic_experiment_dict()

Currently the standard entries for an experiment dictionary are: 

+ `exp_name` - some string that uniquely defines that experiment, e.g. a short version of the CESM run name.  
+ `diagn` - the DART diagnostic that you want to plot: "Prior", "Posterior", or "Truth"  
+ `copystring` - a string corresponding to one of the available copies in DART output, e.g. "ensemble mean"
+ `variable` - the model variable that you want to be plot  
+ `levrange` - the range of model levels that you want to plot or average over, in the units of the DART output files
+ `lonrange` - the range of longitudes that you want to plot or average over, in the units of the DART output files
+ `latrange` - the range of latitudes that you want to plot or average over, in the units of the DART output files
+ `extras` - other interesting quantities to compute, e.g. "MSE" (mean squared error)  
+ `obs_name` - an observation you might be interested in, e.g. when computing correlations to observations  
+ `run_category` - an extra category that you can give different runs, for creating different file paths  
+ `daterange` - the range of dates over which to load data, in Python `datetime` format  
+ `clim` - the limits of the values to be plotted (depending on the plot, this could be the y-limits, x-limits, or color limits)  

## Examples  

## Modules 
