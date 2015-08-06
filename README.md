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

## Modules 

### `DART.py`  

This module contains what you need to read in DART outout.  
This module has the following subroutines:  

+ `load_covariance_file`  loads netcdf files of covariance and correlation between the model state and a given observation  
+ `load_DART_obs_epoch_series_as_dataframe` runs through DART `obs_epoch` files corresponding to a given date range, and turns them into a Pandas dataframe  
+ `load_DART_obs_epoch_file_as_dataframe` read in a DART `obs_epoch` files and retuns a dataframe 
+ `load_DART_obs_epoch_file` reads in a DART `obs_epoch` files and retuns a dataframe 
+ `load_DART_diagnostic_file` read in a DART `Posterior_Diag` or `Prior_Diag` file and return the desired variable field. 
+ `get_ensemble_size` given a DART output diagnostic netcdf file that is already open, find the number of ensemble members in the output
+ `get_obs_type_number` having opened a DART output diagnostic netcdf file, find the obs_type number that corresponds to a given obs_typestring
+ `get_copy` having opened a DART output diagnostic netcdf file, find the copy number that corresponds to a given copystring
+ `basic_experiment_dict` loads a default Python dictionary containing the details of an experiment that we look at -- 
+ `date_to_gday` convert a datetime date to gregorian day count the way it is counted in DART  (i.e. number of days since 1601-01-01
+ `daterange` generate a range of dates (in python datetime format), given some start date, a time delta, and the numper of periods
+ `rank_hist` given a 1-D ensemble time series and a verification (usually the truth), compute the rank histogram over the desired block of time
+ `kurtosis`  given a 1D ensemble of numbers (obs space, state space, whatever) return the kurtosis of the PDF represented by the ensemble
+ `skewness`  given a 1D ensemble of numbers (obs space, state space, whatever) return the skewness of the PDF represented by the ensemble
+ `point_check_dictionaries` pre-defined experiment dictionaries that give various averaging regions 
+ `climate_index_dictionaries` returns experiment dictionaries with the lat, long, and levranges needed to compute certain climate indices.  
	

## Dependencies  

+ netCDF4 python library  
+ datetime 
+ pandas 
+ [dayconv.py](http://www.astrobetter.com/wiki/Python+Switchers+Guide) tool for converting Gregorian to Julian day  

## Examples  
For examples of how to use this code, see [this iPython notebook](https://github.com/LisaNeef/DART-state-space/blob/master/DART_state_space.py)
