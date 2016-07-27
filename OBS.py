# THis module contaiins codes for retrieving and visualizing various observation types. 

# load the required packages  
import numpy as np
import pandas as pd  
import datetime
#import os.path
#from netCDF4 import Dataset

def HRRS_as_DF(OBS,hostname='taurus'):

	"""
	Loop over a set of dates and a specified latitude- and longitude range, and return 
	the available high-resolution radiosonde data as a pandas data frame  
	
	Input OBS has to be a dictionary with the following entries:  
		daterange: a list of datetime objects that give the desired date range  
		latrange: a list giving the bounding latitudes of the desired range 
		lonrange: a list giving the bounding longitudes of the desired range 
	Note that OBS can be a DART experiment dictionary (see DART.py), but the DART/model 
		specific entries are ignored. 
	"""

	# because the HRRS data are sorted by years, loop over the years in the daterange
	y0 = DR[0].year
	yf = DR[len(DR)-1].year
	years = range(y0,yf+1,1)
	for YY in years:  

