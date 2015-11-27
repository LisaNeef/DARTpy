# Python module for DART diagnostic plots in observation space.
#
# Lisa Neef, started 12 Nov 2015

# here are all the relevant packages to load 
import numpy as np
from mpl_toolkits.basemap import Basemap
import DART as dart
import brewer2mpl
import pandas as pd

#import matplotlib as mpl
#import matplotlib.pyplot as plt
#import matplotlib.dates as mdates
#import datetime
#from netCDF4 import Dataset
#import ERP as erp
#import WACCM as waccm
#import re
#import ERA as era
#import TEM as tem
#import experiment_settings as es


def plot_DARTobs_scatter_globe(E,projection='miller',coastline_width=0,water_color="#CCF3FF",land_color="#996600",obs_colors=None,hostname='taurus',debug=False):

	"""
	This code plots a scatterplot of the horizontal distribution of DART assimilated or evaluated
	observations on a map.
	
	INPUTS:
	E: a DART experiment dictionary. The relevant keys are: 
		'latrange' : gives the lat limits of the plot
		'lonrange' : gives the lon limits of the plot
		'levrange' : only observations between these levels are shown 
		'copystring': a string giving the DART copies to show. If it's a list, we loop over the list
		'obs_name': a string giving the DART observation to show. If it's a list, we loop over the list
		'daterange': range of dates over which to plot the observations 
	projection: the map projection to use (default is "miller")
	coastline_width: width of the coastlines; default is 0 (no coastlines)
	water_color: color given to oceans and lakes (default is cyan/blue)
	land_color: color given to continents (default is an earthy brown)
	obs_color: list of colors assigned to the different types of obs plotted. Default is colorbrewer Paired 
	"""

	#---------set up the map-----------------
	# if plotting a polar stereographic projection, it's better to return all lats and lons, and then 
	# cut off the unwanted regions with map limits -- otherwise we get artifical circles on a square map
	if (projection == 'npstere'): 
		if E['latrange'][0] < 0:
			boundinglat = 0
		else:
			boundinglat =  E['latrange'][0]
		E['latrange'] = [-90,90]
		E['lonrange'] = [0,361]
	if (projection == 'spstere'):
		boundinglat = E['latrange'][1]
		E['latrange'] = [-90,90]
		E['lonrange'] = [0,361]

 	# set up a map projection
	if projection == 'miller':
		maxlat = np.min([E['latrange'][1],90.0])
		minlat = np.max([E['latrange'][0],-90.0])
		map = Basemap(projection='mill',llcrnrlat=minlat,urcrnrlat=maxlat,\
			    llcrnrlon=E['lonrange'][0],urcrnrlon=E['lonrange'][1],resolution='l')
	if 'stere' in projection:
		map = Basemap(projection=projection,boundinglat=boundinglat,lon_0=0,resolution='l')
	if projection == None:
		map = Basemap(projection='ortho',lat_0=54,lon_0=10,resolution='l')

        # draw coastlines, country boundaries, fill continents.
        map.drawcoastlines(linewidth=coastline_width)
	map.drawmapboundary(fill_color=water_color)
		
        # draw lat/lon grid lines every 30 degrees.
        map.drawmeridians(np.arange(0,360,30),linewidth=0.25)
        map.drawparallels(np.arange(-90,90,30),linewidth=0.25)
	map.fillcontinents(color=land_color,lake_color=water_color,alpha=0.2)

	#--------- load the obs on the given day 
	OBS,copy_names,obs_names,lons,lats,levs = dart.load_DART_obs_epoch_file(E,debug=debug,hostname=hostname)

	#---------loop over obs types-----------------
	# loop over obs types given in E
	if type(E['obs_name']) is not list:
		obs_type_list = [E['obs_name']]
	else:
		obs_type_list = E['obs_name']

	# define a list of colors if needed 
	if obs_colors is None:
		bmap = brewer2mpl.get_map('Set1', 'qualitative', 7)
		obs_colors = bmap.mpl_colors

	for obs_type,ii in zip(obs_type_list,range(len(obs_type_list))):
		lons_obstype = [lons[i] for i,x in enumerate(obs_names) if obs_type in x]
		lats_obstype = [lats[i] for i,x in enumerate(obs_names) if obs_type in x]

		# scatter the obs over the map 
		x, y = map(lons_obstype,lats_obstype)
		map.scatter(x,y,3,marker='o',color=obs_colors[ii])

	return 

def count_Nobs_in_time(E,output_interval=0,DART_qc_flags_list=[0]):

	"""
	For a given experiment and list of observations, cycle through a daterange 
	and count the number of each observation type made at each time, 
	then return a Pandas dataframe that gives observations as a function of 
	observation type and name. 

	INPUTS:
	E: A standard DART experiment dictionary. The following keys are important:
		daterange: the range of dates over which to count the observations
		obs_name: a list of the observations to retrieve 
	output_interval: the number of steps (in the daterange) where we print output. 
		This can be helpful for checking progress, because this process can take a while. 
		The default is 0: no output printed to the screen at all.  
	DART_qc_flags_list: a list of DART quality control flags indicating which obs to retrieve. 
		The default is [0], i.e. only retrieving the assimilated observations. 
		Here is what the DART QC flags mean:  
			0= Assimilated  
			1= Evaluated only  
			2= Assimilated but posterior forward observation operator(s) failed  
			3= Evaluated only but posterior forward observation operator(s) failed  
			4= Not used, prior forward observation operator(s) failed  
			5= Not used because not selected in obs_kind_nml  
			6= Not used, failed prior quality control check  
			7= Not used, violated outlier threshold  
	"""

	# copy the input dict into a temporary one, and make sure that the right diagnostic (prior)
	# and copy (observation) are retrieved. 
	P = E.copy()
	P['diagn']='prior'
	P['copystring']='observation'

	# save the daterange and create an empy dictionary with the dates as the keys 
	DR = E['daterange']
	Sdict = dict.fromkeys(DR)

	#  loop over dates
	for D,ii in zip(DR,range(len(DR))):
		if output_interval != 0:
			if np.mod(ii,output_interval) == 0:
				print D
		P['daterange'] = [D]

		# for each date, load all the obs types
		OBS,copy_names,obs_names,lons,lats,levs,QCdict = dart.load_DART_obs_epoch_file(P,debug=False)

		# initialize an empty dictionary comprised of a list for each obs type
		obscount = {k:[] for k in P['obs_name']}

		# select only the obs where the DART quality control is 0
		QCD = QCdict['DART quality control            ']
		assimilated = [obs_names[i] for i, j in enumerate(QCD) if j ==0]

		# for each obs type, count how many times it occurs, and store in the dictionary
		for ObsName in P['obs_name']:
			nobs = assimilated.count(ObsName)
			obscount[ObsName] = nobs

		# now turn the dictionary into a Series
		S = pd.Series(obscount)

		# store the series for this date in a dictionary 
		Sdict[D] = S


	# turn the dictionary into a pandas dataframe 
	DF = pd.DataFrame(Sdict)

	return DF
