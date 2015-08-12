## Python module for various diagnostics related to the stratospheric polar vortex  

# load necessary modules  

import DART as dart
import MJO as mjo
import numpy as np
import matplotlib.pyplot as plt
import brewer2mpl
import plot_tools 
#--------------------------------------------------------------------

def plot_climate_indices(E,index_name,copies_to_plot,climatology_option = 'NODA',hostname='taurus',verbose=False):

	"""
	This subroutine computes a bunch of simple climate indices 
	for a given experiment, and plots them for all copies given by 'copies_to_plot'

	INPUTS:
	copies_to_plot: list containing keywords for what copies to plot. Here are the options:  
	+ any valid copystring in DART output data  (e.g. "ensemble member 1")
	+ 'ensemble' = plot the entire ensemble  
	+ 'ensemble mean' = plot the ensemble mean  
	+ 'operational' = plot the operational value of this index 
	"""

	# create a list of copies to load
	copy_list = []

	if "copystring" in copies_to_plot:
		copy_list.append(E['copystring'])

	if ("ensemble" in copies_to_plot): 
		N = dart.get_ensemble_size_per_run(E['exp_name'])
		for iens in np.arange(1,N+1):
			if iens < 10:
				spacing = '      '
			else:
				spacing = '     '
			copy_list.append("ensemble member"+spacing+str(iens))		
	if ("ensemble mean" in copies_to_plot): 
			copy_list.append("ensemble mean")

	# retrieve desired index for all the copies in the list
	L = []
	for copy in copy_list:
		E['copystring'] = copy
		x = compute_climate_indices(E,index_name,climatology_option,hostname,verbose)
		L.append(x)

	# plot it  
	for copy,climate_index in zip(copy_list,L):

		# define a color for the ensemble mean - depending on experiment   
		lcolor = "#000000"
		if E['exp_name'] == 'W0910_NODA':
			bmap = brewer2mpl.get_map('Dark2', 'qualitative', 7)
			lcolor = bmap.hex_colors[1]
		if E['exp_name'] == 'W0910_GLOBAL':
			# make this one black, since it's sort of the reference  
			lcolor = "#000000"
		if E['exp_name'] == 'W0910_TROPICS':
			bmap = brewer2mpl.get_map('Dark2', 'qualitative', 7)
			lcolor = bmap.hex_colors[0]

		# make the ensemble a lighter version of their original color  
		ensemble_color = plot_tools.colorscale(lcolor, 1.6)

		# here is the plot  
		if (copy == 'ensemble mean'):
			plt.plot(E['daterange'],climate_index,color=lcolor,linewidth=2)
		if "ensemble member" in copy:
			plt.plot(E['daterange'],climate_index,color=ensemble_color,linewidth=1)



def compute_climate_indices(E,index_name,climatology_option = 'NODA',hostname='taurus',verbose=False):  

	"""
	This subroutine computes various simple climate indices for a dataset 
	defined by an experiment dictionary.  

	Currently supporting the following indices:  
	+ 'Aleutian Low' index of Garfinkel et al. (2010)
	+ 'East European High' index of Garfinkel et al. (2010)
	+ 'AO Proxy' -- Polar Cap GPH Anomaly at 500hPa -- it's a  proxy for the AO suggested by Cohen et al. (2002)   
		* note however that we define the polar cap as everything north of 70N, I think Cohen et al do 60N
	+ 'Vortex Strength' -- Polar Cap GPH Anomaly averaged 3-30hPa -- it's a measure of vortex strength suggested by Garfinkel et al. 2012

	"""

	# modify the experiment dictionary to retrieve the right index 
	EI = dart.climate_index_dictionaries(index_name)
	E['levrange'] = EI['levrange']
	E['latrange'] = EI['latrange']
	E['lonrange'] = EI['lonrange']
	E['variable'] = EI['variable']

	# for all indices defined so far, compute the anomaly
	# with respect to climatology  
	# this uses an anomaly subroutine from the MJO module  
	A,C,lat,lon,lev = mjo.ano(E,climatology_option = climatology_option,verbose=verbose)

	# Aleutian Low and East European high indices are single points, so just return the anomaly
	if (index_name == 'Aleutian Low') or (index_name == 'East European High'):
		index_out = A

	# for the Polar Cap GPH -based indices, average over latitude and longitude  
	# here can use a subroutine written for MJO stuff in the MJO module  
	if (index_name == 'AO Proxy') or (index_name == 'Vortex Strength'):
		lat1,lon1,Aave = mjo.aave(E,A,lat,lon,season=None,variable_name=None,averaging_dimension='all')

	# for the AO proxy, reverse the sign so that it's more intuitive -- a positive GPH anomaly is related to a negative AO	
	if (index_name == 'AO Proxy') or (index_name == 'Vortex Strength'):
		index_out = -Aave

	# for vortex strength, average between 3 and 30 hPa  
	if (index_name == 'Vortex Strength'):
		index_out = np.nanmean(Aave,axis=0)

	# return index over desired daterange
	return index_out
