# Python module for MJO diagnostics 
# Lisa Neef, 27 Jan 2015

# to find the Pyclimate package that works for this version of Python, need to append this odd path to the system path.\n",
import sys
sys.path.append("/home/lneef/anaconda/pkg/lib/python/") 

# load the required packages  
import numpy as np
import datetime
import time as time
import os.path
import pandas as pd
import DART as dart
import experiment_settings as es
#from calendar import monthrange
#from netCDF4 import Dataset
import WACCM as waccm
import DART_state_space as DSS
import pyclimate.LanczosFilter as LF
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from scipy.stats import nanmean


def plot_RMM(E,copies_to_plot,climatology_option='NODA',plot_type='polar',hostname='taurus',verbose=False):

	"""
	given a certain experiment dictionary, compute the Wheeler and Hendon (2004)
	RMM index (by projecting the modeled fields onto Wheeler and Hendon's multivariate EOFs),
	with or without the "true" (i.e. operational) value.  

	INPUTS:
	copies_to_plot: list containing keywords for what copies to plot. Here are the options:  
		+ any valid copystring in DART output data  (e.g. "ensemble member 1")
		+ 'ensemble' = plot the entire ensemble  
		+ 'ensemble mean' = plot the ensemble mean  
		+ 'operational' = plot the operational value of this index 
	climatology_option: options for how to compute the climatology needed for anomaly computation: 
		+ 'NODA'  - use the analogue for the desired experiment that has no assimilation  
		+ 'F_W4_L66' - WACCM run with F_W4_L66 compset  
	plot_type: choose 'polar' to draw the standard RMM circle diagram, or 'linear' to plot 
		the two components linearly 
	"""

	# given the chosen plot variation, define a list of copies to load
	copy_list = []

	if "copystring" in copies_to_plot:
		copy_list.append(E['copystring'])

	if ("ensemble" in copies_to_plot): 
		N = es.get_ensemble_size_per_run(E['exp_name'])
		for iens in np.arange(1,N+1):
			if iens < 10:
				spacing = '      '
			else:
				spacing = '     '
			copy_list.append("ensemble member"+spacing+str(iens))		
	if ("ensemble mean" in copies_to_plot): 
			copy_list.append("ensemble mean")
	if ("operational" in copies_to_plot): 
			copy_list.append("operational")

	# compute the PCs for the desired timespan and list of copies 
	# or load the observatinal value 
	RMM1list = []
	RMM2list = []
	bad_copies = []		# start a list of the copies that are unavailable  
	for copy in copy_list:

		print(copy)
		if copy == "operational":
			date_limits = (E['daterange'][0],E['daterange'][len(E['daterange'])-1])
			dates,RMM1,RMM2 = read_RMM_true(date_limits,hostname='taurus')
			RMM1list.append(RMM1)
			RMM2list.append(RMM2)
		
		else:
			E['copystring'] = copy
			pc = RMM(E,climatology_option=climatology_option,hostname='taurus',verbose=verbose)
			if pc is None:
				# if we don't have enough data to compute the RMM index for this experiment, 
				# add it to the list of bad copies:
				print('     Unable to compute RMM index for '+copy)
				bad_copies.append(copy)
			else:
				RMM1list.append(pc[0,:])
				RMM2list.append(pc[1,:])

	# remove the "bad" copies from the list
	[copy_list.remove(bc) for bc in bad_copies]

	if plot_type == 'polar':
		# pimp out the plot a little bit  
		plt.plot([-4,4],[-4,4],linewidth=0.2,linestyle='--',color='k')
		plt.plot([-4,4],[4,-4],linewidth=0.2,linestyle='--',color='k')
		plt.plot([-4,4],[0,0],linewidth=0.2,linestyle='--',color='k')
		plt.plot([0,0],[-4,4],linewidth=0.2,linestyle='--',color='k')
		plt.xlim([-4.0,4.0])
		plt.ylim([-4.0,4.0])

		# circle in the center of the plot to denote weak index  
		circle = plt.Circle((0, 0), radius=1.0, fc='k', ec='k', alpha=0.2)
		plt.gca().add_patch(circle)

	# cycle over copies and plot the two princial components against each other  
	for copy,RMM1,RMM2 in zip(copy_list,RMM1list,RMM2list):
		
		# choose the color based on the copy string
		if "ensemble member" in copy:
			lcolor = "#848484"
			alph=0.8
			time = E['daterange']
		if copy == "ensemble mean":
			lcolor = "#70B8FF"
			alph=1.0
			time = E['daterange']
		if copy == "operational":
			lcolor = "#000000"
			alph=1.0
			time=dates

		# plot desired copy
		if plot_type=='polar':
			plt.plot(RMM1,RMM2,'-',color=lcolor,alpha=alph)
		if plot_type=='RMM1':
			plt.plot(time,RMM1,'-',color=lcolor,alpha=alph)
		if plot_type=='RMM2':
			plt.plot(time,RMM2,'-',color=lcolor,alpha=alph)
		if plot_type=='linear':
			plt.subplot(2,1,1)
			plt.plot(time,RMM1,'-',color=lcolor,alpha=alph)
			plt.subplot(2,1,2)
			plt.plot(time,RMM2,'-',color=lcolor,alpha=alph)

	# labels and stuff  
	plt.xlabel('RMM1')
	plt.ylabel('RMM2')

	return copy,RMM1,RMM2

def plot_correlations_lag_lat_or_lon(E,climatology_option='NODA',maxlag=25,lag_versus_what='lon',nilter_order=50,cbar=True,hostname="taurus",debug=False):

	"""
	 given a certain experiment or dataset over a certain daterange, 
	 plot the correlation between wind or precip anomalies in one reference
	 region, relative to everywhere else, either 
	 as a function of latitude and longite, and Lag.  
	 this should produce figures like Figs. 5-6 of Waliser et al. 

	INPUTS:  
	E - a standard DART experiment dictionary, with the variable field and level range corresponding to some MJO variable  
	maxlag: the limit of the lag (in days) that we look at 
	lag_versus_what: choose 'lat' or 'lon'  
	cbar: set to True to have a colorbar  
	hostname: computer name - default is Taurus  
	climatology_option: choose which climatology to take the anomalies to respect with -- default is "NODA"  
	"""

	# load the correlation field 
	R,S,L,x = correlations_lag_lat_or_lon(E,maxlag,lag_versus_what,filter_order,climatology_option,hostname=hostname,verbose=debug)

        # choose color map based on the variable in question
	E['extras'] = 'Correlation'
	colors,cmap,cmap_type = DSS.state_space_HCL_colormap(E,reverse=True)
	
	# choose axis labels  
	plt.ylabel('Lag (days)')
	if lag_versus_what=='lat':
		plt.xlabel('Latitude')
	if lag_versus_what=='lon':
		plt.xlabel('Longitude')

	# set the contour levels - it depends on the color limits and the number of colors we have  
	clim = 1.0
	if cmap_type == 'divergent':
		clevels  = np.linspace(start=-clim,stop=clim,num=11)
	else:
		clevels  = np.linspace(start=0,stop=clim,num=11)

        # contour plot of the chosen variable
	cs = plt.contourf(x,L,R,levels=clevels,cmap=cmap)
	plt.clim([-1.0,1.0])

	if (cbar is not None):
		CB = plt.colorbar(cs, shrink=0.6, extend='both', orientation=cbar)

	return x,L,R,S

def plot_variance_maps(E,cbar=True,hostname="taurus"):

	# given a certain experiment or dataset over a certain daterange, 
	# plot the MJO-related variance on a map

	# load the variance map  
	VV,lat,lon = variance_maps(E,hostname=hostname)  

 	# set up the  map projection
	map = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,\
		    #llcrnrlon=-180,urcrnrlon=180,resolution='c')
		    llcrnrlon=0,urcrnrlon=360,resolution='c')

        # draw coastlines, country boundaries, fill continents.
	map.drawcoastlines(linewidth=0.25)
	map.drawcountries(linewidth=0.25)

        # draw lat/lon grid lines every 30 degrees.
	map.drawmeridians(np.arange(0,360,30),linewidth=0.25)
	map.drawparallels(np.arange(-90,90,30),linewidth=0.25)

        # compute native map projection coordinates of lat/lon grid.
	X,Y = np.meshgrid(lon,lat)
	x, y = map(X, Y)

        # choose color map based on the variable in question
	E['extras'] = 'MJO variance'
	colors,cmap,cmap_type = DSS.state_space_HCL_colormap(E)

	## if no color limits are specified, at least make them even on each side
        #if clim is None:
        #        clim = np.nanmax(np.absolute(VV))
	#print('------clim----------')
	#print(clim)


        # contour data over the map.
	cs = map.contourf(x,y,VV,15,cmap=cmap)
	#cs = map.contourf(x,y,M,len(colors)-1,colors=colors)
	#cs = map.contourf(X,Y,M,len(colors)-1,cmap=cmap,extend="both",vmin=-clim,vmax=clim)

	# apply color limits, but not if it's a log contour plot
	#if log_levels is None:
	#	print('applying color limits')
	#	if cmap_type == 'divergent':
	#		plt.clim([-clim,clim])
	#	else:
	#		plt.clim([0,clim])

	#if cbar:
	#	if (clim > 1000):
	#		CB = plt.colorbar(cs, shrink=0.6, extend='both',format='%.1e')
	#	if (clim < 0.001):
	#		CB = plt.colorbar(cs, shrink=0.6, extend='both',format='%.1e')
	#	else:
	CB = plt.colorbar(cs, shrink=0.6, extend='both')


def variance_maps(E,climatology_option = 'NODA',hostname='taurus',verbose=False):  

	# given a certain experiment or dataset (E) over a certain daterange,  
	# retrieve the data, then 
	# calculate the daily climatology and anomaly wrt climatology, 
	# then filter the daily anomaly using a Lanczos filter, 
	# then calculate the variance of the filtered anomaly.  
	# based on code from the CLIVAR MJO diagnostics, 

	# make sure that the vertical level range is set to something around 850 hPa
	# the vertical levels we select depend on the model
	# **right now only have settings for L66 WACCM
	E['levrange'] = [850,850]

	# compute or load the daily climatology and deviation from climatology  
	anomalies,climatology,lat,lon,lev = ano(E,climatology_option = climatology_option,hostname=hostname,verbose=verbose)

	# filter daily anomalies using a Lanczos filter
	AA,FA = filter(anomalies,return_as_vector=True)

	# compute the variance of these filtered anomaly fields
	VV = var(AA,variable_dimensions=anomalies.shape,return_as_vector=False)

	return VV,lat,lon

def correlations_lag_lat_or_lon(E,maxlag,lat_or_lon = 'lon',filter_order=50,climatology_option='NODA',hostname='taurus',verbose=False):

	"""
	compute correlations between U850 or OLR in a reference are and everywhere else, 
	as a function of lag and either latitude or longitude 

	INPUTS:  
	E - a standard DART experiment dictionary, with the variable field and level range corresponding to some MJO variable  
	maxlag: the limit of the lag (in days) that we look at 
	lat_or_lon: choose dimension to preserve after averaging -- 'lat' or 'lon'  
	climatology_option: choose which climatology to take the anomalies to respect with -- default is "NODA"  
	"""

	# change the given daterange to daily resolution, because the lag is specified in days  
	E['daterange'] = dart.change_daterange_to_daily(E['daterange'])

	# compute or load the daily climatology and deviation from climatology  
	anomalies,climatology,lat,lon,lev,DRnew = ano(E,climatology_option = climatology_option,hostname=hostname,verbose=verbose)

	# filter daily anomalies using a Lanczos filter
	AA,FA = filter(anomalies,filter_order,return_as_vector=False)
	
	if E['variable'] == 'U':
		variable_name = 'U'+str(E['levrange'][0])
	else:
		variable_name = E['variable']

	# compute the zonal and meridional mean of the resulting field 
	# the regions we average over depend on whether we want lag-lat, or lag-lon plots
	# also, note thatm by how the filtered anomalies are constructed, the 3rd dimension is always time  
	if lat_or_lon == "lon":
		# select latitudes 10S-10N and average meridionally, then plot correlations as a function of lon  	
		lat1,lon1,FAm = aave('TB',FA,lat,lon,None,variable_name,averaging_dimension='lat')
	if lat_or_lon == "lat":
		# average over the longitude corridor 80-100E and plot correlations as a function of lat
		lat1,lon1,FAm = aave('ZB',FA,lat,lon,None,variable_name,averaging_dimension='lon')


	# area averaging  the desired variable over the Indian Ocean reference point
	if (E['daterange'][0].month  >= 10) or (E['daterange'][0].month  <= 2):
		season = 'winter'
	else:
		season = 'summer'
	lat0,lon0,FA0 = aave('IO',FA,lat,lon,season,variable_name,averaging_dimension="all")

	#------ compute field of correlation coefficients   	
	# empty array size Lag by Lat
	# plus an array to keep track of sample size
	Lag_range = range(-maxlag,maxlag+1)
	nlag = len(Lag_range)
	n = FAm.shape[0]
	R = np.zeros(shape=(nlag,n))
	S = np.zeros(shape=(nlag,n))

	# loop over latitudes
	T = len(FA0)
	for ii in range(n):
		# loop over lags  
		for ilag,L in zip(range(nlag),Lag_range):
			# the time points that we can check go from L to T-L
			# so shorter lags have a larger sample size and are more significant.  
			if L < 0:
				Tsel = range(-L,T)
			if L > 0:
				Tsel = range(0,T-L)
			if L == 0:
				Tsel = range(0,T)

			# loop over the available time points and gather values to compare
			IO = []
			X  = []
			for k in Tsel:
				IO.append(FA0[k+L])
				X.append(FAm[ii,k])

			# now compute the correlation from this list of samples and store in the lag vs lat array  
			rho = np.corrcoef(X,IO)
			if rho != []:
				R[ilag,ii] = rho[1,0]
				S[ilag,ii] = len(IO)
			else:
				R[ilag,ii] = np.nan
				S[ilag,ii] = np.nan
	if lat_or_lon == 'lon':
		space_dim = lon1
	if lat_or_lon == 'lat':
		space_dim = lat1

	L = np.array(Lag_range)

	return R,S,L,space_dim

def RMM(E,climatology_option = 'NODA',hostname='taurus',verbose=False):

	"""
	this subroutine computes the real-time multivariate MJO (RMM) indices defined by Wheeler and Hendon (2004)
	this is done by reading in the multivariate EOF of OLR, U850, and U200 (computed from 
	satellite data and NCEP reanalysis), and then projecting our model's anomaly 
	fields onto these EOFs.  

	This code is pretty clunky, because it computes the RMM index over the daterange in E['daterange'], but 
	it uses all available data in the experiment given by E to compute the standard deviations of anomalies in 
	each variable. -- so it's best to run this all at once over long spans of time.  

	"""

	# read in the multivariate EOFs (eigenvectors)  
	if hostname == 'taurus':
		data_dir = '/data/c1/lneef/MJOindex/'
	fname = 'WH04_EOFstruc.txt'  
	ff = data_dir+fname  

	EVEC = pd.read_csv(ff,sep=' ',skiprows=9,nrows=432,header=None,engine='python')
	EVEC.columns=['blank','EV1','EV2']
	EOF = [EVEC.EV1, EVEC.EV2]

	# read in the normalization factors  
	#NORM = pd.read_csv(ff,skiprows=442,sep='  ',engine='python')
	#NORM.columns = ['normalization_factors']  
	#normfac = NORM.normalization_factors
	#NF_FLUT = normfac[0:144]
	#NF_U850 = normfac[144:288]
	#NF_U200 = normfac[288:432]
	#NF_list = [NF_FLUT,NF_U850,NF_U200]

	# read in the eigenvalues  
	f = open(ff, "r")
	lines = f.readlines()
	eigenvalues = lines[4].split()
	eigenval1 = float(eigenvalues[0])
	eigenval2 = float(eigenvalues[1])
	evalues = [eigenval1,eigenval2]
	f.close()
	
	# load anomalies of the three MJO variable (OLR, U850, U200) for this experiment
	variable_list = ['FLUT','U','U']
	levrange_list = [None,[850,850],[200,200]]
	Anomaly_list = []
	for variable,levrange in zip(variable_list,levrange_list):
		Etemp = E.copy()
		Etemp['variable'] = variable
		Etemp['levrange'] = levrange

		# for the MJO, we are only interested in anomalies between 15S and 15N
		Etemp['latrange'] = [-15,15]

		if Etemp['variable'] == 'U':
			variable_name = 'U'+str(levrange[0])
		else:
			variable_name = Etemp['variable']

		# load variable anomaly field for each variable
		anomalies,climatology,lat,lon,lev,DRnew = ano(Etemp,climatology_option=climatology_option,hostname=hostname,verbose=verbose)
		if anomalies is None:
			print('not enough data to compute RMM index -- returning')
			return None

		# average the normalized anomalies over the 15S-15N latitude band  
		lat1,lon1,ave_anom = aave('WH',anomalies,lat,lon,None,variable_name,averaging_dimension='lat')

		# normalize the anoamlies of each variable field by its standard deviation 
		if climatology_option=='NODA':
			# if we want standard deviations given by the No-DA experiment, do that here 
			# TODO: subroutine that loads correposnind No-DA experiment for a given experiment 
			Etemp = E.copy()
			Etemp['exp_name'] = 'W0910_NODA'
			S,lat,lon,lev,DR = load_std(Etemp,'ensemble',hostname)
		else:
			S,lat,lon,lev,DR = load_std(E,climatology_option,hostname)

		# the compute the "global" std for that variable (it's actually over the vertical and 
		# horizontal domain given in E)
		std = np.nanmean(S)

		# for each time in the array of anomalies, divide out the normalization factor for each MJO variable  
		nT = np.squeeze(ave_anom).shape[1]
		ave_anom_norm = 0*ave_anom
		for aa,iT in zip(ave_anom,range(nT)):
			ave_anom_norm[:,iT] = np.squeeze(ave_anom[:,iT])/std

		# put everything into a list
		Anomaly_list.append(ave_anom_norm)


	# concatenate the 3 anomaly fields so that we have a length (144x3) vector for nT points in time  
	AA = np.concatenate([A for A in Anomaly_list], axis=0)

	# compute the principal components  
	N = EVEC.EV1.shape[0]
	nT = AA.shape[1]
	pc = np.zeros(shape=(2,nT))

	for k in range(nT):	# loop over time  
		for eof,iev,eigval in zip(EOF,range(2),evalues):	# loop over eigenvectors and eigenvalues
			for ii in range(N):				# loop over 3xlongitudes  
				pc[iev,k] += (AA[ii,k]*eof[ii])/np.sqrt(eigval)

	return pc

def load_climatology(E,climatology_option = 'NODA',hostname='taurus',verbose=False):

	"""
	Load a climatology option for a given DART experiment. 
	The choice of climatology is given by 'climatology_option'. Choices are:  
	'NODA' (default): take the ensemble mean of the corresponding no-DA experiment as a N-year climatology  
	'F_W4_L66': CESM-WACCM simulation with observed forcings, 1951-2010 (perfomed by Wuke Wang)  
	"""
	climatology_option_not_found = True

	if climatology_option == 'NODA' :
		climatology_option_not_found = False
		# cycle over the dates in the experiment dictionary 
		# and load the ensemble mean of the corresponding No-assimilation case 
		# TODO: a subroutine that returns the corresponding NODA experiment for each case  
		Xlist = []	
		ECLIM = E.copy()
		ECLIM['exp_name'] = 'W0910_NODA'
		ECLIM['diagn'] = 'Prior'
		ECLIM['copystring'] = 'ensemble mean'
		Xclim,lat,lon,lev,DRnew = DSS.DART_diagn_to_array(ECLIM,hostname=hostname,debug=verbose)
		if len(DRnew) != len(ECLIM['daterange']):
			print('NOTE: not all requested data were found; returning a revised datarange')
		if Xclim is None:
			print('Cannot find data for climatology option '+climatology_option+' and experiment '+E['exp_name'])
			return None, None, None, None

	if climatology_option == 'F_W4_L66' :
		from netCDF4 import Dataset
		climatology_option_not_found = False
		# in this case, load a single daily climatology calculated from this CESM-WACCM simulation  
		ff = '/data/c1/lneef/CESM/F_W4_L66/atm/climatology/F_W4_L66.cam.h1.1951-2010.daily_climatology.nc'
		f = Dataset(ff,'r')
		lat = f.variables['lat'][:]
		lon = f.variables['lon'][:]
		lev = f.variables['lev'][:]
		time = f.variables['time'][:]

		# load climatology of the desired model variable  
		variable = E['variable']
		if E['variable'] == 'US':
			variable = 'U'
		if E['variable'] == 'VS':
			variable = 'V'
		if E['variable'] == 'OLR':
			variable = 'FLUT'
		VV = f.variables[variable][:]
		f.close()

		# choose the times corresponding to the daterange in E
		d0 = E['daterange'][0].timetuple().tm_yday	# day in the year where we start  
		nT = len(E['daterange'])
		df = E['daterange'][nT-1].timetuple().tm_yday	# day in the year where we start  

		# if df<d0, we have to cycle back to the beginning of the year
		if df < d0:
			day_indices = list(range(d0-1,365))+list(range(0,df))
		else:
			day_indices = list(range(d0-1,df))

		# also choose the lat, lon, and level ranges corresponding to those in E
		if E['levrange'] is not None:
			if E['levrange'][0] == E['levrange'][1]:
				ll = E['levrange'][0]
				idx = (np.abs(lev-ll)).argmin()
				lev2 = lev[idx]
				k1 = idx
				k2 = idx
			else:
				k2 = (np.abs(lev-E['levrange'][1])).argmin()
				k1 = (np.abs(lev-E['levrange'][0])).argmin()
				lev2 = lev[k1:k2+1]

		j2 = (np.abs(lat-E['latrange'][1])).argmin()
		j1 = (np.abs(lat-E['latrange'][0])).argmin()
		lat2 = lat[j1:j2+1]
		i2 = (np.abs(lon-E['lonrange'][1])).argmin()
		i1 = (np.abs(lon-E['lonrange'][0])).argmin()
		lon2 = lon[i1:i2+1]

		if len(VV.shape) == 4:
			Xclim = VV[day_indices,k1:k2+1,j1:j2+1,i1:i2+1]
		else:
			Xclim = VV[day_indices,j1:j2+1,i1:i2+1]

		# in this case, we don't need to change the daterange  
		DRnew = E['daterange']

	if climatology_option_not_found:
		print('Climatology option '+climatology_option+' has not been coded yet. Returning None for climatology.')
		return None, None, None, None

	return Xclim,lat,lon,lev,DRnew

def load_std(E,std_mode = 'NODA',hostname='taurus',verbose=False):

	"""
	This subroutine returns the standard deviation of whatever variable is given in E['variable'], 
	for each time given in E['daterange'].
	There are several ways to compute the standard deviation, and that's determined by the input 
	'std_mode':
		std_mode='ensemble' simply computes the standard deviation of the DART ensemble  
			at each time 
		if you set std_mode to any other string, it looks up the multi-year experiment corresponding 
			to that string using the subroutine 'std_runs' in the user 
			module experiment_settings. 
			In this case, the standard deviation  
			is computed for each time over several years, rather than an ensemble 

	"""
	if std_mode == 'ensemble' :
		# cycle over the dates in the experiment dictionary 
		# and load the ensemble mean of the corresponding No-assimilation case 
		# TODO: a subroutine that returns the corresponding NODA experiment for each case  
		Xlist = []	
		ECLIM = E.copy()
		ECLIM['copystring'] = 'ensemble std'
		Xclim,lat,lon,lev,DRnew = DSS.DART_diagn_to_array(ECLIM,hostname=hostname,debug=verbose)
		if len(DRnew) != len(ECLIM['daterange']):
			print('NOTE: not all requested data were found; returning a revised datarange')
		if Xclim is None:
			print('Cannot find data for experiment '+E['exp_name'])
			return None, None, None, None

	if std_mode == 'F_W4_L66' :

		# find the corresponding dataset  
		ff = es.std_runs(std_mode,hostname=hostname,debug=verbose)

		# load the desired variables 
		from netCDF4 import Dataset
		f = Dataset(ff,'r')
		lat = f.variables['lat'][:]
		lon = f.variables['lon'][:]
		lev = f.variables['lev'][:]
		time = f.variables['time'][:]

		variable = E['variable']
		if E['variable'] == 'US':
			variable = 'U'
		if E['variable'] == 'VS':
			variable = 'V'
		if E['variable'] == 'OLR':
			variable = 'FLUT'
		VV = f.variables[variable][:]
		f.close()

		# choose the times corresponding to the daterange in E
		d0 = E['daterange'][0].timetuple().tm_yday	# day in the year where we start  
		nT = len(E['daterange'])
		df = E['daterange'][nT-1].timetuple().tm_yday	# day in the year where we start  

		# if df<d0, we have to cycle back to the beginning of the year
		if df < d0:
			day_indices = list(range(d0-1,365))+list(range(0,df))
		else:
			day_indices = list(range(d0-1,df))

		# also choose the lat, lon, and level ranges corresponding to those in E
		if E['levrange'] is not None:
			if E['levrange'][0] == E['levrange'][1]:
				ll = E['levrange'][0]
				idx = (np.abs(lev-ll)).argmin()
				lev2 = lev[idx]
				k1 = idx
				k2 = idx
			else:
				k2 = (np.abs(lev-E['levrange'][1])).argmin()
				k1 = (np.abs(lev-E['levrange'][0])).argmin()
				lev2 = lev[k1:k2+1]

		j2 = (np.abs(lat-E['latrange'][1])).argmin()
		j1 = (np.abs(lat-E['latrange'][0])).argmin()
		lat2 = lat[j1:j2+1]
		i2 = (np.abs(lon-E['lonrange'][1])).argmin()
		i1 = (np.abs(lon-E['lonrange'][0])).argmin()
		lon2 = lon[i1:i2+1]

		if len(VV.shape) == 4:
			Xclim = VV[day_indices,k1:k2+1,j1:j2+1,i1:i2+1]
		else:
			Xclim = VV[day_indices,j1:j2+1,i1:i2+1]

		# in this case, we don't need to change the daterange  
		DRnew = E['daterange']

	return Xclim,lat,lon,lev,DRnew

def ano(E,climatology_option = 'NODA',hostname='taurus',verbose=False):

	"""
	Compute anomaly fields relative to some climatology

	Inputs allowed for climatology_option:  
	'NODA': take the ensemble mean of the corresponding no-DA experiment as a 40-year climatology  
	'F_W4_L66': daily climatology of a CESM+WACCM simulation with realistic forcings, 1951-2010
	None: don't subtract out anything -- just return the regular fields in the same shape as other "anomalies"  
	"""

	# load climatology 
	Xclim,lat,lon,lev,DR = load_climatology(E,climatology_option,hostname)

	# change the daterange in the anomalies to suit what was found for climatology  
	if len(DR) != len(E['daterange']):
		print('Changing the experiment daterange to the dates found for the requested climatology')
		E['daterange'] = DR
		d1 = DR[0].strftime("%Y-%m-%d")
		d2 = DR[len(E['daterange'])-1].strftime("%Y-%m-%d")
		print('new daterange goes from '+d1+' to '+d2)

	# some climatologies are only available at daily resolution, so 
	# in that case we have to change the daterange in E to be daily  
	if (climatology_option == 'F_W4_L66'):
		d0 = E['daterange'][0]
		df = E['daterange'][len(E['daterange'])-1]
		days = df-d0
		DRnew =  dart.daterange(date_start=d0, periods=days.days+1, DT='1D')
		E['daterange'] = DRnew

	# load the desired model fields for the experiment
	Xlist = []	# empty list to hold the fields we retrieve for every day  
	for date in E['daterange']:
		X,lat0,lon0,lev0 = DSS.compute_DART_diagn_from_model_h_files(E,date,hostname=hostname,verbose=verbose)
		if X is not None:
			Xs = np.squeeze(X)
			Xlist.append(Xs)
			lat = lat0
			lon = lon0
			lev = lev0

	# check that the right vertical levels were loaded
	if verbose:
		print('------computing daily anomalies for the following vertical levels and variable:-------')
		print(lev)
		print(E['variable'])

	# compute anomalies:
	# for this we turn the model fields into a matrix and subtract from the climatology
	XX = np.concatenate([X[..., np.newaxis] for X in Xlist], axis=len(Xs.shape))
	if climatology_option == None:
		AA = XX
	else:
		# if the climatology does not have shape lat x lon x lev x time, 
		# run swapaxes 2x to get it as such  
		# NOTE: this is still a kludge and probably wont work with all datasets - check this carefully 
		# with your own data 
		XclimS = np.squeeze(Xclim)
		nT = len(DRnew)
		lastdim = len(XclimS.shape)-1
		for s,ii in zip(XclimS.shape,range(len(XclimS.shape))):
			if s == nT:
				time_dim = ii

		# if only retrieveing a single date, don't need to do any reshaping
		# but might need to squeeze out a length-one time dimension
		if nT == 1:
			XclimR = XclimS
			XX = np.squeeze(XX)
		else:
			# if time is the last dimension, don't need to reshape Xclim 
			if time_dim == lastdim: 
				XclimR = XclimS
			# if time is the first dimension, need to reshape Xclim
			if time_dim == 0:	
				Xclim2 = XclimS.swapaxes(0,lastdim)
				XclimR = Xclim2.swapaxes(0,1)


		AA = XX-XclimR

	return AA,XclimR,lat,lon,lev,DR

def filter(daily_anomalies,filter_order = 50, return_as_vector = True):

	"""
 	given 3D or 2D anomaly fields (e.g. of zonal wind)
	apply a Lanczos filter to isolate the 20-100 day MJO signal 

	note that here the input data have to have DAILY resolution  

	input filter_order gives the order of the Lanczos Filter - it's defaults is 50 , for a 201-point filter (not sure about this yet-- need to check)  
	"""

	# turn the anomaly field into a vectors in time 
	if len(daily_anomalies.shape)==3:
		[n1,n2,nt] = daily_anomalies.shape
		L = n1*n2
		A = np.reshape(daily_anomalies,(L,nt))
	if len(daily_anomalies.shape)==4:
		[n1,n2,n3,nt] = daily_anomalies.shape
		L = n1*n2*n3
		A = np.reshape(daily_anomalies,(L,nt))

	f_low = 0.01		# 100 days
	f_high = 0.05		# 20 days 
	n = filter_order  

	fil = LF.LanczosFilter("bp",f_low,f_high,n)

	FA = A*0
	for ii in range(len(A)):
		FA[ii] = fil.getfiltered(A[ii,:])
    
	# if return_as_vector is false, reshape the filtered fields to 3D
	if return_as_vector:
		filtered_anomalies = FA
		daily_anomalies = A
	else:
		if len(daily_anomalies.shape)==3:
			filtered_anomalies = np.reshape(FA,(n1,n2,nt))
		if len(daily_anomalies.shape)==4:
			filtered_anomalies = np.reshape(FA,(n1,n2,n3,nt))

	return daily_anomalies,filtered_anomalies

def var(filtered_anomalies,variable_dimensions,return_as_vector=False):

	# compute the variance of 3Dxtime or 2Dxtime anomaly fields that have been 
	# Lanczos-filtered to isolate the MJO time window, 
	# here the input fields should be a vector varying in time, i.e. with dimension N x nt
	# where N = nlat x nlon x nlev

	N = filtered_anomalies.shape[0]
	VA = np.zeros(shape=(N,1))

	for ii in range(N):
		VA[ii] = np.var(filtered_anomalies[ii,:])

	# if return_as_vector is false, reshape the filtered fields to 3D
	if return_as_vector:
		Vout = VA
	else:
		Vout = np.reshape(VA,variable_dimensions[0:2])

	return(Vout)

def aave(region,FA,lat,lon,season,variable_name,averaging_dimension='all'):

	"""
	average diagnostic model variables over a region  
	
	INPUTS:  
	+ region: either a pre-defined region (retrieved from function averaging_regions) or 
		an experiment dictionary, in which case we use the lat and lonrage there  
	+ FA: anomaly field over which to average. For standard MJO diagnostics this field should be 
		filtered for the MJO time window  
	+ lat,lon: the lat and lon arrays that go with FA  
	+ variable_name: the name of the variable over which we average  
		this is only relevant if we have a pre-named averaging region  
	+ averaging_dimension: the dimension we average over. There are 3 options:  
		 all : over lat and lon
		 lat: over lat only
		 lon: over lon only
	"""


	# retrieve the averaging region limits
	if isinstance(region,dict):
		# if 'region' is given by an experiment dictionary, read the lat and lonranges from the dictionary itself  
		latrange = region['latrange']
		lonrange = region['lonrange']
	else:
		# otherwise, retrieve the right averaging region  
		latrange,lonrange = averaging_regions(region,season,variable_name)

	# figure out how the anomaly field FA is shaped
	# the way FA is calculated, it's last dim is always time  

	shape_tuple = FA.shape
	for dimlength,ii in zip(shape_tuple,range(len(shape_tuple))):
		if dimlength == len(lon):
			londim = ii
		if dimlength == len(lat):
			latdim = ii


	# find the lat and lon indices that those ranges correspond to the region limits
	i1 = (np.abs(lon-lonrange[0])).argmin()	
	i2 = (np.abs(lon-lonrange[1])).argmin()	
	j1 = (np.abs(lat-latrange[0])).argmin()	
	j2 = (np.abs(lat-latrange[1])).argmin()	

	lat_out = lat[j1:j2+1]
	lon_out = lon[i1:i2+1]

	# select the focus region  
	# here I just brute-forced it, manually checking various array shapes that I've had  
	if (latdim == 0) and (londim == 1):
		if len(FA.shape)==2:
			FAsel = FA[j1:j2+1,i1:i2+1]
		if len(FA.shape)==3:
			FAsel = FA[j1:j2+1,i1:i2+1,:]
	if (latdim == 1) and (londim == 0):
		if len(FA.shape)==2:
			FAsel = FA[i1:i2+1,j1:j2+1,:]
		if len(FA.shape)==3:
			FAsel = FA[i1:i2+1,j1:j2+1,:]
	if (latdim == 1) and (londim == 2) and (len(shape_tuple) == 4):
		FAsel = FA[:,j1:j2+1,i1:i2+1,:]
	
	# average  
	if averaging_dimension == "all":
		FAave1 = np.nanmean(FAsel,axis=latdim,keepdims=True)
		FAave2 = np.nanmean(FAave1,axis=londim,keepdims=True)
		FAave = np.squeeze(FAave2)
	if averaging_dimension == "lat": # meridional mean only
		FAave = nanmean(FAsel,axis=latdim)
	if averaging_dimension == "lon": # zonal mean only
		FAave = nanmean(FAsel,axis=londim)

	return lat_out,lon_out,FAave

def astd(region,FA,lat,lon,season,variable_name,averaging_dimension='all'):

	"""
	compute the standard deviations of diagnostic model variables over a region  
	
	INPUTS:  
	+ region: either a pre-defined region (retrieved from function averaging_regions) or 
		an experiment dictionary, in which case we use the lat and lonrage there  
	+ FA: anomaly field over which to average. For standard MJO diagnostics this field should be 
		filtered for the MJO time window  
	+ lat,lon: the lat and lon arrays that go with FA  
	+ variable_name: the name of the variable over which we average  
		this is only relevant if we have a pre-named averaging region  
	+ averaging_dimension: the dimension we compute the STD over. There are 3 options:  
		 all : over lat and lon
		 lat: over lat only
		 lon: over lon only
	"""


	# retrieve the averaging region limits
	if isinstance(region,dict):
		# if 'region' is given by an experiment dictionary, read the lat and lonranges from the dictionary itself  
		latrange = region['latrange']
		lonrange = region['lonrange']
	else:
		# otherwise, retrieve the right averaging region  
		latrange,lonrange = averaging_regions(region,season,variable_name)

	# figure out how the anomaly field FA is shaped
	# the way FA is calculated, it's last dim is always time  

	shape_tuple = FA.shape
	for dimlength,ii in zip(shape_tuple,range(len(shape_tuple))):
		if dimlength == len(lon):
			londim = ii
		if dimlength == len(lat):
			latdim = ii


	# find the lat and lon indices that those ranges correspond to the region limits
	i1 = (np.abs(lon-lonrange[0])).argmin()	
	i2 = (np.abs(lon-lonrange[1])).argmin()	
	j1 = (np.abs(lat-latrange[0])).argmin()	
	j2 = (np.abs(lat-latrange[1])).argmin()	

	lat_out = lat[j1:j2+1]
	lon_out = lon[i1:i2+1]

	# select the focus region  
	# here I just brute-forced it, manually checking various array shapes that I've had  
	if (latdim == 0) and (londim == 1):
		if len(FA.shape)==2:
			FAsel = FA[j1:j2+1,i1:i2+1]
		if len(FA.shape)==3:
			FAsel = FA[j1:j2+1,i1:i2+1,:]
	if (latdim == 1) and (londim == 0):
		if len(FA.shape)==2:
			FAsel = FA[i1:i2+1,j1:j2+1,:]
		if len(FA.shape)==3:
			FAsel = FA[i1:i2+1,j1:j2+1,:]
	if (latdim == 1) and (londim == 2) and (len(shape_tuple) == 4):
		FAsel = FA[:,j1:j2+1,i1:i2+1,:]
	
	# average  
	if averaging_dimension == "all":
		FAstd1 = np.nanstd(FAsel,axis=latdim,keepdims=True)
		FAstd2 = np.nanstd(FAstd1,axis=londim,keepdims=True)
		FAstd = np.squeeze(FAstd2)
	if averaging_dimension == "lat": # meridional std only
		FAstd = np.nanstd(FAsel,axis=latdim)
	if averaging_dimension == "lon": # zonal std only
		FAstd = np.nanstd(FAsel,axis=londim)

	return lat_out,lon_out,FAstd

def averaging_regions(region,season,variable):  

	"""
	lat and lon limits of the averaging regions for CLIVAR MJO diagnostics
	these are taken from Waliser et al. 2009 (J. Clim)  
	"""

	if region is 'WH':	# this latitude band is used in the Wheeler and Hendon MJO index  
		latrange = [-15,15]
		lonrange = [0,360]
		return latrange, lonrange
	if region is 'TB':
		latrange = [-10,10]
		lonrange = [0,360]
		return latrange, lonrange
	if region is 'ZB':
		latrange = [-30,30]
		lonrange = [80,100]
		return latrange, lonrange

	# variables that count as "precipitation"
	precip_variables = ['OLR','precip','FLUT']

	# regions specific for seasons------

	#boreal winter
	if season is 'winter':

		# indian ocean  
		if region is 'IO':  
			if (variable in precip_variables):
				latrange = [-10,5]
				lonrange = [75,100]
			if (variable == 'U850'):
				latrange = [-16.25,-1.25]
				lonrange = [68.75,96.25]
			if (variable is 'U200'):
				latrange = [3.75,21.25]
				lonrange = [56.25,78.75]
		
		# west pacific  
		if region is 'WP':  
			if (variable in precip_variables):
				latrange = [-20,-5]
				lonrange = [160,185]
			if (variable is 'U850'):
				latrange = [-13.75,1.25]
				lonrange = [163.75,191.25]
			if (variable is 'U200'):
				latrange = [3.75,21.25]
				lonrange = [123.75,151.25]
		
		# maritime continent
		if region is 'MC':  
			if (variable in precip_variables):
				latrange = [-17.5,-2.5]
				lonrange = [115,145]
			else:
				print('averaging over the Maritime Continent for zonal winds is not part of the CLIVAR diagnostics.')
				return 
		
		# east pacific
		if region is 'EP':  
			if (variable in precip_variables) or (variable is 'U850'):
				print('averaging over the East Pacific for anything other than U200 is not part of the CLIVAR diagnostics.')
				return 
			if (variable is 'U200'):
				latrange = [1.25,16.25]
				lonrange = [256.25,278.75]

	# boreal summer
	if season is 'summer':

		# indian ocean  
		if region is 'IO':  
			if (variable in precip_variables):
				latrange = [-10,5]
				lonrange = [75,100]
			if (variable is 'U850'):
				latrange = [3.75,21.25],
				lonrange = [68.75,96.25]
			if (variable is 'U200'):
				latrange = [1.25,16.25]
				lonrange = [43.75,71.25]
	
		# bay of Bengal
		if region is 'BB':
			if (variable in precip_variables):
				latrange = [10,20]
				lonrange = [80,100]		
			else:
				print('averaging over the bay of Bengal for anything other than U200 and U850 is not part of the CLIVAR diagnostics.')
				return 

		# west pacific  
		if region is 'WP':  
			if (variable in precip_variables):
				latrange = [10,25]
				lonrange = [115,140]
			if (variable is 'U850'):
				latrange = [3.75,21.25]
				lonrange = [118.75,146.25]
			if (variable is 'U200'):
				latrange = [3.75,21.25]
				lonrange = [123.75,151.25]
		
		# maritime continent
		if region is 'MC':  
			print('averaging over the Maritime Continent for boreal summer is not part of the CLIVAR diagnostics.')
			return 
		
		# east pacific
		if region is 'EP':  
			if (variable in precip_variables):
				print('averaging over the East Pacific for OLD and precip during boreal summer is not part of the CLIVAR diagnostics.')
				return 
			if (variable is 'U850'):
				latrange = [6.25,16.25]
				lonrange = [241.25,266.25]
				
			if (variable is 'U200'):
				latrange = [1.25,16.25]
				lonrange = [238.75,266.25]

	# throw an error if latrange and lonrange didn't get defined
	try:
		latrange
		lonrange
	except NameError:
		print('MJO.averaging_regions Nothing defined for region '+region+', season '+season,', and variable '+variable)
		return

	return latrange,lonrange		

def read_RMM_true(date_limits,hostname='taurus'):

	"""
	Read in the observed real-time multivariate MJO index and return RMM1 and RMM2 
	between a start date and end date

	INPUTS:
	date_limits: a tuple of datetime.datetime objects giving the start and end dates  

	"""


	# read in the real-time operational MJO index  
	hostname_not_found = True
	if hostname == 'taurus':
		hostname_not_found = False
		data_dir = '/data/c1/lneef/MJOindex/'
	fname = 'RMM1RMM2.74toRealtime.txt'
	ff = data_dir+fname  
	na_values = [9.9999996E+35,999]
	DF = pd.read_csv(ff,skiprows=2,header=None,delim_whitespace=True,na_values=na_values)
	DF.columns=['Year','Month','Day','RMM1','RMM2','phase','amplitude','description']
	DF.dtype = {'Year':np.int32, 'Month':np.int32, 'Day':np.int32, 'RMM1':np.float64, 
	'RMM2':np.float64, 'phase':np.int32, 'amplitude':np.float64, 'description':object}
	
	# remove the rows with bad values from the data frame  
	DF2 = DF.dropna() 

	# turn the year, mmonth, and day columns into a datetime array  
	# and collect the dates and RMM values that fit into the date range  
	# there is probably a more elegant way to do this.  
	ylist = list(DF2['Year'])
	mlist = list(DF2['Month'])
	dlist = list(DF2['Day'])
	rmm1list = list(DF2['RMM1'])
	rmm2list = list(DF2['RMM2'])
	dates = []
	RMM1 = []
	RMM2 = []

	for y,m,d,r1,r2 in zip(ylist,mlist,dlist,rmm1list,rmm2list):
		d = datetime.datetime(int(y),int(m),int(d))
		cond = (d > date_limits[0]) and (d < date_limits[1])
		if cond:
			dates.append(d)
			RMM1.append(r1)
			RMM2.append(r2)


	if hostname_not_found:
		print('Do not have file paths set for hostname  ',hostname)
		return

	# return the RMM1 and RMM2 indices  
	return dates, RMM1, RMM2


def compute_RMM(E,climatology_option='NODA',hostname='taurus',verbose=False):

	"""
	given a certain experiment dictionary, compute the Wheeler and Hendon (2004)
	RMM index by projecting the modeled fields onto Wheeler and Hendon's multivariate EOFs.
	
	The subroutine loops over the dates and times given in E['daterange'] and, for each day,
	creates a Pandas dataframe that holds the RMM1 and RMM2 values for each DART copy. 
	This datarame is then printed to a CSV file, which is stored under /csv/MJO in that 
	experiment's directory. 

	"""



	# compute the PCs for the desired timespan and list of copies 
	RMM1list = []
	RMM2list = []
	bad_copies = []		# start a list of the copies that are unavailable  

	for copy in copy_list:
		if copy == "operational":
			date_limits = (E['daterange'][0],E['daterange'][len(E['daterange'])-1])
			dates,RMM1,RMM2 = read_RMM_true(date_limits,hostname='taurus')
			RMM1list.append(RMM1)
			RMM2list.append(RMM2)
		
		else:
			E['copystring'] = copy
			pc = RMM(E,climatology_option=climatology_option,hostname='taurus',verbose=verbose)
			if pc is None:
				# if we don't have enough data to compute the RMM index for this experiment, 
				# add it to the list of bad copies:
				print('     Unable to compute RMM index for '+copy)
				bad_copies.append(copy)
			else:
				RMM1list.append(pc[0,:])
				RMM2list.append(pc[1,:])

	# remove the "bad" copies from the list
	[copy_list.remove(bc) for bc in bad_copies]

	# pimp out the plot a little bit  
	plt.plot([-4,4],[-4,4],linewidth=0.2,linestyle='--',color='k')
	plt.plot([-4,4],[4,-4],linewidth=0.2,linestyle='--',color='k')
	plt.plot([-4,4],[0,0],linewidth=0.2,linestyle='--',color='k')
	plt.plot([0,0],[-4,4],linewidth=0.2,linestyle='--',color='k')
	plt.xlim([-4.0,4.0])
	plt.ylim([-4.0,4.0])

	# circle in the center of the plot to denote weak index  
	circle = plt.Circle((0, 0), radius=1.0, fc='k', ec='k', alpha=0.2)
	plt.gca().add_patch(circle)

	# cycle over copies and plot the two princial components against each other  
	for copy,RMM1,RMM2 in zip(copy_list,RMM1list,RMM2list):
		
		# choose the color based on the copy string
		if "ensemble member" in copy:
			lcolor = "#848484"
			#lcolor = "#70B8FF"
			plt.plot(RMM1,RMM2,'-',color=lcolor,linewidth=1)
		if copy == "ensemble mean":
			#lcolor = "#636363"
			lcolor = "#70B8FF"
			#c = np.linspace(0, 10, RMM1.shape[0])
			#cmap = plt.cm.jet
			#plt.scatter(RMM1,RMM2,c=c,cmap=cmap,s=10)
			plt.plot(RMM1,RMM2,'-',color=lcolor,linewidth=2)
		if copy == "operational":
			lcolor = "#000000"
			print('plotting operational RMM index in black')
			plt.plot(RMM1,RMM2,'-',color=lcolor,linewidth=2)

	# labels and stuff  
	plt.xlabel('RMM1')
	plt.ylabel('RMM2')

