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
#from calendar import monthrange
#from netCDF4 import Dataset
import WACCM as waccm
import DART_state_space as DSS
import pyclimate.LanczosFilter as LF
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from scipy.stats import nanmean


def plot_RMM(E,copies_to_plot,hostname='taurus',verbose=False):

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

	"""

	# given the chosen plot variation, define a list of copies to load
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
	if ("operational" in copies_to_plot): 
			copy_list.append("operational")

	# compute the PCs for the desired timespan and list of copies 
	# or load the observatinal value 
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
			pc = RMM(E,hostname='taurus',verbose=verbose)
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

def plot_correlations_lag_lat_or_lon(E,maxlag=25,lag_versus_what='lon',cbar=True,hostname="taurus"):

	# given a certain experiment or dataset over a certain daterange, 
	# plot the correlation between wind or precip anomalies in one reference
	# region, relative to everywhere else, either 
	# as a function of latitude and longite, and Lag.  
	# this should produce figures like Figs. 5-6 of Waliser et al. 

	# load the correlation field 
	R,S,L,x = correlations_lag_lat_or_lon(E,maxlag,lag_versus_what)
        #X,Y = np.meshgrid(x,L)

	# note that DART-WACCM output data is in time units of 6 hours
	# for the MJO it makes more sense to look at days, so 
	# convert L to daily
	Ld = L*0.25

        # choose color map based on the variable in question
	E['extras'] = 'Correlation'
	colors,cmap,cmap_type = DSS.state_space_HCL_colormap(E)
	if lag_versus_what=='lat':
		plt.xlabel('Latitude')
	if lag_versus_what=='lon':
		plt.xlabel('Longitude')
	plt.ylabel('Lag (days)')

        # contour plot of the chosen variable
        cs = plt.contourf(x,Ld,R,len(colors)-1,cmap=cmap)
	plt.clim([-1.0,1.0])

	if cbar:
		CB = plt.colorbar(cs, shrink=0.6)

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


def variance_maps(E,hostname='taurus',verbose=False):  

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
	anomalies,climatology,lat,lon = ano(E,climatology_option = 'experiment',hostname=hostname,verbose=verbose)

	# filter daily anomalies using a Lanczos filter
	AA,FA = filter(anomalies,return_as_vector=True)

	# compute the variance of these filtered anomaly fields
	VV = var(AA,variable_dimensions=anomalies.shape,return_as_vector=False)

	return VV,lat,lon

def correlations_lag_lat_or_lon(E,maxlag,lat_or_lon = 'lon',hostname='taurus',verbose=False):

	# compute correlations between U850 or OLR in a reference are and everywhere else, 
	# as a function of lag and either latitude or longitude 

	# compute or load the daily climatology and deviation from climatology  
	anomalies,climatology,lat,lon = ano(E,climatology_option = 'experiment',hostname=hostname,verbose=verbose)

	# filter daily anomalies using a Lanczos filter
	AA,FA = filter(anomalies,return_as_vector=False)

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
	#IOA = aave('IO',FA,lat,lon)
	if (E['daterange'][0].month  >= 10) or (E['daterange'][0].month  < 10):
		season = 'winter'
	else:
		season = 'summer'
	lat0,lon0,FA0 = aave('IO',FA,lat,lon,season,variable_name,averaging_dimension="all")

	#------ compute field of correlation coefficients   	
	# empty array size Lag by Lat
	# plut an array to keep track of sample size
	Lag_range = range(-maxlag,maxlag+1)
	nlag = len(Lag_range)
	n = FAm.shape[0]
	R = np.zeros(shape=(nlag,n))
	S = np.zeros(shape=(nlag,n))

	# loop over latitudes
	T = len(FA0)
	for ii in range(n):
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
				IO.append(FA0[k])
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

def RMM(E,hostname='taurus',verbose=False):

	"""
	this subroutine computes the real-time multivariate MJO (RMM) indices defined by Wheeler and Hendon (2004)
	this is done by reading in the multivariate EOF of OLR, U850, and U200 (computed from 
	satellite data and NCEP reanalysis), and then projecting our model's anomaly 
	fields onto these EOFs.  
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
	NORM = pd.read_csv(ff,skiprows=442,sep='  ',engine='python')
	NORM.columns = ['normalization_factors']  
	normfac = NORM.normalization_factors

	# read in the eigenvalues  
	f = open(ff, "r")
	lines = f.readlines()
	eigenvalues = lines[4].split()
	eigenval1 = float(eigenvalues[0])
	eigenval2 = float(eigenvalues[1])
	evalues = [eigenval1,eigenval2]
	f.close()
	
	# read in the wind and OLR data corresponding to this experiment  
	variable_list = ['FLUT','U','U']
	levrange_list = [None,[850,850],[200,200]]

	Anomaly_field_list = []
	for variable,levrange in zip(variable_list,levrange_list):
		Etemp = E.copy()
		E['variable'] = variable
		E['levrange'] = levrange

		if E['variable'] == 'U':
			variable_name = 'U'+str(E['levrange'][0])
		else:
			variable_name = E['variable']

		# load variable anomaly field for each variable
		anomalies,climatology,lat,lon = ano(E,climatology_option = 'NODA',hostname=hostname,verbose=verbose)
		if anomalies is None:
			print('not enough data to compute RMM index -- returning')
			return None

		# also load an adequate standard deviation for the variable in question
		standard_deviations,lat,lon = stds(E,std_option = 'NODA',hostname=hostname,verbose=verbose)
		if standard_deviations is None:
			print('not enough standard deviation data to compute RMM index -- returning')
			return None

		# average the normalized anomalies over the 15S-15N latitude band  
		norm_anoms = anomalies/standard_deviations
		lat1,lon1,A = aave('WH',norm_anoms,lat,lon,None,variable_name,averaging_dimension='lat')
		Anomaly_field_list.append(A)

	# concatenate the 3 anomaly fields so that we have a length (144x3) vector for nT points in time  
	AA = np.concatenate([A for A in Anomaly_field_list], axis=0)

	# compute the principal components  
	N = EVEC.EV1.shape[0]
	nT = AA.shape[1]
	pc = np.zeros(shape=(2,nT))

	for k in range(nT):	# loop over time  
		for eof,iev in zip(EOF,range(2)):	# loop over eigenvectors
			for ii in range(N):				# loop over longitudes  
				pc[iev,k] += AA[ii,k]*eof[ii]

	return pc



def ano(E,climatology_option = 'NODA',hostname='taurus',verbose=False):

	"""
	Compute anomaly fields relative to some climatology

	Inputs allowed for climatology_option:  
	'NODA': take the ensemble mean of the corresponding no-DA experiment as a 40-year climatology  
	None: don't subtract out anything -- just return the regular fields in the same shape as other "anomalies"  
	"""

	Xclim = None

	if climatology_option is not None:
		# cycle over the dates in the requested experiment and load a corresponding climatology  
		Xlist = []	
		for date in E['daterange']:
			if climatology_option == 'NODA':
				# todo: a subroutine that returns the corresponding NODA experiment for each case  
				ECLIM = E.copy()
				ECLIM['exp_name'] = 'W0910_NODA'
				ECLIM['diagn'] = 'Prior'
				ECLIM['copystring'] = 'ensemble mean'
				X,lat0,lon0,lev0 = DSS.compute_DART_diagn_from_model_h_files(ECLIM,date,hostname=hostname,verbose=verbose)

			if X is None:
				# kill this loop if we can't find data for every time instance for this experiment and clim choice
				datestr = date.strftime("%Y-%m-%d")
				print('Cannot find climatology data for climatology option '+climatology_option+' and date '+datestr)
				return None, None, None, None
			else:	
				# if data are found, add to the list
				Xs = np.squeeze(X)
				Xlist.append(Xs)
				lat = lat0
				lon = lon0
				lev = lev0

		# if we successfully made it through the loop over dates and found climatology files, 
		# put them together  
		Xclim = np.concatenate([X[..., np.newaxis] for X in Xlist], axis=len(Xs.shape))
		if Xclim is None:
			if verbose:
				print('Climatology option '+climatology_option+' not possible for this experiment, either because it is not coded yet, or because we dont have enough data per instance.')
			return None, None, None, None

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
	# the daily climatology should be a matrix of hape lat x lon x time
	# so we turn the model fields into another matrix and simply subtract the two  
	XX = np.concatenate([X[..., np.newaxis] for X in Xlist], axis=len(Xs.shape))
	if climatology_option == None:
		AA = XX
	else:
		AA = XX-Xclim

	return AA,Xclim,lat,lon,lev

def filter(daily_anomalies,return_as_vector = True):

 	# given 3D or 2D anomaly fields (e.g. of zonal wind), band-pass filter to 
	# isolate the MJO signal  

	# turn the anomaly field into a vectors in time 
	if len(daily_anomalies.shape)==3:
		[n1,n2,nt] = daily_anomalies.shape
		L = n1*n2
		A = np.reshape(daily_anomalies,(L,nt))
	if len(daily_anomalies.shape)==4:
		[n1,n2,n3,nt] = daily_anomalies.shape
		L = n1*n2*n3
		A = np.reshape(daily_anomalies,(L,nt))

	#f_low = 0.01		# 100 days
	#f_high = 0.05		# 20 days 
	# ****note that DART-WACCM output data is 6-hourly, and the cutoff frequencies here are
	# defined in terms of DART-WACCM time units of 6 hours
	f_low = (1.0/100)*(6.0/24.0)
	f_high = (1.0/20)*(6.0/24.0)
	n = 50 # 201-point filter (??)

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
		FAave1 = np.nanmean(FA,axis=latdim,keepdims=True)
		FAave2 = np.nanmean(FAave1,axis=londim,keepdims=True)
		FAave = np.squeeze(FAave2)
	if averaging_dimension == "lat": # meridional mean only
		FAave = nanmean(FA,axis=latdim)
	if averaging_dimension == "lon": # zonal mean only
		FAave = nanmean(FA,axis=londim)

	return lat_out,lon_out,FAave

def averaging_regions(region,season,variable):  

	# lat and lon limits of the averaging regions for CLIVAR MJO diagnostics
	# these are taken from Waliser et al. 2009 (J. Clim)  
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


	# boreal winter
	if season is 'winter':

		# indian ocean  
		if region is 'IO':  
			if (variable is 'OLR') or (variable is 'precip'):
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
			if (variable is 'OLR') or (variable is 'precip'):
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
			if (variable is 'OLR') or (variable is 'precip'):
				latrange = [-17.5,-2.5]
				lonrange = [115,145]
			else:
				print('averaging over the Maritime Continent for zonal winds is not part of the CLIVAR diagnostics.')
				return 
		
		# east pacific
		if region is 'EP':  
			if (variable is 'OLR') or (variable is 'precip') or (variable is 'U850'):
				print('averaging over the East Pacific for anything other than U200 is not part of the CLIVAR diagnostics.')
				return 
			if (variable is 'U200'):
				latrange = [1.25,16.25]
				lonrange = [256.25,278.75]

	# boreal summer
	if season is 'summer':

		# indian ocean  
		if region is 'IO':  
			if (variable is 'OLR') or (variable is 'precip'):
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
			if (variable is 'OLR') or (variable is 'precip'):
				latrange = [10,20]
				lonrange = [80,100]		
			else:
				print('averaging over the bay of Bengal for anything other than U200 and U850 is not part of the CLIVAR diagnostics.')
				return 

		# west pacific  
		if region is 'WP':  
			if (variable is 'OLR') or (variable is 'precip'):
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
			if (variable is 'OLR') or (variable is 'precip'):
				print('averaging over the East Pacific for OLD and precip during boreal summer is not part of the CLIVAR diagnostics.')
				return 
			if (variable is 'U850'):
				latrange = [6.25,16.25]
				lonrange = [241.25,266.25]
				
			if (variable is 'U200'):
				latrange = [1.25,16.25]
				lonrange = [238.75,266.25]

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


def stds(E,std_option = 'NODA',hostname='taurus',verbose='False'):

	"""
	this subroutine loads a time series of standard deviation fields for a certain 
	DART experiment.  
	This can be used in the RMM index computation.  

	`std_option` chooses how we compute the standard deviation. 
	So far the only choice is NODA, which takes the STD of the corresponding free-running
	ensemble.
	"""

	std_option_not_found = True
	if std_option == 'NODA':
		ESTD = E.copy()
		ESTD['exp_name'] = 'W0910_NODA'
		ESTD['copystring'] = 'ensemble std'
		std_option_not_found = False

	if std_option_not_found:
		print('std option '+std_option+' is not a valid choice.')
		print('right now only supportting std_option NODA.')
		return None, None, None, None

	Slist = []
	for date in E['daterange']:
		S,lat,lon,lev = DSS.compute_DART_diagn_from_model_h_files(ESTD,date,verbose=verbose)

		if S is None:
			# kill this loop if we can't find data for every time instance for this experiment and clim choice
			datestr = date.strftime("%Y-%m-%d")
			print('Cannot find climatology data for std option '+std_option+' and date '+datestr)
			return None, None, None, None
		else:	
			# if data are found, add to the list
			Ss = np.squeeze(S)
			Slist.append(Ss)

	# if we successfully made it through the loop over dates and found std files, 
	# put them together  
	STD = np.concatenate([S[..., np.newaxis] for S in Slist], axis=len(Ss.shape))

	# return lat, lon, and STD field
	return STD,lat,lon
