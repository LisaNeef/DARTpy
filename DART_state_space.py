# Python module for DART diagnostic plots in state space.
#
# Lisa Neef, 4 June 2014


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.basemap import Basemap
import datetime
import pandas as pd
import DART as dart
import brewer2mpl
from netCDF4 import Dataset
import ERP as erp
import WACCM as waccm
import re
import ERA as era
import TEM as tem
import experiment_settings as es

## here are some common settings for the different subroutines

# list the 3d, 2d, 1d variables 
# TODO: fill this in with other common model variables 
var3d = ['U','US','V','VS','T','Z3','DELF']
var2d = ['PS','FLUT']
var1d = ['hyam','hybm','hyai','hybi']



def plot_diagnostic_globe(E,Ediff=None,projection='miller',clim=None,cbar='vertical',log_levels=None,hostname='taurus',debug=False,colorbar_label=None,reverse_colors=False,stat_sig=None):

	"""
	plot a given state-space diagnostic on a given calendar day and for a given variable 
	field on the globe.
	We can also plot the difference between two fields by specifying another list of Experiment
	dictionaries called Ediff  


	To plot climatology fields or anomalies wrt climatology, make the field E['diagn'] 'climatology.XXX' or 'anomaly.XXX', 
	where 'XXX' some option for loading climatologies accepted by the subroutine ano in MJO.py (see that code for options)

	INPUTS:  
	E: an experimend dictionary given the variable and diagnostic to be plotted, along with level, lat, lon ranges, etc. 
	Ediff: the difference experiment to subtract out (default is None)
	projection: the map projection to use (default is "miller")
	clim: the colorbar limits. If the scale is divergent, the limits are set as [-clim,clim]. If it's sequential, we do [0,clim].
	cbar: the orientation of the colorbar. Allowed values are 'vertical'|'horizontal'|None
	log_levels: a list of the (logarithmic) levels to draw the contours on. If set to none, just draw regular linear levels. 
	hostname
	taurus
	debug
	colorbar_label: string with which to label the colorbar  
	reverse_colors: set to false to reverse the colors in the 
	stat_sig: a dictionary giving the settings for estimating statistical significance with boostrap.
		Entries in this dict are: 
			P: the probability level at which we estimate the confidence intervals
			nsamples: the number of bootstrap samples 
		If these things are set, we add shading to denote fields that are statistically significantly 
			different from zero -- so this actually only makes sense for anomaies. 
		if stat_sig is set to "None" (which is the default), just load the data and plot. 
	"""

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


	##-----load data------------------
	if stat_sig is None:
		# turn the requested diagnostic into an array 
		Vmatrix,lat,lon,lev,DRnew = DART_diagn_to_array(E,hostname=hostname,debug=debug)

		# average over the last dimension, which is time
		if len(DRnew) > 1:
			VV = np.nanmean(Vmatrix,axis=len(Vmatrix.shape)-1)	
		else:
			VV = np.squeeze(Vmatrix)

		# average over vertical levels  if the variable is 3D
		# -- unless we have already selected a single level in DART_diagn_to_array
		if (E['variable'] in var3d) and (type(lev) != np.float64) and (E['levrange'][0] != E['levrange'][1]):
			# find the level dimension
			nlev = len(lev)
			for dimlength,idim in zip(VV.shape,range(len(VV.shape))):
				if dimlength == nlev:
					levdim = idim
			M1 = np.mean(VV,axis=levdim)
		else:
			M1 = np.squeeze(VV)

		# if computing a difference to another field, load that here  
		if (Ediff != None):
			Vmatrix,lat,lon,lev,DRnew = DART_diagn_to_array(Ediff,hostname=hostname,debug=debug)
			if len(DRnew) > 1:
				VV = np.nanmean(Vmatrix,axis=len(Vmatrix.shape)-1)	
			else:
				VV = np.squeeze(Vmatrix)
			# average over vertical levels  if the variable is 3D
			if (E['variable'] in var3d) and (type(lev) != np.float64) and (E['levrange'][0] != E['levrange'][1]):
				M2 = np.mean(VV,axis=levdim)
			else:
				M2 = np.squeeze(VV)
			# subtract the difference field out from the primary field  
			M = M1-M2
		else:
			M = M1
	else:
		# if statistical significance stuff was defined, loop over entire ensemble 
		# and use bootstrap to compute confidence intervals

		# first look up the ensemble size for this experiment from an internal subroutine:
		N = es.get_ensemble_size_per_run(E['exp_name'])

		# initialize an empty list to hold the ensemble of averaged fields 
		Mlist = []

		# loop over the ensemble  
		for iens in range(N):
			import bootstrap as bs

			E['copystring'] = 'ensemble member '+str(iens+1)
			# retrieve data for this ensemble member
			Vmatrix,lat,lon,lev,DRnew = DART_diagn_to_array(E,hostname=hostname,debug=debug)
			# if there is more than one time, average over this dimension (it's always the last one)
			if len(DRnew) > 1:
				VV = np.nanmean(Vmatrix,axis=len(Vmatrix.shape)-1)	
			else:
				VV = Vmatrix
			# average over vertical levels  if the variable is 3D and hasn't been averaged yet 
			if E['variable'] in var3d and type(lev) != np.float64:
				# find the level dimension
				nlev = len(lev)
				for dimlength,idim in zip(VV.shape,len(VV.shape)):
					if dimlength == nlev:
						levdim = idim
				M1 = np.mean(VV,axis=levdim)
			else:
				M1 = np.squeeze(VV)
			# if computing a difference to another field, load that here  
			if (Ediff != None):
				Ediff['copystring'] = 'ensemble member '+str(iens+1)
				Vmatrix,lat,lon,lev,DRnew = DART_diagn_to_array(Ediff,hostname=hostname,debug=debug)
				if len(DRnew) > 1:
					VV = np.nanmean(Vmatrix,axis=len(Vmatrix.shape)-1)	
				else:
					VV = Vmatrix
				# average over vertical levels  if the variable is 3D
				if E['variable'] in var3d and type(lev) != np.float64:
					M2 = np.mean(VV,axis=levdim)
				else:
					M2 = np.squeeze(VV)
				# subtract the difference field out from the primary field  
				M = M1-M2
			else:
				M = M1
			
			# store the difference (or plain M1 field) in a list 
			Mlist.append(M)

		# turn the list of averaged fields into a matrix, where ensemble index is the first dimension
		Mmatrix = np.concatenate([M[np.newaxis,...] for M in Mlist], axis=0)

		# now apply bootstrap over the first dimension, which by construction is the ensemble  
		CI = bs.bootstrap(Mmatrix,stat_sig['nsamples'],np.mean,stat_sig['P'])

		# anomalies are significantly different from 0 if the confidence interval does not cross zero
		# we can estimate this by checking if there is a sign change
		LU = CI.lower*CI.upper
		sig = LU > 0		# this mask is True when CI.lower and CI.upper have the same sign  
		
		# also compute the ensemble average for plotting
		M = np.mean(Mmatrix,axis=0)

	##-----done loading data------------------

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
	coastline_width = 0.25
	if projection == 'miller':
		coastline_width = 1.0
        map.drawcoastlines(linewidth=coastline_width)
		

        # draw lat/lon grid lines every 30 degrees.
        map.drawmeridians(np.arange(0,360,30),linewidth=0.25)
        map.drawparallels(np.arange(-90,90,30),linewidth=0.25)

        # compute native map projection coordinates of lat/lon grid.
        X,Y = np.meshgrid(lon,lat)
        x, y = map(X, Y)

        # choose color map based on the variable in question
	colors,cmap,cmap_type = state_space_HCL_colormap(E,Ediff,reverse=reverse_colors)

	# specify the color limits 
	if clim is None:
		clim = np.nanmax(np.absolute(M))
	if debug:
		print('++++++clim+++++')
		print(clim)

	# set the contour levels - it depends on the color limits and the number of colors we have  
	if cmap_type == 'divergent':
		L  = np.linspace(start=-clim,stop=clim,num=11)
	else:
		L  = np.linspace(start=0,stop=clim,num=11)


        # contour data over the map.
	if (projection == 'ortho') or ('stere' in projection):
		if log_levels is not None:
			cs = map.contourf(x,y,M, norm=mpl.colors.LogNorm(vmin=log_levels[0],vmax=log_levels[len(log_levels)-1]),levels=log_levels,cmap=cmap)
		else:
			cs = map.contourf(x,y,M,levels=L,cmap=cmap,extend="both")
	if projection is 'miller':
		cs = map.contourf(x,y,M,L,cmap=cmap,extend="both")

	if (cbar is not None):
		if (clim > 1000) or (clim < 0.001):
			CB = plt.colorbar(cs, shrink=0.6, extend='both',format='%.1e', orientation=cbar)
		else:
			CB = plt.colorbar(cs, shrink=0.6, extend='both', orientation=cbar)
	if colorbar_label is not None:
		CB.set_label(colorbar_label)

	else:
		CB = None

	# if desired, add shading for statistical significance - this only works for when we plot anomalies
	if stat_sig is not None:
		colors = ["#ffffff","#636363"]
		cmap = mpl.colors.ListedColormap(colors, name='my_cmap')
		map.contourf(x,y,sig,cmap=cmap,alpha=0.3)
	else:
		sig = None

	# return the colorbar handle if available, the map handle, and the data
	return CB,map,M,sig

def plot_diagnostic_hovmoeller(E,Ediff=None,clim=None,cbar='vertical',log_levels=None,hostname='taurus',debug=False,colorbar_label=None):

	"""
	plot a given state-space diagnostic on a Hovmoeller plot, i.e. with time on the y-axis and 
	longitudeo on the x-axis.  
	We can also plot the difference between two fields by specifying another list of Experiment
	dictionaries called Ediff.  

	To plot climatology fields or anomalies wrt climatology, make the field E['diagn'] 'climatology.XXX' or 'anomaly.XXX', 
	where 'XXX' some option for loading climatologies accepted by the subroutine ano in MJO.py (see that code for options)

	INPUTS:  
	log_levels: a list of the (logarithmic) levels to draw the contours on. If set to none, just draw regular linear levels. 

	"""

	# generate an array from the requested diagnostic  
	Vmatrix,lat,lon,lev,DRnew = DART_diagn_to_array(E,hostname=hostname,debug=debug)

	# find the latidue dimension and average over it
	shape_tuple = Vmatrix.shape
	for dimlength,ii in zip(shape_tuple,range(len(shape_tuple))):
		if dimlength == len(lat):
			latdim = ii
	Mlat = np.nanmean(Vmatrix,axis=latdim)

	# if it's a 3d variable, also average over the selected level range  
	if len(shape_tuple) > 3: 
		shape_tuple_2 = Mlat.shape
		for dimlength,ii in zip(shape_tuple_2,range(len(shape_tuple_2))):
			if dimlength == len(lev):
				levdim = ii
		M1 = np.nanmean(Mlat,axis=levdim)
	else:
		M1 = Mlat

	# if computing a difference to another field, load that here  
	if (Ediff != None):
		Vmatrix,lat,lon,lev = DART_diagn_to_array(Ediff,hostname=hostname,debug=debug)

		# find the latidue dimension and average over it
		shape_tuple = Vmatrix.shape
		for dimlength,ii in zip(shape_tuple,range(len(shape_tuple))):
			if dimlength == len(lat):
				latdim = ii
		Mlat = np.nanmean(Vmatrix,axis=latdim)

		# if it's a 3d variable, also average over the selected level range  
		if lev is not None:
			shape_tuple_2 = Mlat.shape
			for dimlength,ii in zip(shape_tuple_2,range(len(shape_tuple_2))):
				if dimlength == len(lev):
					levdim = ii
			M2 = np.nanmean(Mlat,axis=levdim)
		else:
			M2 = Mlat

		# subtract the difference field out from the primary field  
		M = M1-M2
	else:
		M = M1

	#---plot settings----------------
	time = DRnew

        # choose color map based on the variable in question
	colors,cmap,cmap_type = state_space_HCL_colormap(E,Ediff)

	# specify the color limits 
	if clim is None:
		clim = np.nanmax(np.absolute(M))
	if debug:
		print('++++++clim+++++')
		print(clim)

	# set the contour levels - it depends on the color limits and the number of colors we have  
	if cmap_type == 'divergent':
		L  = np.linspace(start=-clim,stop=clim,num=11)
	else:
		L  = np.linspace(start=0,stop=clim,num=11)

        # contour plot 
	MT = np.transpose(M)
        cs = plt.contourf(lon,time,MT,L,cmap=cmap,extend="both")

	# date axis formatting 
	if len(time)>30:
		fmt = mdates.DateFormatter('%b-%d')
		plt.gca().yaxis.set_major_locator(mdates.AutoDateLocator())
		plt.gca().yaxis.set_major_formatter(fmt)
	else:
		fmt = mdates.DateFormatter('%b-%d')
		plt.gca().yaxis.set_major_locator(mdates.AutoDateLocator())
		plt.gca().yaxis.set_major_formatter(fmt)

	if cbar is not None:
		if (clim > 1000) or (clim < 0.001):
			CB = plt.colorbar(cs, shrink=0.8, extend='both',orientation=cbar,format='%.3f')
		else:
			CB = plt.colorbar(cs, shrink=0.8, extend='both',orientation=cbar)
	else: 
		CB = None

	if colorbar_label is not None:
		CB.set_label(colorbar_label)

	#plt.gca().invert_yaxis()
        plt.ylabel('Time')
        plt.xlabel('Longitude')
	#plt.axis('tight')
	return CB,cs,M


def plot_diagnostic_lev_time(E=dart.basic_experiment_dict(),Ediff=None,clim=None,cbar='vertical',colorbar_label=None,reverse_colors=False,scaling_factor=1.0,hostname='taurus',debug=False):

	"""
	Given a DART experiment dictionary E, plot the desired diagnostic as a function of vertical level and time, 
	averaging over the selected latitude and longitude ranges. 

	INPUTS:
	E: experiment dictionary defining the main diagnostic  
	Ediff: experiment dictionary for the difference experiment
	clim: color limits (single number, applied to both ends if the colormap is divergent)
	hostname: name of the computer on which the code is running
	cbar: how to do the colorbar -- choose 'vertical','horiztonal', or None
	reverse_colors: set to True to flip the colormap
	scaling_factor: factor by which to multiply the array to be plotted 
	"""

	# throw an error if the desired variable is 2 dimensional 
	if E['variable'].upper() not in var3d:
		print('Attempting to plot a two dimensional variable ('+E['variable']+') over level and latitude - need to pick a different variable!')
		return

	# load the desired DART diagnostic for the desired variable and daterange:
	Vmatrix,lat,lon,lev,new_daterange = DART_diagn_to_array(E,hostname=hostname,debug=debug)

	# figure out which dimension is longitude and then average over that dimension 
	# unless the data are already in zonal mean, in which case DART_diagn_to_array should have returned None for lon
	shape_tuple = Vmatrix.shape
	if debug:
		print('shape of array after concatenating dates:')
		print shape_tuple
	if lon is not None:
		for dimlength,ii in zip(shape_tuple,range(len(shape_tuple))):
			if dimlength == len(lon):
				londim = ii
		Vlon = np.squeeze(np.mean(Vmatrix,axis=londim))
	else:
		Vlon = np.squeeze(Vmatrix)  
	if debug:
		print('shape of array after averaging out longitude:')
		print Vlon.shape

	# figure out which dimension is longitude and then average over that dimension 
	# unless the data are already in zonal mean, in which case DART_diagn_to_array should have returned None for lon
	shape_tuple = Vlon.shape
	if lat is not None:
		for dimlength,ii in zip(shape_tuple,range(len(shape_tuple))):
			if dimlength == len(lat):
				latdim = ii
		Vlonlat = np.squeeze(np.mean(Vlon,axis=latdim))
	else:
		Vlonlat = Vlon
	if debug:
		print('shape of array after averaging out latitude:')
		print Vlonlat.shape

	# if computing a difference to another field, load that here  
	if (Ediff != None):

		# load the desired DART diagnostic for the difference experiment dictionary
		Vmatrix,lat,lon,lev,new_daterange = DART_diagn_to_array(Ediff,hostname=hostname,debug=debug)

		# average over longitudes 
		if lon is not None:
			Vlon2 = np.squeeze(np.mean(Vmatrix,axis=londim))
		else:
			Vlon2 = np.squeeze(Vmatrix)

		# average over latitudes
		if lat is not None:
			Vlonlat2 = np.squeeze(np.mean(Vlon2,axis=latdim))
		else:
			Vlonlat2 = np.squeeze(Vlon2)

		# subtract the difference field out from the primary field  
		M = Vlonlat-Vlonlat2
	else:
		M = Vlonlat

        # choose color map based on the variable in question
	colors,cmap,cmap_type = state_space_HCL_colormap(E,Ediff,reverse=reverse_colors)

	# set the contour levels - it depends on the color limits and the number of colors we have  
	if clim is None:
		clim = scaling_factor*np.nanmax(np.absolute(M[np.isfinite(M)]))

	if cmap_type == 'divergent':
		L  = np.linspace(start=-clim,stop=clim,num=11)
	else:
		L  = np.linspace(start=0,stop=clim,num=11)

        # contour data 
	t = new_daterange
	if debug:
		print('shape of the array to be plotted:')
		print M.shape
        cs = plt.contourf(t,lev,M*scaling_factor,L,cmap=cmap,extend="both")

	# fix the date exis
	if len(t)>30:
		fmt = mdates.DateFormatter('%b-%d')
		plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
		plt.gca().xaxis.set_major_formatter(fmt)
	else:
		fmt = mdates.DateFormatter('%b-%d')
		plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
		plt.gca().xaxis.set_major_formatter(fmt)
	#plt.xticks(rotation=45)

	# add a colorbar if desired 
	if cbar is not None:
		if (clim > 1000) or (clim < 0.001):
			CB = plt.colorbar(cs, shrink=0.8, extend='both',orientation=cbar,format='%.0e')
		else:
			CB = plt.colorbar(cs, shrink=0.8, extend='both',orientation=cbar)
	else: 
		CB = None

	if colorbar_label is not None:
		CB.set_label(colorbar_label)

        plt.xlabel('time')
        plt.ylabel('Pressure (hPa)')
	plt.yscale('log')
	plt.gca().invert_yaxis()
	plt.axis('tight')
	return cs,CB

def plot_diagnostic_lat_time(E=dart.basic_experiment_dict(),Ediff=None,daterange = dart.daterange(date_start=datetime.datetime(2009,1,1), periods=81, DT='1D'),clim=None,hostname='taurus',cbar=True,debug=False):

	# loop over the input date range
	for date, ii in zip(daterange,np.arange(0,len(daterange))):  


		# load the data over the desired latitude and longitude range  
		if (E['diagn'].lower() == 'covariance') or (E['diagn'].lower() == 'correlation') :
			if ii == 0:
				lev,lat,lon,Cov,Corr = dart.load_covariance_file(E,date,hostname,debug=debug)
				nlat = len(lat)
				refshape = Cov.shape
			else:
				dum1,dum2,dum3,Cov,Corr = dart.load_covariance_file(E,date,hostname,debug=debug)


			if E['diagn'].lower() == 'covariance':
				VV = Cov
			if E['diagn'].lower() == 'correlation':
				VV = Corr
		else:
			if ii == 0:
				lev,lat,lon,VV,P0,hybm,hyam = dart.load_DART_diagnostic_file(E,date,hostname=hostname,debug=debug)
				nlat = len(lat)
				refshape = VV.shape
			else:
				dum1,dum2,dum3,VV,P0,hybm,hyam = dart.load_DART_diagnostic_file(E,date,hostname=hostname,debug=debug)

		# if the file was not found, VV will be undefined, so put in empties
		if VV is None:
			VV = np.empty(shape=refshape)

		# average over latitude and (for 3d variables) vertical levels 
		if (E['variable']=='PS'):
			Mlonlev = np.mean(VV,axis=1)
		else:
			Mlon = np.mean(VV,axis=1)
			Mlonlev = np.mean(Mlon,axis=1)
		

		M1 = Mlonlev


		# repeat for the difference experiment
		if (Ediff != None):
			lev2,lat2,lon2,VV,P0,hybm,hyam = dart.load_DART_diagnostic_file(Ediff,date,hostname=hostname,debug=debug)
			if (E['variable']=='PS'):
				M2lonlev = np.mean(VV,axis=1)
			else:
				M2lon = np.mean(VV,axis=1)
				M2lonlev = np.mean(M2lon,axis=1)
			M2 = M2lonlev
			M = M1-M2
		else:
			M = M1


		# append the resulting vector to the larger array (or initialize it)
		if (ii==0) :
			MM = np.zeros(shape=(nlat, len(daterange)), dtype=float)
			names=[]
		MM[:,ii] = M

	# make a grid of levels and days
	t = daterange

        # choose color map based on the variable in question
	colors,cmap,cmap_type = state_space_HCL_colormap(E,Ediff)


        # contour data over the map.
        cs = plt.contourf(t,lat,MM,len(colors)-1,cmap=cmap,extend="both")
	plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
	plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
	plt.axis('tight')
        if cmap_type == 'divergent':
		if clim is None:
			clim = np.nanmax(np.absolute(MM))
                plt.clim([-clim,clim])
	if debug:
		print(cs.get_clim())
	if cbar:
		if (clim > 1000) or (clim < 0.001):
			CB = plt.colorbar(cs, shrink=0.8, extend='both',orientation='vertical',format='%.3f')
		else:
			CB = plt.colorbar(cs, shrink=0.8, extend='both',orientation='vertical')
	else:
		CB = None
        plt.xlabel('time')
        plt.ylabel('Latitude')

	# fix the date exis
	if len(t)>30:
		fmt = mdates.DateFormatter('%b-%d')
		plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
		plt.gca().xaxis.set_major_formatter(fmt)
	else:
		fmt = mdates.DateFormatter('%b-%d')
		plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
		plt.gca().xaxis.set_major_formatter(fmt)

	return cs,CB

def retrieve_state_space_ensemble(E,averaging=True,include_truth=False,hostname='taurus',debug=False):

	"""
	retrieve the prior or posterior ensemble averaged over some region of the state,
	along with the truth (if desired), 
	for some DART experiment
	
	INPUTS:
	E: standard experiment dictionary 
	averaging: set to True to average over the input latitude, longitude, and level ranges (default=True).
	include_truth: set to True to include the true state for this run. Note that if the truth does not exist but is requested, this 
		subroutine will throw an error. 
	hostname
	debug
	"""

	# query the daterange of E
	daterange = E['daterange']

	# query the ensemble size for this experiment
	N = es.get_ensemble_size_per_run(E['exp_name'])

	# if the input daterange is a single date, we don't have to loop over files
	if not isinstance(daterange,list):
		nT = 1
		sample_date = daterange
	else:
		nT = len(daterange)
		sample_date = daterange[0]

	# initialize an empty array to hold the ensemble
	if averaging:
		VE = np.zeros(shape=(N,nT))+np.NaN
		VT = np.zeros(shape=(1,nT))+np.NaN
	else:
		lev,lat,lon,VV,P0,hybm,hyam = dart.load_DART_diagnostic_file(E,sample_date,hostname=hostname,debug=debug)
		if VV is None:
			return None, None, None, None, None
		nlev = len(lev)
		nlat = len(lat)
		nlon = len(lon)
		if E['variable']=='PS':
			VE = np.zeros(shape=(N,nlat,nlon,nT))+np.NaN
			VT = np.zeros(shape=(1,nlat,nlon,nT))+np.NaN
		else:
			VE = np.zeros(shape=(N,nlev,nlat,nlon,nT))+np.NaN
			VT = np.zeros(shape=(1,nlev,nlat,nlon,nT))+np.NaN
	

	if not isinstance(daterange,list):
		date = daterange
		ii=0

		# load the ensemble over the desired latitude and longitude range, by looping over the ensemble  
		Eens = E.copy()
		Eens['extras'] = 'None'
		
		for iens in np.arange(1,N+1):
			if iens < 10:
				spacing = '      '
			else:
				spacing = '     '
			copystring = "ensemble member"+spacing+str(iens)		
			Eens['copystring'] = copystring
			lev,lat,lon,VV,P0,hybm,hyam = dart.load_DART_diagnostic_file(Eens,date,hostname=hostname,debug=debug)
			# exit if the data are not available  
			if VV is None:
				return None, None, None, None, None

			# average over latitude, longitude, and level  
			if averaging:
				Mlat = np.mean(VV,axis=0)
				Mlatlon = np.mean(Mlat,axis=0)
				if E['variable'] != 'PS':
					Mlatlonlev = np.mean(Mlatlon,axis=0)
				else:
					mlatlonlev = Mlatlon
				VE[iens-1,ii] = np.mean(Mlatlonlev,axis=0)
			else:
				if E['variable'] == 'PS':
					VE[iens-1,:,:,ii] = VV
				else:
					VE[iens-1,:,:,:,ii] = np.transpose(VV,(2,0,1))

		# load the corresponding truth, if desired or if it exists
		if include_truth:
			Etr = E.copy()
			Etr['diagn'] = 'Truth'
			Etr['copystring'] = 'true state'
			lev,lat,lon,VV,P0,hybm,hyam = dart.load_DART_diagnostic_file(Etr,date,hostname=hostname,debug=debug)
			if VV is None:
				VT = None
			else:	
				if averaging:
					Mlat = np.mean(VV,axis=0)
					Mlatlon = np.mean(Mlat,axis=0)
					if E['variable'] != 'PS':
						Mlatlonlev = np.mean(Mlatlon,axis=0)
					else:
						Mlatlonlev = Mlatlon
					VT = np.mean(Mlatlonlev,axis=0)
				else:
					if E['variable'] == 'PS':
						VT[0,:,:,ii] = VV
					else:
						VT[0,:,:,:,ii] = np.transpose(VV,(2,0,1))

	else:

		# loop over the input date range
		for date, ii in zip(daterange,np.arange(0,len(daterange))):  

			# load the ensemble over the desired latitude and longitude range, by looping over the ensemble  
			Eens = E.copy()
			Eens['extras'] = 'None'
			
			for iens in np.arange(1,N+1):
				if iens < 10:
					spacing = '      '
				else:
					spacing = '     '
				copystring = "ensemble member"+spacing+str(iens)		
				Eens['copystring'] = copystring
				lev,lat,lon,VV,P0,hybm,hyam = dart.load_DART_diagnostic_file(Eens,date,hostname=hostname,debug=debug)
				if VV is None:
					# end the loop if we have no data 
					break
				# average over latitude, longitude, and level  
				if averaging:
					Mlat = np.mean(VV,axis=0)
					Mlatlon = np.mean(Mlat,axis=0)
					if E['variable'] != 'PS':
						Mlatlonlev = np.mean(Mlatlon,axis=0)
					else:
						Mlatlonlev = Mlatlon
					#---mistake?---VE[iens-1,ii] = np.mean(Mlatlonlev,axis=0)
					VE[iens-1,ii] = Mlatlonlev
				else:
					if E['variable'] == 'PS':
						VE[iens-1,:,:,ii] = VV
					else:
						VE[iens-1,:,:,:,ii] = np.transpose(VV,(2,0,1))

			# load the truth if it exists 
			if include_truth:
				Etr = E.copy()
				Etr['diagn'] = 'Truth'
				Etr['copystring'] = 'true state'
				lev,lat,lon,VV,P0,hybm,hyam = dart.load_DART_diagnostic_file(Etr,date,hostname=hostname,debug=debug)
				if VV is not None:
					if averaging:
						Mlat = np.mean(VV,axis=0)
						Mlatlon = np.mean(Mlat,axis=0)
						if E['variable'] != 'PS':
							Mlatlonlev = np.mean(Mlatlon,axis=0)
						else:
							Mlatlonlev = Mlatlon
						#---mistake?---VT[0,ii] = np.mean(Mlatlonlev,axis=0)
						VT[0,ii] = Mlatlonlev
					else:
						if E['variable'] == 'PS':
							VT[0,:,:,ii] = VV
						else:
							VT[0,:,:,:,ii] = np.transpose(VV,(2,0,1))
				else:
					VT = None
			else:
				VT = None

		# output
	return VE,VT,lev,lat,lon


def plot_state_space_ensemble(E=None,truth_option='ERA',color_choice=1,hostname='taurus',debug=False,show_legend=False):

	"""
	plot the prior or posterior ensemble averaged over some region of the state,
	along with the truth (if available), 
	for some DART experiment

	input truth_option chooses what to plot as the "truth" to compare the ensemble to: 
	'pmo': plots the reference (or "truth") state, available only for PMO experiments. 
	'ERA': plots corresponding ERA-40 or ERA-Interin data  
	None: plots no true state.  

	input color_choice chooses a different color palette: 
	1 = gray ensemble with black ensemble mean (boring but straightforward)
	2 = "helmholtz" blue (sort of)

	"""

	# retrieve the ensemble
	if truth_option == 'pmo':
		include_truth = True
	else:
		include_truth = False
	VE,VT,lev,lat,lon = retrieve_state_space_ensemble(E=E,averaging=True,include_truth=include_truth,hostname=hostname,debug=debug)

	# retrieve ERA data if desired
	if truth_option=='ERA':
		VT,t_tr,lat2,lon2,lev2 = era.retrieve_era_averaged(E)

	# set up a  time grid 
	t = E['daterange']
	if truth_option=='pmo':
		t_tr = t
		VT = VT[0,:]

	# if no color limits are specified, at least make them even on each side
	# change the default color cycle to colorbrewer colors, which look a lot nicer
	if color_choice == 1:
		bmap = brewer2mpl.get_map('Dark2', 'qualitative', 7)
		color_ensemble = "#878482"
		color_truth = bmap.mpl_colors[3]
		color_mean = "#000000"
	if color_choice == 2:
		bmap = brewer2mpl.get_map('YlGnBu', 'sequential', 9)
		color_ensemble = bmap.mpl_colors[4]
		color_mean = bmap.mpl_colors[7]
		color_truth = "#000000"

        # plot global diagnostic in in time
	N = VE.shape[0]
	VM = np.mean(VE,axis=0)
	cs = plt.plot(t,VE[0,:],color=color_ensemble,label='Ensemble')
	for iens in np.arange(1,N):
		cs = plt.plot(t,VE[iens,:],color=color_ensemble,label='_nolegend_')
	plt.hold(True)
	if truth_option is not None:
		cs = plt.plot(t_tr,VT,color=color_truth,linewidth=2.0,label='Truth')
	plt.plot(t,VM,color=color_mean,label='Ensemble Mean')

	# show a legend if desired
	if show_legend:
		lg = plt.legend(loc='best')
		lg.draw_frame(False)
	else: 
		lg=None

	clim = E['clim']
	if E['clim'] is not None:
		plt.ylim(clim)
        plt.xlabel('time')

	# format the y-axis labels to be exponential if the limits are quite high
	if clim is not None:
		if (clim[1] > 100):
			ax = plt.gca()
			ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

	# format the x-axis labels to be dates
	if len(t) > 30:
		#plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1,interval=1))
		plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
		#plt.gca().xaxis.set_minor_locator(mdates.DayLocator(interval=1))
	if len(t) < 10:
		plt.gca().xaxis.set_major_locator(mdates.DayLocator(bymonthday=range(len(t))))
	fmt = mdates.DateFormatter('%b-%d')
	plt.gca().xaxis.set_major_formatter(fmt)

	return VE,VT,t,lg

def plot_diagnostic_global_ave(EE=[],EEdiff=None,ylim=None,xlim=None,include_legend=True,colors=None,linestyles=None,markers=None,x_as_days=False,hostname='taurus',debug=False):

	"""
	plot a given state-space diagnostic for a given variable field,
	as a function of time only (averaging spatially)  
	We can also plot the difference between two fields by specifying another experiment structure 
	called Ediff  

	INPUTS:
	EE: a list of experiment dictionaries to loop over an plot
	EEdiff: a list of experiments to subtract from the experiments in EE
	ylim: y-limits of the figure
	xlim: x-limits of the figure - note that this is tricky if we use dates instead of numbers 
	include_legend: set to False to get rid of the legennd (default is True)
	colors: input a list of hex codes that give the colors of the experiments to plot 
		the default is "None" -- in this case, choose Colorbrewer qualitative colormap "Dark2"
	linestyles: input a list of linestyle strings that give the styles for each line plotted. 
		the default is "None" - in this case, all lines are plotted as plain lines  
	markers: input a list of marker strings that give the markers for each line plotted. 
		the default is "None" - in this case, all lines are plotted as plain lines  
	x_as_days: set to True to plot a count of days on the x-axis rather than dates

	"""

	# set up an array of global averages that's the length of the longest experiment  
	DR_all = []
	for E in EE:
		DR_all.append(len(E['daterange']))
	max_length_time = max(DR_all)
	nE = len(EE)
	MM = np.zeros(shape=(nE, max_length_time), dtype=float)

	# also set up an array that holds the day count for each experiment  
	if x_as_days:
		x = np.zeros(shape=(nE, max_length_time), dtype=float)
		x[:,:] = np.NAN
	else: 
		x = E['daterange']

	# loop over experiment dictionaries and load the timeseries of the desired diagnostic
	names = []
	for iE,E in zip(range(nE),EE):

		# store the name of this experiment
		names.append(E['title'])

		# for each experiment loop over the input date range
		for date, ii in zip(E['daterange'],range(len(E['daterange']))):  

			# fill in the day count (if desired) 
			if x_as_days:
				dt = date-E['daterange'][0]	
				dtfrac = dt.days + dt.seconds/(24.0*60.0*60.0)
				x[iE,ii] = dtfrac

			# load the data over the desired latitude and longitude range  
			lev,lat,lon,VV,P0,hybm,hyam = dart.load_DART_diagnostic_file(E,date,hostname=hostname,debug=debug)

			# compute global average only if the file was found
			if VV is not None:
				# average over latitude, longitude, and level  
				Mlat = np.mean(VV,axis=0)
				Mlatlon = np.mean(Mlat,axis=0)
				if E['variable'] in var3d:
					Mlatlonlev = np.mean(Mlatlon,axis=0)
				if E['variable'] in var2d:
					Mlatlonlev = Mlatlon
				M1 = Mlatlonlev

				# repeat for the difference experiment
				if (EEdiff != None):
					Ediff = EEdiff[iE]
					lev2,lat2,lon2,VV,P0,hybm,hyam = dart.load_DART_diagnostic_file(Ediff,date,hostname=hostname,debug=debug)
					if VV is not None:
						M2lat = np.mean(VV,axis=0)
						M2latlon = np.mean(M2lat,axis=0)
						if E['variable'] in var3d:
							M2latlonlev = np.mean(M2latlon,axis=0)
						if E['variable'] in var2d:
							M2latlonlev = M2latlon
						M2 = M2latlonlev
						M = M1-M2
					else:
						M = np.NAN
				else:
					M = M1
			else:
				# if no file was found, just make the global average a NAN
				M = np.NAN

			# append the resulting vector to the larger array (or initialize it)
			MM[iE,ii] = M


	#------plotting----------

	# change the default color cycle to colorbrewer Dark2, or use what is supplied
	if colors is None:
		bmap = brewer2mpl.get_map('Dark2', 'qualitative', 7)
		colors = bmap.mpl_colors

	# set all line styles to a plain line if not previous specified  
	if linestyles == None:
		linestyles = ['-']*nE

	# set all markers to None unless previously specified  
	if markers is None:
		markers = [None]*nE

        # plot global diagnostic in in time
	MT = np.transpose(MM)
	if x_as_days:
		xT = np.transpose(x)
	for iE in np.arange(0,nE):
		y0 = MT[:,iE]
		y = y0[~np.isnan(y0)]
		if x_as_days:
			x0 = xT[:,iE]
			x = x0[~np.isnan(y0)]
		else:
			x = E['daterange']
		cs = plt.plot(x,y,color=colors[iE],linewidth=2,linestyle=linestyles[iE],marker=markers[iE])

	# include legend if desire
	if include_legend:
		lg = plt.legend(names,loc='best')
		lg.draw_frame(False)

        plt.xlabel('Time (Days)')
	if ylim is not None:
		plt.ylim(ylim)
	if xlim is not None:
		plt.xlim(xlim)

	# format the y-axis labels to be exponential if the limits are quite high
	if (ylim > 100):
		ax = plt.gca()
		ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

	if not x_as_days:
		# format the x-axis labels to be dates
		if len(x) > 30:
			plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
		if len(x) < 10:
			plt.gca().xaxis.set_major_locator(mdates.DayLocator(bymonthday=range(len(t))))
		fmt = mdates.DateFormatter('%b-%d')
		plt.gca().xaxis.set_major_formatter(fmt)

	return MT,x


def state_space_colormap(E,Ediff=None):

	# appropriate color maps for state space plots
	cmap_sequential = False

	# for square error plots, we want a sequential color map, but only if not taking a difference
	if (E['extras']=='MSE')and (Ediff == None): 
		cmap_sequential = True

	# for ensemble spread plots, we want a sequential color map, but only if not taking a diff
	if (E['copystring']=='ensemble spread') and (Ediff == None): 
		cmap_sequential = True


	# choose sequential or diverging colormap
	if cmap_sequential:
		bmap = brewer2mpl.get_map('GnBu', 'sequential', 9)
		#---for newer matplotlib---cmap = bmap.get_mpl_colormap(N=15, gamma=4.0)
		cmap = bmap.get_mpl_colormap(N=9)
		if debug:
			print('loading a sequential colormap')
	else:
		bmap = brewer2mpl.get_map('PiYG', 'diverging', 11)
		#---for newer matplotlib---cmap = bmap.get_mpl_colormap(N=15, gamma=1.0)
		cmap = bmap.get_mpl_colormap(N=9)
		if debug:
			print('loading a diverging colormap')


	return cmap

def state_space_HCL_colormap(E,Ediff=None,reverse=False,ncol=18,debug=False):

	"""
	loads colormaps (not a matplotlib colormap, but just a list of colors)
	based on the HCL theory put forth in Stauffer et al 2014
	other sample color maps are available here:  http://hclwizard.org/why-hcl/  

	INPUTS:
	E: a DART experiment dictionary. Relevant entries are:
		variable: if the variable is a positive definite quantity, use a sequential colormap  
		extras: if this indicates a postitive definite quantity, use a sequential colormap  
	Ediff: the differenence experiment dictionary. Used to determine if we are taking a difference, in 
		which case we would want a divergent colormap. 
	reverse: flip the colormap -- default is False 
	ncol: how many colors? Currently only 11 and 18 are supported for divergent maps, and only 11 for 
		sequential maps. Default is 18. 
	"""

        # appropriate color maps for state space plots
        colors_sequential = False

	# sequential plot if plotting positive definite variables and not taking a difference  
	post_def_variables = ['Z3','PS','FLUT','T','Nsq']
	if (Ediff == None) and (E['variable'] in post_def_variables):
                colors_sequential = True

        # for square error plots, we want a sequential color map, but only if not taking a difference
        if (E['extras']=='MSE')and (Ediff == None):
                colors_sequential = True

        # for ensemble spread plots, we want a sequential color map, but only if not taking a diff
        if (E['copystring']=='ensemble spread') and (Ediff == None):
                colors_sequential = True

        # for ensemble variance plots, we want a sequential color map, but only if not taking a diff
        if (E['extras']=='ensemble variance scaled') and (Ediff == None):
                colors_sequential = True

	# if plotting the MJO variance, wnat a sequential colormap
	if (E['extras'] == 'MJO variance'):
                colors_sequential = True

	# if the diagnostic includes a climatological standard deviation, turn on sequential colormap
	if 'climatological_std' in E['diagn']:
                colors_sequential = True

	# if any of the above turned on the sequential colormap but we are looking at anomalies or correlations, turn it back off  
	if E['extras'] is not None:
		if 'Correlation' in E['extras']:
			colors_sequential = False
	if 'anomaly' in E['diagn']:
                colors_sequential = False

        # choose sequential or diverging colormap
        if colors_sequential:
		# yellow to blue
		colors = ("#F4EB94","#CEE389","#A4DA87","#74CF8C","#37C293","#00B39B",
			  "#00A1A0","#008CA1","#00749C","#005792","#202581")

		if debug:
			print('loading a sequential HCL colormap')
		type='sequential'
        else:
		#---red negative and blue positive with white center instead of gray--
		colordict = {11:("#D33F6A","#DB6581","#E28699","#E5A5B1","#E6C4C9","#FFFFFF","#FFFFFF","#C7CBE3","#ABB4E2","#8F9DE1","#7086E1","#4A6FE3"),
				 19:("#D33F6A","#DA5779","#E26C88","#E88197","#EE94A7","#F3A8B6",
					  "#F7BBC6","#FBCED6","#FDE2E6","#FFFFFF","#FFFFFF","#E4E7FB",
					  "#D3D8F7","#C1C9F4","#AFBAF1","#9DABED","#8B9CEA","#788DE6",
					  "#637EE4","#4A6FE3")}
		colors=colordict[ncol]

		if debug:
			print('loading a diverging HCL colormap')
		type='divergent'

	if reverse:
		colors = colors[::-1]

	cmap = mpl.colors.ListedColormap(colors, name='my_cmap')

        return colors,cmap,type



def compute_rank_hist(E=dart.basic_experiment_dict(),daterange=dart.daterange(datetime.datetime(2009,1,1),10,'1D'),space_or_time='both',hostname='taurus'):

	# given some experiment E, isolate the ensemble at the desired location  
	# (given by E's entried latrange, lonrange, and levrange), retrieve 
	# the truth at the same location, and compute a rank histogram over the 
	# desired date range.  
	# 
	# the paramter space_or_time determines whether we count our samples over a blog of time, or in space 
	# if the choice is 'space', the time where we count is the first date of the daterange
	if (space_or_time == 'space'):
		dates = daterange[0]
		averaging = False

	if (space_or_time == 'time'):
		averaging = True
		dates = daterange
	
	if (space_or_time == 'both'):
		averaging = False
		dates = daterange


	# loop over dates and retrieve the ensemble
	VE,VT,lev,lat,lon = retrieve_state_space_ensemble(E,dates,averaging,hostname)

	# from this compute the rank historgram
	bins,hist = dart.rank_hist(VE,VT[0,:])

	return bins,hist,dates


def plot_rank_hist(E=dart.basic_experiment_dict(),daterange=dart.daterange(datetime.datetime(2009,1,1),81,'1D'),space_or_time='space',hostname='taurus'):


	# compute the rank historgram over the desired date range
	bins,hist,dates = compute_rank_hist(E,daterange,space_or_time,hostname)

	# plot the histogram
	plt.bar(bins,hist,facecolor='#9999ff', edgecolor='#9999ff')
	plt.axis('tight')
	plt.xlabel('Rank')

	return bins,hist,dates

def compute_state_to_obs_covariance_field(E=dart.basic_experiment_dict(),date=datetime.datetime(2009,1,1),obs_name='ERP_LOD',hostname='taurus'):

	# Given a DART experiment, load the desired state-space diagnostic file and corresponding obs_epoch_XXX.nc file,
	# and then compute the field of covariances between every point in the field defined by latrange, lonrange, and levrange
	# (these are entries in the experiment dictionary, E), and the scalar observation.

	# first load the entire ensemble for the desired variable field
	#lev,lat,lon,VV = dart.load_DART_diagnostic_file(E,date,hostname)
        VV,VT,lev,lat,lon = retrieve_state_space_ensemble(E,date,False,hostname)

	# now load the obs epoch file corresponding to this date
	obs,copynames = dart.load_DART_obs_epoch_file(E,date,[obs_name],['ensemble member'], hostname)
 
	# compute the ensemble mean value for each point in the variable field
	VM = np.mean(VV,0)

	# compute the mean and standard deviation of the obs predicted by the ensemble
	obsM = np.mean(obs)
	eobs = obs-obsM
	sobs = np.std(obs)
	

	# 3D variables: loop over lev, lat, lon and compute the covariance with the observation
	if len(VV.shape) == 5:
		[N,nlev,nlat,nlon,nt] = VV.shape
		C = np.zeros(shape=(nlat,nlon,nlev,nt))
		R = C.copy()
		for ilev in range(nlev):
			for ilat in range(nlat):
				for ilon in range(nlon):
					Ctemp = np.zeros(shape = (1,N))
					for ii in range(N):
						dx = VV[ii,ilev,ilat,ilon,:]-VM[ilev,ilat,ilon,:]
						Ctemp[0,ii] = dx*eobs[0,ii]
					C[ilat,ilon,ilev,:] = np.mean(Ctemp)/(float(N)-1.)
					sx = np.std(VV[:,ilev,ilat,ilon,0])
					R[ilat,ilon,ilev,:] = C[ilat,ilon,ilev,:]/(sx*sobs)
						

	# 2D variables: loop over  lat and lon and compute the covariance with the observation
	if len(VV.shape) == 4:
		lev = np.nan
		[N,nlat,nlon,nt] = VV.shape
		C = np.zeros(shape=(nlat,nlon,nt))
		R = C.copy()
		for ilat in range(nlat):
			for ilon in range(nlon):
				c = 0
				for ii in range(N):
					dx = VV[ii,ilat,ilon,:]-VM[ilat,ilon,:]
					C[ilat,ilon,:] += (1/(N-1))*dx*eobs[0,ii]
				sx = np.std(VV[:,ilat,ilon,:])
				R[ilat,ilon,:] = C[ilat,ilon,:]/(sx*sobs)

	return C,R,lev,lat,lon


def make_state_to_obs_covariance_file(E,date=datetime.datetime(2009,1,1,0,0,0),obs_name='ERP_LOD',hostname='taurus'):

	# run through a set of DART runs and dates and compute the covariances between the state variables  
	# and a given observation, then save it as a netcdf file  

	# Compute the covariance and correlation fields
	C, R, lev0,lat0,lon0 = compute_state_to_obs_covariance_field(E,date,obs_name,hostname)

	# compute the gregorian day number for this date
	# note: we can also go higher res and return the 12-hourly analysis times, but that requires changing several other routines
	dt = date - datetime.datetime(1600,1,1,0,0,0)
	time0 = dt.days

	# save a netcdf file for each date and observation variable
	fname = E['exp_name']+'_'+'covariance_'+obs_name+'_'+E['variable']+'_'+date.strftime('%Y-%m-%d')+'.nc'
	ff = Dataset(fname, 'w', format='NETCDF4')
	lat = ff.createDimension('lat', len(lat0))
	lon = ff.createDimension('lon', len(lon0))
	time = ff.createDimension('time', 1)
	longitudes = ff.createVariable('lon','f4',('lon',))
	latitudes = ff.createVariable('lat','f4',('lat',))
	times = ff.createVariable('time','f4',('time',))
	if E['variable']=='PS':
		covar = ff.createVariable('Covariance','f8',('lat','lon','time'))
		correl = ff.createVariable('Correlation','f8',('lat','lon','time'))
	else:
		lev = ff.createDimension('lev', len(lev0))
		levels = ff.createVariable('lev','f4',('lev',))
		covar = ff.createVariable('Covariance','f8',('lat','lon','lev','time'))
		correl = ff.createVariable('Correlation','f8',('lat','lon','lev','time'))

	# fill in the variables
	latitudes[:] = lat0
	longitudes[:] = lon0
	times[:] = time0
	if E['variable']=='PS':
		covar[:,:,:] = C
		correl[:,:,:] = R
	else:
		levels[:] = lev0
		covar[:,:,:,:] = C
		correl[:,:,:,:] = R

	# add file attributes
	ff.description = 'Covariance and Correlatin between variable field '+E['variable']+' and observation ',obs_name
	ff.history = 'Created ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	ff.source = 'Python modlue DART_state_space.py'  
	latitudes.units = 'degrees north'
	longitudes.units = 'degrees west'
	times.units = 'Days since 1601-01-01'  
	if not E['variable']=='PS':
		levels.units = 'hPa'

	# close the file
	ff.close()
	print('Created file '+fname)
	
def compute_aefs_as_csv(E = dart.basic_experiment_dict(),date=datetime.datetime(2009,1,1),hostname='taurus',debug=False):

	# given a DART experiment, compute the three AEF excitation functions, and save as a csvfile  

	# list of excitation functions to compute
	AEFlist = ['X1','X2','X3']

	# list of the 3 variables which contribute to AAM excitation
	variable_list = ['US','VS','PS']

	# figure out which copy strings are in the state space vector
	copystring_list = es.get_expt_CopyMetaData_state_space(E)

	# initialize lists for the three AEFs
	X = []
	aef_name_list = []
	copystring_list_long = []

	# loop over the AEFs and compute each one  
	for AEF in AEFlist:

		# cycle over all the copies that are available in the state space vector
		for copystring in copystring_list:
			print('+++computing AEF '+AEF+' for '+copystring+' for experiment '+E['exp_name']+'------')

			E['copystring'] = copystring
			dum,Xtemp = aef_from_model_field(E,date,variables=variable_list,ERP=AEF,levels_mistake=False,integral_type='mass',hostname=hostname)
			X.append(sum(Xtemp))  
			aef_name_list.append(AEF)
			copystring_list_long.append(copystring)

	# also make columns for experiment name, diagnostic, and date
	nC = len(copystring_list)
	nAEF = len(AEFlist)
	datelist = np.repeat(date,nC*nAEF)
	exp_name_list = np.repeat(E['exp_name'],nC*nAEF)
	diagn_list = np.repeat(E['diagn'],nC*nAEF)

	# now stick every thing into a dictionary
	D = {'time':datelist,
		'experiment':exp_name_list,
		'diagnostic':diagn_list,
		'copystring':copystring_list_long,
		'AEF':X,
		'Parameter_Name':aef_name_list}

	# turn the dictionary into a pandas dataframe and export to a csv file
	DF = pd.DataFrame(D)
	file_out_name = E['exp_name']+'_'+'AEFs_'+E['diagn']+'_'+date.strftime('%Y-%m-%d-%H-%M-%S')+'.csv'
	DF.to_csv(file_out_name, sep='\t')
	print('Created file '+file_out_name)



def aef_from_model_field(E = dart.basic_experiment_dict(),date=datetime.datetime(2009,1,1),variables=['U'],ERP='X3',levels_mistake=False,integral_type='mass',hostname='taurus',debug=False):

	# given a DART experiment dictionary and a date, retrieve the model field for that day 
	# and compute the AAM excitation function (AEF) from that field.
	# the keyword levels_mistake is set to true to simulate a possible code mistake where pressure levels were flipped the wrong way 
	# relative to the wind fields

	# the  output AEFs will be in a list, corresponding to the variables given as input
	Xout = []

	# cycle over the input list of variables and compute the AEF for each one
	for variable in variables:
		# retrieve the desired field for tha day
		E['variable'] = variable
		if variable == 'U':
			E['variable'] = 'US'
		if variable == 'V':
			E['variable'] = 'VS'

		# load the variable field
		lev,lat,lon,VV,P0,hybm,hyam = dart.load_DART_diagnostic_file(E,date,hostname=hostname,debug=debug)

		# if doing the mass integral, we have to recreate the 3d pressure field from hybrid model levels
		if (integral_type is 'mass'):

			# recreate the 3D pressure field
			E2 = E.copy()
			E2['variable'] = 'PS'
			dum1,latps,lonps,PS,dum4,dum5,dum6 = dart.load_DART_diagnostic_file(E2,date,hostname=hostname,debug=debug)
			nlev = len(lev)
			nlat = len(latps)
			nlon = len(lonps)
			P = np.zeros(shape=(nlat,nlon,nlev))
			for k in range(nlev):
				for j in range(nlat):
					for i in range(nlon):
						dum = None
						P[j,i,k] = hyam[k]*P0[0] + hybm[k] * PS[j,i]
					
			
			# compute the integral
			#Xtemp = erp.aef_massintegral(field=VV,PS=PS,p=P,lat=lat,lon=lon,variable_name=variable,ERP=ERP)
			Xtemp = erp.aef_massintegral(VV=VV,PS=PS,p=P,lat=latps,lon=lonps,variable_name=variable,ERP=ERP)

		# if doing a volume integral, we need to make sure the levels array is in Pascal 
		if (integral_type is 'volume'):
			lev_Pa = lev*100
			# simulate a flipped levels error if desired
			if levels_mistake:
				lev_Pa = np.flipud(lev_Pa)

			# compute the integral
			Xtemp = erp.aef(field=VV,lev=lev_Pa,lat=lat,lon=lon,variable_name=variable,ERP=ERP)

		# append integrated value to the list
		Xout.append(Xtemp)


	# temporary test plot
	#plt.figure(1)
	#dm = stuff[94,142,:]
	#dm = stuff[2,2,:]
	#plt.plot(range(nlev),dm)
        #fig_name = 'test.pdf'
        #plt.savefig(fig_name, dpi=96)
        #plt.close(1)



	# return ouput
	return variables,Xout


def plot_compare_AEFintegrals_to_obs(E = dart.basic_experiment_dict(),daterange = dart.daterange(periods=30),ERP='X1',hostname='taurus'):

	# for a given DART experiment, integrate the dynamical fields to get AAM excitation functions (AEFs), 
	# and compare these to the AEF observations produced by the obs operator (obs_def_eam.f90) 
	#
	# this is mostly to check that the AEF operator was coded correctly.  
	E['copystring'] = 'ensemble member     29'
	variables=['U','V','PS']
	X = np.zeros(shape=(4,len(daterange)))
	Xbad = np.zeros(shape=(4,len(daterange)))
	Y = np.zeros(shape=(1,len(daterange)))

	# choose the observation name that goes with the desired ERP
	if ERP == 'X1':
		obs_name = 'ERP_PM1'
	if ERP == 'X2':
		obs_name = 'ERP_PM2'
	if ERP == 'X3':
		obs_name = 'ERP_LOD'

	# loop over the daterange and create timeseries of the AEFs, 
	# and load the corresponding observations
	for date,ii in zip(daterange,range(len(daterange))):
		vars,XX = aef_from_model_field(E,date,variables,ERP,False,hostname)
		vars,XXbad = aef_from_model_field(E,date,variables,ERP,True,hostname)
		X[0,ii] = XX[0]
		X[1,ii] = XX[1]
		X[2,ii] = XX[2]
		X[3,ii] = sum(XX)
		Xbad[0,ii] = XXbad[0]
		Xbad[1,ii] = XXbad[1]
		Xbad[2,ii] = XXbad[2]
		Xbad[3,ii] = sum(XXbad)

		# load the corresponding observation
		obs,cs = dart.load_DART_obs_epoch_file(E,date,[obs_name],None, hostname)
		Y[0,ii] = obs


	# plot it and export as pdf
        plt.close('all')
        plt.figure(1)
        plt.clf()

	ax1 = plt.subplot(121)
        t = [d.date() for d in daterange]
        bmap = brewer2mpl.get_map('Dark2', 'qualitative', 7)
	plt.plot(t,Y[0,:],color=bmap.mpl_colors[0])
        plt.hold(True)
	plt.plot(t,X[3,:],color=bmap.mpl_colors[1])
	plt.plot(t,Xbad[3,:],color=bmap.mpl_colors[2])
        plt.legend(['EAM Code','My integral','Integral with flipped p levels'],loc='best')


	ax2 = plt.subplot(121)
	plt.plot(t,Y[0,:]-np.mean(Y[0,:]),color=bmap.mpl_colors[0])
        plt.hold(True)
	plt.plot(t,X[0,:]-np.mean(X[0,:]),color=bmap.mpl_colors[1])
	plt.plot(t,Xbad[0,:]-np.mean(Xbad[0,:]),color=bmap.mpl_colors[2])
        plt.legend(['EAM Code Anomaly','U integral anomaly','U integral anom with error'],loc='best')

        fig_name = 'EAM_obs_operator_error_check_'+ERP+'.pdf'
        plt.savefig(fig_name, dpi=96)
        plt.close()


	#return X,Y,t,vars

def retrieve_obs_space_ensemble(E=dart.basic_experiment_dict(),daterange = dart.daterange(date_start=datetime.datetime(2009,1,1), periods=5, DT='1D'),averaging=True,hostname='taurus'):

	# retrieve the prior or posterior ensemble for some observation, given by E['obs_name'],
	# along with the truth (if available), 
	# for some DART experiment

	# NOTE: so far I'm just writing this for Earth rotation parameter obs, which have no spatial location -- 
	# still need to expand the code for spatially-distributed obs, and potentially add averaging  

	# query the ensemble size for this experiment
	N = dart.get_ensemble_size_per_run(E['exp_name'])

	# if the input daterange is a single date, we don't have to loop over files
	nT = len(daterange)
	sample_date = daterange[0]

	# initialize an empty array to hold the ensemble
	VE = np.zeros(shape=(N,nT))
	VT = np.zeros(shape=(1,nT))

	# loop over the input date range
	for date, ii in zip(daterange,np.arange(0,len(daterange))):  

		# load the ensemble  
		obs_ensemble,copynames = dart.load_DART_obs_epoch_file(E,date,[E['obs_name']],['ensemble member'], hostname)
		VE[:,ii] = obs_ensemble

		# load the true state  
		Etr = E.copy()
		Etr['diagn'] = 'Truth'
		obs_truth,copynames = dart.load_DART_obs_epoch_file(Etr,date,[Etr['obs_name']],['Truth'], hostname)
		print(obs_truth)
		VT[0,ii] = obs_truth

	# output
	return VE,VT

def plot_obs_space_ensemble(E = dart.basic_experiment_dict(),daterange = dart.daterange(periods=30),clim=None,hostname='taurus'):

	# plot the prior or posterior ensemble averaged over some region of the state,
	# along with the truth (if available), 
	# for some DART experiment

	# retrieve the ensemble
	VE,VT = retrieve_obs_space_ensemble(E=E,daterange=daterange,hostname=hostname)
#	if E['exp_name'] == 'PMO32':

	# compute the ensemble mean
	VM = np.mean(VE,axis=0)

	# set up a  time grid 
	t = daterange

	# if no color limits are specified, at least make them even on each side
	# change the default color cycle to colorbrewer colors, which look a lot nicer
	bmap = brewer2mpl.get_map('Dark2', 'qualitative', 7)
	colors = bmap.mpl_colors

        # plot global diagnostic in in time
	N = VE.shape[0]
	for iens in np.arange(0,N):
		cs = plt.plot(t,VE[iens,:],color="#878482")
	plt.hold(True)
	cs = plt.plot(t,VT[0,:],color=bmap.mpl_colors[3])
	cm = plt.plot(t,VM,color="#000000")
	#lg = plt.legend(names,loc='best')
	#lg.draw_frame(False)

	if clim is not None:
		plt.ylim([-clim,clim])
        plt.xlabel('time')

	# format the y-axis labels to be exponential if the limits are quite high
	if (clim > 100):
		ax = plt.gca()
		ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

	# format the x-axis labels to be dates
	if len(t) > 30:
		#plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1,interval=1))
		plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
		plt.gca().xaxis.set_minor_locator(mdates.DayLocator(interval=1))
	if len(t) < 10:
		plt.gca().xaxis.set_major_locator(mdates.DayLocator(bymonthday=range(len(t))))
	fmt = mdates.DateFormatter('%b-%d')
	plt.gca().xaxis.set_major_formatter(fmt)

def plot_diagnostic_lon_time(E=dart.basic_experiment_dict(),Ediff=None,clim=None,hostname='taurus',cbar=True,debug=False):

	# loop over the input date range
	daterange=E['daterange']
	for date, ii in zip(daterange,np.arange(0,len(daterange))):  


		# load the data over the desired latitude and longitude range  
		if (E['diagn'].lower() == 'covariance') or (E['diagn'].lower() == 'correlation') :
			if ii == 0:
				lev,lat,lon,Cov,Corr = dart.load_covariance_file(E,date,hostname,debug=debug)
				nlat = len(lat)
				refshape = Cov.shape
			else:
				dum1,dum2,dum3,Cov,Corr = dart.load_covariance_file(E,date,hostname,debug=debug)


			if E['diagn'].lower() == 'covariance':
				VV = Cov
			if E['diagn'].lower() == 'correlation':
				VV = Corr
		else:
			if ii == 0:
				lev,lat,lon,VV,P0,hybm,hyam = dart.load_DART_diagnostic_file(E,date,hostname=hostname,debug=debug)
				nlon = len(lon)
				refshape = VV.shape
			else:
				dum1,dum2,dum3,VV,P0,hybm,hyam = dart.load_DART_diagnostic_file(E,date,hostname=hostname,debug=debug)

		# if the file was not found, VV will be undefined, so put in empties
		if VV is None:
			VV = np.empty(shape=refshape)

		# average over latitude and (for 3d variables) vertical levels 
		if (E['variable']=='PS'):
			Mlatlev = np.mean(VV,axis=0)
		else:
			Mlat = np.mean(VV,axis=0)
			Mlatlev = np.mean(Mlat,axis=1)
		

		M1 = Mlatlev


		# repeat for the difference experiment
		if (Ediff != None):
			lev2,lat2,lon2,VV,P0,hybm,hyam = dart.load_DART_diagnostic_file(Ediff,date,hostname=hostname,debug=debug)
			if (E['variable']=='PS'):
				M2latlev = np.mean(VV,axis=0)
			else:
				M2lat = np.mean(VV,axis=0)
				M2latlev = np.mean(M2lat,axis=1)
			M2 = M2latlev
			M = M1-M2
		else:
			M = M1


		# append the resulting vector to the larger array (or initialize it)
		if (ii==0) :
			MM = np.zeros(shape=(nlon, len(daterange)), dtype=float)
			names=[]
		MM[:,ii] = M

	# make a grid of levels and days
	#day = daterange.dayofyear
	#t = [d.date() for d in daterange]
	t = daterange

        # choose color map based on the variable in question
	#cmap = state_space_colormap(E,Ediff)
	colors,cmap,cmap_type = state_space_HCL_colormap(E,Ediff)


        # contour data over the map.
        #cs = plt.contourf(t,lat,MM,15,cmap=cmap)
        #cs = plt.contourf(t,lat,MM,len(colors)-1,colors=colors)
	MT = np.transpose(MM)
        cs = plt.contourf(lon,t,MT,len(colors)-1,cmap=cmap,extend="both")
	plt.axis('tight')
        if cmap_type == 'divergent':
		if clim is None:
			clim = np.nanmax(np.absolute(MM))
                plt.clim([-clim,clim])
	print(cs.get_clim())
	if cbar:
		if (clim > 1000) or (clim < 0.001):
			CB = plt.colorbar(cs, shrink=0.8, extend='both',orientation='vertical',format='%.3f')
		else:
			CB = plt.colorbar(cs, shrink=0.8, extend='both',orientation='vertical')
	else:
		CB = None
        plt.ylabel('time')
        plt.xlabel('Longitude')

	# fix the date exis
	if len(t)>30:
		fmt = mdates.DateFormatter('%b-%d')
		plt.gca().yaxis.set_major_locator(mdates.AutoDateLocator())
		#plt.gca().yaxis.set_major_locator(mdates.MonthLocator())
		#plt.gca().yaxis.set_minor_locator(mdates.DayLocator())
		plt.gca().yaxis.set_major_formatter(fmt)
	else:
		fmt = mdates.DateFormatter('%b-%d')
		plt.gca().yaxis.set_major_locator(mdates.AutoDateLocator())
		#plt.gca().yaxis.set_minor_locator(mdates.DayLocator())
		plt.gca().yaxis.set_major_formatter(fmt)

	return cs,CB

def read_aefs_from_csv_to_dataframe(E=dart.basic_experiment_dict(), hostname='taurus', debug=False):

	#read in pre-computed angular momentum excitation functions (AEFs) for a DART run defined in the dictionary E
	# the AEFs are stored in csv files computed using the subroutine compute_aefs_as_csv

	# find the file path for the given experiment
	if E['run_category'] == None:
		path_list,dum = dart.exp_paths(hostname,E['exp_name'])
	if E['run_category'] == 'NCAR':
		path,dum = dart.exp_paths_NCAR(hostname,E['exp_name'])
		path_list = [path]
	if E['run_category'] == 'ERPDA':
		path_list,dum = dart.exp_paths_old(hostname,E['exp_name'])

	# select the first date and figure out which directory has the files we need in it  
	date = E['daterange'][0]
	correct_filepath_found = False
	import os.path
	for path in path_list:
		if debug:
			print path
		ff = E['exp_name']+'_'+'AEFs_'+E['diagn']+'_'+date.strftime('%Y-%m-%d-%H-%M-%S')+'.csv'
		filename = path+'/'+ff
		if os.path.exists(filename):
			correct_filepath_found = True
			break

	if correct_filepath_found is False:
		if debug:
			print("***cannot find files that look like  "+ff+' in any of the above directories')
		return
	
	if debug:
		print("loading files from the following directory: ")
		print(path)

	# cycle over the given dates, load data, and stick into a dataframe
	for date in E['daterange']:

		ff = E['exp_name']+'_'+'AEFs_'+E['diagn']+'_'+date.strftime('%Y-%m-%d-%H-%M-%S')+'.csv'
		filename = path+'/'+ff
		if date == E['daterange'][0]:
			DF = pd.read_csv(filename,sep='\t')
		else:
			DF2 = pd.read_csv(filename,sep='\t')
			DF = pd.merge(DF,DF2,how='outer')

	return(DF)

def compute_DART_diagn_from_Wang_TEM_files(E,datetime_in,hostname='taurus',debug=False):

	"""
	For a given experiment dictionary and datetime, load the transformed Eulerian mean (TEM) 
	diagnostics 
	corresponging to the desired DART diagnostic.  

	This code is designed to read in TEM diagnostics computed by Wuke Wang, GEOMAR Kiel 
	"""

	import TEM as tem

	# load the file corresponding to the desired date 
	X,lat,lev = tem.load_Wang_TEM_file(E,datetime_in,hostname=hostname,verbose=debug)
	CS = E['copystring']

	# if looking at ERA data, we don't have ensemble members. Here just return the array
	if E['exp_name'] == 'ERA':
		Dout = np.squeeze(X)	
	else:
		# if the diagnostic is a single ensemble member, simply choose it out of the array and return 
		if 'ensemble member' in CS:
			ensindex = re.sub(r'ensemble member*','',CS).strip()
			Dout = np.squeeze(X[:,:,:,int(ensindex)-1])	
		# can also compute simple ensemble statistics: mean, standard deviation, etc (other still need to be added)
		if CS == 'ensemble mean':
			Dout = np.squeeze(np.nanmean(X,axis=3))
		if CS == 'ensemble std':
			Dout = np.squeeze(np.nanstd(X,axis=3))
		

	return Dout,lat,lev

def compute_DART_diagn_from_model_h_files(E,datetime_in,hostname='taurus',verbose=True):

	# compute ensemble mean or spread, or just retrieve an ensemble member  
	# from variables that are found in WACCM or CAM history files 
	CS = E['copystring']

	Xout = None
	lat = None
	lon = None

	# it's easy if the copy we want is a single ensemble member  
	if 'ensemble member' in CS:
		ensindex = re.sub(r'ensemble member*','',CS).strip()
		instance = int(ensindex)
		Xout,lat,lon,lev = waccm.load_WACCM_multi_instance_h_file(E,datetime_in,instance,hostname=hostname,verbose=verbose)
		if (Xout is None) or (lat is None) or (lon is None):
			datestr = datetime_in.strftime("%Y-%m-%d")
			if verbose:
				print("filling in None for experiment "+E['exp_name']+', instance '+str(instance)+', and date '+datestr)

	# ensemble mean also has a special precomputed file
	if (CS == 'ensemble mean'):
		instance = 'ensemble mean'
		Xout,lat,lon,lev = waccm.load_WACCM_multi_instance_h_file(E,datetime_in,instance,hostname=hostname,verbose=verbose)
		if (Xout is None) or (lat is None) or (lon is None):
			datestr = datetime_in.strftime("%Y-%m-%d")
			if verbose:
				print("filling in None for experiment "+E['exp_name']+', instance '+str(instance)+', and date '+datestr)

	# ensemble standard deviation also has a special precomputed file
	if (CS == 'ensemble std'):
		instance = 'ensemble std'
		Xout,lat,lon,lev = waccm.load_WACCM_multi_instance_h_file(E,datetime_in,instance,hostname=hostname,verbose=verbose)
		if (Xout is None) or (lat is None) or (lon is None):
			datestr = datetime_in.strftime("%Y-%m-%d")
			if verbose:
				print("filling in None for experiment "+E['exp_name']+', instance '+str(instance)+', and date '+datestr)

	return Xout,lat,lon,lev

def plot_diagnostic_lev_lat(E=dart.basic_experiment_dict(),Ediff=None,clim=None,hostname='taurus',cbar='vertical',reverse_colors=False,ncolors=18,scaling_factor=1.0,debug=False):

	"""
	Retrieve a DART diagnostic (defined in the dictionary entry E['diagn']) over levels and latitude.  
	Whatever diagnostic is chosen, we average over all longitudes in E['lonrange'] and 
	all times in E['daterange']

	INPUTS:
	E: basic experiment dictionary
	Ediff: experiment dictionary for the difference experiment
	clim: color limits (single number, applied to both ends if the colormap is divergent)
	hostname: name of the computer on which the code is running
	cbar: how to do the colorbar -- choose 'vertical','horiztonal', or None
	reverse_colors: set to True to flip the colormap
	ncolors: how many colors the colormap should have. Currently only supporting 11 and 18. 
	scaling_factor: factor by which to multiply the array to be plotted 
	debug: set to True to get extra ouput
	"""

	# throw an error if the desired variable is 2 dimensional 
	if (E['variable'] == 'PS') or (E['variable'] == 'FLUT'):
		print('Attempting to plot a two dimensional variable ('+E['variable']+') over level and latitude - need to pick a different variable!')
		return

	# load the desired DART diagnostic for the desired variable and daterange:
	Vmatrix,lat,lon,lev,new_daterange = DART_diagn_to_array(E,hostname=hostname,debug=debug)

	# and average over the last dimension, which is always time (by how we formed this array) 
	VV = np.nanmean(Vmatrix,axis=len(Vmatrix.shape)-1)	

	# figure out which dimension is longitude and then average over that dimension 
	# unless the data are already in zonal mean, in which case DART_diagn_to_array should have returned None for lon
	shape_tuple = VV.shape
	if lon is not None:
		for dimlength,ii in zip(shape_tuple,range(len(shape_tuple))):
			if dimlength == len(lon):
				londim = ii
		M1 = np.squeeze(np.mean(VV,axis=londim))
	else:
		M1 = np.squeeze(VV)

	# if computing a difference to another field, load that here  
	if (Ediff != None):

		# load the desired DART diagnostic for the difference experiment dictionary
		Vmatrix,lat,lon,lev,new_daterange = DART_diagn_to_array(Ediff,hostname=hostname,debug=debug)

		# average over time 
		VV = np.nanmean(Vmatrix,axis=len(Vmatrix.shape)-1)	

		# average over longitudes 
		# as before, look for the londim (it might be different this time) 
		shape_tuple = VV.shape
		if lon is not None:
			for dimlength,ii in zip(shape_tuple,range(len(shape_tuple))):
				if dimlength == len(lon):
					londim = ii
			M2 = np.squeeze(np.mean(VV,axis=londim))
		else:
			M2 = np.squeeze(VV)

		# subtract the difference field out from the primary field  
		M = M1-M2
	else:
		M = M1


        # choose a color map based on the variable in question
	colors,cmap,cmap_type = state_space_HCL_colormap(E,Ediff,reverse=reverse_colors,ncol=ncolors)

	# set the contour levels - it depends on the color limits and the number of colors we have  
	if clim is None:
		clim = np.nanmax(np.absolute(M[np.isfinite(M)]))

	if cmap_type == 'divergent':
		L  = np.linspace(start=-clim,stop=clim,num=ncolors)
	else:
		L  = np.linspace(start=0,stop=clim,num=ncolors)

	# transpose the array if necessary  
	if M.shape[0]==len(lat):
		MT = np.transpose(M)
	else:
		MT = M

        # plot
        cs = plt.contourf(lat,lev,scaling_factor*MT,L,cmap=cmap,extend="both")

	# add a colorbar if desired 
	if cbar is not None:
		if (clim > 1000) or (clim < 0.01):
			CB = plt.colorbar(cs, shrink=0.8, extend='both',orientation=cbar,format='%.0e')
		else:
			CB = plt.colorbar(cs, shrink=0.8, extend='both',orientation=cbar)
	else: 
		CB = None

	# axis labels 
        plt.xlabel('Latitude')
        plt.ylabel('Pressure (hPa)')
	plt.yscale('log')
	plt.gca().invert_yaxis()

	# make sure the axes only go as far as the ranges in E
	plt.ylim(E['levrange'])
	plt.xlim(E['latrange'])

	# return the colorbar handle if available, so we can adjust it later
	return CB,M

def plot_diagnostic_lev_lat_quiver(E=dart.basic_experiment_dict(),Ediff=None,alpha=(1,1),scale_by_pressure=False,hostname='taurus',debug=False):

	"""
	Retrieve TWO DART diagnostics (defined in the dictionary entry E['diagn']) over levels and latitude,  
	and then plot them as a "quiver" plot (i.e. vector field). 
	In this case, E['variable'] should be a LIST or TUPLE of the two variable that we plot, e.g. FPHI and FZ for the components
	of EP flux, e.g. E['variable'] = (x,y), where x is the x-component of the vectors, and y the y-component. 
	Whatever diagnostic is chosen, we average over all longitudes in E['lonrange'] and 
	all times in E['daterange']

	INPUTS:
	E - experiment dictionary  
	Ediff - dictionary for the difference experiment (default is None)
	alpha - tuple of scaling factors for the horizontal and vertical components, 
		e.g. for EP flux alpha should be (4.899E-3,0)
	scale_by_pressure: set to True to scale the arrows by the pressure at each point
	"""

	# throw an error if the desired variable is 2 dimensional 
	if (E['variable'] == 'PS') or (E['variable'] == 'FLUT'):
		print('Attempting to plot a two dimensional variable ('+E['variable']+') over level and latitude - need to pick a different variable!')
		return

	# throw an error if E['variable'] is not a list or a tuple
	if (type(E['variable']) != tuple) and (type(E['variable']) != list):
		print('----Trying to make a vector field plot but the requested variable is not a tuple or a list, but rather:')	
		print(type(E['variable']))
		return

	# loop over the two variables and 
	# load the desired DART diagnostic for each variable and daterange:
	Mlist = []
	for vv in E['variable']:
		Etemp = E.copy()
		Etemp['variable'] = vv
		Vmatrix,lat,lon,lev,new_daterange = DART_diagn_to_array(Etemp,hostname=hostname,debug=debug)

		# if desired, scale the array by pressure (this is useful for EP flux vector)
		if scale_by_pressure:
			EP = E.copy()
			EP['variable'] = 'P'
			VP,dumlat,lonP,dumlev,dumdaterange = DART_diagn_to_array(EP,hostname=hostname,debug=debug)
			shape_tuple = VP.shape
			for dimlength,ii in zip(shape_tuple,range(len(shape_tuple))):
				if dimlength == len(lonP):
					londim = ii
			VPlonave = np.squeeze(np.mean(VP,axis=londim))
			Vnorm = Vmatrix/VPlonave
		else:
			Vnorm = Vmatrix

		# average over the last dimension, which is always time (by how we formed this array) 
		VV = np.nanmean(Vnorm,axis=len(Vnorm.shape)-1)	
		
		# figure out which dimension is longitude and then average over that dimension 
		# unless the data are already in zonal mean, in which case DART_diagn_to_array should have returned None for lon
		shape_tuple = VV.shape
		if lon is not None:
			for dimlength,ii in zip(shape_tuple,range(len(shape_tuple))):
				if dimlength == len(lon):
					londim = ii
			M1 = np.squeeze(np.mean(VV,axis=londim))
		else:
			M1 = np.squeeze(VV)

		# if computing a difference to another field, load that here  
		if (Ediff != None):
			Edtemp = Ediff.copy()
			Edtemp['variable'] = vv

			# load the desired DART diagnostic for the difference experiment dictionary
			Vmatrix,lat,lon,lev,new_daterange = DART_diagn_to_array(Edtemp,hostname=hostname,debug=debug)

			# if desired, scale the array by pressure (this is useful for EP flux vector)
			if scale_by_pressure:
				EdiffP = Ediff.copy()
				EdiffP['variable'] = 'P'
				VP,dumlat,lonP,dumlev,dumdaterange = DART_diagn_to_array(EdiffP,hostname=hostname,debug=debug)
				shape_tuple = VP.shape
				for dimlength,ii in zip(shape_tuple,range(len(shape_tuple))):
					if dimlength == len(lonP):
						londim = ii
				VPlonave = np.squeeze(np.mean(VP,axis=londim))
				Vnorm = Vmatrix/VPlonave
			else:
				Vnorm = Vmatrix

			# average over time 
			VV = np.nanmean(Vnorm,axis=len(Vnorm.shape)-1)	

			# average over longitudes 
			if lon is not None:
				M2 = np.squeeze(np.mean(VV,axis=londim))
			else:
				M2 = np.squeeze(VV)

			# subtract the difference field out from the primary field  
			M = M1-M2
		else:
			M = M1

		# transpose the array if necessary  
		if M.shape[0]==len(lat):
			MT = np.transpose(M)
		else:
			MT = M

		# MT is the field we want to plot --> append it to the list
		Mlist.append(MT)

	# create a mesh
	X,Y = np.meshgrid(lat,lev)


        # plot
	plt.quiver(X,Y,alpha[0]*Mlist[0],alpha[1]*Mlist[1],pivot='mid', units='inches')


	# axis labels 
        plt.xlabel('Latitude')
        plt.ylabel('Pressure (hPa)')
	plt.yscale('log')
	plt.gca().invert_yaxis()

	# make sure the axes only go as far as the ranges in E
	plt.ylim(E['levrange'])
	plt.xlim(E['latrange'])

	# return the colorbar handle if available, so we can adjust it later
	return Mlist

def Nsq(E,date,hostname='taurus',debug=False):

	"""
	given a DART experiment dictionary on a certain date and time, compute the buoyancy frequency as a 3D field 

	**main calculation:**  
	N2 = (g/theta)*dtheta/dz 
	where theta = T(p_ref/p)^K is the potential temperature 
	K = R/cp 
	T = Temperature 
	p_ref = reference pressure (here using P0 = 1000.0 in WACCM data) 
	p = pressure  
	"""


	# reconstruct the pressure field at each point from hybrid model variables 
	varlist = ['hyam','hybm','P0','PS','T','Z3']
	H = dict()
	for vname in varlist:
	    Ehyb = E.copy()
	    Ehyb['variable'] = vname
	    field,lat,lon,lev = compute_DART_diagn_from_model_h_files(Ehyb,date,verbose=debug)
	    if vname == 'PS':
		H['lev'] = lev
		H['lat'] = lat
		H['lon'] = lon        
	    H[vname]=field

	nlev = len(lev)
	nlat = len(lat)
	nlon = len(lon)
	P = np.zeros(shape = (nlev,nlat,nlon))
	for k in range(nlev):
	    for i in range(nlon):
		for j in range(nlat):
			P[k,j,i] = H['hyam'][k]*H['P0'] + H['hybm'][k]* np.squeeze(H['PS'])[j,i]


	# compute potential temperature  
	Rd = 286.9968933                # Gas constant for dry air        J/degree/kg
	g = 9.80616                     # Acceleration due to gravity       m/s^2
	cp = 1005.0                     # heat capacity at constant pressure    m^2/s^2*K
	theta = H['T']*(H['P0']/P)**(Rd/cp)

	# compute the vertical gradient in potential temperature 
	dZ = np.gradient(np.squeeze(H['Z3']))	# 3D gradient of geopotential height (with respect to model level) 
	dthetadZ_3D = np.gradient(np.squeeze(theta),dZ[0])
	dthetadZ = dthetadZ_3D[0] # this is the vertical temperature gradient with respect to pressure 

	# compute the buoyancy frequency 
	N2 = (g/np.squeeze(theta))*dthetadZ

	return N2,lat,lon,lev


def P_from_hybrid_levels(E,date,hostname='taurus',debug=False):

	"""
	given a DART experiment dictionary on a certain date and time,
	recreate the pressure field given the hybrid model level parameters 
	"""

	# check whether the requested experiment uses a model with hybrid levels. 
	# right now this just returns if the experiment is ERA-Interm.
	# TODO: subroutine with a dictionary that shows 
	# whether a given experiment has hybrid levels 
	if E['exp_name'] == 'ERA':
		print('ERA data are not on hybrid levels --need to retrieve ERA pressure data instead of calling P_from_hybrid_levels')
		return None,None,None,None

	# reconstruct the pressure field at each point from hybrid model variables 
	varlist = ['hyam','hybm','P0','PS','T','Z3']
	H = dict()
	for vname in varlist:
	    Ehyb = E.copy()
	    Ehyb['variable'] = vname
	    field,lat,lon,lev = compute_DART_diagn_from_model_h_files(Ehyb,date,verbose=debug)
	    if vname == 'PS':
		H['lev'] = lev
		H['lat'] = lat
		H['lon'] = lon        
	    H[vname]=field

	nlev = len(lev)
	nlat = len(lat)
	nlon = len(lon)
	P = np.zeros(shape = (nlev,nlat,nlon))
	for k in range(nlev):
	    for i in range(nlon):
		for j in range(nlat):
			P[k,j,i] = H['hyam'][k]*H['P0'] + H['hybm'][k]* np.squeeze(H['PS'])[j,i]

	return P,lat,lon,lev

def bootstrapci_from_anomalies(E,P=95,nsamples=1000,hostname='taurus',debug=False):

	"""
	Given some DART experiment dictionary, retrieve anomalies with respect 
	to a certain climatology and for the entire ensemble, and 
	then use a bootstrap method to compute the confidence interval 
	of those anomalies.  

	To make this work, the diagnostic given in E needs to specify the percentage of 
	the confidence interval that we want, and what climatology we are computing the anomalies
	with respect to.  

	E['diagn'] should have the form 'anomaly.XXXX.bootstrapci.NN' where 
		XXXX = the code for the climatology being used ("NODA" is a good choice)  
		NN = the percentage where we want the confidence interval (e.g. '95'

	INPUTS:  
	E: a standard DART experiment dictionary 
	P: the percentage where we want the confidence interval  - default is 95
	nsamples: the number of samples for the boostrap algorithm - default is 10000

	"""

	# look up the ensemble size for this experiment
	N = es.get_ensemble_size_per_run(E['exp_name'])

	# extract the climatology option for the anomalies from the diagnostic
	climatology_option = E['diagn'].split('.')[1]
	
	# loop over the entire ensemble, compute the anomalies with respect to
	# the desired climatology, and append to a list  
	Alist = []
	for iens in range(N):
	    E['copystring'] = 'ensemble member '+str(iens+1)
	    AA,Xclim,lat,lon,lev,new_daterange = mjo.ano(E,climatology_option)
	    Alist.append(AA)

	# turn the arrays in the list into a matrix
	Amatrix = np.concatenate([A[np.newaxis,...] for A in Alist], axis=0)

	# now apply bootstrap.
	# note that this function applies np.mean over the first dimension, which we made the ensemble
	CI = bs.bootstrap(Amatrix,nsamples,np.mean,P)
	
	# we can also make a mask for statistical significance. 
	# anomalies where the confidence interval includes zero are not considered statistically significant at the P% level. 
	# we can tell where the CI crosses zero by there the lower and upper bounds have opposite signs, which means that 
	# their product will be negative
	L = CI.lower
	U = CI.upper
	LU = L*U
	sig = LU > 0

	return CI,sig

def DART_diagn_to_array(E,hostname='taurus',debug=False):

	"""
	This subroutine loops over the dates given in E['daterange'] and load the appropriate DART diagnostic for each date, 
	returning a numpy matrix of the relevant date.  

	The files we load depend on the desired DART diagnostic (given in E['diagn']), variable (E['variable']), and 
	any extra computations needed (E['extras'])  
	"""

	Vmatrix_found = False
	# if plotting anomalies from climatology, climatology, or a climatological standard deviation, 
	# can load these using the `stds` and `ano` rubroutines in MJO.py  
	if ('climatology' in E['diagn']) or ('anomaly' in  E['diagn']) or ('climatological_std' in E['diagn']):
		from MJO import ano,stds
		climatology_option = E['diagn'].split('.')[1]
		AA,Xclim,lat,lon,lev,new_daterange = ano(E,climatology_option,hostname,debug)	
		if 'climatology' in E['diagn']:
			Vmatrix = Xclim
		if 'anomaly' in E['diagn']:
			Vmatrix = AA
		if 'climatological_std' in E['diagn']:
			S,lat,lon,lev = stds(E,climatology_option,hostname,debug)	
			Vmatrix = S.reshape(AA.shape)
		Vmatrix_found = True

	# if loading regular variables from ERA data, can load those using a subroutine from the ERA module.
	# in this case, we also don't have to loop over dates.
	era_variables_list = ['U','V','Z','T','MSLP']
	if (E['exp_name']=='ERA') and (E['variable'].upper() in era_variables_list):
		import ERA as era
		VV,new_daterange,lat,lon,lev = era.retrieve_era_averaged(E,False,False,False,hostname,debug)
		# this SR returns an array with time in the first dimension. We want it in the last
		# dimension, so transpose
		Vmatrix = VV.transpose()
		Vmatrix_found = True

	if not Vmatrix_found:
	# if neither of the above worked, we have to 
	# loop over the dates given in the experiment dictionary and load the desired data  
		Vlist = []
		Vshape = None
		for date in E['daterange']:

			# for covariances and correlations
			if (E['diagn'].lower() == 'covariance') or (E['diagn'].lower() == 'correlation') :
				lev,lat,lon,Cov,Corr = dart.load_covariance_file(E,date,hostname,debug=debug)
				if E['diagn'].lower() == 'covariance':
					V = Cov
				if E['diagn'].lower() == 'correlation':
					V = Corr

			# for regular diagnostic, the file we retrieve depends on the variable in question  
			else:
				file_type_found = False
				# here are the different categories of variables:
				# TODO: subroutine that reads the control variables specific to each model/experiment
				dart_control_variables_list = ['US','VS','T','PS']
				tem_variables_list = ['VSTAR','WSTAR','FPHI','FZ','DELF']
				dynamical_heating_rates_list = ['VTY','WS']

				# DART control variables are in the Prior_Diag and Posterior_Diag files 
				if E['variable'] in dart_control_variables_list:
					lev,lat,lon,V,P0,hybm,hyam = dart.load_DART_diagnostic_file(E,date,hostname=hostname,debug=debug)
					file_type_found = True

				# transformed Eulerian mean diagnostics have their own routine 
				if E['variable'].upper() in tem_variables_list+dynamical_heating_rates_list:
					V,lat,lev = compute_DART_diagn_from_Wang_TEM_files(E,date,hostname=hostname,debug=debug)
					lon = None
					file_type_found = True

				# another special case is buoyancy frequency -- this is computed in a separate routine 
				if E['variable'] == 'Nsq':
					V,lat,lon,lev = Nsq(E,date,hostname=hostname,debug=debug)
					file_type_found = True
					
				# pressure needs to be recreated from the hybrid model levels -- this is done in a separate routine 
				if E['variable'] == 'P':
					V,lat,lon,lev = P_from_hybrid_levels(E,date,hostname=hostname,debug=debug)
					file_type_found = True
		
				# for all other variables, compute the diagnostic from model h files 
				if not file_type_found:
					V,lat,lon,lev = compute_DART_diagn_from_model_h_files(E,date,hostname=hostname,verbose=debug)

			# add the variable field just loaded to the list:
			Vlist.append(V)

			# store the dimensions of the array V one time 
			if (V is not None) and (Vshape is None):
				Vshape = V.shape

			# if Vlist still has length 0, we didn't find any data -- abort 
			if len(Vlist)==0:
				d1 = E['daterange'][0].strftime("%Y-%m-%d")
				d2 = E['daterange'][len(E['daterange'])-1].strftime("%Y-%m-%d")
				print('Could not find any data for experiment '+E['exp_name']+' and variable '+E['variable']+' between dates '+d1+' and '+d2)
				Vmatrix = None
			else:
				# first remove and Nones that might be in there  
				Vlist2 = [V for V in Vlist if V is not None]
				bad = [i for i, j in enumerate(Vlist) if j is None]
				new_daterange = [i for j, i in enumerate(E['daterange']) if j not in bad]

				# turn the list of variable fields into a matrix 
				Vmatrix = np.concatenate([V[..., np.newaxis] for V in Vlist2], axis=len(V.shape))

	return Vmatrix,lat,lon,lev,new_daterange

def plot_diagnostic_profiles(E=dart.basic_experiment_dict(),Ediff=None,color="#000000",linestyle='-',linewidth = 2,alpha=1.0,hostname='taurus',log_levels=True,debug=False):

	"""
	Plot a vertical profile of some DART diagnostic / variable, 
	averaged over the date, latitude, and longitude ranges given in the 
	experiment dictionary 
	"""


	daterange = E['daterange']

	# throw an error if the desired variable is 2 dimensional 
	if (E['variable'] == 'PS') or (E['variable'] == 'FLUT'):
		print('Attempting to plot a two dimensional variable ('+E['variable']+') over level and latitude - need to pick a different variable!')
		return

	# load the timeseries of data from either reanalysis (ERA-Interim) or DART  
	# TODO: check to make sure this returns the same kind of array as DART_diagn_to_array
	#if E['exp_name'] == 'ERA':
	#	M0,t,lat,lon,lev = era.retrieve_era_averaged(E,average_levels=False,hostname=hostname,verbose=debug)
	#	Vmatrix = np.transpose(M0)	
	#else:
	Vmatrix,lat,lon,lev,new_daterange = DART_diagn_to_array(E,hostname=hostname,debug=debug)

	# average over the last dimension, which is always time (by how we formed this array) 
	VV = np.nanmean(Vmatrix,axis=len(Vmatrix.shape)-1)	

	# find the latidue and longitude dimensions and average 
	shape_tuple = VV.shape
	for dimlength,ii in zip(shape_tuple,range(len(shape_tuple))):
		if lat is not None:
			if dimlength == len(lat):
				latdim = ii
		if lon is not None:
			if dimlength == len(lon):
				londim = ii
	if lat is not None:
		Mlat = np.nanmean(VV,axis=latdim)
	else:
		Mlat = VV
	if lon is not None:
		M1 = np.nanmean(Mlat,axis=latdim)
	else:
		M1 = Mlat


	# repeat everything for the difference experiment
	if (Ediff != None):
		# TODO: check to make sure this returns the same kind of array as DART_diagn_to_array
		if Ediff['exp_name'] == 'ERA':
			M0,t,lat,lon,lev = era.retrieve_era_averaged(Ediff,average_levels=False,hostname=hostname,verbose=debug)
			Vmatrix = np.transpose(M0)	

		else:
			Vmatrix,lat,lon,lev,new_daterange = DART_diagn_to_array(Ediff,hostname=hostname,debug=debug)

		# average over the last dimension, which is always time (by how we formed this array) 
		VV = np.nanmean(Vmatrix,axis=len(Vmatrix.shape)-1)	

		# find the latidue and longitude dimensions and average 
		shape_tuple = VV.shape
		for dimlength,ii in zip(shape_tuple,range(len(shape_tuple))):
			if lat is not None:
				if dimlength == len(lat):
					latdim = ii
			if lon is not None:
				if dimlength == len(lon):
					londim = ii
		if lat is not None:
			Mlat = np.nanmean(VV,axis=latdim)
		else:
			Mlat = VV
		if lon is not None:
			M2 = np.nanmean(Mlat,axis=latdim)
		else:
			M2 = Mlat
		
		# take the difference
		M = M1-M2
	else:
		M = M1

        # plot the profile 
        plt.plot(M,lev,color=color,linestyle=linestyle,linewidth=linewidth,label=E['title'],alpha=alpha)

	# improve axes and labels
	ax = plt.gca()
	xlim = ax.get_xlim()[1]
	ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        plt.ylabel('Level (hPa)')
	if log_levels:
		plt.yscale('log')
	plt.gca().invert_yaxis()

	# make sure the axes only go as far as the ranges in E
	plt.ylim(E['levrange'])
	return M
