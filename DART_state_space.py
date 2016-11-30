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
from netCDF4 import Dataset
import WACCM as waccm
import re
import ERA as era
import TEM as tem
import experiment_settings as es
import palettable as pb

## here are some common settings for the different subroutines

# list the 3d, 2d, 1d variables 
# TODO: fill this in with other common model variables 
var3d = ['U','US','V','VS','T','Z3','DELF','Q','CH4','OH','Nsq']
var2d = ['PS','FLUT','ptrop','ztrop']
var1d = ['hyam','hybm','hyai','hybi']

# constants
H = 7.0    # 7.0km scale height 

def plot_diagnostic_globe(E,Ediff=None,projection='miller',clim=None,cbar='vertical',log_levels=None,ncolors=19,hostname='taurus',debug=False,colorbar_label=None,reverse_colors=False,stat_sig=None):

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
	ncolors: how many colors in the contours - default is 19
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
		L  = np.linspace(start=-clim,stop=clim,num=ncolors)
	else:
		L  = np.linspace(start=0,stop=clim,num=ncolors)


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

def plot_diagnostic_hovmoeller(E,Ediff=None,clim=None,cbar='vertical',log_levels=None,hostname='taurus',debug=False,scaling_factor=1.0,reverse_colors=False,cmap_type='sequential'):

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
	D = DART_diagn_to_array(E,hostname=hostname,debug=debug)

	# load the difference array and subtract (if requested)
	if Ediff is not None:
		D2 = DART_diagn_to_array(Ediff,hostname=hostname,debug=debug)
		Vmatrix = D['data']-D2['data']
	else:	
		Vmatrix = D['data']

	# find the lat and level dimensions and average 
	V1 = average_over_named_dimension(Vmatrix,D['lat'])
	if 'lev' in D:
		if D['lev'] is not None:
			V2 = average_over_named_dimension(V1,D['lev'])
		else:
			V2=V1
	else:
		V2=V1

	# multiply by a scaling factor if needed 
	M = scaling_factor*np.squeeze(V2)

	#---plot setup----------------
	time = D['daterange']
	lon = D['lon']

        # choose color map 
	cc = nice_colormaps(cmap_type,reverse_colors)
	cmap=cc.mpl_colormap
	ncolors = cc.number
	colors,cmap,cmap_type = state_space_HCL_colormap(E,Ediff)

	# specify the color limits 
	if clim is None:
		clim = np.nanmax(np.absolute(M))

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
		if 'units' in D:
			CB.set_label(D['units'])
	else: 
		CB = None

	plt.ylabel('Time')
	plt.xlabel('Longitude')

	# put some outputs into a dictionary 
	Mout = dict()
	Mout['data']=MT
	Mout['xname']='Longitude'
	Mout['yname']='Date'
	Mout['x']=lon	
	Mout['y']=time
	Mout['colorbar']=CB
	Mout['contours']=cs

	return Mout


def plot_diagnostic_lev_time(E=dart.basic_experiment_dict(),Ediff=None,vertical_coord='log_levels',clim=None,cbar='vertical',colorbar_label=None,reverse_colors=False,scaling_factor=1.0,cmap_type='sequential',hostname='taurus',debug=False):

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
	cmap_type: what kind of colormap do we want? choose 'sequential' or 'divergent'
	vertical_coord: option for how to plot the vertical coordinate. These are your choices:
		'log_levels' (default) -- plot whatever the variable 'lev' gives (e.g. pressure in hPa) on a logarithmic scale 
		'levels' -- plot whatever the variable 'lev' gives (e.g. pressure in hPa) on a linear scale 
		'z' -- convert lev (assumed to be pressure) into log-pressure height coordinates uzing z=H*exp(p/p0) where p0 = 1000 hPa and H=7km  
		'TPbased': in this case, compute the height of each gridbox relative to the local tropopause and 
			plot everything on a "tropopause-based" grid, i.e. zt = z-ztrop-ztropmean 
	"""

	# throw an error if the desired variable is 2 dimensional 
	if E['variable'].upper() not in var3d:
		print('Attempting to plot a two dimensional variable ('+E['variable']+') over level and latitude - need to pick a different variable!')
		return

	# load the requested array, and the difference array if needed 
	Vmain0,lat,lon,lev0,new_daterange = DART_diagn_to_array(E,hostname=hostname,debug=debug)
	# convert to TP-based coordinates if requested 	
	if vertical_coord=='TPbased': 
		Vmain,lev=to_TPbased(E,Vmain0,lev0,hostname=hostname,debug=debug)
	else:
		Vmain=Vmain0
		lev=lev0
	if Ediff is not None:
		Vdiff0,lat,lon,lev0,new_daterange = DART_diagn_to_array(Ediff,hostname=hostname,debug=debug)
		# convert to TP-based coordinates if requested 	
		if vertical_coord=='TPbased': 
			Vdiff,lev=to_TPbased(E,Vdiff0,lev0,hostname=hostname,debug=debug)
		else:
			Vdiff=Vdiff0
			lev=lev0
		Vmatrix=Vmain-Vdiff
	else:
		Vmatrix=Vmain

	# figure out which dimension is longitude and then average over that dimension 
	# unless the data are already in zonal mean, in which case DART_diagn_to_array should have returned None for lon
	if debug:
		print('shape of array after concatenating dates:')
		print(Vmatrix.shape)
	if lon is not None:
		for idim,dimlen in enumerate(Vmatrix.shape):
			if dimlen == len(lon):
				londim = idim
		Vlon = np.mean(Vmatrix,axis=londim)
	else:
		Vlon = Vmatrix  
	if debug:
		print('shape of array after averaging out longitude:')
		print(Vlon.shape)

	# figure out which dimension is latitude and then average over that dimension 
	if lat is not None:
		for idim,dimlen in enumerate(Vlon.shape):
			if dimlen == len(lat):
				latdim = idim
		Vlonlat = np.squeeze(np.mean(Vlon,axis=latdim))
	else:
		Vlonlat = Vlon
	if debug:
		print('shape of array after averaging out latitude:')
		print(Vlonlat.shape)

	# squeeze out any leftover length-1 dimensions
	M = scaling_factor*np.squeeze(Vlonlat)

        # choose color map 
	cc = nice_colormaps(cmap_type,reverse_colors)
	cmap=cc.mpl_colormap
	ncolors = cc.number

	# set the contour levels - it depends on the color limits and the number of colors we have  
	if clim is None:
		clim = np.nanmax(np.absolute(M[np.isfinite(M)]))

	if cmap_type == 'divergent':
		L  = np.linspace(start=-clim,stop=clim,num=ncolors)
	else:
		L  = np.linspace(start=0,stop=clim,num=ncolors)

	# compute vertical coordinate depending on choice of pressure or altitude 
	if 'levels' in vertical_coord:
		y=lev
		ylabel = 'Level (hPa)'
	if vertical_coord=='z':
		H=7.0
		p0=1000.0 
		y = H*np.log(p0/lev)
		ylabel = 'log-p height (km)'
	if vertical_coord=='TPbased':
		#from matplotlib import rcParams
		#rcParams['text.usetex'] = True
		y=lev
		ylabel='z (TP-based) (km)'

        # contour data 
	t = new_daterange
	if debug:
		print('shape of the array to be plotted:')
		print(M.shape)
	cs = plt.contourf(t,y,M,L,cmap=cmap,extend="both")

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
		if colorbar_label is not None:
			CB.set_label(colorbar_label)
	else: 
		CB = None


	plt.xlabel('time')
	plt.ylabel(ylabel)
	if vertical_coord=='log_levels':
		plt.yscale('log')
	if 'levels' in vertical_coord:
		plt.gca().invert_yaxis()

	return cs,CB,M

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

def retrieve_state_space_ensemble(E,averaging=True,ensemble_members='all',scaling_factor=1.0,hostname='taurus',debug=False):

	"""
	retrieve the prior or posterior ensemble averaged over some region of the state,
	for some DART experiment
	
	INPUTS:
	E: standard experiment dictionary 
	averaging: set to True to average over the input latitude, longitude, and level ranges (default=True).
	ensemble_members: set to "all" to request entire ensemble, or specify a list with the numbers of the ensemble members you want to plot  
	scaling_factor: factor by which to multiply the array to be plotted 
	hostname
	debug
	"""

	# query the daterange of E
	daterange = E['daterange']

	# decide what ensemble members to loop over here - specific ones, or the whole set?
	if type(ensemble_members) is list:
		ens_list = ensemble_members
	else:
		N = es.get_ensemble_size_per_run(E['exp_name'])
		ens_list = np.arange(1,N+1)

	# loop over the ensemble members and timeseries for each ensemble member, and add to a list
	Eens = E.copy()
	VElist = []
	for iens in ens_list:
		if iens < 10:
			spacing = '      '
		else:
			spacing = '     '
		copystring = "ensemble member"+spacing+str(iens)		
		Eens['copystring'] = copystring

		D = DART_diagn_to_array(Eens,hostname=hostname,debug=debug)

		# if averaging, do that here
		if averaging:

			# average over latitude
			V0 = average_over_named_dimension(D['data'],D['lat'])
			
			# average over longitude  
			V1 = average_over_named_dimension(V0,D['lon'])

			# average over vertical level, if present  

			# for 3d variables, average over level:
			if E['variable'] not in var2d: 
				V2 = average_over_named_dimension(V1,D['lev'])
			else:
				V2=V1

			# thee might be another length-1 dimension left --average that out here  
			VV = scaling_factor*np.squeeze(V2)
		else:
			VV = scaling_factor*D['data']

		# append ensemble member to list
		VElist.append(VV)

	# turn the list of ensemble states into a matrix 
	VE = np.concatenate([V[np.newaxis,...] for V in VElist], axis=0)

	# output
	return VE,lev,lat,lon


def plot_state_space_ensemble(E=None,color_ensemble='#777777',color_mean=None,label_ensemble='Ensemble',label_mean='Mean',scaling_factor=1.0,linewidth=1.0,alpha=1.0,linestyle='-',hostname='taurus',debug=False,show_legend=False,ensemble_members='all'):

	"""
	plot the prior or posterior ensemble averaged over some region of the state,
	for some DART experiment

	INPUTS:
	ensemble_members: set to "all" to request entire ensemble, or specify a list with the numbers of the ensemble members you want to plot  
	color_ensemble: the color that we want to plot the ensemble in -- default is gray  
	color_mean: the color that we want to plot the ensemble mean in -- if this is None (default), then 
		the color of the ensemble is chosen
	label_ensemble: string with which to label the ensemble in the plot 
	label_mean: string with which to label the mean in the plot 
	scaling_factor: factor by which to multiply the array to be plotted 

	"""

	# retrieve the ensemble
	VE,lev,lat,lon = retrieve_state_space_ensemble(E=E,averaging=True,
								hostname=hostname,debug=debug,scaling_factor=scaling_factor,
								ensemble_members=ensemble_members)
	# set up a  time grid 
	t = E['daterange']

	# color for the mean state 
	if color_mean is None:
		color_mean=color_ensemble

        # plot global diagnostic in in time
	N = VE.shape[0]
	VM = np.mean(VE,axis=0)
	cs = plt.plot(t,VE[0,:],color=color_ensemble,label=label_ensemble)
	for iens in np.arange(1,N):
		cs = plt.plot(t,VE[iens,:],color=color_ensemble,label='_nolegend_',linewidth=0.7*linewidth,alpha=alpha)
	plt.plot(t,VM,color=color_mean,label=label_mean,linewidth=linewidth,linestyle=linestyle)

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
	#if len(t) > 30:
	plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
	#if len(t) < 10:
	#	plt.gca().xaxis.set_major_locator(mdates.DayLocator(bymonthday=range(len(t))))
	fmt = mdates.DateFormatter('%b-%d')
	plt.gca().xaxis.set_major_formatter(fmt)

	# put some outputs into a dictionary 
	Mout = dict()
	Mout['data']=VE
	Mout['xname']='Date'
	Mout['yname']=E['variable']
	Mout['x']=t
	Mout['legend_handle']=lg

	return Mout

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
	for iE,E in enumerate(EE):

		# store the name of this experiment
		names.append(E['title'])

		# TODO: instead of looping over dates, load the entire timeseries using this subroutine
		# for each experiment, load the desired DART diagnostic for the desired variable and daterange:
		#Vmatrix,lat,lon,lev,new_daterange = DART_diagn_to_array(E,hostname=hostname,debug=debug)

		# for each experiment loop over the input date range
		for ii,date in enumerate(E['daterange']):

			# fill in the day count (if desired) 
			if x_as_days:
				dt = date-E['daterange'][0]	
				dtfrac = dt.days + dt.seconds/(24.0*60.0*60.0)
				x[iE,ii] = dtfrac

			# load the data over the desired latitude and longitude range  
			lev,lat,lon,VV,P0,hybm,hyam = dart.load_DART_diagnostic_file(E,date,hostname=hostname,debug=debug)

			# load the difference array if desired  
			if EEdiff is not None:
				Ediff = EEdiff[iE]
				lev2,lat2,lon2,VVdiff,P0,hybm,hyam = dart.load_DART_diagnostic_file(Ediff,date,hostname=hostname,debug=debug)
				Vmatrix = VV-VVdiff
			else:
				Vmatrix = VV

			# compute global average only if the file was found
			if Vmatrix is not None:

				# average over latitude, longitude, and vertical level 
				if lat is not None:
					Vlat = average_over_named_dimension(Vmatrix,lat)
				else:	
					Vlat=Vmatrix
				if lon is not None:
					Vlatlon = average_over_named_dimension(Vlat,lon)
				else:
					Vlatlon=Vlat
				if lev is not None:
					Vlatlonlev = average_over_named_dimension(Vlatlon,lev)
				else:
					Vlatlonlev=Vlatlon

				# squeeze out any remaining length-1 dimensions  
				M = np.squeeze(Vlatlonlev)

			else:
				# if no file was found, just make the global average a NAN
				M = np.NAN

			# append the resulting vector to the larger array (or initialize it)
			try:
				MM[iE,ii] = M
			except ValueError:
				print("Array M doesn't seem to fit. Here is it's shape:")
				print(M.shape)


	#------plotting----------

	# change the default color cycle to colorbrewer Dark2, or use what is supplied
	if colors is None:
		cc = nice_colormaps('qualitative')
		colors=cc.mpl_colors

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
		try:
			cs = plt.plot(x,y,color=colors[iE],linewidth=2,linestyle=linestyles[iE],marker=markers[iE])
		except ValueError:
			print("There's a problem plotting the time and global average array. Here are their shapes:")
			print(len(x))
			print(y.shape)

	# include legend if desire
	if include_legend:
		lg = plt.legend(names,loc='best')
		lg.draw_frame(False)

	plt.xlabel('Time (Days)')
	if ylim is not None:
		plt.ylim(ylim)
	else:
		ylim= plt.gca().get_ylim()
	if xlim is not None:
		plt.xlim(xlim)

	# format the y-axis labels to be exponential if the limits are quite high
	if (np.max(ylim) > 1000):
		ax = plt.gca()
		ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

	if not x_as_days:
		# format the x-axis labels to be dates
		#if len(x) > 30:
		plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
		#if len(x) < 10:
		#	plt.gca().xaxis.set_major_locator(mdates.DayLocator(bymonthday=range(len(E['daterange']))))
		fmt = mdates.DateFormatter('%b-%d')
		plt.gca().xaxis.set_major_formatter(fmt)

	return MT,x



def state_space_HCL_colormap(E,Ediff=None,reverse=False,ncol=19,debug=False):

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
	ncol: how many colors? Currently only 11 and 19 are supported for divergent maps, and only 11 for 
		sequential maps. Default is 19. 
	"""

        # appropriate color maps for state space plots
	colors_sequential = False

	# sequential plot if plotting positive definite variables and not taking a difference  
	post_def_variables = ['Z3','PS','FLUT','T','Nsq','Q','O3']
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

	# also turn off the sequential colors if the diagnostic is increment  
	if E['diagn'].lower()=='increment':
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
	obs,copynames = dart.load_DART_obs_epoch_file(E,date,[obs_name],['ensemble member'],hostname)

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

	# TODO: replace with call to moduel palettable to get colorbrewer colors back
	colors,cmap,cmap_type = state_space_HCL_colormap(E,Ediff,reverse=reverse_colors)
	#bmap = brewer2mpl.get_map('Dark2', 'qualitative', 7)
	plt.plot(t,Y[0,:],color=_colors[0])
	plt.hold(True)
	plt.plot(t,X[3,:],color=colors[1])
	plt.plot(t,Xbad[3,:],color=bmap.colors[2])
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
	# TODO: replace with call to moduel palettable to get colorbrewer colors back
	colors,cmap,cmap_type = state_space_HCL_colormap(E,Ediff,reverse=reverse_colors)
	#bmap = brewer2mpl.get_map('Dark2', 'qualitative', 7)

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
			print(path)
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
	if 'ERA' in E['exp_name']:
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
		
		# or return entire ensemble 
		if CS=='ensemble':
			# need to make the ensemble the zeroth dimension insteadd 
			# of the 3rd 
			N = X.shape[3]
			Xlist = []
			for ii in range(N):
				Xlist.append(X[:,:,:,ii])
			Dout = np.concatenate([X[np.newaxis,...] for X in Xlist], axis=0)
			# squeeze out a potential extra time dimension
			Dout = np.squeeze(Dout)

		D=dict()
		D['data']=Dout
		D['lat']=lat
		D['lev']=lev

	return D

def compute_DART_diagn_from_model_h_files(E,datetime_in,hostname='taurus',verbose=True):

	# compute ensemble mean or spread, or just retrieve an ensemble member  
	# from variables that are found in WACCM or CAM history files 
	CS = E['copystring']

	Xout = None
	lat = None
	lon = None
	lev = None

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

	# to return the entire ensemble, retrieve number of ensemble members and loop  
	if (CS == 'ensemble'):
		N = es.get_ensemble_size_per_run(E['exp_name'])
		Xlist=[]
		for iens in range(N):
			instance = iens+1
			Xs,lat,lon,lev = waccm.load_WACCM_multi_instance_h_file(E,datetime_in,instance,hostname=hostname,verbose=verbose)
			Xlist.append(Xs)
		# turn the list of arrays into a new array 
		Xout = np.concatenate([X[np.newaxis,...] for X in Xlist], axis=0)

	# print an error message if none of these worked 
	if Xout is None:
		print('compute_DART_diagn_from_model_h_files does not know what to do with copystring '+copystring)
		print('Returning None')
		return None
	else:
		D = dict()
		D['data']=Xout
		D['lat']=lat
		D['lon']=lon
		D['lev']=lev
		return D

def plot_diagnostic_lev_lon(E=dart.basic_experiment_dict(),Ediff=None,clim=None,L=None,hostname='taurus',cbar='vertical',cmap_type='sequential',reverse_colors=False,colorbar_label=None,vertical_coord='log_levels',scaling_factor=1.0,debug=False):

	"""
	Retrieve a DART diagnostic (defined in the dictionary entry E['diagn']) 
	and plot if over longtitude and vertical levels.  
	Whatever diagnostic is chosen, we average over all longitudes in E['lonrange'] and 
	all times in E['daterange']

	INPUTS:
	E: basic experiment dictionary
	Ediff: experiment dictionary for the difference experiment
	clim: color limits (single number, applied to both ends if the colormap is divergent)
	L: list of contour levels - default is none, which choses the levels evenly based on clim 
	hostname: name of the computer on which the code is running
	cbar: how to do the colorbar -- choose 'vertical','horiztonal', or None
	cmap_type: choose a sequential, qualitative, or diverging colormap
	reverse_colors: set to True to flip the colormap
	colorbar_label: string with which to label the colorbar  
	scaling_factor: factor by which to multiply the array to be plotted 
	vertical_coord: option for how to plot the vertical coordinate. These are your choices:
		'log_levels' (default) -- plot whatever the variable 'lev' gives (e.g. pressure in hPa) on a logarithmic scale 
		'levels' -- plot whatever the variable 'lev' gives (e.g. pressure in hPa) on a linear scale 
		'z' -- convert lev (assumed to be pressure) into log-pressure height coordinates uzing z=H*exp(p/p0) where p0 = 1000 hPa and H=7km  
		'TPbased': in this case, compute the height of each gridbox relative to the local tropopause and 
			plot everything on a "tropopause-based" grid, i.e. zt = z-ztrop-ztropmean 
	debug: set to True to get extra ouput
	"""

	# throw an error if the desired variable is 2 dimensional 
	if E['variable'] in var2d:
		print('Attempting to plot a two dimensional variable ('+E['variable']+') over level and longitude - need to pick a different variable!')
		return

	# load the requested array, and the difference array if needed 
	D = DART_diagn_to_array(E,hostname=hostname,debug=debug)
	
	# convert to TP-based coordinates if requested 	
	if vertical_coord=='TPbased': 
		Vmain,lev=to_TPbased(E,D['data'],D['lev'],hostname=hostname,debug=debug)
	else:
		Vmain=D['data']
		lev=D['lev']
	if Ediff is not None:
		Ddiff = DART_diagn_to_array(Ediff,hostname=hostname,debug=debug)
		# convert to TP-based coordinates if requested 	
		if vertical_coord=='TPbased': 
			Vdiff,lev=to_TPbased(E,Ddiff['data'],Ddiff['lev'],hostname=hostname,debug=debug)
		else:
			Vdiff=Ddiff['data']
		Vmatrix=Vmain-Vdiff
	else:
		Vmatrix=Vmain

	# average over time and latitude  
	lat = D['lat']
	lon = D['lon']
	V0 = average_over_named_dimension(Vmatrix,D['daterange'])
	if lat is not None:
		V1 = average_over_named_dimension(V0,lat)
	else:
		V1 = V0

	# squeeze out any leftover length-1 dimensions,and multiply by scaling factor if needed  
	M = scaling_factor*np.squeeze(V1)

        # choose color map 
	cc = nice_colormaps(cmap_type,reverse_colors)
	cmap=cc.mpl_colormap
	ncolors = cc.number

	# create color limits if they weren't already specified 
	if clim is None:
		clim1 = np.nanmax(M[np.isfinite(M)])
		clim0 = np.nanmin(M[np.isfinite(M)])
		# for divergent colormaps, need to have the color limits even 
		if cmap_type == 'divergent':
			clim1 = np.nanmax(np.absolute(M[np.isfinite(M)]))
			clim0 = -clim1
		clim=[clim0,clim1]

	# create contour levels based on the array that we have  
	if L is None:
		L  = np.linspace(start=clim[0],stop=clim[1],num=ncolors)

	# transpose the array if necessary  
	if M.shape[0]==len(lon):
		MT = np.transpose(M)
	else:
		MT = M

	if len(MT.shape) < 2:
		print('plot_diagnostic_lev_lat: the derived array is not 2-dimensional. This is its shape:')
		print(MT.shape)
		print('Returning with nothing plotted...')
		return None,None

	if (MT.shape[0] != len(lev)) |  (MT.shape[1] != len(lon)):
		print("plot_diagnostic_lev_lat: the dimensions of the derived array don't match the level and latitude arrays we are plotting against. Here are their shapes:")
		print(MT.shape)
		print(len(lev))
		print(len(lon))
		print('Returning with nothing plotted...')
		return None,None

	# compute vertical coordinate depending on choice of pressure or altitude 
	if 'levels' in vertical_coord:
		y=lev
		ylabel = 'Level (hPa)'
	if vertical_coord=='z':
		H=7.0
		p0=1000.0 
		y = H*np.log(p0/lev)
		ylabel = 'log-p height (km)'
	if vertical_coord=='TPbased':
		#from matplotlib import rcParams
		#rcParams['text.usetex'] = True
		y=lev
		ylabel='z (TP-based) (km)'

	cs = plt.contourf(lon,y,MT,L,cmap=cmap,extend="both")

	# add a colorbar if desired 
	if cbar is not None:
		if (np.max(clim) > 1000) or (np.max(clim) < 0.01):
			CB = plt.colorbar(cs, shrink=0.8, extend='both',orientation=cbar,format='%.0e')
		else:
			CB = plt.colorbar(cs, shrink=0.8, extend='both',orientation=cbar)
		if colorbar_label is not None:
			CB.set_label(colorbar_label)
	else: 
		CB = None


	# axis labels 
	plt.xlabel('Longitude')
	plt.ylabel(ylabel)
	if vertical_coord=='log_levels':
		plt.yscale('log')
	if 'levels' in vertical_coord:
		plt.gca().invert_yaxis()

	# make sure the axes only go as far as the ranges in E
	plt.xlim(E['lonrange'])

	# return the colorbar handle if available, so we can adjust it later
	return CB,M,L

def plot_diagnostic_lev_lat(E=dart.basic_experiment_dict(),Ediff=None,clim=None,L=None,hostname='taurus',cbar='vertical',cmap_type='sequential',reverse_colors=False,colorbar_label=None,vertical_coord='log_levels',scaling_factor=1.0,debug=False):

	"""
	Retrieve a DART diagnostic (defined in the dictionary entry E['diagn']) over levels and latitude.  
	Whatever diagnostic is chosen, we average over all longitudes in E['lonrange'] and 
	all times in E['daterange']

	INPUTS:
	E: basic experiment dictionary
	Ediff: experiment dictionary for the difference experiment
	clim: color limits (single number, applied to both ends if the colormap is divergent)
	L: list of contour levels - default is none, which choses the levels evenly based on clim 
	hostname: name of the computer on which the code is running
	cbar: how to do the colorbar -- choose 'vertical','horiztonal', or None
	cmap_type: choose a sequential, qualitative, or diverging colormap -- default is sequential 
	reverse_colors: set to True to flip the colormap
	colorbar_label: string with which to label the colorbar  
	scaling_factor: factor by which to multiply the array to be plotted 
	vertical_coord: option for how to plot the vertical coordinate. These are your choices:
		'log_levels' (default) -- plot whatever the variable 'lev' gives (e.g. pressure in hPa) on a logarithmic scale 
		'levels' -- plot whatever the variable 'lev' gives (e.g. pressure in hPa) on a linear scale 
		'z' -- convert lev (assumed to be pressure) into log-pressure height coordinates uzing z=H*exp(p/p0) where p0 = 1000 hPa and H=7km  
		'TPbased': in this case, compute the height of each gridbox relative to the local tropopause and 
			plot everything on a "tropopause-based" grid, i.e. zt = z-ztrop-ztropmean 
	debug: set to True to get extra ouput
	"""

	# throw an error if the desired variable is 2 dimensional 
	if E['variable'].upper() not in var3d:
		print('Attempting to plot a two dimensional variable ('+E['variable']+') over level and latitude - need to pick a different variable!')
		return

	# load the requested array, and the difference array if needed 
	Vmain0,lat,lon,lev0,new_daterange = DART_diagn_to_array(E,hostname=hostname,debug=debug)
	# convert to TP-based coordinates if requested 	
	if vertical_coord=='TPbased': 
		Vmain,lev=to_TPbased(E,Vmain0,lev0,hostname=hostname,debug=debug)
	else:
		Vmain=Vmain0
		lev=lev0
	if Ediff is not None:
		Vdiff0,lat,lon,lev0,new_daterange = DART_diagn_to_array(Ediff,hostname=hostname,debug=debug)
		# convert to TP-based coordinates if requested 	
		if vertical_coord=='TPbased': 
			Vdiff,lev=to_TPbased(E,Vdiff0,lev0,hostname=hostname,debug=debug)
		else:
			Vdiff=Vdiff0
			lev=lev0
		Vmatrix=Vmain-Vdiff
	else:
		Vmatrix=Vmain

	# average over time and longitude 
	V0 = average_over_named_dimension(Vmatrix,new_daterange)
	if lon is not None:
		V1 = average_over_named_dimension(V0,lon)
	else:
		V1 = V0
	# squeeze out any leftover length-1 dimensions 
	M = np.squeeze(V1)

        # choose color map 
	cc = nice_colormaps(cmap_type,reverse_colors)
	cmap=cc.mpl_colormap
	ncolors = cc.number

	if clim is None:
		clim = np.nanmax(np.absolute(M[np.isfinite(M)]))

	# if not already specified, 
	# set the contour levels - it depends on the color limits and the number of colors we have  
	if L is None:
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
	if len(MT.shape) < 2:
		print('plot_diagnostic_lev_lat: the derived array is not 2-dimensional. This is its shape:')
		print(MT.shape)
		print('Returning with nothing plotted...')
		return None,None

	if (MT.shape[0] != len(lev)) |  (MT.shape[1] != len(lat)):
		print("plot_diagnostic_lev_lat: the dimensions of the derived array don't match the level and latitude arrays we are plotting against. Here are their shapes:")
		print(MT.shape)
		print(len(lev))
		print(len(lat))
		print('Returning with nothing plotted...')
		return None,None

	# compute vertical coordinate depending on choice of pressure or altitude 
	if 'levels' in vertical_coord:
		y=lev
		ylabel = 'Level (hPa)'
	if vertical_coord=='z':
		H=7.0
		p0=1000.0 
		y = H*np.log(p0/lev)
		ylabel = 'log-p height (km)'
	if vertical_coord=='TPbased':
		#from matplotlib import rcParams
		#rcParams['text.usetex'] = True
		y=lev
		ylabel='z (TP-based) (km)'

	cs = plt.contourf(lat,y,scaling_factor*MT,L,cmap=cmap,extend="both")

	# add a colorbar if desired 
	if cbar is not None:
		if (clim > 1000) or (clim < 0.01):
			CB = plt.colorbar(cs, shrink=0.8, extend='both',orientation=cbar,format='%.0e')
		else:
			CB = plt.colorbar(cs, shrink=0.8, extend='both',orientation=cbar)
		if colorbar_label is not None:
			CB.set_label(colorbar_label)
	else: 
		CB = None


	# axis labels 
	plt.xlabel('Latitude')
	plt.ylabel(ylabel)
	if vertical_coord=='log_levels':
		plt.yscale('log')
	if 'levels' in vertical_coord:
		plt.gca().invert_yaxis()

	# make sure the axes only go as far as the ranges in E
	plt.xlim(E['latrange'])

	# return the colorbar handle if available, so we can adjust it later
	return CB,M,L

def plot_diagnostic_lev_lat_quiver(E=dart.basic_experiment_dict(),Ediff=None,alpha=(1,1),narrow=1,arrowscale=1.0,scale_by_pressure=False,vertical_coord='log_levels',hostname='taurus',debug=False):

	"""
	Retrieve TWO DART diagnostic output fields over levels and latitude,  
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
	narrow - only plot every nth arrow. (Default is 1 -- plot every arrow). This number must be an integer. 
	arrowscale - scale the size of the arrows (this feeds directly into the scale argument of the quiver function)
	scale_by_pressure: set to True to scale the arrows by the pressure at each point
	vertical_coord: option for how to plot the vertical coordinate. These are your choices:
		'log_levels' (default) -- plot whatever the variable 'lev' gives (e.g. pressure in hPa) on a logarithmic scale 
		'levels' -- plot whatever the variable 'lev' gives (e.g. pressure in hPa) on a linear scale 
		'z' -- convert lev (assumed to be pressure) into log-pressure height coordinates uzing z=H*exp(p/p0) where p0 = 1000 hPa and H=7km  
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
			try:
				Vnorm = Vmatrix/VPlonave
			except ValueError: 
				# warning: I have my suspicions about this...it might be that 
				# reshape effs it up. 
				VPlonave2 = np.reshape(VPlonave,Vmatrix.shape)
				Vnorm = Vmatrix/VPlonave2
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

	# compute vertical coordinate depending on choice of pressure or altitude 
	if 'levels' in vertical_coord:
		y=lev
		ylabel = 'Level (hPa)'
	if vertical_coord=='z':
		H=7.0
		p0=1000.0 
		y = H*np.log(p0/lev)
		ylabel = 'log-p height (km)'
	if vertical_coord=='TPbased':
		#from matplotlib import rcParams
		#rcParams['text.usetex'] = True
		y=lev
		ylabel='z (TP-based) (km)'

	# create a mesh
	X,Y = np.meshgrid(lat,y)

        # plot
	Q= plt.quiver(X[::narrow,::narrow],Y[::narrow,::narrow],
		alpha[0]*Mlist[0][::narrow, ::narrow],alpha[1]*Mlist[1][::narrow, ::narrow],
		pivot='mid', units='inches', width=0.022,
               scale=arrowscale)
	#qk = plt.quiverkey(Q, 0.5, 0.03, 1, r'$1 \frac{m}{s}$',
	#		   fontproperties={'weight': 'bold'})


	# axis labels 
	plt.xlabel('Latitude')
	plt.ylabel(ylabel)
	if vertical_coord=='log_levels':
		plt.yscale('log')
		plt.ylabel('Pressure (hPa)')
	if 'levels' in vertical_coord:
		plt.gca().invert_yaxis()

	# make sure the axes only go as far as the ranges in E
	plt.xlim(E['latrange'])

	# return the colorbar handle if available, so we can adjust it later
	return Mlist

def Nsq_from_3d(E,date,hostname='taurus',debug=False):

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


	# if the data are on hybrid levels, check if pressure data are available somewhere 
	# otherwise, reconstruct the pressure field at each point from hybrid model variables 
	if (E['levtype']=='hybrid') or (E['levtype']=='model_levels'):
		H = dict()
		EP = E.copy()
		ET = E.copy()
		EP['variable']='P'
		ET['variable']='T'
		# for ERA data, use one of the subroutines in the ERA module to load pressure and temp:
		if 'ERA' in E['exp_name']:
			import ERA as era
			import re
			resol = float(re.sub('\ERA', '',E['exp_name']))
			P,lat,lon,lev,time2 = era.load_ERA_file(EP,date,resol=resol,hostname=hostname,verbose=debug)
			T,lat,lon,lev,time2 = era.load_ERA_file(ET,date,resol=resol,hostname=hostname,verbose=debug)
		else:
			# for DART runs, look for P and T in DART diagnostic files: 
			lev,lat,lon,P,P0,hybm,hyam = dart.load_DART_diagnostic_file(EP,date,debug=debug)
			lev,lat,lon,T,P0,hybm,hyam = dart.load_DART_diagnostic_file(ET,date,debug=debug)
			# TODO: if P is not in a DART diagnostic file, it could also be in a model history file, 
			# so need to add a line of code to try looking for that as well 
		if P is None:
			if debug:
				print('Pressure not available for requested date - recreating from hybrid levels (this takes a while....)')
			# special subroutine if we are dealing with ERA data, where usually log(Ps) is available insted of PS  
			if 'ERA' in E['exp_name']:
				P,lat,lon,lev = era.P_from_hybrid_levels_era(E,date,hostname=hostname,debug=debug)
			else:
				# otherwise, construct pressure the way it's done in CAM/WACCM
				# TODO: make this depend on the model input, so that we can more easily 
				# add settings for other models later  
				P,lat,lon,lev = P_from_hybrid_levels(E,date,hostname=hostname,debug=debug)


# if the data are on pressure levels, simply retrieve the pressure grid and turn it into a 3d field  
# TODO: add code for loading DART/WACCM output on constant pressure levels. Right now this 
	# only works for ERA data. 
	if E['levtype']=='pressure_levels':
		varlist = ['T','Z3']
		H = dict()
		if ('ERA' in E['exp_name']):
			import ERA as era
			for vname in varlist:
				Etemp = E.copy()
				Etemp['variable']=vname
				import re
				resol = float(re.sub('\ERA', '',E['exp_name']))
				field,lat,lon,lev,time_out = era.load_ERA_file(Etemp,date,hostname=hostname,verbose=debug,resol=resol)
				#H[vname]=np.squeeze(field)
				H[vname]=field
			# 3D pressure array from 1D array
			# here "3D" actually si 4D, because we have a dummy dimension for "copy" 
			nlat = len(lat)
			nlon = len(lon)
			P1 = np.repeat(lev[:,np.newaxis],nlat,axis=1)
			P = np.repeat(P1[:,:,np.newaxis],nlon,axis=2)
			T=H['T']

	# choose reference pressure as 1000 hPa, with units based on the max of the P array 
	if np.max(P) > 2000.0:
		P0 = 100000.0			# reference pressure 
	else:
		P0 = 1000.0			# reference pressure 

	# compute potential temperature  
	Rd = 286.9968933                # Gas constant for dry air        J/degree/kg
	g = 9.80616                     # Acceleration due to gravity       m/s^2
	cp = 1005.0                     # heat capacity at constant pressure    m^2/s^2*K
	theta = T*(P0/P)**(Rd/cp)

	# turn the 3d pressure array into a geometric height array 
	z = 7000.0*np.log(P0/P)

	# compute the vertical gradient in potential temperature 
	# this still has issues when lat or lon have length 1 -- need to fix this.
	dZ = np.gradient(np.squeeze(z))	# 3D gradient of height (with respect to model level) 
	try:
		dthetadZ_3D = np.gradient(np.squeeze(theta,axis=0),dZ[0])
	except ValueError:
		print('shape of z and its gradient')
		print(z.shape)
		print(dZ[0].shape)
		print(np.squeeze(theta,axis=0).shape)
	dthetadZ = dthetadZ_3D[0] # this is the vertical temperature gradient with respect to pressure 

	# compute the buoyancy frequency 
	try:
		N2 = (g/theta)*dthetadZ
	except ValueError: 
		print('problemo with shaoes that comprise N2') 
		print(theta.shape)
		print(dthetadZ.shape)
		print(z.shape)

	return N2,lat,lon,lev


def P_from_hybrid_levels(E,date,hostname='taurus',debug=False):

	"""
	given a DART experiment dictionary on a certain date and time,
	recreate the pressure field given the hybrid model level parameters 
	**note:** this code was crafted for WACCM/CAM data, and returns a pressure array 
	that fits te latxlonxlev structure of WACCM/CAM history files. 
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

	if lev is None:
		print(Ehyb)

	nlev = len(lev)
	nlat = len(lat)
	nlon = len(lon)
	P = np.zeros(shape = (nlat,nlon,nlev))
	for k in range(nlev):
		for i in range(nlon):
			for j in range(nlat):
				P[j,i,k] = H['hyam'][k]*H['P0'] + H['hybm'][k]* np.squeeze(H['PS'])[j,i]

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

def DART_diagn_to_array(E,hostname='taurus',debug=False,return_single_variables=False):

	"""
	This subroutine loops over the dates given in E['daterange'] and load the appropriate DART diagnostic for each date, 
	returning a numpy matrix of the relevant date.  

	The file type that we load depends on the entry `file_type` in the dictionary E. 
	Here are the types of files that work:  
	+ monthly: if this string is added to whatever is in E['file_type'], the input daterange is automatically changed to monthly values.  
	+ ANOM: anomalies from climatology, climatologies themselves, or climatological standard deviations. These are loaded using the `ano` and `stds` subroutines in MJO.py  
	+ ERA: ERA-Interim or ERA-40 data  
	+ DART: regular DART diagnostic files (these usually have names like 'Posterior_diagn_XXXX.nc')  
	+ WANG-TEM: For loading transformed Eulerian mean diagnostics made by Wuke Wang. These are pretty ad-hoc files and this should be deprecated eventually.  
	+ COVAR: covariances and correlations, calculated using FILL IN  
	+ SPECIAL: this is for quantities that have to be calculated from other subroutines. This is a good place to add paths to functions that compute obscure quantities. For example, buoyancy frequency forcing due to the residual circulation is one of the things you can compute here.  
	+ WACCM: WACCM and CAM-type history files  


	This code returns a dictionary, Dout, which holds the requested variable, its corresponding 
	spacial dimension arrays (e.g. lat, lon, lev), units, and long name. 
	To get  these as single variables, set the input parameter return_single_variables to True. This will be 
	deprecated eventually when all other visualization codes are changed to deal with single variables.  
	"""
	import pprint

	# read in the file type  and check if it's admissable  
	if 'file_type' in E:
		FT = E['file_type']
	else:
		FT='DART'
	file_types_list = ['ANOM','ERA','DART','WANG-TEM','COVAR','SPECIAL','WACCM']
	if FT not in file_types_list:
		if debug:
			print('DART_diagn_to_array: This code is not set up to handle the given file type:  '+FT)
			print('Assuming that you just want DART-type output data and loading that.')
			FT = 'DART'

	# adjust daterange if loading monthly data 
	if 'monthly' in FT:
		DRm = [datetime.datetime(dd.year,dd.month,1,12,0) for dd in E['daterange']]
		DR = list(set(DRm))
	else:
		DR = E['daterange']

	#----------------ANOMALIES------------------------------
	# if plotting anomalies from climatology, climatology, or a climatological standard deviation, 
	# can load these using the `stds` and `ano` rubroutines in MJO.py  
	if FT is 'ANOM':
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
		return Vmatrix,lat,lon,lev,new_daterange
		#TODO: make this return a dict

	# ------data types that loop over date ranges  
	Vlist = []

	for date in DR:

		# ERA-40 and ERA-Interim data 
		if FT is 'ERA':
			if (E['variable'] == 'Nsq'):
				# ERA buoyancy frequency can be calculated with the Nsq function 
				V,lat,lon,lev = Nsq_from_3d(E,date,hostname=hostname,debug=debug)
			if 'V' not in locals():
				# all other variables are loaded via a function in the ERA module:
				import ERA as era
				import re
				resol = float(re.sub('\ERA', '',E['exp_name']))
				V,lat,lon,lev,dum = era.load_ERA_file(E,date,resol=resol,hostname=hostname,verbose=debug)

		# regular DART diagnostic files (these usually have names like 'Posterior_diagn_XXXX.nc')	
		if FT is 'DART':
			try:
				DD = dart.load_DART_diagnostic_file(E,date,hostname=hostname,debug=debug)
				V = DD['data']
			except RuntimeError:
				error_msg_DART_diagn_to_array(FT,E)
				V = None

		# Wuke Wang TEM diagnostics  
		if FT == 'WANG-TEM':
			try:
				DD = compute_DART_diagn_from_Wang_TEM_files(E,date,hostname=hostname,debug=debug)
				lon = None
				V = DD['data']
			except RuntimeError:
				error_msg_DART_diagn_to_array(FT,E)
				V = None

		# Covariances and correlations 
		if FT is 'COVAR':
			try:
				lev,lat,lon,Cov,Corr = dart.load_covariance_file(E,date,hostname,debug=debug)
				if E['diagn'].lower() == 'covariance':
					V = Cov
				if E['diagn'].lower() == 'correlation':
					V = Corr
			except RuntimeError:
				error_msg_DART_diagn_to_array(FT,E)
				V = None

				
		# other obscure calculations: 	
		if FT is 'SPECIAL':
			# buoyancy frequency forcing due to residual circulation 
			if (E['variable'] == 'Nsq_wstar_forcing') or (E['variable'] == 'Nsq_vstar_forcing'):
				import TIL as til
				DD = til.Nsq_forcing_from_RC(E,date,hostname=hostname,debug=debug)
				lon = None
				V = DD['data']
			# similar buoyancy frequency forcing from diabaitcc heating 
			if 'Nsq_forcing_' in E['variable']: 
				import TIL as til
				DD = til.Nsq_forcing_from_Q(E,date,hostname=hostname,debug=debug)
				lon = None
				V = DD['data']

			# it might be that pressure needs to be recreated from the hybrid model levels 
			# Note that it is easier and faster
			#  to just compute pressure in the format of DART diagnostic files and then read those in. 
			if E['variable'] == 'P':
				V,lat,lon,lev = P_from_hybrid_levels(E,date,hostname=hostname,debug=debug)

			# buoyancy frequency 
			if E['variable'] == 'Nsq':
				V,lat,lon,lev = Nsq(E,date,hostname=hostname,debug=debug)


		if FT is 'WACCM':

				# for WACCM and CAM runs, if we requested US or VS, have to change these to U and V, 
				# because that's what's in the WACCM output 
				if E['variable'] is 'US':
					E['variable'] = 'U'
				if E['variable'] is 'VS':
					E['variable'] = 'V'
				DD = compute_DART_diagn_from_model_h_files(E,date,hostname=hostname,verbose=debug)
				V = DD['data']

		# add the variable field just loaded to the list:
		Vlist.append(V)

	# if Vlist still has length 0, we didn't find any data -- abort 
	if len(Vlist)>0:
	
		# if Vlist has length, 
		# remove any Nones that might be in there and check again 
		Vlist2 = [V for V in Vlist if V is not None]
		if len(Vlist2)>0:
			bad = [i for i, j in enumerate(Vlist) if j is None]
			new_daterange = [i for j, i in enumerate(E['daterange']) if j not in bad]
			# turn the list of variable fields into a matrix 
			Vmatrix = np.concatenate([V[..., np.newaxis] for V in Vlist2], axis=len(V.shape))
			# make sure we transfer fill values if they exist
			if 'FillValue' in DD:
				Vmatrix=np.ma.masked_values(Vmatrix,DD['FillValue'])

		else:
			d1 = E['daterange'][0].strftime("%Y-%m-%d")
			d2 = E['daterange'][len(E['daterange'])-1].strftime("%Y-%m-%d")
			print('Could not find any data for experiment '+E['exp_name']+' and variable '+E['variable']+' between dates '+d1+' and '+d2)
			return None,None,None,None,None
	else:
		d1 = E['daterange'][0].strftime("%Y-%m-%d")
		d2 = E['daterange'][len(E['daterange'])-1].strftime("%Y-%m-%d")
		print('Could not find any data for experiment '+E['exp_name']+' and variable '+E['variable']+' between dates '+d1+' and '+d2)
		return None,None,None,None,None

	if return_single_variables:
		return Vmatrix,lat,lon,lev,new_daterange
	else:
		DD['data']=Vmatrix
		DD['daterange']=new_daterange
		return DD

def error_msg_DART_diagn_to_array(FT,E):

	"""
	Return an error message when a certain experiment, defined by the dictionary E, 
	can't be found for a given file type, given in the string FT 
	"""
	print('DART_diagn_to_array: Cannot find the requested data in file type '+FT)
	print('here is the whole experiment dict:')
	pprint.pprint(E, width=1)


def plot_diagnostic_profiles(E=dart.basic_experiment_dict(),Ediff=None,color="#000000",linestyle='-',linewidth = 2,alpha=1.0,scaling_factor=1.0,hostname='taurus',vertical_coord='log_levels',label_for_legend=True,debug=False):

	"""
	Plot a vertical profile of some DART diagnostic / variable, 
	averaged over the date, latitude, and longitude ranges given in the 
	experiment dictionary.

	Instead of the zonal or meridional mean, we can also take the max of one of those dimensions. 
	To do this, add the words 'lonmax' or 'latmax' to the E['extras'] entry of the experiment 
	dictionary.  

	INPUTS:
	E: DART experiment dictionary of the primary experiment/quantity that we want to plot 
	Ediff: DART experiment dictionary of the experiment/quantity that we want to subtract out  (default is None)  
	color, linestyle, linewidth, alpha: parameters for the plotting (optional) 
	scaling_factor: factor by which we multiply the profile to be plotted (default is 1.0)    
	hostname: the computer this is being run on (default is taurus)  
	vertical_coord: option for how to plot the vertical coordinate. These are your choices:
		'log_levels' (default) -- plot whatever the variable 'lev' gives (e.g. pressure in hPa) on a logarithmic scale 
		'levels' -- plot whatever the variable 'lev' gives (e.g. pressure in hPa) on a linear scale 
		'z' -- convert lev (assumed to be pressure) into log-pressure height coordinates uzing z=H*exp(p/p0) where p0 = 1000 hPa and H=7km  
		'TPbased': in this case, compute the height of each gridbox relative to the local tropopause and 
			plot everything on a "tropopause-based" grid, i.e. zt = z-ztrop-ztropmean 
	label_for_legend: if this is set to true, the profile being plotted is assigned whatever is given in E['title']
		to appear when you do plt.legend(). The default is True; set this to False to ignore certain profiles 
		(e.g. individual ensemble members) in the legend. 
	debug: set to True to print out extra output 
	"""
	daterange = E['daterange']

	# throw an error if the desired variable is 2 dimensional 
	if (E['variable'] == 'PS') or (E['variable'] == 'FLUT'):
		print('Attempting to plot a two dimensional variable ('+E['variable']+') over level and latitude - need to pick a different variable!')
		return

	# check if the desired variable is a sum
	if ('+' in E['variable']):
		variable_list = E['variable'].split('+')
	else:
		variable_list=[E['variable']]
	Vmatrix_list=[]
	for variable in variable_list:
		Etemp=E.copy()
		Etemp['variable']=variable
		# load the requested array, and the difference array if needed 
		D = DART_diagn_to_array(Etemp,hostname=hostname,debug=debug)
		# convert to TP-based coordinates if requested 	
		if 'TPbased' in vertical_coord: 
			vcoord_string=vertical_coord.split('.')
			if len(vcoord_string) == 1:
				# iff the mean tropopause for TP-based coordinates is not defined, 
				# just call a Dec-Feb mean 
				meantrop='DJFmean'
			else:
				meantrop=vcoord_string[1]
			D=to_TPbased(E,D,meantrop=meantrop,hostname=hostname,debug=debug)
		#else:
		#	Vmain=D['data']
		#	try:
		#		lev=D['lev']
		#	except KeyError:
		#		print(D.keys())
		#		return

		if Ediff is not None:
			Etempdiff=Ediff.copy()
			Etempdiff['variable']=variable
			D2 = DART_diagn_to_array(Etempdiff,hostname=hostname,debug=debug)
			# convert to TP-based coordinates if requested 	
			if vertical_coord=='TPbased': 
				D2=to_TPbased(Etempdiff,D2,hostname=hostname,debug=debug)
			Vmatrix=D['data']-D2['data']
		else:
			Vmatrix=D['data']

		Vmatrix_list.append(Vmatrix)

	if ('+' in E['variable']):
		Vmatrix = sum(V for V in Vmatrix_list)

	# make sure bad values are masked out 
	#if 'FillValue' in D:
	#	Vmatrix=np.ma.masked_values(Vmatrix,FillValue)

	# average over time 
	V0 = average_over_named_dimension(Vmatrix,D['daterange'])
	
	# average over latitude 
	if D['lat'] is not None:
		V1 = average_over_named_dimension(V0,D['lat'])
	else:
		V1 = V0

	# average over longitude
	if 'lon' in D:
		if D['lon'] is not None:
			V2 = average_over_named_dimension(V1,D['lon'])
		else:
			V2 = V1
	else:
		V2=V1

	# finally, apply the scaling factor 
	M = scaling_factor*V2

	# compute vertical coordinate depending on choice of pressure or altitude 
	if 'levels' in vertical_coord:
		y=D['lev']
		ylabel = 'Level (hPa)'
	if vertical_coord=='z':
		H=7.0
		p0=1000.0 
		y = H*np.log(p0/D['lev'])
		ylabel = 'log-p height (km)'
	if 'TPbased' in vertical_coord:
		#from matplotlib import rcParams
		#rcParams['text.usetex'] = True
		y=D['lev']
		ylabel='z (TP-based) (km)'

        # plot the profile  - loop over copies if that dimension is there  
	# from the way DART_diagn_to_array works, copy is always the 0th dimension  

	if M.ndim == 2:
		nC = M.shape[0]
		for iC in range(nC):
			if type(color) is 'list':
				color2 = color[iC]
			else:
				color2=color 
			if label_for_legend:
				plt.plot(M[iC,:],y,color=color2,linestyle=linestyle,linewidth=linewidth,label=E['title'],alpha=alpha)
			else:
				plt.plot(M[iC,:],y,color=color2,linestyle=linestyle,linewidth=linewidth,alpha=alpha)
	else:
		plt.plot(M,y,color=color,linestyle=linestyle,linewidth=linewidth,label=E['title'],alpha=alpha)

	# x axis stuff 
	ax = plt.gca()
	xlim = ax.get_xlim()[1]
	ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
	if not 'long_name' in D:
		D['long_name']=''
	if not 'units' in D:
		D['units']=''
	plt.xlabel(D['long_name']+' ('+D['units']+')')

	# y axis stuff 
	plt.ylabel(ylabel)
	if vertical_coord=='log_levels':
		plt.yscale('log')
	if (vertical_coord=='log_levels') or (vertical_coord=='levels'):
		plt.gca().invert_yaxis()

	# make sure the axes only go as far as the ranges in E
	if 'levels' in vertical_coord:
		plt.ylim(E['levrange'])
	else:
		H=7.0
		p0 = 1000.0
		ylim0=H*np.log(p0/E['levrange'][0])
		if E['levrange'][1]==0:
			ylimf = np.max(y)
		else:
			ylimf=H*np.log(p0/E['levrange'][1])
		ylim=(ylim0,ylimf)
		plt.ylim(ylim)
	return M,y
	

def plot_diagnostic_lat(E=dart.basic_experiment_dict(),Ediff=None,color="#000000",linestyle='-',linewidth = 2,alpha=1.0,hostname='taurus',scaling_factor=1.0,invert_yaxis=False,debug=False):

	"""
	Retrieve a DART diagnostic (defined in the dictionary entry E['diagn']) and plot it 
	as a function of latitude 
	Whatever diagnostic is chosen, we average over all longitudes in E['lonrange'] and 
	all times in E['daterange'], and if the quantity is 3d, average over vertical levels  

	INPUTS:
	E: basic experiment dictionary
	Ediff: experiment dictionary for the difference experiment
	hostname: name of the computer on which the code is running
	ncolors: how many colors the colormap should have. Currently only supporting 11 and 18. 
	colorbar_label: string with which to label the colorbar  
	scaling_factor: factor by which to multiply the array to be plotted 
	debug: set to True to get extra ouput
	"""

	# load the desired DART diagnostic for the desired variable and daterange:
	VV,lat,lon,lev,new_daterange = DART_diagn_to_array(E,hostname=hostname,debug=debug)

	# load the difference array if desired  
	if Ediff is not None:
		Vdiff,lat,lon,lev,new_daterange = DART_diagn_to_array(Ediff,hostname=hostname,debug=debug)
		Vmatrix = VV-Vmatrix
	else:
		Vmatrix = VV
		
	# average over time, longitude, and vertical levels  
	V0 = average_over_named_dimension(Vmatrix,new_daterange)
	if lon is not None:
		V1 = average_over_named_dimension(V0,lon)
	else:
		V1 = V0
	if lev is not None:
		V2 = average_over_named_dimension(V1,lev)
	else:
		V2 = V1

	# squeeze out any remaining length-1 dimensions and scale 
	M = scaling_factor*np.squeeze(V2)

	# transpose the array if necessary  
	if len(M.shape)>1:
		if M.shape[0]==len(lat):
			MT = np.transpose(M)
		else:
			MT = M
	else:
		MT = M

	# if we are plotting multiple copies (e.g. the entire ensemble), need to loop over them  
	# otherwise, the plot is simple
	if len(MT.shape) == 2:
		ncopies = MT.shape[0]
		for icopy in range(ncopies):
			plt.plot(lat,MT[icopy,:],color=color,linestyle=linestyle,linewidth=linewidth,label=E['title'],alpha=alpha)
	else:
		plt.plot(lat,MT,color=color,linestyle=linestyle,linewidth=linewidth,label=E['title'],alpha=alpha)

	# axis labels 
	plt.xlabel('Latitude')

	# vertical axis adjustments if desired (e.g. if plotting tropopause height) 
	if invert_yaxis:
		plt.gca().invert_yaxis()

	# make sure the axes only go as far as the ranges in E
	plt.xlim(E['latrange'])

	return MT,lat

def plot_diagnostic_lon(E=dart.basic_experiment_dict(),Ediff=None,color="#000000",linestyle='-',linewidth = 2,alpha=1.0,hostname='taurus',scaling_factor=1.0,invert_yaxis=False,debug=False):

	"""
	Retrieve a DART diagnostic (defined in the dictionary entry E['diagn']) and plot it 
	as a function of longitude
	Whatever diagnostic is chosen, we average over all latgitudes in E['latrange'] and 
	all times in E['daterange'], and if the quantity is 3d, average over vertical levels  

	INPUTS:
	E: basic experiment dictionary
	Ediff: experiment dictionary for the difference experiment
	hostname: name of the computer on which the code is running
	ncolors: how many colors the colormap should have. Currently only supporting 11 and 18. 
	colorbar_label: string with which to label the colorbar  
	scaling_factor: factor by which to multiply the array to be plotted 
	debug: set to True to get extra ouput
	"""

	# load the desired DART diagnostic for the desired variable and daterange:
	D = DART_diagn_to_array(E,hostname=hostname,debug=debug)

	# load the difference array if desired  
	if Ediff is not None:
		Dfiff = DART_diagn_to_array(Ediff,hostname=hostname,debug=debug)
		Vmatrix = D['data']-Ddiff['data']
	else:
		Vmatrix = D['data']
		
	# average over time, latitude, and vertical levels  
	lat = D['lat']
	lev = D['lev']
	lon = D['lon']
	V0 = average_over_named_dimension(Vmatrix,D['daterange'])
	if lat is not None:
		V1 = average_over_named_dimension(V0,lat)
	else:
		V1 = V0
	if lev is not None:
		V2 = average_over_named_dimension(V1,lev)
	else:
		V2 = V1

	# squeeze out any remaining length-1 dimensions and scale 
	M = scaling_factor*np.squeeze(V2)

	# transpose the array if necessary  
	if len(M.shape)>1:
		if M.shape[0]==len(lat):
			MT = np.transpose(M)
		else:
			MT = M
	else:
		MT = M

	# if we are plotting multiple copies (e.g. the entire ensemble), need to loop over them  
	# otherwise, the plot is simple
	if len(MT.shape) == 2:
		ncopies = MT.shape[0]
		for icopy in range(ncopies):
			plt.plot(lon,MT[icopy,:],color=color,linestyle=linestyle,linewidth=linewidth,label=E['title'],alpha=alpha)
	else:
		plt.plot(lon,MT,color=color,linestyle=linestyle,linewidth=linewidth,label=E['title'],alpha=alpha)

	# axis labels 
	plt.xlabel('Latitude')

	# vertical axis adjustments if desired (e.g. if plotting tropopause height) 
	if invert_yaxis:
		plt.gca().invert_yaxis()

	# make sure the axes only go as far as the ranges in E
	plt.xlim(E['lonrange'])

	return MT,lon

def to_TPbased(E,D,meantrop='DJFmean',hostname='taurus',debug=False):

	"""
	This routine takes some multi-dimensional variable field and a corresponding array for vertical levels, 
	and transforms the vertical coordinate into altitudes defined relative to the local tropopause, plus
	the time-mean tropopause in that location, i.e. zt = z-ztrop-ztropmean 
	(See [Birner 2006](http://www.agu.org/pubs/crossref/2006/2005JD006301.shtml))

	After computing the TP-based height at each location, we run through all latitudes, 
	longitudes, times, and copies, and interpolate the tropopause-based heights 
	to a regular grid so that we can average. 
	The `interp1d` function creates a functional relationship between the variable in 
	Vmatrix and the TP-based coordinates, and the grid to which we interpolate has to be 
	within the bounds of this function (i.e. the min and max values of TP-based altitude 
	for each column).  -- Might have to play with this for your own data. 
 
	INPUTS:
	E: a DART experiment dictionary giving the details of the data that we are requesting 
	D: a dictionary holding the data matrix that we want to interpolate under 'data', an 
	array of vertical levels under 'lev', and a 'FillValue' entry to denote bad values 
	lev: a vector of vertical level pressures. These can be in Pascal or hPa. 
	meantrop: a string denoting how we compute the mean tropopause. This has to also appear 
		in the filename that holds mean tropopause height (default is 'DJFmean')
	"""

	# given the data matrix, we have to retrieve several other things: 
	# and the climatological-mean tropopause height for every point on the grid. 
	#   this last one is most easily computed by using the ensemble mean of a corresponding 
	#   No-DA experiment. 
	#
	# First define all the things we need in experiment dictionaries, and then 
	# stick those into a list to loop over 
	Vmatrix = D['data']
	lev = D['lev']

	# Vmatrix should be a masked array, and to make sure interp1d doesn't mess with 
	# the bad values, replace them with NaNs
	Vmatrix = np.ma.fix_invalid(Vmatrix,fill_value=np.nan)

	# tropopause height of the experiment 
	Etrop=E.copy()
	Etrop['variable']='ptrop'
	Etrop['matrix_name']='ztrop'

	# the pressure field of the requested experiment 
	# this is used to compute the altitude of every grid point 
	Ep=E.copy()
	Ep['variable']='P'
	Ep['matrix_name']='z'  

	# the DJF-mean tropopause height
	Etropmean=E.copy()
	Etropmean['variable']='ptrop'
	Etropmean['matrix_name']='ztropmean'  
	Etropmean['daterange']=[meantrop]  

	# now loop over these experiment and retrieve the data, also converting pressures to altitudes 
	# stick these into a dictionary 
	EE = [Etrop,Ep,Etropmean]
	Zdict = dict()
	for Etemp in EE:
		if Etemp is not None:
			# dirty fix: typically we don't want to retrieve these data from "special" versions of the data 
			# denoted by some entry in E['extrastring'] (e.g. filtered data), so set that parameter back here. 
			Etemp['extrastring']=''

			# the pressure field wont be around if the data are on levels 
			# of constant pressure 
			# instead load temp field and then expand the constant levels array to be 
			# of the same shape 
			if (Etemp['variable']=='P') and (Etemp['levtype']=='pressure_levels'):
				Etemp['variable']='T'
				Dtemp= DART_diagn_to_array(Etemp,debug=debug)
				VT = Dtemp['data']
				Px = Dtemp['lev']		# these are the pressures at each level 
				for idim,dimlength in enumerate(VT.shape):
					if dimlength != len(levT):
						Px = np.expand_dims(Px,axis=idim)
				V = np.broadcast_to(Px,VT.shape)
			else:
				# otherwise, load the field  of whatever was requested - pressure, tropopopause pressure, or mean trop pressyre 
				Dtemp= DART_diagn_to_array(Etemp,debug=debug)
				V = Dtemp['data']
			try:
				if (Dtemp['units']=='Pa') or (Dtemp['units']=='Pascal'):
					P0=1.0E5
				else:                        # otherwise assume pressure is in hPa
					P0=1.0E3
			except ValueError:  #raised if `V` has an empty dimension, which sometimes happens when we load netcdf files with time means 
				pass
			Z = H*np.log(P0/V)

			# for tropopause heights, convert 2d to 3d array by adding an additional dimension 
			if 'ztrop' in Etemp['matrix_name']:
				# find which is the vertical levels dimension 
				nlev = len(lev)
				levdim = list(Vmatrix.shape).index(nlev)  
				Zx = np.expand_dims(Z, axis=levdim)
				try:
					Z3d=np.broadcast_to(Zx,Vmatrix.shape)
				except ValueError:
					print('to_TPbased: there is a mismatch between the tropopause height array we created and the matrix we are broadcasting to:')
					print(Zx.shape)
					print(Vmatrix.shape)
			else:
				Z3d=Z
				       
			# add final array to dictionary 
			Zdict[Etemp['matrix_name']]=Z3d

	#now for each point, compute z-ztrop+ztropmean
	ZT = Zdict['z']-Zdict['ztrop']+Zdict['ztropmean']

	# create a regular grid 
	zTPgrid=np.arange(6.0,26.0, 1.0)

	# empty array to hold interpolated data
	Snew = list(Vmatrix.shape)
	Snew[levdim] = len(zTPgrid)
	Vnew = np.empty(shape=Snew)*np.nan

	# note that so far I've only coded this for two array shapes 
	# -- add more if needed 

	S=Vmatrix.shape

	from scipy.interpolate import interp1d
	if levdim==1:
		# if levels is the 1st dimension, then the jj index is for the third dimension
		jjindex = 3
	if levdim==3:
		# if levels is the 3rd dimension, then the jj index is for the first dimension
		jjindex = 1
	for ii in range(S[0]):
		for jj in range(S[jjindex]):
			for kk in range(S[2]):
				for ll in range(S[4]):
					if levdim == 3:
						Vcolumn = Vmatrix[ii,jj,kk,:,ll]
						ZTcolumn = ZT[ii,jj,kk,:,ll]
					if levdim == 1:
						Vcolumn = Vmatrix[ii,:,kk,jj,ll]
						ZTcolumn = ZT[ii,:,kk,jj,ll]

					# here is the interpolation function:
					try:
						f = interp1d(ZTcolumn,Vcolumn, kind='cubic')
					except ValueError:
						print('these data columns give the interpolation trouble:')
						print(Vcolumn)
						print(ZTcolumn)
						print('these are the shapes of the temp and trop heigh arrays:')
						print(Vmatrix.shape)
						print(ZT.shape)
						print(E)

					# check whether the sampled ZTcolumn covers the grid we interpolate to
					select = np.where(np.logical_and(zTPgrid>np.min(ZTcolumn), zTPgrid<np.max(ZTcolumn)))
					zTPnew=zTPgrid[select]
					if levdim == 1:
						Vnew[ii,select,kk,jj,ll] = f(zTPnew)
					if levdim == 3:
						Vnew[ii,jj,kk,select,ll] = f(zTPnew)
					
					# need to check whether the sampled ZTcolumn covers the 
					# grid to which we want to interpolate
					#if (np.min(zTPgrid) < np.min(ZTcolumn)) or (np.max(zTPgrid) > np.max(ZTcolumn)):
					#	Vnew[ii,jj,kk,:,ll] = np.nan
					#else:
					#	Vnew[ii,jj,kk,:,ll] = f(zTPgrid)


	# applying interp1d fucks up the mask over bad values 
	# (I guess it comes up wtih values close to the masked value)
	# to need to reapply it, with some closer values 
	if 'FillValue' in D:
		mask = np.isclose(Vnew,D['FillValue'])
		Vnew2 = np.ma.array(Vnew,mask=mask)
	else:
		Vnew2 = Vnew  

	Dout = D
	Dout['data']=Vnew2
	Dout['lev']=zTPgrid
	
	return D

def nice_colormaps(cmap_type='sequential',reverse_colors=False):

	"""
	A tool for picking colormaps for contour plots that look nice.

	INPUTS:
	cmap_type: choose 'sequential' or 'divergent' (sequential is the default). 
	reverse_colors: set to True to flip the colormap. The default is False. 


	OUTPUT: 
	a palettable object from which you can extract mpl_colors, colors, number of colors, etc. 
	"""


	# choose qualitative, sequential, or divergent colormap  
	cname_dict = dict()
	# sequential maps look cool with the cubehelix-type maps, for which
	# number 2 looks like a good balance of colorful and garish. 
	cname_dict['sequential'] = 'cubehelix.cubehelix2_16'
	# colorbrewer has nice divergent maps -- let's choose a simple red-blue with lots of shades. 
	cname_dict['divergent'] = 'colorbrewer.diverging.RdBu_11'
	cname_dict['qualitative'] = 'colorbrewer.qualitative.Dark2_8'


	cname = cname_dict[cmap_type]

	if reverse_colors:
		rev='_r'
	else:
		rev=''

	cmap = eval('pb.'+cname+rev)
	return cmap

def average_over_named_dimension(V,dim):

	"""
	This subroutine takes a multi-dimensional data matrix and finds the dimension that matches 
	the length of a given input dimension, and then averages over that. 
	So for example, if you have an atmosphere data array shaped like lat x lon x lev 
	and you put in that array and an array with the levels, you will get back an average over the 
	3rd dimension.   

	INPUTS:  
	V: multi dimensional data array  
	dim: dimension array (1xN, where N is the length of the dim in question)  
	"""

	# the input matrix should be a masked array. For some reason, even though values are masked 
	# the mean performed here is also performed on those values, which screws the mask
	# so as a temporary solution, convert the masked values to nans. 
	#V= np.ma.fix_invalid(V,fill_value=np.nan)

	for idim,dimlen in enumerate(V.shape):
		if dimlen == len(dim):
			desired_dimension_number = idim
	if 'desired_dimension_number' not in locals():
		print("Looking for dimension of this shape:")
		print(dim.shape)
		print("In variable of this shape:")
		print(V.shape)
		raise RuntimeError("average_over_named_dimension cannot find the right dimension")
	Vave = np.nanmean(V,axis=desired_dimension_number)

	return(Vave)

