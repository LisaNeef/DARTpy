# Python module for codes that deal with ERA-Interim output
# Lisa Neef, 16 Mar 2015

# load the required packages  
import numpy as np
import datetime
import os.path
import DART as dart
from netCDF4 import Dataset
import experiment_settings as es

#-------reading in merged ERA40/Interim files given a DART experiment dictionary----------------------
def load_ERA_file(E,datetime_in,resol=1.5,hostname='taurus',verbose=False):

	"""
	This subroutine loads a file from our merged ERA-40/Interim data 
	as if it were a DART experiment, i.e. given an "experiment dictionary" 
	as defined in DART.py

	Output is the variable field, and corresponding lat, lon, lev arrays. 
	Only the data over the lat, lon, lev, and date ranges specified in E 
	is returned. 

	Note also that even though the ERA data have levels in Pa, here we 
	convert them to hPa, to make stuff comparable to DART-WACCM.  

	INPUTS:
	E: a standard DART experiment dictionary. Relevant fields are: 
		variable - the variable to load 
		daterange - determines which dates are returned 
		latrange | lonrange | latrange - selects the points that fall into these spatial ranges 
	datetime_in: a datetime-type variable giving the data that we are loading. 
	resol: which resolution should be loaded? Default is 1.5  -- this is actually something 
		that should go in the personal experiment_settings module, so I need to eventually take this out
	"""

	# find the file path corresponding to this experiment  
	ff,dum = es.exp_paths_era(datetime_in,hostname=hostname,resolution=resol,diagnostic=E['diagn'],variable=E['variable'],level_type=E['levtype'])

	variable_found = False

	# load the file  
	if os.path.isfile(ff):
		if verbose:  
			print('Loading ERA file '+ff)
		f = Dataset(ff,'r')
		
		# a list of 2d variables, in which case we don't need to load level  
		# TODO: add other 2d vars to this list 
		variables_2d = ['PS','ptrop']

		# load the grid variables 
		# check whether lat/lon/lev are named as such, or whether the full 
		# words are given 
		if 'latitude' in f.variables:
			lat = f.variables['latitude'][:]
		else:
			lat = f.variables['lat'][:]
		if 'longitude' in f.variables:
			lon = f.variables['longitude'][:]
		else:
			lon = f.variables['lon'][:]
		if E['variable'] in variables_2d:
			lev0 = None
		else:
			if 'level' in f.variables:
				lev0 = f.variables['level']
			else:
				lev0 = f.variables['lev']
			
		time = f.variables['time'][:]
		
		# if the level is in level numbers (rather than approximate pressures) 
		# convert this array to midpoint pressures 
		# (note that these are approximate -- below about 200hPa, the hybrid levels 
		# really follow topography, so there could be use differences in the approximate
		# pressure and the actual pressure at that point  
		if lev0 is not None:
			if lev0.long_name == 'model_level_number':
				levlist = [0.1, 0.292, 0.51, 0.796, 1.151, 1.575, 2.077, 2.666, 3.362, 4.193, 5.201, 6.444, 7.984, 9.892, 12.257, 15.186, 18.815, 23.311, 28.882, 35.784, 44.335, 54.624, 66.623, 80.397, 95.978, 113.421, 132.758, 153.995, 177.118, 202.086, 228.839, 257.356, 287.638, 319.631, 353.226, 388.27, 424.571,461.9,500, 538.591, 577.375, 616.042, 654.273, 691.752, 728.163, 763.205, 796.588, 828.047, 857.342, 884.266, 908.651, 930.37, 949.349, 965.567, 979.063, 989.944, 998.385, 1004.644, 1009.056, 1012.049]
				lev = np.asarray(levlist)
			else:
				lev = lev0[:]
		else:
			lev = lev0

		# first set a general factor that we can scale the variable array by if needed
		prefac = 1.0

		# if the requested variable is available, load it 
		if E['variable'] in f.variables:
			V = f.variables[E['variable']]
		else:
			# if not available, try other names 
			if (E['variable']=='T') or (E['variable']=='TS'):
				possible_varnames = ['T','t','var130']
			if (E['variable']=='U') or (E['variable']=='US'):
				possible_varnames = ['U','u','var131']
			if (E['variable']=='V') or (E['variable']=='VS'):
				possible_varnames = ['V','v','var132']
			if (E['variable']=='Z') or (E['variable']=='geopotential'):
				possible_varnames = ['Z','z','var129']
			if (E['variable']=='GPH') or (E['variable']=='Z3'):
				possible_varnames = ['Z','z','var129']
				prefac = 1/9.8    # convert geopotential to geopotential height
			if (E['variable']=='msl') or (E['variable']=='MSLP'):
				possible_varnames = ['msl','var151']

			if 'possible_varnames' in locals():
				# loop over the list of possible variable names and load the first one we find 
				for varname in possible_varnames:
					if varname in f.variables:
						varname_load = varname
				if 'varname_load' in locals():
					V = f.variables[varname_load]
				else:
					return None,None,None,None,None
			else:
				return None,None,None,None,None

		# replace values with NaNs
		VV = prefac*V[:]
		if hasattr(V,'_FillValue'):
				VV[VV==V._FillValue]=np.nan
		f.close()
	
		# select the vertical and lat/lon ranges specified in E
		# if only one number is specified, find the lev,lat, or lon closest to it

		# note that in the ERA data, levels are given in Pa (whereas in CESM, CAM, WACCM 
		# they are in hPa) -- so multiply the requested range by 100. 
		levrange=E['levrange']
		if lev is not None:
			if levrange is not None:
				# first check whether the levels are given in hPa or Pa 
				# (levrange should be in hPa)
				if max(lev) > 10000:
					# in this casethe levels are in Pascal so levrange must be scaled
					levrange_to_Pa = 100.0
				else:
					# in this case the levels are in hPa, so same units as levrange
					levrange_to_Pa = 1.0

				if levrange[0] == levrange[1]:
					ll = levrange[0]*levrange_to_Pa
					idx = (np.abs(lev-ll)).argmin()
					lev2 = lev[idx]
					k1 = idx
					k2 = idx
				else:
					# level order is reversed in 2.5 and 1.5 degree data 
					# if levels are sorted from surface to TOA
					if lev[0] < lev[len(lev)-1]:
						k1 = (np.abs(lev-levrange[1])).argmin()
						k2 = (np.abs(lev-levrange[0])).argmin()
						# put the output level in hPa
						lev2 = lev[k1:k2+1]
					# if levels are sorted from TOA to bottom:
					if lev[0] > lev[len(lev)-1]:
						k2 = (np.abs(lev-levrange[1]*levrange_to_Pa)).argmin()
						k1 = (np.abs(lev-levrange[0]*levrange_to_Pa)).argmin()
						# put the output level in hPa
						lev2 = lev[k1:k2+1]*(1/levrange_to_Pa)

		latrange=E['latrange']
		j1 = (np.abs(lat-latrange[1])).argmin()
		j2 = (np.abs(lat-latrange[0])).argmin()
		lat2 = lat[j1:j2+1]

		lonrange=E['lonrange']
		i2 = (np.abs(lon-lonrange[1])).argmin()
		i1 = (np.abs(lon-lonrange[0])).argmin()
		lon2 = lon[i1:i2+1]

		# convert time to datetime and select only the ones that fit into the requested range
                #time:units = "hours since 1958-01-01 00:00:00" ;
		reftime = datetime.datetime(1958,1,1,0,0,0)
		time_dates = [reftime+datetime.timedelta(hours=x) for x in list(time)]
		t0 = E['daterange'][0]
		tF = E['daterange'][len(E['daterange'])-1]
		D0 = np.array([t-t0 for t in time_dates])
		DF = np.array([t-tF for t in time_dates])
		t1 = abs(D0).argmin()
		t2 = abs(DF).argmin()
		time2 = time_dates[t1:t2+1]

		if len(VV.shape)==3:
			# 2D variables have shape time x lat x lon
			Vout = VV[t1:t2+1,j1:j2+1,i1:i2+1]
			lev2 = None
		if len(VV.shape)==4:
			# 3D variables have shape time x lev x lat x lon
			Vout = VV[t1:t2+1,k1:k2+1,j1:j2+1,i1:i2+1]
		if len(VV.shape)==1:
			# some variables are just vertical
			Vout = VV[k1:k2+1]
		if 'Vout' not in locals():
			print('unable to deal with the variable shape for variable '+E['variable']+':')
			print(VV.shape)
			return


	# for file not found 
	else:
		if verbose: 
			print('Unable to find ERA-Interim or ERA-40 file '+ff)
		Vout = None
		lat2 = None
		lon2 = None
		lev2 = None
		time2 = None

	return Vout,lat2,lon2,lev2,time2

def retrieve_era_averaged(E,average_latitude=True,average_longitude=True,average_levels=True,hostname='taurus',verbose=False):

	"""
	Given a certain DART experiment dictionary, retrieve the ERA-Interim or ERA-40
	data over the date range given in E['daterange'], and 
	averaged over the spatial ranges defined in E.

	The following parameters can be set to False to eliminate averaging in that dimension (but their defaults are True):  
	average_latitude  
	average_longitude 
	average_levels

	"""


	# -- this next part is deprecated - it loops over files separated by year. 
	# -- but it's more efficient to put files in days and only load the days we want 
	## make a list of the years in this daterange  
	#D0 = E['daterange'][0]
	#DF = E['daterange'][len(E['daterange'])-1]
	#y0 = D0.year
	#yF = DF.year

	## if the desired daterange is all within one year, it's easy
	#if yF==y0:
	#	V,lat,lon,lev,time = load_ERA_file(E,yF,hostname=hostname,verbose=verbose)
	#	time_out = np.array(time)
	#else:
		## loop over years and load the data 
		#VV=[]
		#tt=[]
		#for year in range(y0,yF+1):
		#	Vtemp,lat,lon,lev,time = load_ERA_file(E,year,hostname=hostname,verbose=verbose)
		#	VV.append(Vtemp)
		#	tt.extend(time)

	## turn the list of arrays into an extra long array  
	#V = np.concatenate([v for v in VV],axis=0)
	#time_out = np.array(tt)

	# loop over requested daterange and append arrays 
	VV=[]
	tt=[]
	for date in E['daterange']:
		# decide which resolution of ERA data to load. this is a kludge and highly specific to Lisa's 
		# way of doing things -- need to make this more flexible  
		if E['exp_name']=='ERA':
			resol=2.5
		else:
			resol=1.5
		Vtemp,lat,lon,lev,time_out = load_ERA_file(E,date,resol=resol,hostname=hostname,verbose=verbose)
		VV.append(Vtemp)
		tt.append(time_out)
	V = np.concatenate([v for v in VV],axis=0)
	time = np.array(tt)
	
	# if desired, average over lat, lon, and lev  
	if average_latitude:
		V = np.mean(V,axis=2,keepdims=True)
	if average_longitude:
		V = np.mean(V,axis=3,keepdims=True)
	if average_levels:
		V = np.mean(V,axis=1,keepdims=True)

	# squeeze out any dimensions that have been reduced to one by averaging
	Vout = np.squeeze(V)

	return Vout,time,lat,lon,lev

def construct_era_pressures_from_hybrid(E,datetime_in,resol=2.5,hostname='taurus',verbose=False):

	"""
	Given a certain date, construct the 3d grid of pressure from the hybrid model variables and the 
	ERA surface pressure field 

	This is based on code by Robin Pilch writtin in R. 
	It requires presence of a file called 

	a_b_file="/data/c1/rpilch/ERAint/ERAint_L60/ERAint_L60_hybrid_constants"
	a_b_file=scan(a_b_file,sep="\n",what='raw',quiet=TRUE)a_b=array(NA,dim=c(61,3)) ## Nlev,a,b
	for (i in 1:61){
	a_b[i,1]=as.numeric(substr(a_b_file[i],4,5))
	a_b[i,2]=as.numeric(substr(a_b_file[i],11,22))
	a_b[i,3]=as.numeric(substr(a_b_file[i],27,36))
	}
	##############################################################
	12:33
	## ...
	library(ncdf)
	## ...
	###### ---> for a given lon-lat-time:
	surface_pres=get.var.ncdf(nc_surface,varid="lnsp",start=c(lon_slot,lat_slot,time_slot_surface),count=c(1,1,1))
	surface_pres=(exp(1)^surface_pres)/100 ## to hPa### pressure at 61 interfaces
	p_interfaces=array(NA,dim=c(61))for (i in 1:61){
	p_interfaces[i]=(a_b[i,2]+a_b[i,3]*surface_pres*100)/100 ## output in hPa
	}p_full_lev=array(NA,dim=c(60))
	### pressure at full levels
	for (i in 1:60){
	N=61-i
	p_full_lev[N]=(p_interfaces[N+1]+p_interfaces[N])/2
	}
	"""


	
def P_from_hybrid_levels_era(E,date,hostname='taurus',debug=False):

	"""
	for a given ERA data subset (given in a DART experiment dictionary, E)
	on a certain date and time,
	recreate the pressure field given the hybrid model level parameters 
	"""

	# read in the hybrid level parameters and ln of surface pressure  
	# kludge: obviously all ERA-Interim data are "posterior", but it's possible to calculate priors by subtracting 
	# out the ERA-Interim DA increments. Those come in pretty shitty netcdf files on model levels without the hybrid 
	# paremeters given, so any prior files you might produce will be wonky -- so just to be sure, load "posterior" 
	# files here instead
	varlist= ['LNSP','T']
	varlist_posterior = ['hyam','hybm']
	H = dict()
	import re
	resol = float(re.sub('\ERA', '',E['exp_name']))
	# variables loaded from whatever file is requested 
	for vname in varlist:
		Ehyb = E.copy()
		Ehyb['variable'] = vname
		Ehyb['levtype']='model_levels'
		field,lat,lon,lev,time2 = load_ERA_file(Ehyb,date,resol=resol,hostname=hostname,verbose=debug)
		if vname == 'LNSP':
			H['lev'] = lev
			H['lat'] = lat
			H['lon'] = lon        
		H[vname]=field
	for vname in varlist_posterior:
		Epo = Ehyb.copy()
		Epo['diagn']='posterior'
		Epo['variable']=vname
		field,lat,lon,lev,time2 = load_ERA_file(Epo,date,resol=resol,hostname=hostname,verbose=debug)
		H[vname]=field

	# loop over all grid points and compute the pressure there: 
	# note that here we squeeze out the time dimension in T, which should be 1. 
	# further we assume that levels are the first dimension, and lat/lon the other two. 
	vshape = np.squeeze(H['T']).shape
	P = np.zeros(shape = vshape)

	for i in range(vshape[1]):
		for j in range(vshape[2]):
			ps = np.exp(np.squeeze(H['LNSP'])[i,j])
			# hyam+hybm*PS
			P[:,i,j] = H['hyam'] + H['hybm']*ps

	return P,lat,lon,lev
