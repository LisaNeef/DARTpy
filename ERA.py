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
def load_ERA_file(E,datetime_in,resol=2.5,hostname='taurus',verbose=False):

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
	resol: which resolution should be loaded? Default is 2.5 
	"""

	# find the file path corresponding to this experiment  
	#file_path_list,dum = es.exp_paths_era(hostname=hostname,resolution=resol)
	ff,dum = es.exp_paths_era(datetime_in,hostname=hostname,resolution=resol,diagnostic=E['diagn'])

	variable_found = False

	# load the file  
        if os.path.isfile(ff):
		if verbose:  
			print('Loading ERA file '+ff)
		f = Dataset(ff,'r')
		if resol == 1.5:
			lat = f.variables['latitude'][:]
			lon = f.variables['longitude'][:]
			lev = f.variables['level'][:]
		if resol == 2.5:
			lat = f.variables['lat'][:]
			lon = f.variables['lon'][:]
			lev = f.variables['lev'][:]
		time = f.variables['time'][:]
		
		if (E['variable']=='T') or (E['variable']=='TS'):
			VV = f.variables['var130'][:]
			variable_found = True
		if (E['variable']=='U') or (E['variable']=='US'):
			VV = f.variables['var131'][:]
			variable_found = True
		if (E['variable']=='V') or (E['variable']=='VS'):
			VV = f.variables['var132'][:]
			variable_found = True
		if (E['variable']=='Z') or (E['variable']=='GPH') or (E['variable']=='Z3'):
			if resol == 1.5:
				# for ERA we have Geopotential (z) but not Geopotential height
				VV = (1/9.8)*f.variables['z'][:]
			if resol == 2.5:
				VV = f.variables['var129'][:]
			variable_found = True
		if (E['variable']=='msl') or (E['variable']=='MSLP'):
			VV = f.variables['var151'][:]
			variable_found = True

		if (variable_found is False):
			print('Unable to find variable '+E['variable']+' in file '+ff)
			f.close()
			return
		f.close()
	
		# select the vertical and lat/lon ranges specified in E
		# if only one number is specified, find the lev,lat, or lon closest to it

		# note that in the ERA data, levels are given in Pa (whereas in CESM, CAM, WACCM 
		# they are in hPa) -- so multiply the requested range by 100. 
		levrange=E['levrange']
	
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
		Vtemp,lat,lon,lev,time_out = load_ERA_file(E,date,hostname=hostname,verbose=verbose)
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
