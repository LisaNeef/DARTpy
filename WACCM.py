# Python module for codes that deal with WACCM output
# Lisa Neef, 26 Jan 2015

# load the required packages  
import numpy as np
import datetime
import time as time
import os.path
#import pandas as pd
import DART as dart
#from calendar import monthrange
from netCDF4 import Dataset
import glob
import experiment_settings as es

#-------reading in WACCM history files--------------------------------------
def load_WACCM_multi_instance_h_file(E,datetime_in,instance,hostname='taurus',verbose=False,special_flag=None):

	"""
	This subroutine loads an h file from a WACCM multi-instance run (so far just
	used for DART).  

	Input instance is usually an integer, but if it's "ensemble mean", we load 
	files that are the mean computed over all instances at one time. 

	Inputs:  
	E
	datetime_in  
	instance  
	hostname: default is taurus  
	verbose: default if False  
	special_flag: a variable to set for files that deviate from the regular output, e.g.: 
		'lowpass6day' loads the output that has been low-pass filtered with a 6 day cutoff  
		default is 'None'  

	"""

	# find the history file path corresponding to this experiment  
	h_file_path_list,truth_path_list = es.exp_paths(hostname,E['exp_name'])

	# find the history file number corresponding to the desired variable field  
	hnum = history_file_lookup(E)

	# define the string corresponding to the given instance  
	# here we use the python function 'isinstance', but that's just a coincidence.  
	if isinstance(instance,str):
		if instance == 'ensemble mean':
			instance_str = 'ensemble_mean'
		if instance == 'ensemble std':
			instance_str = 'ensemble_std'
	if isinstance(instance,int):
		if instance < 10:
			instance_str = '000'+str(instance)
		if (instance >= 10) and (instance < 100):
			instance_str = '00'+str(instance)
		if instance >= 100:
			instance_str = '0'+str(instance)

	# loop through the available file paths and look for the right kind of files
	correct_filepath_found = False
	for h_file_path in h_file_path_list:
		p2 = h_file_path+'/atm/hist/'
		filename_test = glob.glob(p2+'*.cam_'+instance_str+'.h'+str(hnum)+'*.nc')
		if len(filename_test) >0:
			correct_filepath_found = True
			break
	if correct_filepath_found is False:
		stub = '/atm/hist/'+'*.cam_'+instance_str+'.h'+str(hnum)+'*.nc'
		print("+++cannot find files that look like "+stub+' in any of these directories:')
		print(h_file_path_list)
		return


	# put the filename together from the available information  
	stub = waccm_experiment_full_names(E)
	if (special_flag == None):
		if hnum == 0:
			# h0 files are monthly means
			datestring = datetime_in.strftime("%Y-%m")
			fname = stub+'.cam_'+instance_str+'.h'+str(hnum)+'.'+datestring+'.nc'

		else:
			# currently this is set up for h1 files, which we've saved daily 
			datestring = datetime_in.strftime("%Y-%m-%d")
			seconds = datetime_in.hour*60*60
			if seconds == 0:
				timestring = '00000'
			else:
				timestring = str(seconds)
			fname = stub+'.cam_'+instance_str+'.h'+str(hnum)+'.'+datestring+'-'+timestring+'.nc'
	if (special_flag == 'lowpass6day'):
		month = datetime_in.month
		day = datetime_in.day
		if month < 10:
			monthstr = '0'+str(month)
		else:
			monthstr = str(month)
		if day < 10:
			daystr = '0'+str(day)
		else:
			daystr = str(day)
		fname = stub+'.cam_'+instance_str+'.h'+str(hnum)+'.lowpass6day.'+monthstr+'.'+daystr+'.nc'
	ff = p2+'/'+fname

	# load the file  
        if os.path.isfile(ff):
		if verbose:  
			print('Loading WACCM file '+ff)
		f = Dataset(ff,'r')
		lat = f.variables['lat'][:]
		lon = f.variables['lon'][:]
		lev = f.variables['lev'][:]
		time = f.variables['time'][:]
		VV = f.variables[E['variable']][:]
		if VV is None:
			print('Unable to find variable '+E['variable']+' in file '+ff)
		f.close()


		# select the vertical and lat/lon ranges specified in E
		# if only one number is specified, find the lev,lat, or lon closest to it
		# TODO: need to add this option for lev and lat
		levrange=E['levrange']
		if levrange is not None:
			if levrange[0] == levrange[1]:
				ll = levrange[0]
				idx = (np.abs(lev-ll)).argmin()
				lev2 = lev[idx]
				k1 = idx
				k2 = idx
			else:
				k1 = (np.abs(lev-levrange[1])).argmin()
				k2 = (np.abs(lev-levrange[0])).argmin()
				lev2 = lev[k1:k2+1]

		latrange=E['latrange']
		j2 = (np.abs(lat-latrange[1])).argmin()
		j1 = (np.abs(lat-latrange[0])).argmin()
		lat2 = lat[j1:j2+1]

		lonrange=E['lonrange']
		i2 = (np.abs(lon-lonrange[1])).argmin()
		i1 = (np.abs(lon-lonrange[0])).argmin()
		lon2 = lon[i1:i2+1]

		# now select the relevant lat, lon, and lev regions -- different variables have different shapes, 
		# so this depends on variable 
		scalar_variables = ['P0']
		variables_1d = ['hyam','hybm','hyai','hybi']
		variables_2d = ['PS','FLUT']
		variables_3d = ['US','VS','T','Z3','QRS_TOT','QRL','QRL_TOT','U','V','Q']

		if E['variable'] in scalar_variables:
			# scalar 
			Vout = VV

		if E['variable'] in variables_1d:
			# these variables just have dimensionality lev
			Vout = VV[k1:k2+1]

		if E['variable'] in variables_2d:
			# 2D variables have shape time x lat x lon
			Vout = VV[:,j1:j2+1,i1:i2+1]
			lev2 = None
		
		if E['variable'] in variables_3d:
			# 3D variables have shape time x lev x lat x lon
			Vout = VV[:,k1:k2+1,j1:j2+1,i1:i2+1]

		# throw an error if variable has not been defined. 
		if 'Vout' not in locals():
			print "WACCM.py doesnt know what to do with variable "+E['variable']+" -- still need to code its shape in subroutine load_WACCM_multi_instance_h_file" 
			return None,None,None,None

		# if loading low-pass filtered data, several times are in one file -- choose the first.  
		# (It shouldn't matter because we filtered fast stuff anyway)  
		if (special_flag == 'lowpass6day'):
			if len(VV.shape)==3:
				Vout = VV[0,j1:j2+1,i1:i2+1]	
			if len(VV.shape)==4:
				Vout = VV[0,k1:k2+1,j1:j2+1,i1:i2+1]

	# for file not found 
	else:
		print('Unable to find WACCM file '+ff)
		Vout = None
		lat2 = None
		lon2 = None
		lev2 = None

	return Vout,lat2,lon2,lev2


#-------full names of waccm experiments--------------------------------------
def waccm_experiment_full_names(E):

	if E['exp_name'] == 'PMO32':
		full_name = 'waccm-dart-assimilate-pmo-32'
	if E['exp_name'] == 'NODA_WACCM':
		full_name = 'nechpc-waccm-dart-gpsro-ncep-no-assim'
	if E['exp_name'] == 'W0910_NODART':
		full_name = 'nechpc-waccm-dart-gpsro-ncep-no-dart'
	if E['exp_name'] == 'W0910_NODA':
		full_name = 'nechpc-waccm-dart-gpsro-ncep-no-assim-01'
	if E['exp_name'] == 'W0910_GLOBAL':
		full_name = 'nechpc-waccm-dart-gpsro-ncep-global-01'
	if E['exp_name'] == 'W0910_TROPICS':
		full_name = 'nechpc-waccm-dart-gpsro-ncep-30S-30N-01'

	return full_name
#-------full names of waccm experiments--------------------------------------


#-------dictionaries of WACCM h-file numbers--------------------------------------
def history_file_lookup(E):  

	# these are the default settings   -- might not be the same for all experiments  
	exp_name_found = True
	H = {'O3':1,
	'T':1,
	'PS':1,
	'U':1,
	'FLUT':1,
	'Z3':1,
	'hyam':1,
	'hybm':1,
	'P0':1,
	'PS':1,
	'Q':1,
	'QRL':0,
	'QRL_TOT':0,
	'QRS_TOT':0
	}
		
	# if the desired key does not exist, set output hnumber to None
	if H.has_key(E['variable']):
		hnumber = H[E['variable']]
	else:
		hnumber=None

	return hnumber  




