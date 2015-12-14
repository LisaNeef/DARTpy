# Python module for dealing with TEM diagnostics 
# Lisa Neef, 26 Aug 2015

# load the required packages  
import numpy as np
#import datetime
import os.path
from netCDF4 import Dataset

def load_Wang_TEM_file(E,datetime_in,hostname='taurus',verbose=False):

	"""
	Load the file of TEM diagnostics computed for a certain experiment and a 
	given month, 
	 using code by Wuke Wang. 


	Inputs:  
	E		: experiment dictionary 
	datetime_in  	: datetime.datetime object for the file to load 
	hostname	: default is taurus  
	verbose		: default is False  

	These are the variables that this subroutine can read (and what is allowed 
	in E['variable']:

        float VTy(time, lev, lat, ens) ;
                VTy:long_name = "VTy: Vstar*dT/dy" ;
                VTy:units = "K/day" ;
        float WS(time, lev, lat, ens) ;
                WS:long_name = "WS: Wstar*S, S=H*N2/R" ;
                WS:units = "K/day" ;

        float VSTAR(time, lev, lat, ens) ;
                VSTAR:long_name = "VSTAR" ;
                VSTAR:units = "M/S" ;
        float WSTAR(time, lev, lat, ens) ;
                WSTAR:long_name = "WSTAR" ;
                WSTAR:units = "M/S" ;
        float FPHI(time, lev, lat, ens) ;
                FPHI:long_name = "FPHI" ;
                FPHI:units = "KG/S2" ;
        float FZ(time, lev, lat, ens) ;
                FZ:long_name = "FZ" ;
                FZ:units = "KG/S2" ;
        float DELF(time, lev, lat, ens) ;
                DELF:long_name = "DELF" ;
                DELF:units = "M/S2" ;


	"""
	# if the variable given in E isn't a TEM diagnostic, change it to the default variable 
	# wstar (residual vertical velocity)
	tem_variables_list = ['VSTAR','WSTAR','FPHI','FZ','DELF']
	dynamical_heating_rates_list = ['VTY','WS']
	variable_name = E['variable']
	if variable_name.upper() not in tem_variables_list+dynamical_heating_rates_list:
		print(variable_name+' is not a valid diagnostic -- retrieving w* instead')
		variable_name = 'WSTAR'

	# find the file path corresponding to this experiment  
	import experiment_settings as es 
	ff = es.exp_paths_TEM(E,datetime_in)

	# load the file  
        if os.path.isfile(ff):
		VV = None
		if verbose:  
			print('Loading TEM diagnostics file file '+ff)
		f = Dataset(ff,'r')
		lat = f.variables['lat'][:]
		lev = f.variables['lev'][:]
		time = f.variables['time'][:]
		VV = f.variables[variable_name][:]
		if VV is None:
			if verbose:
				print('Unable to find variable '+E['variable']+' in file '+ff)
		f.close()

		# bad flag is -999 -- turn it into np.nan
		# actually there seem to be other large negative numbers in here that aren't physical - 
		# maybe they were created by the daysplit step in CDO
		VV[np.abs(VV)>900.]=np.nan
		#VV[VV==-999]=np.nan
		#VV[VV<-500]=np.nan

		# select the vertical and lat ranges specified in E
		# if only one number is specified, find the lev,lat, or lon closest to it
		# TODO: make this an external subroutine 
		levrange=E['levrange']
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
		if latrange[0] == latrange[1]:
			ll = latrange[0]
			idx = (np.abs(lat-ll)).argmin()
			lat2 = lat[idx]
			k1 = idx
			k2 = idx
		else:
			# selection of the right latitude range depends on whether they go from south to north, or vice versa 
			nlat = len(lat)
			if lat[0] < lat[nlat-1]:
				j2 = (np.abs(lat-latrange[1])).argmin()
				j1 = (np.abs(lat-latrange[0])).argmin()
				lat2 = lat[j1:j2+1]
			else:
				j1 = (np.abs(lat-latrange[1])).argmin()
				j2 = (np.abs(lat-latrange[0])).argmin()
				lat2 = lat[j1:j2+1]


		# now select the relevant lat and lev regions 
		if E['exp_name'] == 'ERA':
			# wuke's TEM diagnostics for ERA have shape time x lev x lat
			Vout = VV[:,k1:k2+1,j1:j2+1]
		else:
			# All the variables in Wuke's TEM diagnostics for WACCM have shape time x lev x lat x ensemble_member 
			Vout = VV[:,k1:k2+1,j1:j2+1,:]

		# finally, for dynamical heating due to vertical residual circulation, we are actually interested in -wstar*S, 
		# whereas Wuke's data just has wstar*S -- so reverse the sign here. 
		if variable_name is 'WS':
			Vout = -Vout
		
	# for file not found 
	else:
		if verbose: 
			print('Unable to find TEM diagnostic file '+ff)
		Vout = None
		lat2 = None
		lev2 = None

	return Vout,lat2,lev2

