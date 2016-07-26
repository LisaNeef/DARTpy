# Module TIL: 
# this module holds Python routines for diagnosing the tropopause inversion layer 
# (TIL) in atmosphere models with DART data assimilation 
# Lisa Neef, 27 Jun 2016

# load the required packages  
import numpy as np
import os.path
from netCDF4 import Dataset
import DART_state_space as DSS
import DART as dart

def Nsq_forcing_from_RC(E,datetime_in=None,debug=False,hostname='taurus'):

	"""
	Birner (2010) used the thermodynamic equation in the TEM form to derive an expression 
	for the rate of change of static stability (N2) due to residual motion and diabatic heating. 

	This subroutine compares those terms from the dynamical heating rates computed by Wuke Wang. 
	The vertical motion (wstar) term is -d(wsar*Nsq)/dz.  
	Wuke already computed WS = -wstar*HNsq/R, so it's easiest to load that data, divide out H and R, and then take the vertical gradient. 

	The horizontal term is -g d(vstar/theta * d(theta)dy)/dz. 
	Wuke already computed the heating rate term v*dtheta/dy = v*dTdy, 
	so the easiest thing to do is to multiply the heating rates by g/theta
	and then take the vertical gradient. 

	INPUTS:
	E: a DART experiment dictionary. Relevant fields are:
		E['exp_name'] - the experiment name
		E['daterange'] - helps to choose which date to load in case this isn't specifically given
		E['variable'] - if this is set to N2_forcing_vstar, the code returns the N2 forcing due to 
			meridional residual circulation. For anything else, it returns the forcing 
			due to vertical residual circulation. 
	datetime_in: the date for which we want to compute this diagnostic. 
		default is None -- in this case, just choose the fist date in E['daterange']


	OUTPUTS:
	N2_forcing: Nsquared forcing term  in s^2/day
	lev
	lat 
	"""

	# necessary constants  
	H=7000.0	# scale height in m  
	g = 9.80
	p0=1000.0	# reference pressure in hPa  

	if datetime_in is None:
		datetime_in = E['daterange'][0]

	# depending on which term we want, need to load the residual circulation component and some other stuff, 
	# and then derive a quantity for which we take the vertical gradient 
	ERC = E.copy()
	ET=E.copy()
	if E['variable'] == 'Nsq_vstar_forcing':
		ET['variable']='theta'
		lev,lat,lon,theta,P0,hybm,hyam = dart.load_DART_diagnostic_file(ET,datetime_in,hostname=hostname,debug=debug)
		ERC['variable']='VSTAR'
		vstar,lat,lev = DSS.compute_DART_diagn_from_Wang_TEM_files(ERC,datetime_in,hostname=hostname,debug=debug)

		# the above routines do not return arrays of consistent shape, so have to do 
		# some acrobatics to get everything to match up. 

		# find how the dimensions fit to the shape 
		nlon=len(lon)
		nlat=len(lat)
		nlev=len(lev)
		for idim,s in enumerate(theta.shape):
			if s==nlon:
				londim=idim
				latdim=idim
				levdim=idim
			
		# take the zonal mean of potential temp  - this should make its shape copy x lat x lev
		thetam = np.average(theta,axis=londim)

		# next step is to find the meridional gradient of theta 
		# latitude steps --> convert to distance (arclength)
		rlat = np.deg2rad(lat)
		Re = 6371000.0		# radius of Earth in m 
		y = Re*rlat
		dy = np.gradient(y)
		# need to replicate dy to suit the shape of zonal mean theta 
		dym = dy[None,:,None]
		dy3 = np.broadcast_to(dym,thetam.shape)
		# here is the gradient - need to squeeze out a possible length-1 
		# copy dimension 
		dthetady_list = np.gradient(np.squeeze(thetam),np.squeeze(dy3))

		# now find which dimension of _squeezed_ thetam corresponds to latitude - 
		# that's the gradient that we want
		# (is this a pain in the ass? Yes! But I haven't yet found a more clever approach) 
		for idim,s in enumerate(np.squeeze(thetam).shape):
			if s==nlat:
				newlatdim=idim
		dthetady = dthetady_list[newlatdim]

		# the meridional gradient of zonal mean theta then gets multiplied by vstar and g/theta. But...  
		
		# the subroutine compute_DART_diagn_from_Wang_TEM_files delivers an array with 
		# dimensions lev x lat x copy (or just levxlat)
		# whereas N2 should come out as copy x lat x lev (or simply lat x lev)
		# need to transpose this, but I don't trust np.reshape - do it manually 
		vstar2 = np.zeros(shape=dthetady.shape)
		if vstar2.ndim==3:
			for icopy in range(dthetady.shape[0]):
				for ilat in range(dthetady.shape[1]):
					for ilev in range(dthetady.shape[2]):
						vstar2[icopy,ilat,ilev]=vstar[icopy,ilev,ilat]
		else:
			for ilat in range(dthetady.shape[0]):
				for ilev in range(dthetady.shape[1]):
					vstar2[ilat,ilev]=vstar[ilev,ilat]

		X = (g/np.squeeze(thetam))*vstar2*dthetady

	else:
		
		ET['variable']='Nsq'
		lev,lat,lon,Nsq,P0,hybm,hyam = dart.load_DART_diagnostic_file(ET,datetime_in,hostname=hostname,debug=debug)
		ERC['variable']='WSTAR'
		wstar,lat,lev = DSS.compute_DART_diagn_from_Wang_TEM_files(ERC,datetime_in,hostname=hostname,debug=debug)

		# find how the dimensions fit to the shape 
		nlon=len(lon)
		nlat=len(lat)
		nlev=len(lev)
		for idim,s in enumerate(Nsq.shape):
			if s==nlon:
				londim=idim
				latdim=idim
				levdim=idim
        
		# take the zonal mean of buoyancy frequency 
		Nsqm = np.average(Nsq,axis=londim)
		
		# might have to squeeze out a length-1 copy dimension 
		Nsqm2 = np.squeeze(Nsqm)

		# the subroutine compute_DART_diagn_from_Wang_TEM_files delivers an array with dimensions lev x lat x copy (or just levxlat)
		# whereas N2 should come out as copy x lat x lev (or simply lat x lev)
		# need to transpose this, but I don't trust np.reshape - do it manually 
		wstar2 = np.zeros(shape=Nsqm2.shape)
		if wstar2.ndim==3:
			for icopy in range(Nsqm2.shape[0]):
				for ilat in range(Nsqm2.shape[1]):
					for ilev in range(Nsqm2.shape[2]):
						wstar2[icopy,ilat,ilev]=wstar[icopy,ilev,ilat]
		else:
			for ilat in range(Nsqm2.shape[0]):
				for ilev in range(Nsqm2.shape[1]):
					wstar2[ilat,ilev]=wstar[ilev,ilat]

		X = Nsqm2*wstar2

	# convert pressure levels to approximate altitude and take the vertical gradient  
	zlev = H*np.log(p0/lev)
	dZ = np.gradient(zlev)   # gradient of vertical levels in m

	# now X *should* have shape (copy x lat x lev) OR (lat x lev)
	# so need to copy dZ to look like this 
	if X.ndim==3:
		dZm = dZ[None,None,:]
		levdim=2
	if X.ndim==2:
		dZm = dZ[None,:]
		levdim=1
	dZ3 = np.broadcast_to(dZm,X.shape)
	dXdZ_3D = np.gradient(X,dZ3)
	dxdz = dXdZ_3D[levdim] # this is the vertical gradient with respect to height 

	# the above calculation yields a quantity in units s^-2/s, but it makes more sense 
	# in the grand scheme of things to look at buoyancy forcing per day, so here 
	# is a conversion factor.
	seconds_per_day = 60.*60.*24.0

	N2_forcing = -dxdz*seconds_per_day

	return N2_forcing,lat,lev

def Nsq_forcing_from_Q(E,datetime_in=None,debug=False,hostname='taurus'):

	"""
	Birner (2010) used the thermodynamic equation in the TEM form to derive an expression 
	for the rate of change of static stability (N2) due to residual motion and diabatic heating. 

	This subroutine compares the term due to diabatic heating, i.e.: 
	g d(Q/theta)dz

	INPUTS:
	E: a DART experiment dictionary. Relevant fields are:
		E['exp_name'] - the experiment name
		E['daterange'] - helps to choose which date to load in case this isn't specifically given
		E['variable'] - this determines what kind of diabatic heating we use:
			the value of E['variable'] should be a string like 'Nsq_forcing_XXXXX'
			where XXXXX is the model variable corresponding to whatever diabatic 
			heating type we are looking for. 
			For example, in WACCM, 'QRL_TOT' is the total longwave heating, so to get the 
			N2 forcing from that, just set E['variable']='Nsq_forcing_QRL_TOT'
	datetime_in: the date for which we want to compute this diagnostic. 
		default is None -- in this case, just choose the fist date in E['daterange']


	OUTPUTS:
	N2_forcing: Nsquared forcing term  in s^2/day
	lev
	lat 
	"""

	# necessary constants  
	H=7000.0	# scale height in m  
	p0=1000.0	# reference pressure in hPa  
	g=9.8		# acceleration of gravity 

	# load the desired diabatic heating term
	# this is not typically part of the DART output, so load from model history files
	# (right now this really only works for WACCM/CAM)  
	Qstring = E['variable'].strip('Nsq_forcing_')
	EQ = E.copy()
	EQ['variable']=Qstring
	Q2,lat,lon,lev = DSS.compute_DART_diagn_from_model_h_files(EQ,datetime_in,verbose=debug)
	# remove the time dimension, which should have length 1 
	Q = np.squeeze(Q2)

	# also load potential temperature 
	ET = E.copy()
	ET['variable']='theta'
	lev,lat,lon,theta2,P0,hybm,hyam = dart.load_DART_diagnostic_file(ET,datetime_in,hostname=hostname,debug=debug)
	# squeeze out extra dims, which we get if we load single copies (e.g. ensemble mean)
	theta = np.squeeze(theta2)

	# now find the longitude dimension and average over it  
	# for both Q and theta  
	nlon=len(lon)
	Mean_arrays = []
	for A in [Q,theta]:
		for idim,s in enumerate(A.shape):
			if s == nlon:
				londim=idim
		Mean_arrays.append(np.average(A,axis=londim))
	Q_mean=Mean_arrays[0]
	# if the shapes don't match up, might have to transpose one of them
	if Mean_arrays[1].shape[0] != Q_mean.shape[0]:
		theta_mean=np.transpose(Mean_arrays[1])
	else:
		theta_mean=Mean_arrays[1]
		
	# divide Q by theta
	X = Q_mean/theta_mean

	# convert pressure levels to approximate altitude and take the vertical gradient  
	zlev = H*np.log(p0/lev)
	dZ = np.gradient(zlev)   # gradient of vertical levels in m

	# copy dZ over lat so it's a 2D array (lev x lat) 
	nlat = len(lat)
	dZ2d = np.repeat(dZ[:,np.newaxis],nlat,axis=1)

	# if looking at more than one copy (e.g. the entire ensemble) 
	# copy dxdz N times, so that it's a lev x lat x N array 
	if len(X.shape)==3:
		N = X.shape[2]
		dZ3d = np.repeat(dZ2d[:,:,np.newaxis],N,axis=2)
	else:
		dZ3d=dZ2d

	# now compute gradient wrt height 
	dXdZ_3D = np.gradient(np.squeeze(X),dZ3d)
	dxdz = dXdZ_3D[0] # this is the vertical gradient with respect to height 

	# the above calculation yields a quantity in units s^-2/s, but it makes more sense 
	# in the grand scheme of things to look at buoyancy forcing per day, so here 
	# is a conversion factor.
	seconds_per_day = 60.*60.*24.0

	# now loop over ensemble members and compute the n2 forcing for each one
	N2_forcing = g*dxdz*seconds_per_day

	return N2_forcing,lat,lev
