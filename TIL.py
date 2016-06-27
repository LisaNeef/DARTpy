# Module TIL 
# This module contains subroutines for diagnosing the tropopause inversion layer (TIL) 
# in DART applied to atmosphere models 


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
	p0=1000.0	# reference pressure in hPa  

	# load the dynamical heating due to residual vertical velocity
	ERC = E.copy()
	if E['variable'] == 'Nsq_vstar_forcing':
		ERC['variable']='VTy'
		theta,lat,lon,lev,new_daterange = DSS.DART_diagn_to_array(E2)
		theta_zm = np.average(np.squeeze(theta),axis=1)
		# todo: make sure that this works
		factor = g/theta_zm
		RC,lat,lev = DSS.compute_DART_diagn_from_Wang_TEM_files(ERC,datetime_in,hostname=hostname,debug=debug)
		X = factor*RC
	else:
		ERC['daterange']=[datetime_in]
		ERC['variable']='WSTAR'
		wstar,lat,lev = DSS.compute_DART_diagn_from_Wang_TEM_files(ERC,datetime_in)
