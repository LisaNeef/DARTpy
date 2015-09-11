## Python module for Earth rotation data
## Lisa Neef, 22 April 2014

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.io import netcdf
import csv 
import experiment_settings as es

#---also include some constants in this module
# these are all taken from Gross, 2009
M_atm   =  5.1441e18           # mass of the atmosphere in kg
M_oc     =  1.4e21             # mass of the ocena in kg
Q       =  7.292115e-5         # rot. rate of the earth in rad/s
M       =  5.9737e24           # mass of the earth in kg
C       =  8.0365e37           # axial principal moment of inertia  (kg m2)
B       =  8.0103e37           # next-largest principal MOI (kg m2)
A       =  8.0101e37           # next-largest principal MOI (kg m2)
CminusA =  2.6398e35           # kg m2
CminusB =  2.6221e35           # kg m2
BminusA =  1.763e33            # kg m2
Re_m    = 6.371e6              # radius of earth (m)
Re_km   = 6371.0               # radius of earth (km)
g       = 9.81                 # grav constant in m/s2
# crust and mantle parameters
Mm      = 4.0337e24            # mass of mantle (km)
Cm      = 7.1236e37            # principal MOI of mantle (kgm^2)
Am      = 7.0999e37            # next-largest MOI (kgm^2)
# other conversions, etc.
rad2microas = (180/np.pi)*60*60*1e6	#  radians to micro arcseconds
rad2mas = (180/np.pi)*60*60*1e3    	#  radians to milli arcseconds
LOD0_ms = 86160*1e3     		# sidereal LOD in milliseconds.
LOD0_nominal_ms = 86400*1e3     	# nominal LOD in milliseconds.

# parameters used in DART AAM observation operator (IERS conventions 2003)
k2 = 0.295          # rotational Love number degree 2
ks = 0.9383          # secular (fluid limit) Love number
kl =-0.301          # load Love number
C  = 8.0365E37      # (3,3) component Earth tensor of inertia
A  = 8.0101E37      # (1,1) component Earth tensor of inertia
Cm = 7.1237E37      # (3,3) component core tensor of inertia
Am = 7.0999E37      # (1,1) component core tensor of inertia
g = 9.81E0          # Earth's mean gravity acceleration [m/s^2]
omega = 7.292115E-5 # Earth's mean angular velocity [s-1]








def read_aefs_iers(hostname):

	# read in the AAM excitation function data from the IERS

        # file path depends on the host
        FP = es.iers_file_paths(hostname,'AAM')
        ff = FP

	# read in the ERP data
	data = np.genfromtxt(ff, dtype=float, skip_header=2)
	mjd   = data[:,0]
	x     = data[:,1]        
	y     = data[:,3]        
	dlod       = data[:,9]

	# define some constants
	sigc = 2*np.pi/433;      


	# rotate the polar motion terms to the proper reference frame	
	# where X(t) = p(t)+(i/sigma)*deriv(p(t))
	xdot = np.gradient(x)
	ydot = np.gradient(y)
	X1 = x+ydot/sigc
	X2 = -y+xdot/sigc

	return mjd,X1,X2,dlod

def read_erps(hostname):

	# read in the Earth Rotation Parameter data from the IERS

        # file path depends on the host
        ff = es.iers_file_paths(hostname,'ERP')

	# read in the ERP data
	data = np.genfromtxt(ff, dtype=float, skip_header=2)
	mjd   = data[:,0]
	x     = data[:,1]        
	y     = data[:,3]        
	dlod       = data[:,9]

	# define some constants
	sigc = 2*np.pi/433;      


	# rotate the polar motion terms to the proper reference frame	
	# where X(t) = p(t)+(i/sigma)*deriv(p(t))
	xdot = np.gradient(x)
	ydot = np.gradient(y)
	X1 = x+ydot/sigc
	X2 = -y+xdot/sigc


	return mjd,X1,X2,dlod

def iers_file_paths(hostname,data_type):

	if (data_type == 'ERP'):
		# path to the IERS earth rotation data
		FP = {'blizzard' : '/work/bb0519/b325004/IERS-ERP/C04_1962_2010_notides.txt'
		}
	if (data_type == 'AAM'):
		# path to the IERS earth rotation data
		FP = {'blizzard' : '/work/bb0519/b325004/IERS-ERAinterim/'
		}

        return FP[hostname]


def eam_weights(lat,lon,comp,variable):
	# retrieve a lat/lon matrix of the weights needed to compute atmospheric excitation functions 
	# given input latitude and longitude arrays, a vector component of AAM, and the variable
	# that we want to apply the weights to


	# temp inputs
	#lon = np.arange(0,361.,1.)
	#lat = np.arange(-90,91.,1.)
	#comp = 'X1'
	#variable = 'U'

	# what factor do we want to compute the weights for?
	cc = comp+variable

	# make radian arrays
	rlon = lon*np.pi/180.;
	rlat = lat*np.pi/180.;
	[LAT,LON] = np.meshgrid(rlat,rlon);


	# list the possible conditions we can have
	condX1 = [cc == 'X1PS',cc == 'X1U',cc == 'X1V']
	condX2 = [cc == 'X2PS',cc == 'X2U',cc == 'X2V']
	condX3 = [cc == 'X3PS',cc == 'X3U',cc == 'X3V']
	condlist = condX1+condX2+condX3

	# list the possible outcomes
	choiceX1 = [np.sin(LAT)*np.cos(LAT)*np.cos(LAT)*np.cos(LON),np.sin(LAT)*np.cos(LAT)*np.cos(LON),-np.cos(LAT)*np.sin(LON)]
	choiceX2 = [np.sin(LAT)*np.cos(LAT)*np.cos(LAT)*np.sin(LON),np.sin(LAT)*np.cos(LAT)*np.sin(LON),np.cos(LAT)*np.cos(LON)]
	choiceX3 = [np.cos(LAT)*np.cos(LAT)*np.cos(LAT),np.cos(LAT)*np.cos(LAT),LAT*0]
	choicelist = choiceX1+choiceX2+choiceX3

	return np.select(condlist,choicelist)

def aef_massintegral(VV,PS,p,lat,lon,variable_name,ERP='X3'):

	# goven a grid of U,V, or surface pressure, plus a 3D pressure grid, integrate 
	# the variable field to get the corresponding AAM term.


	# figure out some stuff concerning the grid
	nlat,nlon,nlev = p.shape

	# some abbreves
	radian = np.pi/180.
	rlat = radian*lat
	rlon = radian*lon
	coslat = np.cos(rlat)
	sinlat = np.sin(rlat)
	coslon = np.cos(rlon)
	sinlon = np.sin(rlon)

	# find the area of gridboxes, which is only a function of latitude
	dlon = radian*np.abs(lon[0]-lon[1])
	area_horiz = np.zeros(shape=lat.shape)

	# interior latitude bands  
	for j in range(1,nlat-1):
		dlat = lat*0
		dlat = radian*abs(lat[j-1] - lat[j+1])/2.0	
		area_horiz[j] = dlat*dlon*(Re_m**2)*np.cos(lat[j]*radian)

	# north pole latband
	if (np.abs(lat[0]) + np.abs(lat[0]-lat[1])/2.0 > 90.0):
		alpha = radian*(90.0-np.abs(lat[1]))
		area_horiz[0] = 2.0*np.pi*(Re_m**2)*(1.-np.cos(alpha))/nlon
	else:
	# if the pole is the actual boundary, then it's just a square pixel
		dlat=radian*abs(lat[0]-lat[1])
		area_horiz[0]=dlat*dlon*(Re_m**2)*np.cos(lat[0]*radian)

	# south pole latband
	if (np.abs(lat[nlat-1]) + np.abs(lat[nlat-1]-lat[nlat-2])/2.0 > 90.0):
		# in this case, make south pole into southern boundary, compute area of circle
		alpha = radian*(90.0-np.abs(lat[nlat-2]))
		area_horiz[nlat-1] = 2.0*np.pi*(Re_m**2)*(1.-np.cos(alpha))/nlon
	else:  
		dlat=radian*abs(lat[nlat-2]-lat[nlat-1])
		area_horiz[nlat-1]=dlat*dlon*(Re_m**2)*np.cos(lat[nlat-1]*radian)
			
	# prefactors -- this is taken straight from the EAM observation operator
	alp1 = 1E0 / ( 1E0 - k2/ks)                          # rotational deformation
	alp2 = 1E0 / ( 1E0 + (4E0/3E0) * (k2/ks) * (C-A)/C ) # rotational deformation
	alp3 = 1E0 + kl                                      # loading
	alp4 = (C-A) / (Cm-Am)                               # core decoupling
	alp5 = C / Cm                                        # core decoupling
	chifacprs1 = alp1 * alp3 * alp4 / (C-A)
	chifacprs2 = alp1 * alp3 * alp4 / (C-A)
	chifacprs3 = alp2 * alp3 * alp5 /  C
	chifacwin1 = alp1 * alp4 / (C-A) / omega
	chifacwin2 = alp1 * alp4 / (C-A) / omega
	chifacwin3 = alp2 * alp5 /  C    / omega

	# start with zero AAM and then add up
	AAM = 0.0

	print(variable_name)
	if variable_name is 'PS':
		nlat2,nlon2 = VV.shape
		dm = np.zeros(shape=(nlat2,nlon2))
		x = np.zeros(shape=(nlat2,nlon2))
		# for mass term, loop over lat and lon and add up the mass for each column
		for ilat in range(nlat2):
			for ilon in range(nlon2):
				dm[ilat,ilon] = VV[ilat,ilon] * area_horiz[ilat] / g
				# mass increment for each column:
				if ERP is 'X1':
					x[ilat,ilon] = -(Re_m**2)*coslat[ilat]*sinlat[ilat]*coslon[ilon]*chifacprs1*dm[ilat,ilon]
				if ERP is 'X2':
					x[ilat,ilon] = -(Re_m**2)*coslat[ilat]*sinlat[ilat]*sinlon[ilon]*chifacprs2*dm[ilat,ilon]
				if ERP is 'X3':
					x[ilat,ilon] = (Re_m**2)*coslat[ilat]*coslat[ilat]*chifacprs3*dm[ilat,ilon]
				AAM = AAM + x[ilat,ilon]

	else:
		nlat2,nlon2,nlev2 = VV.shape
		# compute the pressure increments for each layer
		dlev = p*0
		for j in range(nlat):
			for i in range(nlon):
				dlev[j,i,0] = 0.5*(p[j,i,0] + p[j,i,1])		# top layer
				dlev[j,i,nlev-1] = PS[j,i] - 0.5*(p[j,i,nlev-2]+p[j,i,nlev-1]  )	# bottom layer
				for k in range(1,nlev-1):
					dlev[j,i,k] = 0.5*(p[j,i,k+1] - p[j,i,k-1] )	# inner layers

		# for wind terms, loop over lat, lon, lev, and compute the AM contribution for each box
		dm = dlev*0
		x = np.zeros(shape=(nlat2,nlon2))
		dm = np.zeros(shape=(nlat2,nlon2,nlev))
		#print('checking stats for latitude = ',str(lat[45]),' and lon = ',str(lon[90]))
		#print('k, dlev, dm,U, aam')
		for k in range(nlev):
			for ilat in range(nlat2):
				for ilon in range(nlon2):
					dm[ilat,ilon,k] = dlev[ilat,ilon,k] * area_horiz[ilat] / g
					if (variable_name is 'US') or (variable_name is 'U'):
						if ERP is 'X1':
							x[ilat,ilon] = x[ilat,ilon] - Re_m*VV[ilat,ilon,k]*sinlat[ilat]*coslon[ilon]*dm[ilat,ilon,k]
						if ERP is 'X2':
							x[ilat,ilon] = x[ilat,ilon] - Re_m*VV[ilat,ilon,k]*sinlat[ilat]*sinlon[ilon]*dm[ilat,ilon,k]
						if ERP is 'X3':
							x[ilat,ilon] = x[ilat,ilon] + Re_m*VV[ilat,ilon,k]*coslat[ilat]*dm[ilat,ilon,k]
					if (variable_name is 'VS') or (variable_name is 'V'):
						if ERP is 'X1':
							x[ilat,ilon] = x[ilat,ilon] + Re_m*VV[ilat,ilon,k]*sinlon[ilon]*dm[ilat,ilon,k]
						if ERP is 'X2':
							x[ilat,ilon] = x[ilat,ilon] - Re_m*VV[ilat,ilon,k]*coslon[ilon]*dm[ilat,ilon,k]
						if ERP is 'X3':
							x[ilat,ilon] = 0.0
					#aamu = + Re_m*VV[45,90,k]*coslat[45]*dm[45,90,k]

			#print(str(k)+' '+str(dlev[45,90,k])+' '+str(dm[45,90,k])+' '+str(VV[45,90,k])+' '+str(aamu))

		# now loop over lat and lon and add up AAM:
		for ilat in range(nlat2):
			for ilon in range(nlon2):
				if ERP is 'X1':
					AAM = AAM + x[ilat,ilon]*chifacwin1
				if ERP is 'X2':
					AAM = AAM + x[ilat,ilon]*chifacwin2
				if ERP is 'X3':
					AAM = AAM + x[ilat,ilon]*chifacwin3
		
	print('the total AAM for this variable and case is:  '+str(AAM))
	return AAM


def aef(field,lev,lat,lon,variable_name,ERP='X3'):

	# given some variable field (U, V, or surface pressure), compute the AAM excitation function for the desired Earth rotation parameters

	# check whether pressure levels are in Pascale (not hPa) -- send an alert if this is not the case
	if np.max(lev) < 8E4:
		print('the pressure at the surface is '+str(np.max(lev))+' which means these levels are probably not in Pascal')
		print('returning...')
		return None


	# reshape the variable array to be lev x lat x lon, or lat x lon
	ndim = len(field.shape)
	nlat = len(lat)
	nlon = len(lon)
	if ndim == 2:
		V = np.reshape(field,(nlat,nlon))
	if ndim == 3:
		nlev = len(lev)
		V = np.reshape(field,(nlat,nlon,nlev))


	# load the appropriate geographic weighting function for the desired variable and ERP, and 
	# multiply the field by this
	W0 = eam_weights(lat,lon,ERP,variable_name)
	#W = np.reshape(W0,(nlat,nlon))
	W = np.transpose(W0)
	if ndim == 2:
		Vw = V*W
	else:
		Vw = V*0
		for ilev in range(nlev):
			Vw[:,:,ilev] = V[:,:,ilev]*W	


	# convert lat and lon to radians
	rlon=lon*np.pi/180
	rlat=lat*np.pi/180



	# the integrals have to be multiplied by -1 if the lat, lon, or lev arrays are defined in the other direction
	# the "right" direction is lat -90-90, lon 0-360, and ps to ptop
	latfac = 1.0
	lonfac = 1.0
	levfac = 1.0
	if lat[0] > lat[nlat-1]:
		latfac = -1.
	if lon[0] > lon[nlon-1]:
		lonfac = -1.
	if not (variable_name == 'PS'):
		if lev[0] < lev[nlev-1]:
			levfac = -1.

	# I have no idea why this needs to be positive for CAM, but that seems to be the only way to rectivy the diff between the vol and mass integral
	levfac = 1.0

	if variable_name == 'PS':
		# surface pressure: integrate over lat and lon
		Vlon = lonfac*np.trapz(Vw,rlon)
		Vint = latfac*np.trapz(Vlon,rlat)

	else:
		# meridional and zonal wind: integrate over lat,lon,lev
		# integral over lon:
		Vlon = np.zeros(shape=(nlat,nlev))
		for k in range(nlev):
			Vlon[:,k] = lonfac*np.trapz(Vw[:,:,k],rlon)

		# integral over lev:
		Vlevlon = levfac*np.trapz(Vlon,lev)

		# integral over lat:
		Vint = latfac*np.trapz(Vlevlon,rlat)
	
	# multiply by the prefactors that give us AAM:
	fac = aam_prefactors(ERP,variable_name)
	AAM = fac*Vint

	return AAM

def aam_prefactors(comp,variable_name):

	# return the geophysical prefactors needed to compute the angular momentum components.

	# this is taken straight from the EAM observation operator
	#alp1 = 1E0 / ( 1E0 - k2/ks)                          # rotational deformation
	#alp2 = 1E0 / ( 1E0 + (4E0/3E0) * (k2/ks) * (C-A)/C ) # rotational deformation
	#alp3 = 1E0 + kl                                      # loading
	#alp4 = (C-A) / (Cm-Am)                               # core decoupling
	#alp5 = C / Cm                                        # core decoupling
	#chifacprs1 = alp1 * alp3 * alp4 / (C-A)
	#chifacprs2 = alp1 * alp3 * alp4 / (C-A)
	#chifacprs3 = alp2 * alp3 * alp5 /  C
	#chifacwin1 = alp1 * alp4 / (C-A) / omega
	#chifacwin2 = alp1 * alp4 / (C-A) / omega
	#chifacwin3 = alp2 * alp5 /  C    / omega

	if (comp=='X1') or (comp=='X2'):
		if (variable_name == 'U') or (variable_name == 'V'):
			fac = (-1.591*Re_m**3)/(Q*g*CminusA)
			ff = 1/(1-k2/ks)
			fac2 = (-ff*Re_m**3)/(Q*g*CminusA)
		if (variable_name == 'PS'):
			fac = (-1.098*(Re_m**4))/(g*CminusA)
			ff = (1+kl)/(1-k2/ks)
			fac2 = (-ff*(Re_m**4))/(g*CminusA)

	if (comp=='X3'):
		if (variable_name == 'U') or (variable_name == 'V'):
			fac = (0.997*(Re_m**3))/(Q*g*Cm);
			top = 1.0
			bot = (1.+(4./3.)*(k2/ks)*(C-A)/C)
			ff = top/bot
			fac2 = (ff*(Re_m**3))/(Q*g*Cm)
		if (variable_name == 'PS'):
			fac = (0.748*(Re_m**4))/(g*Cm)
			top = (1.+kl)
			bot = (1.+(4./3.)*(k2/ks)*(C-A)/C)
			ff = top/bot
			fac2 = (ff*(Re_m**4))/(g*Cm)

	#print ff
	return fac2

