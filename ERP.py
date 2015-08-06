## Python module for Earth rotation data
## Lisa Neef, 22 April 2014

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.io import netcdf
import csv 
import experiment_settings as es
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

