## Python module for DART-WACCM paths and stuff
## Lisa Neef, 4 June 2014


import numpy as np
from netCDF4 import Dataset
import datetime as datetime
import dayconv 
import os.path
import pandas as pd
import re
import experiment_settings as es

def load_covariance_file(E,date,hostname='taurus',debug=False):

	"""
	this subroutine loads in a pre-computed file of state-to-observation covariances and correlations.
	the state variable is given by E['variable']
	the observation is given by E['obs_name']
	"""

	# find the directory for this run   
	# this requires running a subroutine called `find_paths`, stored in a module `experiment_datails`, 
	# but written my each user -- it should take an experiment dictionary and the hostname 
	# as input, and return as output 
	# the filepath that corresponds to the desired field, diagnostic, etc. 
	filename = es.find_paths(E,date,file_type='covariance',hostname=hostname)
	if not os.path.exists(filename):
		print("+++cannot find files that look like  "+filename+' -- returning None')
		return None, None, None, None, None

	# load the netcdf file 
	f = Dataset(filename,'r')
	lat = f.variables['lat'][:]
	lon = f.variables['lon'][:]
	if E['variable']!='PS':
		lev = f.variables['lev'][:]
	time = f.variables['time'][:]
	Correlation = f.variables['Correlation'][:]
	Covariance = f.variables['Covariance'][:]
	f.close()

	# squeeze out the time dimension -- for now. Might make this longer than 1 later
	# also select the right level, lat, and lon ranges
	# figure out which vertical level range we want
	if E['variable'] !='PS':
		levrange=E['levrange']
		k1 = (np.abs(lev-levrange[1])).argmin()
		k2 = (np.abs(lev-levrange[0])).argmin()
		lev2 = lev[k1:k2+1]

	# figure out which latitude range we want
	latrange=E['latrange']
	j2 = (np.abs(lat-latrange[1])).argmin()
	j1 = (np.abs(lat-latrange[0])).argmin()
	lat2 = lat[j1:j2+1]

	# figure out which longitude range we want
	lonrange=E['lonrange']
	i2 = (np.abs(lon-lonrange[1])).argmin()
	i1 = (np.abs(lon-lonrange[0])).argmin()
	lon2 = lon[i1:i2+1]

	if E['variable']=='PS':
		R = np.squeeze(Correlation[j1:j2+1,i1:i2+1,0])
		C = np.squeeze(Covariance[j1:j2+1,i1:i2+1,0])
		lev2 = None
	else:
		R = np.squeeze(Correlation[j1:j2+1,i1:i2+1,k1:k2+1,0])
		C = np.squeeze(Covariance[j1:j2+1,i1:i2+1,k1:k2+1,0])

	# return covariance and correlation grids 
	return  lev2, lat2, lon2, C, R

def load_DART_obs_epoch_series_as_dataframe(E,obs_type_list=['ERP_PM1','ERP_LOD'],ens_status_list=['ensemble member'], hostname='taurus'):

	"""
	this function scoots through a set of dates and returns a (sometimes very huge) dataframe of information
	"""
	daterange = E['daterange']
	for date in daterange:

		if date == daterange[0]:
			DF =  load_DART_obs_epoch_file_as_dataframe(E,date,obs_type_list,ens_status_list,hostname)
		else:
			DF2 = load_DART_obs_epoch_file_as_dataframe(E,date,obs_type_list,ens_status_list,hostname)
			DF = pd.merge(DF,DF2,how='outer')

	return DF


def load_DART_obs_epoch_file_as_dataframe(E,date=datetime.datetime(2009,1,1,0,0,0),obs_type_list=['ERP_PM1','ERP_LOD'],ens_status_list=['ensemble member'], hostname='taurus',debug=False):

	"""
	 read in a DART obs epoch file, defined by its date and the Experiment E, and return as a Pandas data frame, in which al the observations 
	 that have ensemble status and obs types given in ens_status_list and obs_type_list, respectively, 
	 are ordered according to ObsIndex.  
	 this should eventually replace the SR load_DART_obs_epoch_file  
	"""

	# find the directory for this run   
	# this requires running a subroutine called `find_paths`, stored in a module `experiment_datails`, 
	# but written my each user -- it should take an experiment dictionary and the hostname 
	# as input, and return as output 
	# the filepath that corresponds to the desired field, diagnostic, etc. 
	filename = es.find_paths(E,date,hostname=hostname)
	if not os.path.exists(filename):
		print("+++cannot find files that look like  "+filename+' -- returning None')
		return None

	# load the file and select the observation we want
        else:
		f = Dataset(filename,'r')
		CopyMetaData = f.variables['CopyMetaData'][:]
		ObsTypesMetaData = f.variables['ObsTypesMetaData'][:]
		observations = f.variables['observations'][:]
		time = f.variables['time'][:]
		copy = f.variables['copy'][:]
		obs_type = f.variables['obs_type'][:]
		location = f.variables['location'][:]
		ObsIndex = f.variables['ObsIndex'][:]
		qc = f.variables['qc'][:]

		# find the obs_type number corresponding to the desired observations
		obs_type_no_list = []
		for obs_type_string in obs_type_list:
			obs_type_no_list.append(get_obs_type_number(f,obs_type_string))
		
		# expand "CopyMetaData" into lists that hold ensemble status and diagnostic
		diagn = []
		ens_status = []
		CMD = []
		# loop over the copy meta data and record the ensemble status and diagnostic for reach copy
		for icopy in copy:
			temp = CopyMetaData[icopy-1,].tostring()
			CMD.append(temp.rstrip())

			if 'prior' in temp:
				diagn.append('Prior')
			if 'posterior' in temp:
				diagn.append('Posterior')
			if 'truth' in temp:
				diagn.append('Truth')
				ens_status.append('Truth')
			if 'observations' in temp:
				diagn.append('Observation')
				ens_status.append('Observation')
			if 'ensemble member' in temp:
				ens_status.append('ensemble member')
			if 'ensemble mean' in temp:
				ens_status.append('ensemble mean')
			if 'ensemble spread' in temp:
				ens_status.append('ensemble spread')
			if 'observation error variance' in temp:
				ens_status.append(None)
				diagn.append(None)
			
                f.close()

	# return the desired observations and copys, and the copy meta data
	#for obs_type_no in obs_type_no_list:
	iobs=[]
	iensstatus=[]
	if debug:
		print('selecting the following obs type numbers')
		print(obs_type_no_list)
	for OTN in obs_type_no_list:
		itemp = np.where(obs_type == OTN)
		if itemp is not None:
			# itemp is a tuple - the first entry is the list of indices (I know - this is fucked)
			itemp2 = itemp[0]
			# now scoot through itemp2 (which is an ndarray...wtf?) and store the entires in a list
			for i in itemp2:
				iobs.append(i)


	# select the copys correposnind go the right ensemble status (or just copystring if the list isn;t give) and diagnostic
	if ens_status_list is None:
		ens_status_list = []
		ens_status_list.append(E['copystring'])
		if debug:
			print(ens_status_list)

	for ES in ens_status_list:
		indices = [i for i, x in enumerate(ens_status) if x == ES]
		iensstatus.extend(indices)
	iensstatus.sort()	# this is the list of copies with the right ensemble status
	idiagn = [i for i,x in enumerate(diagn) if x == E['diagn']]	# this is the list of copies with the right diagnostic

	# we are interested in the indices that appear in both iensstatus and idiagn
	sdiagn = set(idiagn)
	cc = [val for val in iensstatus if val in sdiagn]
	if debug:
		print('these are the copies that suit both the requested ensemble status and the requested diagnostic:')
		print(cc)

	# given the above copy numbers, find the names that suit them
	copynames = [CMD[ii] for ii in cc]

	# turn the array obs_type from numbers to words
	OT = []
	for ii in obs_type:
		temp = ObsTypesMetaData[ii-1,].tostring()
		OT.append(temp)
	# these are the obs types for the observations we select out
	OT_select = [OT[ii] for ii in iobs]


	# now select the observations corresponding to the selected copies and obs types
	i1 = np.array(iobs)
	i2 = np.array(cc)
	obs_select = observations[i1[:,None],i2]
	location_select = location[i1,]
	obs_type_select = obs_type[i1,]
	qc1_select = qc[i1,0]
	qc2_select = qc[i1,1]
	time_select = time[i1]
	ObsIndex_select = ObsIndex[i1]
	

	# for the arrays that are only defined by obs index, replicate for each copy
	loc1 = location_select[:,0]
	loc2 = location_select[:,1]
	loc3 = location_select[:,2]
	loc1_copies= np.repeat(loc1[:,np.newaxis],len(i2),1)
	loc2_copies= np.repeat(loc2[:,np.newaxis],len(i2),1)
	loc3_copies= np.repeat(loc3[:,np.newaxis],len(i2),1)
	qc1_copies =  np.repeat(qc1_select[:,np.newaxis],len(i2),1)
	qc2_copies =  np.repeat(qc2_select[:,np.newaxis],len(i2),1)
	obs_type_copies=  np.repeat(obs_type_select[:,np.newaxis],len(i2),1)
	ObsIndex_copies =  np.repeat(ObsIndex_select[:,np.newaxis],len(i2),1)

	# reshape the output from arrays to vectors
	# also have to squeeze out the empty dimension -- this seems really inelegant, but I don't know a better way to do it!
	L = len(iobs)*len(cc)		# length of the data vector
	date_out = np.repeat(date,L)
	obs_out = np.squeeze(np.reshape(obs_select,(L,1)))
	lon_out = np.squeeze(np.reshape(loc1_copies,(L,1)))
	lat_out = np.squeeze(np.reshape(loc2_copies,(L,1)))
	lev_out = np.squeeze(np.reshape(loc3_copies,(L,1)))
	qc1_out = np.squeeze(np.reshape(qc1_copies,(L,1)))
	qc2_out = np.squeeze(np.reshape(qc2_copies,(L,1)))
	obs_type_out = np.squeeze(np.reshape(obs_type_copies,(L,1)))
	ObsIndex_out = np.squeeze(np.reshape(ObsIndex_copies,(L,1)))



	# for each of the selected obs, report its copystring, ensemble status, and obs type
	copynames_out = []
	for ii in range(len(iobs)):
		for cn in copynames:
			copynames_out.append(cn)


	# round the location values because otherwise pandas fucks up the categorial variable aspect of them
	lat_out = np.round(lat_out,1)
	lon_out = np.round(lon_out,1)
	lev_out = np.round(lev_out)

	# return data frame
	data = {'QualityControl':qc1_out,
		'DARTQualityControl':qc2_out,
		'Value':obs_out,
		'Latitude':lat_out,
		'Longitude':lon_out,
		'Level':lev_out,
		'Date':date_out,
		'CopyName':copynames_out	
		}

	DF = pd.DataFrame(data,index=ObsIndex_out)

	# turn categorical data into categories
	#DF['QualityControl'] = DF['QualityControl'].astype('category')
	#DF['Latitude'] = DF['Latitude'].astype('category')
	#DF['Longitude'] = DF['Longitude'].astype('category')
	#DF['Level'] = DF['Level'].astype('category')
	#DF['CopyName'] = DF['CopyName'].astype('category')

	#return ObsIndex_out, loc1_out, loc2_out, loc3_out, qc_out, obs_out, copynames
	return DF

def load_DART_obs_epoch_file(E,date_in=None, hostname='taurus',debug=False):

	"""
	 this function reads in an obs_epoch_XXX.nc file for a certain DART experiment, with the obs that we want 
	 given in obs_type_list, and returns a vector of the desired observation. 

	INPUTS:
	E: an experiment dictionary 
		if E['copystring'] is a list of copystrings, we cycle through them. 
		if one of the strings in E['copystring'] is 'ensemble member', then return all the ensemble members. 
		if E['obs_name'] is a list of observation types, we cycle through and load them all. 
	date: the date on which we want to load the obs 
		the default for this is None -- in this case, just choose the first entry of E['daterange']
	hostname: computer name - default is Taurus 
	debug: debugging flag; default is False. 

	"""
	# select the date 
	if date_in is None:
		date_in = E['daterange'][0]

	# find the directory for this run   
	# this requires running a subroutine called `find_paths`, stored in a module `experiment_datails`, 
	# but written my each user -- it should take an experiment dictionary and the hostname 
	# as input, and return as output 
	# the filepath that corresponds to the desired field, diagnostic, etc. 
	filename = es.find_paths(E,date_in,hostname=hostname,file_type='obs_epoch',debug=debug)
	if not os.path.exists(filename):
		print("+++cannot find files that look like  "+filename+' -- returning None')
		return None,None

	# load the file and select the observation we want
        else:
		f = Dataset(filename,'r')
		if debug:
			print('Loading file '+filename)
			for key in f.variables.iterkeys():
			    print key
		observations = f.variables['observations'][:]
		time = f.variables['time'][:]
		copy = f.variables['copy'][:]
		CopyMetaData = f.variables['CopyMetaData'][:]
		ObsTypesMetaData = f.variables['ObsTypesMetaData'][:]
		obs_type = f.variables['obs_type'][:]

		# find the obs_type number corresponding to the desired observations
		if type(E['obs_name']) is list:
			obs_type_no_list = []
			for obs_type_string in E['obs_name']:
				obs_type_no_list.append(get_obs_type_number(f,obs_type_string))
		else:
			obs_type_no = get_obs_type_number(f,obs_type_string)
			obs_type_no_list = [obs_type_no]
		
		if type(E['copystring']) is not list:
			# if E['copystring'] is not a list and not 'ensemble', 
			# we only have one copy number to get -- cc tells us the number of it 
			cc = get_copy(f,E['diagn'].lower()+' '+E['copystring'])

		else:
			# if we have to retrieve more than one copy, 
			# expand "CopyMetaData" into lists that hold ensemble status and diagnostic
			diagn = []
			ens_status = []
			CMD = []
			for icopy in copy:
				temp = CopyMetaData[icopy-1,].tostring()
				CMD.append(temp.rstrip())

				if 'prior' in temp:
					diagn.append('Prior')
				if 'posterior' in temp:
					diagn.append('Posterior')
				if 'truth' in temp:
					diagn.append('Truth')
					ens_status.append('Truth')
				if 'observation' in temp:
					diagn.append('Observation')
					ens_status.append('Observation')
				if 'ensemble member' in temp:
					ens_status.append('ensemble member')
				if 'ensemble mean' in temp:
					ens_status.append('ensemble mean')
				if 'ensemble spread' in temp:
					ens_status.append('ensemble spread')
				if 'observation error variance' in temp:
					ens_status.append(None)
					diagn.append(None)
			
                f.close()

	# return the desired observations and copys, and the copy meta data
	iobs=[]
	iensstatus=[]
	if debug:
		print('this is the list of obs type numbers')
		print(obs_type_no_list)
	for OTN in obs_type_no_list:
		itemp = np.where(obs_type == OTN)
		if itemp is not None:
			iobs.append(list(np.squeeze(itemp)))
	# iobs is a list of lists -- turn it into a single list of indices
	iobs2 = [ii for sublist in iobs for ii in sublist]
	print('retrieving '+str(len(iobs))+' observations')

	if type(E['copystring']) is not list:
		# in this case only a single copy, which is defined in E, is returned
		obs_out = observations[iobs,cc]
		copy_names = E['diagn'].lower()+' '+E['copystring']


	else:
		# in this case several copies are returned
		for CS in E['copystring']:

			# ensemble member names are stored weirdly in DART output -- convert here
			if 'ensemble member ' in CS:
				import re
				ensindex = re.sub(r'ensemble member*','',CS).strip()
				if int(ensindex) < 10:
					spacing = '      '
				else:
					spacing = '     '
				CS = "ensemble member"+spacing+str(ensindex)		
			if debug:
				print('looking for copy '+CS)
			if CS is 'ensemble':
				# in this case look for all the copies that have ensemble status = "ensemble member"	
				indices = [i for i, x in enumerate(ens_status) if x == 'ensemble member']
			else:
				# for all other copystrings, just look for the CopyMetaData entries that contrain that copystring
				indices = [i for i, x in enumerate(CMD) if CS in x]
			if debug:
				print('here are the copy indices that fit this copystring')
				print indices
			iensstatus.extend(indices)
		iensstatus.sort()	# this is the list of copies with the right ensemble status
		idiagn = [i for i,x in enumerate(diagn) if x == E['diagn']]	# this is the list of copies with the right diagnostic
		if debug:
			print('here are the copy indices that fit the requested diagnostic')
			print idiagn

		# we are interested in the indices that appear in both iensstatus and idiagn
		sdiagn = set(idiagn)
		jj = [val for val in iensstatus if val in sdiagn]
		if debug:
			print('here are the copy indices that fit both the requested copystrings and the requested diagnostic')
			print jj
			print('this corresponds to the following:')
			for j in jj:
				print CMD[j]

		# now select the observations corresponding to these copies 
		obs1 = observations[iobs2,:]
		obs2 = obs1[:,jj]
		obs_out = obs2
		copy_names = [ CMD[i] for i in jj ]

	return obs_out,copy_names



def load_DART_diagnostic_file(E,date=datetime.datetime(2009,1,1,1,0,0),hostname='taurus',debug=False):

	# if debugging, print out what we're doing  
	if debug:
		print('+++++++++++++++++++++++++++++++++++++++++')
		print("Retrieving experiment "+E['exp_name'])
		print("for diagnostic "+E['diagn'])
		print("copy "+E['copystring'])
		print("extras "+E['extras'])
		datestr = date.strftime("%Y-%m-%d")
		print("and date "+datestr)
		print('+++++++++++++++++++++++++++++++++++++++++')


	# retrieve the entries of the experiment dictionary, E:
	variable = E['variable']
	experiment = E['exp_name']

	# if the diagnostic is the Truth, then the copy string can only be one thing
	copystring = E['copystring']
	if (E['diagn'] == 'Truth'):
		copystring = 'true state'

	# if we want the ensemble variance, copystring has to be the ensemble spread
	if (E['extras'] == 'ensemble variance') or (E['extras'] == 'ensemble variance scaled'):
		copystring = 'ensemble spread'

	# find the directory for this run   
	# this requires running a subroutine called `find_paths`, stored in a module `experiment_datails`, 
	# but written my each user -- it should take an experiment dictionary and the hostname 
	# as input, and return as output 
	# the filepath that corresponds to the desired field, diagnostic, etc. 
	filename = es.find_paths(E,date,'diag',hostname=hostname,debug=debug)
	if not os.path.exists(filename):
		print("+++cannot find files that look like  "+filename+' -- returning None')
		return None,None,None,None,None,None,None
	else:
		if debug:
			print('opening file  '+filename)
		f = Dataset(filename,'r')
		lev = f.variables['lev'][:]
		P0 = f.variables['P0'][:]
		hybm = f.variables['hybm'][:]
		hyam = f.variables['hyam'][:]
		VV = f.variables[variable]
		if (variable=='US'):
			lat = f.variables['slat'][:]
		else:
			lat = f.variables['lat'][:]
		if (variable=='VS'):
			lon = f.variables['slon'][:]
		else:
			lon = f.variables['lon'][:]

	
		# figure out which copy to load
		if debug:
			print(copystring)
		copy = get_copy(f,copystring)

		# figure out which vertical level range we want
		levrange=E['levrange']
		k1 = (np.abs(lev-levrange[1])).argmin()
		k2 = (np.abs(lev-levrange[0])).argmin()
		lev2 = lev[k1:k2+1]

		# figure out which latitude range we want
		latrange=E['latrange']
		j2 = (np.abs(lat-latrange[1])).argmin()
		j1 = (np.abs(lat-latrange[0])).argmin()
		lat2 = lat[j1:j2+1]

		# figure out which longitude range we want
		lonrange=E['lonrange']
		i2 = (np.abs(lon-lonrange[1])).argmin()
		i1 = (np.abs(lon-lonrange[0])).argmin()
		lon2 = lon[i1:i2+1]


		if (variable=='PS'):
			VV2 = VV[0,copy,j1:j2+1,i1:i2+1]
		else:
			VV2 = VV[0,copy,j1:j2+1,i1:i2+1,k1:k2+1]

		# if the ensemble variance was requested, square it here
		if (E['extras'] == 'ensemble variance'): 
			VVout = np.square(VV2)

		# if the copystring is ensemble variance scaled, square the ensemble spread and scale by ensemble size
		if (E['extras'] == 'ensemble variance scaled'):
			if debug:
				print('squaring and scaling ensemble spread to get scaled variance')
			N = get_ensemble_size(f)
			fac = (N+1)/N
			VVout = fac*np.square(VV2)
		else:
			VVout = VV2

		# close the primary file  
		f.close()

		# if requestiing the mean square error (MSE), load the corresponding truth run and subtract it out, then square  
		if (E['extras'] == 'MSE'):
			Etr = E.copy()
			Etr['diagn'] = 'Truth'
			filename_truth = es.find_paths(Etr,date,'truth',hostname=hostname,debug=debug)
			if not os.path.exists(filename):
				print("+++cannot find files that look like  "+filename_truth+' -- returning None')
				return None,None,None,None,None,None,None
			else:
				if debug:
					print('opening file  '+filename_truth)

			# open the truth file and load the field
			ft = Dataset(filename_truth,'r')
			VT = ft.variables[variable]

			# select the true state as the right copy
			copyt = get_copy(ft,'true state',debug)
			if (variable=='PS'):
				VT2 = VT[0,copy,j1:j2+1,i1:i2+1]
			else:
				VT2 = VT[0,copy,j1:j2+1,i1:i2+1,k1:k2+1]

			# close the truth file
			ft.close()

			# compute the square error
			SE = np.square(VV2-VT2)
			VVout = SE

	return lev2,lat2,lon2,VVout,P0,hybm,hyam

def get_ensemble_size(f):

	"""
	given a DART output diagnostic netcdf file that is already open, 	
	find the number of ensemble members in the output  
	"""

        CMD = f.variables['CopyMetaData'][:]
	CopyMetaData = []
	for ii in range(0,len(CMD)):
		temp = CMD[ii,].tostring()
		CopyMetaData.append(temp.rstrip())
	ens_members = [x for x in CopyMetaData if 'ensemble member' in x]
	return len(ens_members)


def get_obs_type_number(f,obs_type_string):

	"""
	having opened a DART output diagnostic netcdf file, find the obs_type number
	that corresponds to a given obs_typestring
	"""

        # figure out which obs_type to load
        OTMD = f.variables['ObsTypesMetaData'][:]
        ObsTypesMetaData = []
        for ii in range(0,len(OTMD)):
                temp = OTMD[ii,].tostring()
                ObsTypesMetaData.append(temp.rstrip())

        obs_type = ObsTypesMetaData.index(obs_type_string)+1

	return obs_type

def get_copy(f,copystring,debug=False):

	"""
	having opened a DART output diagnostic netcdf file, find the copy number that corresponds to a given copystring
	"""
	
	# DART copy strings for individual ensemble members have extra spaces in them -- account for that here:
	if 'ensemble member' in copystring:
		ensindex = re.sub(r'ensemble member*','',copystring).strip()
		if int(ensindex) < 10:
			spacing = '      '
		else:
			spacing = '     '
		copystring = "ensemble member"+spacing+ensindex

        # figure out which copy to load
        CMD = f.variables['CopyMetaData'][:]
        CopyMetaData = []
        for ii in range(0,len(CMD)):
                temp = CMD[ii,].tostring()
                CopyMetaData.append(temp.rstrip())
	if debug:
		print(CopyMetaData.index)
        copy = CopyMetaData.index(copystring)

	return copy

def basic_experiment_dict():

	"""
	this is a default Python dictionary containing the details of an experiment that we look at -- 
	the purpose of this is only to make the inputs to diagnostic codes shorter.
	"""

	E = {'exp_name' : 'generic_experiment_name',
	'diagn' : 'Prior',
	'copystring' : 'ensemble mean',
	'variable' : 'US',
	'levrange' : [1000, 0], 
	'latrange' : [-90,91],
	'lonrange' : [0,361],
	'extras' : None,
	'obs_name':'T',
	'run_category' : None ,
	'daterange':daterange(),
	'clim':None
	}

	return E

def date_to_gday(date=datetime.datetime(2009,10,2,12,0,0)):

	"""
	convert a datetime date to gregorian day count the way it's counted in DART  (i.e. number of days since 1601-01-01
	"""

        datestr = date.strftime("%Y-%m-%d")
	jd = dayconv.gd2jd(datestr)

	# reference date: 
	jd_ref = dayconv.gd2jd("1601-01-01")

	gday_out = jd-jd_ref

	return gday_out

def daterange(date_start=datetime.datetime(2009,1,1), periods=5, DT='1D'):

	"""
        generate a range of dates (in python datetime format), given some start date, a time delta, and the numper of periods
	"""

        # the last character of DT indicates how we are counting time
        time_char = DT[len(DT)-1]
        time_int = int(DT[0:len(DT)-1])

	#base = datetime.datetime.today()
	#date_list = [base - datetime.timedelta(days=x) for x in range(0, numdays)]
        if (time_char == 'D') or (time_char == 'd'):
		date_list = [date_start + time_int*datetime.timedelta(days=x) for x in range(0, periods)]
        if (time_char == 'H') or (time_char == 'h'):
		#date_list = [date_start + datetime.timedelta(hours=x) for x in range(0, periods)]
		date_list = [date_start + time_int*datetime.timedelta(hours=x) for x in range(0, periods)]


	return date_list

def rank_hist(VE,VT):

	"""
	given a 1-D ensemble time series and a verification (usually the truth), compute the
	rank histogram over the desired block of time  
	"""

	# query the ensemble size
	N = VE.shape[0]

	# turn the ensemble and truth into vectors (in case they are 3D fields)
	if len(VE.shape)== 6:
		nentries = VE.shape[1]*VE.shape[2]*VE.shape[3]*VE.shape[4]*VE.shape[5]
		ens = np.zeros((N,nentries))
		for iens in range(N):
			ens[iens,:] = np.ravel(VE[iens,:,:,:,:,:])	
	if len(VE.shape)== 5:
		nentries = VE.shape[1]*VE.shape[2]*VE.shape[3]*VE.shape[4]
		ens = np.zeros((N,nentries))
		for iens in range(N):
			ens[iens,:] = np.ravel(VE[iens,:,:,:,:])	
	if len(VE.shape)== 4:
		nentries = VE.shape[1]*VE.shape[2]*VE.shape[3]
		ens = np.zeros((N,nentries))
		for iens in range(N):
			ens[iens,:] = np.ravel(VE[iens,:,:,:])	
	if len(VE.shape)== 3:
		nentries = VE.shape[1]*VE.shape[2]
		ens = np.zeros((N,nentries))
		for iens in range(N):
			ens[iens,:] = np.ravel(VE[iens,:,:])	

	# determine the length of the ensemble series
	nT = ens.shape[1]

	# the verification series is the true state, raveleed over all spatial or time directions
	verif = np.ravel(VT)

	# given an N-member ensemble and a verification, there are N+1 possible ranks
	# the number of bins is the ensemble size plus 1
	bins = range(1,N+2)
	hist = [0]*len(bins)

	# loop through the times, and get a count for each bin
	for i in range(nT):

		# for each time, reset the counter to zero
		count = 0

		for j in range(N):
			# count every ensemble member that is less than the true value
			if (verif[i] > ens[j,i]):
				count += 1
		# after checking all ensemble members, add the resulting 
		# count to the appropriate bin:
		hist[count] += 1 

	return bins,hist

def kurtosis(ens):  

	"""
	given a 1D ensemble of numbers (obs space, state space, whatever) return the kurtosis of the PDF represented by the ensemble
	"""

	# compute the standard deviation and mean
	sigma = np.std(ens)
	mu = np.mean(ens)
	N = len(ens)

	# this is the factor that multiplies the summation in the kurtosis:
	fac = 1/((N-1)*sigma**4)

	kurtosis = 0
	for iens in range(N):
		dev = (ens[iens]-mu)**4
		kurtosis += fac*dev

	return kurtosis

def skewness(ens):  

	"""
	given a 1D ensemble of numbers (obs space, state space, whatever) return the skewness of the PDF represented by the ensemble
	"""

	# compute the standard deviation and mean
	sigma = np.std(ens)
	mu = np.mean(ens)
	N = len(ens)

	# this is the factor that multiplies the summation in the kurtosis:
	fac = 1/((N-1)*sigma**3)

	kurtosis = 0
	for iens in range(N):
		dev = (ens[iens]-mu)**3
		kurtosis += fac*dev

	return kurtosis


def point_check_dictionaries(return_as_list=True):

	"""
	pre-defined experiment dictionaries that give various averaging regions 
	over which to check the ensemble.
	"""

	E = basic_experiment_dict()

	E1 = E.copy()
	E1['latrange'] = [-5,5]
	E1['levrange'] = [900,800]
	E1['lonrange'] = [120,170]
	E1['title'] = '850 hPa, Nino 3.4 region'

	E2 = E.copy()
	E2['latrange'] = [-5,5]
	E2['lonrange'] = [0,360]
	E2['levrange'] = [4E-5,0]
	E2['title'] = 'Model Top, Tropics'

	E3 = E.copy()
	E3['latrange'] = [30,40]
	E3['lonrange'] = [280,360]
	E3['levrange'] = [300,200]
	E3['title'] = 'Atlantic Jet Stream'

	E4 = E.copy()
	E4['latrange'] = [60,90]
	E4['lonrange'] = [0,360]
	E4['levrange'] = [30,24]
	E4['title'] = 'North Polar Vortex'

	E5 = E.copy()
	E5['latrange'] = [-90,-60]
	E5['lonrange'] = [0,360]
	E5['levrange'] = [30,24]
	E5['title'] = 'South Polar Vortex'

	E6 = E.copy()
	E6['latrange'] = [0,30]
	E6['levrange'] = [1000,500]
	E6['lonrange'] = [180,210]
	E6['title'] = 'North Subtropical Pacific'

	E7 = E.copy()
	E7['latrange'] = [-20,20]
	E7['levrange'] = [105,90]
	E7['lonrange'] = [0,360]
	E7['title'] = 'Tropical Tropopause'

	E8 = E.copy()
	E8['latrange'] = [50,90]
	E8['levrange'] = [370,320]
	E8['lonrange'] = [0,360]
	E8['title'] = 'NHET Tropopause'

	if return_as_list:
		GG = [E1,E2,E3,E4,E5,E6,E7,E8]
		return GG
	else:
		return E1,E2,E3,E4,E5,E6,E7,E8

def climate_index_dictionaries(index_name):

	"""
	This function returns experiment dictionaries with the lat, long, and levranges needed to compute certain climate indices.  
	Eventually this should replace my point_check_dictionaries above.  

	Currently supporting the following indices:  
	+ 'Aleutian Low' index of Garfinkel et al. (2010)
		* note: Garfinkel et al 2010 give this as 175E, but that is closer to Russia than Alaska. 175W (=185E) is closer to the Aleutian islands. 
	+ 'East European High' index of Garfinkel et al. (2010)
	+ 'AO Proxy' -- Polar Cap GPH Anomaly at 500hPa -- it's a  proxy for the AO suggested by Cohen et al. (2002)   
		* note however that we define the polar cap as everything north of 70N, I think Cohen et al do 60N
	+ 'Vortex Strength' -- Polar Cap GPH Anomaly averaged 3-30hPa -- it's a measure of vortex strength suggested by Garfinkel et al. 2012

	"""
	index_name_found = False

	E = basic_experiment_dict()
	
	if index_name == 'Aleutian Low':
		index_name_found = True
		E['latrange'] = [55,55]
		E['lonrange'] = [185,185]
		E['levrange'] = [500,500]
		E['variable'] = 'Z3'

	if index_name == 'East European High':
		index_name_found = True
		E['latrange'] = [60,60]
		E['lonrange'] = [40,40]
		E['levrange'] = [500,500]
		E['variable'] = 'Z3'

	if index_name == 'AO Proxy':
		index_name_found = True
		E['latrange'] = [70,90]
		E['lonrange'] = [0,360]
		E['levrange'] = [500,500]
		E['variable'] = 'Z3'

	if index_name == 'Vortex Strength':
		index_name_found = True
		E['latrange'] = [70,90]
		E['lonrange'] = [0,360]
		E['levrange'] = [30,3]
		E['variable'] = 'Z3'

	if index_name_found == False:
		print('Do not have a definition for climate index  '+index_name)
		print('returning generic experiment dictionary.')
	return E

def change_daterange_to_daily(DR):

	"""
	Change a DART experiment daterange to daily resolution  
	"""

	d0 = DR[0]
	df = DR[len(DR)-1]
	days = df-d0
	DR_out = daterange(date_start=d0, periods=days.days+1, DT='1D')
	return DR_out  
