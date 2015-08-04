## Python module for DART-WACCM paths and stuff
## Lisa Neef, 4 June 2014


import numpy as np
from netCDF4 import Dataset
import datetime as datetime
import dayconv 
import os.path
import pandas as pd
import re
import experiment_details 

def load_covariance_file(E,hostname='taurus',debug=False):

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
	filename = find_paths(E,hostname)
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
	filename = find_paths(E,hostname)
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
#BINK
def load_DART_obs_epoch_file(E,date=datetime.datetime(2009,1,1,0,0,0),obs_type_list=['ERP_PM1','ERP_LOD'],ens_status_list=['ensemble member'], hostname='taurus',debug=False):

	# this function reads in an obs_epoch_XXX.nc file for a certain DART experiment, with the obs that we want 
	# given in obs_type_list.
	# the parameter ens_status_list= gives a list of all copy types to return, so for example if 
	# ens_status_list=['ensemble member'], then return all the ensemble members, regardless of what E['copystring'] says.
	# if this list is left blank, only return what's specified by the E['copystring'] entry in the experiment dictionary

	# find the directory corresponding to this run  
	if E['run_category'] == 'ERPDA':
		run_branch,truth_branch = exp_paths_old(hostname,E['exp_name'])
		run_dir = run_branch+'/'
	if E['run_category'] == 'NCAR':
		run_branch,truth_branch = exp_paths_NCAR(hostname,E['exp_name'])
		run_dir = run_branch+'/'
	if E['run_category'] == None:
		run_branch,truth_branch = exp_paths(hostname,E['exp_name'])
		run_dir = run_branch+'../obs_epoch/'

	# find the obs epoch file corresponding to this run  
	start_date = get_expt_start_date(E)
	delta_time = date-start_date
	obs_epoch_no = delta_time.days+1
	if obs_epoch_no < 10:
		obs_epoch_name = 'obs_epoch_00'+str(obs_epoch_no)+'.nc'
	else:
		obs_epoch_name = 'obs_epoch_0'+str(obs_epoch_no)+'.nc'
	filename = run_dir+obs_epoch_name
	if debug:
		print(filename)

	# load the file and select the observation we want
        if os.path.isfile(filename):
		f = Dataset(filename,'r')
		observations = f.variables['observations'][:]
		time = f.variables['time'][:]
		copy = f.variables['copy'][:]
		CopyMetaData = f.variables['CopyMetaData'][:]
		obs_type = f.variables['obs_type'][:]

		# find the obs_type number corresponding to the desired observations
		obs_type_no_list = []
		for obs_type_string in obs_type_list:
			obs_type_no_list.append(get_obs_type_number(f,obs_type_string))
		
		# if ens_status_list is empty, simply find the copy corresponding to copystring, and we're done
		if ens_status_list is None:

			cc = get_copy(f,E['diagn'].lower()+' '+E['copystring'])

		else:

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
	else:
		print("Cannot find the observation file "+filename)
		return

	# return the desired observations and copys, and the copy meta data
	#for obs_type_no in obs_type_no_list:
	iobs=[]
	iensstatus=[]
	if debug:
		print('this is the list of obs type numbers')
		print(obs_type_no_list)
	for OTN in obs_type_no_list:
		itemp = np.where(obs_type == OTN)
		if itemp is not None:
			iobs.append(itemp[0][0])

	if ens_status_list is None:
		# in this case only a single copy, which is defined in E, is returned
		obs_out = observations[iobs,cc]
		copy_names = E['diagn'].lower()+' '+E['copystring']


	else:
		# in this case several copies are returned, everything corresponding to 
		# the diagnostic in E and the ensemble statuses in ens_status_list
		for ES in ens_status_list:
			indices = [i for i, x in enumerate(ens_status) if x == ES]
			iensstatus.extend(indices)
		iensstatus.sort()	# this is the list of copies with the right ensemble status
		idiagn = [i for i,x in enumerate(diagn) if x == E['diagn']]	# this is the list of copies with the right diagnostic

		# we are interested in the indices that appear in both iensstatus and idiagn
		sdiagn = set(idiagn)
		jj = [val for val in iensstatus if val in sdiagn]

		# now select the observations corresponding to these copies 
		obs1 = observations[iobs,:]
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
		print("and copy "+E['copystring'])
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

	# find the date and second strings that give us the filename
	datestr = date.strftime("%Y-%m-%d")
	seconds = date.hour*60*60
	if seconds == 0:
		timestr = '00000'
	else:
		timestr = str(seconds)

	# define the filenames to load based on what category of run this is  
	if E['run_category'] is None:
		run_dir_list,truth_dir_list = exp_paths(hostname,experiment)
		ff_run = '/'+'/dart/hist/cam_'+E['diagn']+'_Diag.'+datestr+'-'+timestr+'.nc'
		ff_truth = '/'+'/dart/hist/'+'True_State'+'.'+datestr+'-'+timestr+'.nc'

	if E['run_category'] == 'ERPDA':
		run_dir_list,truth_dir_list = exp_paths_old(hostname,experiment)
		gday = date_to_gday(date)
		# for all my (Lisa's) old experiments, obs sequence 1 is 1 Jan 2009
		gday1 = date_to_gday(datetime.datetime(2009,1,1,0,0,0))
		obs_seq_no = int(gday-gday1+1)
		if (obs_seq_no < 10):
			mid = 'obs_000'+str(obs_seq_no)
		else:
			mid = 'obs_00'+str(obs_seq_no)
		ff_truth = mid+'/'+'True_State.nc'
		ff_run = mid+'/'+E['diagn']+'_Diag.nc'
		
		# the filename and run dir are different if the diagnostic is a CAM input file  
		if E['diagn'] == 'caminput':
			mid = 'inputs'  
			ff_run = mid+'/'+E['diagn']+'_.nc'

	if E['run_category']=='NCAR':
		run_dir_list,truth_dir_list = exp_paths_NCAR(hostname,experiment)
		prefix = ''
		if E['exp_name'] == 'NCAR_LAONLY':
			suffix = '_LAONLY'
		else:
			suffix = ''
		ff_truth = '/'+'True_State'+'_'+datestr+'-'+timestr+'.nc'+suffix
		ff_run = '/'+prefix+E['diagn']+'_Diag.'+datestr+'-'+timestr+'.nc'+suffix
		ff_covars = '/'+E['exp_name']+'_covariance_'+E['obs_name']+'_'+E['variable']+'_'+date.strftime('%Y-%m-%d')+'.nc'

	# loop over the path list and look for the correct file  
	correct_filepath_found = False
	data_dir_list = run_dir_list
	ff = ff_run
	if E['diagn'] == 'Truth':
		if truth_dir_list is None:
			if debug:
				print 'This run has no truth associated with it.'
			return None, None, None, None, None, None, None
		else:
			data_dir_list = truth_dir_list
			ff = ff_truth
			copystring = "true state"
	if (E['diagn'] == 'Covariance') or (E['diagn'] == 'Correlation'):
		filename = filename_covars

	if debug:
		print('checking for DART diagnostic files in these directories:')
		print data_dir_list
	for data_dir in data_dir_list:
		if debug:
			print 'Looking for file in  '+data_dir
		filename = data_dir+ff
		if os.path.exists(filename):
			correct_filepath_found = True
			break

	if correct_filepath_found is False:
		print("+++cannot find files that look like  "+ff+' for experiment '+E['exp_name']+' -- returning NA')
		return None, None, None, None, None, None, None

	# open the file if available; return NaNs if not
	if filename is not None:
		if os.path.isfile(filename):
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
			#print('loading the following vertical levels:------')
			#print(lev2)

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
				# in this case we also need to search the right directories  
	
				correct_filepath_found = False
				for truth_dir in truth_dir_list:
					if debug:
						print 'Looking for file in  '+truth_dir
					filename_truth = truth_dir+ff_truth
					if os.path.exists(filename_truth):
						correct_filepath_found = True
						break
				if not correct_filepath_found:
					print('Cannot find the true state file  '+filename_truth)
					return
				if debug:
					print('Opening true state file: '+filename_truth)

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

			
		else:
			print('--------------------------------------Cannot find file:  '+filename)
			VVout = np.nan
			lev2 = np.nan
			lat2 = np.nan
			lon2 = np.nan
			P0 = np.nan
			hybm = np.nan
			hyam = np.nan
	else:
		VVout = np.nan
		lev2 = np.nan
		lat2 = np.nan
		lon2 = np.nan
		P0 = np.nan
		hybm = np.nan
		hyam = np.nan

	return lev2,lat2,lon2,VVout,P0,hybm,hyam

def get_ensemble_size(f):

	# given a DART output diagnostic netcdf file that is already open, 	
	# find the number of ensemble members in the output  

        CMD = f.variables['CopyMetaData'][:]
	CopyMetaData = []
	for ii in range(0,len(CMD)):
		temp = CMD[ii,].tostring()
		CopyMetaData.append(temp.rstrip())
	ens_members = [x for x in CopyMetaData if 'ensemble member' in x]
	return len(ens_members)

def get_ensemble_size_per_run(exp_name):

	# given some existing DART experiment, look up which ensemble size was used there
	N = {'ERPALL' : 80,
	'NODA' : 80,
	'RST' : 80,
	'ERPRST' : 80,
	'SR' : 64,
	'STINFL' : 64,
	'OBSINFL' : 64,
	'PMO27' : 38,
	'PMO28' : 38,
	'PMO32' : 40,
	'NCAR_FULL' : 40,
	'NCAR_LAONLY' : 40, 
	'NCAR_PMO_CONTROL' : 40, 
	'NCAR_PMO_LA' : 40, 
	'NCAR_PMO_LAS' : 40,
	'W0910_NODA' : 40,
	'W0910_TROPICS' : 40,
	'W0910_GLOBAL' : 40
	}

	return(N[exp_name])

def get_expt_CopyMetaData_state_space(E):

	# this code stores a dictionary for each experiment, that connects the copy numbers to their 
	# CopyMetaData -- this is easier than retrieving this information each time.

	exp_found = False

	if E['diagn'] == 'Truth':
		CopyMetaData = ["true state"]
		exp_found = True
	else:
		if E['run_category'] == 'NCAR':
			exp_found = True
			CopyMetaData = ["ensemble mean",
				"ensemble spread",
				"ensemble member      1",
				"ensemble member      2",
				"ensemble member      3",
				"ensemble member      4",
				"ensemble member      5",
				"ensemble member      6",
				"ensemble member      7",
				"ensemble member      8",
				"ensemble member      9",
				"ensemble member     10",
				"ensemble member     11",
				"ensemble member     12",
				"ensemble member     13",
				"ensemble member     14",
				"ensemble member     15",
				"ensemble member     16",
				"ensemble member     17",
				"ensemble member     18",
				"ensemble member     19",
				"ensemble member     20",
				"ensemble member     21",
				"ensemble member     22",
				"ensemble member     23",
				"ensemble member     24",
				"ensemble member     25",
				"ensemble member     26",
				"ensemble member     27",
				"ensemble member     28",
				"ensemble member     29",
				"ensemble member     30",
				"ensemble member     31",
				"ensemble member     32",
				"ensemble member     33",
				"ensemble member     34",
				"ensemble member     35",
				"ensemble member     36",
				"ensemble member     37",
				"ensemble member     38",
				"ensemble member     39",
				"ensemble member     40",
				"inflation mean",
				"inflation sd" ]

		if E['run_category'] == None:
			exp_found = True
			CopyMetaData = ["ensemble mean",
				"ensemble spread",
				"ensemble member      1",
				"ensemble member      2",
				"ensemble member      3",
				"ensemble member      4",
				"ensemble member      5",
				"ensemble member      6",
				"ensemble member      7",
				"ensemble member      8",
				"ensemble member      9",
				"ensemble member     10",
				"ensemble member     11",
				"ensemble member     12",
				"ensemble member     13",
				"ensemble member     14",
				"ensemble member     15",
				"ensemble member     16",
				"ensemble member     17",
				"ensemble member     18",
				"ensemble member     19",
				"ensemble member     20",
				"ensemble member     21",
				"ensemble member     22",
				"ensemble member     23",
				"ensemble member     24",
				"ensemble member     25",
				"ensemble member     26",
				"ensemble member     27",
				"ensemble member     28",
				"ensemble member     29",
				"ensemble member     30",
				"ensemble member     31",
				"ensemble member     32",
				"ensemble member     33",
				"ensemble member     34",
				"ensemble member     35",
				"ensemble member     36",
				"ensemble member     37",
				"ensemble member     38",
				"ensemble member     39",
				"ensemble member     40",
				"inflation mean",
				"inflation sd" ]

		if (E['run_category'] == 'ERPDA'): 
			exp_found = True
			CopyMetaData = ["ensemble mean",
				"ensemble spread",
				"ensemble member      1",
				"ensemble member      2",
				"ensemble member      3",
				"ensemble member      4",
				"ensemble member      5",
				"ensemble member      6",
				"ensemble member      7",
				"ensemble member      8",
				"ensemble member      9",
				"ensemble member     10",
				"ensemble member     11",
				"ensemble member     12",
				"ensemble member     13",
				"ensemble member     14",
				"ensemble member     15",
				"ensemble member     16",
				"ensemble member     17",
				"ensemble member     18",
				"ensemble member     19",
				"ensemble member     20",
				"ensemble member     21",
				"ensemble member     22",
				"ensemble member     23",
				"ensemble member     24",
				"ensemble member     25",
				"ensemble member     26",
				"ensemble member     27",
				"ensemble member     28",
				"ensemble member     29",
				"ensemble member     30",
				"ensemble member     31",
				"ensemble member     32",
				"ensemble member     33",
				"ensemble member     34",
				"ensemble member     35",
				"ensemble member     36",
				"ensemble member     37",
				"ensemble member     38",
				"ensemble member     39",
				"ensemble member     40",
				"ensemble member     41",
				"ensemble member     42",
				"ensemble member     43",
				"ensemble member     44",
				"ensemble member     45",
				"ensemble member     46",
				"ensemble member     47",
				"ensemble member     49",
				"ensemble member     50",
				"ensemble member     51",
				"ensemble member     52",
				"ensemble member     53",
				"ensemble member     54",
				"ensemble member     55",
				"ensemble member     56",
				"ensemble member     57",
				"ensemble member     58",
				"ensemble member     59",
				"ensemble member     60",
				"ensemble member     61",
				"ensemble member     62",
				"ensemble member     63",
				"ensemble member     64",
				"ensemble member     65",
				"ensemble member     66",
				"ensemble member     67",
				"ensemble member     68",
				"ensemble member     69",
				"ensemble member     70",
				"ensemble member     71",
				"ensemble member     72",
				"ensemble member     73",
				"ensemble member     74",
				"ensemble member     75",
				"ensemble member     76",
				"ensemble member     77",
				"ensemble member     78",
				"ensemble member     79",
				"ensemble member     80"]

	if exp_found:
		return CopyMetaData
	else:
		print('Still need to store the CopyMetaData for experiment '+E['exp_name']+' in subroutine DART.py')
		return None

def get_expt_start_date(E):

	# this code stores a dictionary of what the first day was for different experiments.
	# this makes it easier to figure out which obs epoch file to load for a given date

	D = {'ERPALL' : datetime.datetime(2009,1,1),
	'NODA' : datetime.datetime(2009,1,1),
	'RST' : datetime.datetime(2009,1,1),
	'ERPRST' : datetime.datetime(2009,1,1),
	'PMO32' : datetime.datetime(2009,10,1),
	'W0910_NODA' : datetime.datetime(2009,10,1),
	'W0910_TROPICS' : datetime.datetime(2009,10,1),
	'W0910_GLOBAL' : datetime.datetime(2009,10,1)
	}

	return(D[E['exp_name']])

def get_obs_type_number(f,obs_type_string):

	# having opened a DART output diagnostic netcdf file, find the obs_type number
	# that corresponds to a given obs_typestring

        # figure out which obs_type to load
        OTMD = f.variables['ObsTypesMetaData'][:]
        ObsTypesMetaData = []
        for ii in range(0,len(OTMD)):
                temp = OTMD[ii,].tostring()
                ObsTypesMetaData.append(temp.rstrip())

        obs_type = ObsTypesMetaData.index(obs_type_string)+1

	return obs_type

def get_copy(f,copystring,debug=False):

	# having opened a DART output diagnostic netcdf file, find the copy number
	# that corresponds to a given copystring
	
	# DART copy strings for individual ensemble members have extra spaces in them -- account for that here:
	if 'ensemble member' in copystring:
		ensindex = re.sub(r'ensemble member*','',copystring).strip()
		if int(ensindex) < 10:
			spacing = '      '
		else:
			spacing = '     '
		copystring = "ensemble member"+spacing+ensindex

	#print("+++++Loading copy "+copystring)

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

def exp_paths_era(hostname='taurus'):

	"""
	Paths to ERA-Interm and ERA-40 data  
	"""
	truth_dir_list = None
	run_dir_list = None

	if (hostname=='taurus'):
		run_dir = '/data/c1/lneef/ERA/'
		run_dir_list = [run_dir]
	
        return run_dir_list, truth_dir_list


def exp_paths(hostname='blizzard',experiment='PMO32'):

	# store the location of the output for individual DART experiments
	if (hostname=='blizzard'):
		branch='/work/bb0519/CESM/cesm1_2_0/archive/b350071/'
		branch_list = [branch]

	if (hostname=='taurus'):
		# there are several places on Taurus where our experiments live...
		branch1='/data/a1/swahl/cesm1_2_0/archive/'
		branch2='/data/c1/lneef/DART-WACCM/'
		branch_list = [branch1,branch2]

	# list of the full names for each experiment  
	names = {'PMO18' : 'waccm-dart-assimilate-pmo-18',
	'PMO27' : 'waccm-dart-assimilate-pmo-27',	
	'PMO28' : 'waccm-dart-assimilate-pmo-28',	
	'PMO32' : 'waccm-dart-assimilate-pmo-32',	
	'W0910_NODA' : 'nechpc-waccm-dart-gpsro-ncep-no-assim-01',			
	'W0910_TROPICS' : 'nechpc-waccm-dart-gpsro-ncep-30S-30N-01',			
	'W0910_GLOBAL' : 'nechpc-waccm-dart-gpsro-ncep-global-01'			
	}

	# list of the truth runs that were used to generate the obs for each (PMO) experiment
	truth_names = {'PMO18' : 'f55wcn-pmo-cosmic-erp-01',
	'PMO27' : 'f55wcn-pmo-cosmic-erp-01',
	'PMO28' : 'f55wcn-pmo-cosmic-erp-01',
	'PMO32' : 'f55wcn-pmo-cosmic-erp-01',
	'W0910_NODA'  : None,
	'W0910_TROPICS'  : None,
	'W0910_GLOBAL'  : None
	}

	#run_dir = branch+names[experiment]+'/dart/hist/'
	#truth_dir = branch+truth_names[experiment]+'/dart/hist/'
	name = names[experiment]
	truth_name = truth_names[experiment]
	run_dir_list =  [branch+name for branch in branch_list]
	if truth_name is None:
		truth_dir_list = None
	else:
		truth_dir_list =  [branch+truth_name for branch in branch_list]

        return run_dir_list, truth_dir_list

def exp_paths_NCAR(hostname='taurus',experiment='NCAR_FULL'):

	branch = None

	# this is a place to store and retrieve the locations of the DART-WACCM runs performed
	# by Nick Pedatella at NCAR
	if (hostname=='taurus'):
		branch1 ='/data/a1/swahl/DART/'
		branch_list = [branch1]

	# list of the full names for each experiment
	names = {'NCAR_FULL' : 'FULL/',
	'NCAR_LAONLY' : 'LAONLY/',  
	'NCAR_PMO_CONTROL' : 'NCAR_PMO_CONTROL/',  
	'NCAR_PMO_LA' : 'NCAR_PMO_LA/',  
	'NCAR_PMO_LAS' : 'NCAR_PMO_LAS/'  
	}

	# list of the truth runs that were used to generate the obs for each (PMO) experiment
	truth_names = {'NCAR_FULL' : None, 
	'NCAR_LAONLY' : None,  
	'NCAR_PMO_CONTROL' : 'NCAR_TRUE_STATE/',  
	'NCAR_PMO_LA' : 'NCAR_TRUE_STATE/',  
	'NCAR_PMO_LAS' : 'NCAR_TRUE_STATE/'  
	}

	name = names[experiment]
	truth_name = truth_names[experiment]
	run_dir_list =  [branch+name for branch in branch_list]
	if truth_name is None:
		truth_dir_list = None
	else:
		truth_dir_list =  [branch+truth_name for branch in branch_list]

	return run_dir_list, truth_dir_list

def exp_paths_old(hostname='taurus',experiment='ERPALL'):

	branch = None

	# this is a place to store and retrieve the locations of my old DART-CAM experiments  
        if (hostname=='blizzard'):
                branch='/work/scratch/b/b325004/DART_ex/'

	if (hostname=='taurus'):
		branch='/data/c1/lneef/ERP_DA/'

	branch_list = [branch]

	# list of the full names for each experiment
	names = {'ERPALL' : 'ERPALL_2009_N80/',
	'NODA' : 'NODA_2009_N80/',
	'RST' : 'RS_TEMPS_2009_N80/',
	'ERPRST' : 'RS_TEMPS_ERPS_2009_N80/',
	'SR' : 'ERPALL_2001_N64_SR/',
	'STINFL' : 'ERPALL_2001_N64_stinfl_adap_sd0p1/',
	'OBSINFL' : 'ERPALL_2001_N64_obsinfl_adap_sd0p1/',
	'TEST' : 'TEST_EAM_OPERATOR/'  
	}

	# list of the truth runs that were used to generate the obs for each (PMO) experiment
	truth_names = {'ERPALL' : 'PMO_ERPRS_2009/',
	'NODA' : 'PMO_ERPRS_2009/',
	'RST' : 'PMO_ERPRS_2009/',
	'ERPRST' : 'PMO_ERPRS_2009/',
	'TEST' : 'NONE'  
	}

	if branch is None:
		print('The settings for hostname '+hostname+' are not defined yet')
		return
	else:
		name = names[experiment]
		truth_name = truth_names[experiment]
		run_dir_list =  [branch+name for branch in branch_list]
		if truth_name is None:
			truth_dir_list = None
		else:
			truth_dir_list =  [branch+truth_name for branch in branch_list]
			run_dir = branch+names[experiment]
			truth_dir = branch+truth_names[experiment]

	return run_dir_list, truth_dir_list


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

	# convert a datetime date to gregorian day count the way it's counted in DART  
	# (i.e. number of days since 1601-01-01
        datestr = date.strftime("%Y-%m-%d")
	jd = dayconv.gd2jd(datestr)

	# reference date: 
	jd_ref = dayconv.gd2jd("1601-01-01")

	gday_out = jd-jd_ref

	return gday_out

def daterange(date_start=datetime.datetime(2009,1,1), periods=5, DT='1D'):


        # generate a range of dates (in python datetime format), given some start date, 
        # a time delta, and the numper of periods

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

	# given a 1-D ensemble time series and a verification (usually the truth), compute the
	# rank histogram over the desired block of time  

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

	# given a 1D ensemble of numbers (obs space, state space, whatever) return the 
	# kurtosis of the PDF represented by the ensemble

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

	# given a 1D ensemble of numbers (obs space, state space, whatever) return the 
	# skewness of the PDF represented by the ensemble

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
	# pre-defined experiment dictionaries that give various averaging regions 
	# over which to check the ensemble.

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

	if return_as_list:
		GG = [E1,E2,E3,E4,E5]
		return GG
	else:
		return E1,E2,E3,E4,E5

def point_check_dictionaries_2(return_as_list=True):
	# a second set of pre-defined experiment dictionaries that give various averaging regions 
	# over which to check the ensemble.

	E = basic_experiment_dict()


	E1 = E.copy()
	E1['latrange'] = [0,30]
	E1['levrange'] = [1000,500]
	E1['lonrange'] = [180,210]
	E1['title'] = 'North Subtropical Pacific'

	E2 = E.copy()
	E2['latrange'] = [-20,20]
	E2['levrange'] = [105,90]
	E2['lonrange'] = [0,360]
	E2['title'] = 'Tropical Tropopause'

	E3 = E.copy()
	E3['latrange'] = [50,90]
	E3['levrange'] = [370,320]
	E3['lonrange'] = [0,360]
	E3['title'] = 'NHET Tropopause'



	if return_as_list:
		GG = [E1]
		return GG
	else:
		return E1,E2,E3

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

