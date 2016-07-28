import DART as dart
import os.path
import datetime as datetime

def get_experiment_date_ranges(exp_name):

	# stored date ranges for various DART experiments  
	DR = None

	# CAM experiments for ERP assimilation study  
	if exp_name == 'NODA':
		DR = dart.daterange(date_start=datetime.datetime(2009,1,1,0,0,0), periods=31, DT='1D')
	if exp_name == 'ERPALL':
		DR = dart.daterange(date_start=datetime.datetime(2009,1,1,0,0,0), periods=31, DT='1D')
	if exp_name == 'RST':
		DR = dart.daterange(date_start=datetime.datetime(2009,1,1,0,0,0), periods=17, DT='1D')
	if exp_name == 'ERPRST':
		DR = dart.daterange(date_start=datetime.datetime(2009,1,1,0,0,0), periods=17, DT='1D')

	# DART-WACCM runs performed at GEOMAR  
	if exp_name == 'PMO32':
		DR = dart.daterange(date_start=datetime.datetime(2009,10,1,6,0,0), periods=31, DT='6H')
	if exp_name == 'W0910_NODA':
		DR = dart.daterange(date_start=datetime.datetime(2009,10,1,12,0,0), periods=596, DT='6H')
	if exp_name == 'W0910_GLOBAL':
		DR = dart.daterange(date_start=datetime.datetime(2009,10,1,12,0,0), periods=596, DT='6H')
	if exp_name == 'W0910_TROPICS':
		DR = dart.daterange(date_start=datetime.datetime(2009,10,1,12,0,0), periods=596, DT='6H')
	if exp_name == 'W0910_NODART':
		DR = dart.daterange(date_start=datetime.datetime(2009,10,1,12,0,0), periods=10, DT='6H')
	if exp_name == 'W0910_NOSTOP':
		DR = dart.daterange(date_start=datetime.datetime(2009,10,1,12,0,0), periods=64, DT='6H')
	
	# WACCM PMO runs performed by Nick Pedatella at NCAR
	if exp_name == 'NCAR_PMO_CONTROL':
		DR = dart.daterange(date_start=datetime.datetime(2008,11,6,6,0,0), periods=72, DT='6H')
	if exp_name == 'NCAR_PMO_LAS':
		DR = dart.daterange(date_start=datetime.datetime(2008,11,6,6,0,0), periods=72, DT='6H')
	if exp_name == 'NCAR_PMO_LA':
		DR = dart.daterange(date_start=datetime.datetime(2008,11,6,6,0,0), periods=72, DT='6H')

	# WACCM real-obs runs performed by Nick Pedatella at NCAR
	if exp_name == 'NCAR_FULL':
		DR = dart.daterange(date_start=datetime.datetime(2009,1,1,6,0,0), periods=204, DT='6H')
	if exp_name == 'NCAR_LAONLY':
		DR = dart.daterange(date_start=datetime.datetime(2009,1,1,6,0,0), periods=204, DT='6H')

	if DR is None:
		print('find_paths Cannot find experiment '+exp_name+' returning...')

	return DR

def find_paths(E,date,file_type='diag',hostname='taurus',debug=False):

	import DART as dart
	"""
	This subroutine takes a DART experiment dictionary and returns the file path for the 
	needed diagnostic. 

	The optional input, `file_type`, can have one of these values:  
	+ 'covariance' -- then we load pre-computed data of covariances between state variables and a given obs  
	+ 'obs_epoch' -- load obs_epoch_XXXX.nc files  
	+ 'diag' -- load standard  DART Posterior_Diag or Prior_Diag files 
	+ 'truth' -- load true state files from a perfect-model simulation

	"""

	path_found = False
	if E['run_category'] == 'NCAR':
		data_dir_list,truth_dir_list = exp_paths_NCAR(hostname,E['exp_name'])
		path_found = True
	if E['exp_name'] == 'ERA':
		data_dir_list,truth_dir_list = exp_paths_era(date,hostname,diagnostic=E['diagn'])
		path_found = True
	if not path_found:
		data_dir_list,truth_dir_list = exp_paths(hostname,E['exp_name'])


	#------------COVARIANCE FILES  
	if file_type == 'covariance':
		fname = E['exp_name']+'_'+'covariance_'+E['obs_name']+'_'+E['variable']+'_'+date.strftime('%Y-%m-%d')+'.nc'


	#------------OBS EPOCH FILES
	if file_type == 'obs_epoch':
		DR = get_experiment_date_ranges(E['exp_name'])
		delta_time = date-DR[0]
		obs_epoch_no = delta_time.days+1
		if obs_epoch_no < 10:
			obs_epoch_name = 'obs_epoch_00'+str(obs_epoch_no)+'.nc'
		if (obs_epoch_no >= 10) and (obs_epoch_no < 100): 
			obs_epoch_name = 'obs_epoch_0'+str(obs_epoch_no)+'.nc'
		if (obs_epoch_no >= 100): 
			obs_epoch_name = 'obs_epoch_'+str(obs_epoch_no)+'.nc'
		if E['run_category'] is None:
			fname = '/dart/hist/'+obs_epoch_name
		if E['run_category'] == 'ERPDA':
			fname = '/../obs_epoch/'+obs_epoch_name

	#------------regular DART output files or true state files 
	if (file_type == 'diag') or (file_type == 'truth'):
		if E['diagn']=='Truth':
			file_type='truth'
		datestr = date.strftime("%Y-%m-%d")
		seconds = date.hour*60*60
		if seconds == 0:
			timestr = '00000'
		else:
			timestr = str(seconds)
		if E['run_category'] is None:
			diagstring = 'Diag'
			# additional diagnostics files have the 'Diag' string replaced with something else. 
			TIL_variables = ['theta','ptrop','Nsq','P','brunt']
			if E['variable'] in TIL_variables:
				diagstring='TIL'
			fname = '/dart/hist/cam_'+E['diagn']+'_'+diagstring+'.'+datestr+'-'+timestr+'.nc'
			fname_truth = '/dart/hist/cam_'+'True_State'+'.'+datestr+'-'+timestr+'.nc'
		if E['run_category'] == 'ERPDA':
			gday = dart.date_to_gday(date)
			# for all my (Lisa's) old experiments, obs sequence 1 is 1 Jan 2009
			gday1 = dart.date_to_gday(datetime.datetime(2009,1,1,0,0,0))
			obs_seq_no = int(gday-gday1+1)
			if (obs_seq_no < 10):
				mid = 'obs_000'+str(obs_seq_no)
			else:
				mid = 'obs_00'+str(obs_seq_no)
			fname_truth = mid+'/'+'True_State.nc'
			fname = mid+'/'+E['diagn']+'_Diag.nc'
		if E['run_category']=='NCAR':
			if E['exp_name'] == 'NCAR_LAONLY':
				suffix = '_LAONLY'
			else:
				suffix = ''
			fname_truth = '/'+'True_State'+'_'+datestr+'-'+timestr+'.nc'+suffix
			fname = '/'+E['diagn']+'_Diag.'+datestr+'-'+timestr+'.nc'+suffix
		if file_type == 'truth':
			fname = fname_truth
			data_dir_list = truth_dir_list

	# if data_dir_list was not found, throw an error
	if data_dir_list is None:
		print('experiment_settings.py cannot find settings for the following experiment dict:')
		print(E)
		return None


	#-----search for the right files 
	correct_filepath_found = False
	for data_dir in data_dir_list:
		filename = data_dir+fname
		if debug:
			print('Looking for file  '+filename)
		if os.path.exists(filename):
			correct_filepath_found = True
			break

	# return the file filename with path
	return filename

def get_ensemble_size_per_run(exp_name):

	"""
	given some existing DART experiment, look up which ensemble size was used there
	"""
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
	'W0910_NODA_OLDensemble' : 40,
	'W0910_NODART_OLDensemble' : 40,
	'W0910_NOSTOP_OLDensemble' : 40,
	'W0910_TROPICS_OLDensemble' : 40,
	'W0910_GLOBAL_OLDensemble' : 40,
	'W0910_GLOBAL' : 40,
	'W0910_NODA' : 40
	}
	return(N[exp_name])

def get_available_date_range(exp_name):

	"""
	given some existing DART experiment, return the daterange of all currently available data 
	"""

	N = {'W0910_GLOBAL' : dart.daterange(date_start=datetime.datetime(2009,10,1,0,0,0), periods=380, DT='6H'),
		'W0910_NODA' :dart.daterange(date_start=datetime.datetime(2009,10,1,0,0,0), periods=640, DT='6H'),
	}
	return N[exp_name]
	
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


def exp_paths_era(datetime_in,hostname='taurus',resolution=0.75,diagnostic=None,variable='U',level_type='pressure_levels'):

	"""
	Paths to ERA-Interm and ERA-40 data  
	"""

	truth_dir_list = None
	run_dir_list = None

	# find the year, month, and date requested 
	y = str(datetime_in.year)
	month = datetime_in.month
	day = datetime_in.day
	if month < 10:
		m = '0'+str(month)
	else:
		m = str(month)
	if day < 10:
		d = '0'+str(day)
	else:
		d = str(day)

	if (hostname=='taurus'):
		stub = '/data/c1/lneef/ERA/'
		mid = str(resolution)+'deg/'
		# the way the filenames start depends on the resolution
		if resolution == 2.5:
			model_level_parameters_list = ['hyam','hybm']
			if variable in model_level_parameters_list:
				variable_str='T'
			else:
				variable_str=variable
			fstub='ERA_'+variable_str+'_'+diagnostic.lower()+'_'+y+'-'+m+'-'+d+'.nc'
		if resolution == 0.75:
			# the "pure" ERA-Interim files are separated by variable
			varname=variable
			if (variable=='GPH') or (variable=='geopotential') or (variable=='Z') or  (variable=='Z3'):
				varname='z'
			if (variable=='U') or (variable=='US'): 
				varname='u'
			if (variable=='T'): 
				varname='t'
			fstub = 'ERA_'+varname+'_dm_1.5deg_'+y+'-'+m+'-'+d+'.nc'
			# different files loaded for increments - these are actually 2.5 degree:
			if diagnostic.lower() == 'increment':
				fstub = '../2.5deg/ERA_TUV_increments_'+y+'-'+m+'-'+d+'.nc'

		if 'fstub' not in locals():
			print('Cannot find path to requested ERA data:')
			print('resolution: '+str(resolution))
			print('variable: '+variable)
			print('diagnostic: '+diagnostic)
			print('level type: '+level_type)

		# finally here is the full path
		ff = stub+mid+level_type+'/'+fstub
	
	return ff,truth_dir_list

def exp_paths_TEM(E,datetime_in,hostname='taurus'):

	"""
	this subroutine returns the path to the TEM diagnostics 
	for a given DART-WACCM experiment 
	"""

	# list of the full names for each experiment  
	if 'ERA' not in E['exp_name']:
		long_name = get_long_names(E['exp_name'])

	datestr = datetime_in.strftime("%Y-%m-%d")
	tem_variables_list = ['VSTAR','WSTAR','FPHI','FZ','DELF']
	dynamical_heating_rates_list = ['VTy','WS']
	
	hostname_not_found = True

	if hostname == 'taurus':
		hostname_not_found = False

		# Files with TEM diagnostics start with TEM_, while 
		# dynamical heating rate files start with WS_VTy_
		if E['variable'].upper() in tem_variables_list:
			prefix = 'TEM_'
		if E['variable'] in dynamical_heating_rates_list :
			prefix = 'WS_VTy_'

		if 'ERA' in E['exp_name']:
			branch = '/data/c1/lneef/'
			path_out = branch+'ERA/0.75deg/'+'/TEM/'+prefix+'ERA-Interim_dm_'+datestr+'.nc'
		else:
			branch = '/data/c1/lneef/DART-WACCM/'
			# this experiment dictionary relates the short names that I gave my runs 
			# to those that Wuke gave them  

			short_names = {'W0910_NODA':'DW-NODA-02',
			'W0910_GLOBAL':'DW-Global-02',
			'W0910_GLOBAL_OLDensemble':'DW-Global',
			'W0910_NODA_OLDensemble':'DW-NODA',
			'W0910_TROPICS':'DW-Trop'}

			path_out = branch + long_name+'/atm/TEM/'+prefix+short_names[E['exp_name']]+'.cam.h1.'+datestr+'.nc'

	# throw error if hostname is wrong  
	if hostname_not_found:
		print('Hostname '+hostname+' settings not coded yet')
		return None

	return path_out


def exp_paths(hostname='taurus',experiment='PMO32'):

	# store the location of the output for individual DART experiments
	if (hostname=='blizzard'):
		branch='/work/bb0519/CESM/cesm1_2_0/archive/b350071/'
		branch_list = [branch]

	if (hostname=='taurus'):
		# there are several places on Taurus where our experiments live...
		branch1='/data/b4/swahl/cesm1_2_0/archive/'
		branch2='/data/c1/lneef/DART-WACCM/'
		#branch3='/data/c1/lneef/ERP_DA/'
		branch_list = [branch1,branch2]

	# retrieve the full name of the desired experiment
	name = get_long_names(experiment)

	# retrieve name of the corresponding true state run, if available 
	truth_name = get_truth_names(experiment)

	run_dir_list =  [branch+name for branch in branch_list]
	if truth_name is None:
		truth_dir_list = None
	else:
		truth_dir_list =  [branch+truth_name for branch in branch_list]

	return run_dir_list, truth_dir_list


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

def get_long_names(exp_name):

	"""
	returns the true experiment name for a given abbreviation 
	"""

	# retrieve list of the full names for each experiment  
	# note that all the ones with 'OLDensemble' in the short name were run with a flawed initial-time 
	# ensemble (see this note: https://www.evernote.com/shard/s215/sh/9bcc2659-068e-43ee-a604-1a0004b9c076/92188427d3e672203b47b7c6de83356a)
	long_names = {'PMO18' : 'waccm-dart-assimilate-pmo-18',
	'PMO27' : 'waccm-dart-assimilate-pmo-27',	
	'PMO28' : 'waccm-dart-assimilate-pmo-28',	
	'PMO32' : 'waccm-dart-assimilate-pmo-32',	
	'W0910_NODA_OLDensemble' : 'nechpc-waccm-dart-gpsro-ncep-no-assim-01',		
	'W0910_NODART_OLDensemble' : 'nechpc-waccm-dart-gpsro-ncep-no-dart',			
	'W0910_NOSTOP_OLDensemble' : 'nechpc-waccm-dart-gpsro-ncep-nostop',			
	'W0910_TROPICS_OLDensemble' : 'nechpc-waccm-dart-gpsro-ncep-30S-30N-01',			
	'W0910_GLOBAL_OLDensemble' : 'nechpc-waccm-dart-gpsro-ncep-global-01',			
	'W0910_GLOBAL' : 'nechpc-waccm-dart-gpsro-ncep-global-02',			
	'W0910_NODA' : 'nechpc-waccm-dart-gpsro-ncep-no-assim-02',			
	'ERPALL' : 'ERPALL_2009_N80/',
	'NODA' : 'NODA_2009_N80/',
	'RST' : 'RS_TEMPS_2009_N80/',
	'ERPRST' : 'RS_TEMPS_ERPS_2009_N80/',
	'SR' : 'ERPALL_2001_N64_SR/',
	'STINFL' : 'ERPALL_2001_N64_stinfl_adap_sd0p1/',
	'OBSINFL' : 'ERPALL_2001_N64_obsinfl_adap_sd0p1/',
	'TEST' : 'TEST_EAM_OPERATOR/'  
	}

	long_name_out = long_names[exp_name]
	return(long_name_out)

def get_truth_names(exp_name):
	truth_names = {'PMO18' : 'f55wcn-pmo-cosmic-erp-01',
	'PMO27' : 'f55wcn-pmo-cosmic-erp-01',
	'PMO28' : 'f55wcn-pmo-cosmic-erp-01',
	'PMO32' : 'f55wcn-pmo-cosmic-erp-01',
	'W0910_NODA_OLDensemble'  : None,
	'W0910_NODART_OLDensemble'  : None,
	'W0910_NOSTOP_OLDensemble'  : None,
	'W0910_TROPICS_OLDensemble'  : None,
	'W0910_GLOBAL_OLDensemble'  : None,
	'W0910_NODA'  : None,
	'W0910_NODART'  : None,
	'W0910_NOSTOP'  : None,
	'W0910_TROPICS'  : None,
	'W0910_GLOBAL'  : None,
	'ERPALL' : 'PMO_ERPRS_2009/',
	'NODA' : 'PMO_ERPRS_2009/',
	'RST' : 'PMO_ERPRS_2009/',
	'ERPRST' : 'PMO_ERPRS_2009/',
	'TEST' : 'NONE'  
	}

	truth_name_out = truth_names[exp_name]
	return(truth_name_out)


def climatology_runs(clim_name,hostname='taurus',debug=False):

	"""
	This subroutine holds a dictionary of multi-year experiments that can be used as climatologies 
	for other DART runs. 
	"""

	long_names={'F_W4_L66':'/data/c1/lneef/CESM/F_W4_L66/atm/climatology/F_W4_L66.cam.h1.1951-2010.daily_climatology.nc'}

	return long_names[clim_name]

def std_runs(clim_name,hostname='taurus',debug=False):

	"""
	This subroutine holds a dictionary of multi-year experiments that can be used as standard deviation data 
	for other DART runs. 
	"""

	long_names={'F_W4_L66':'/data/c1/lneef/CESM/F_W4_L66/atm/climatology/F_W4_L66.cam.h1.1951-2010.daily_std.nc'}

	return long_names[clim_name]

def obs_data_paths(obs_type,hostname):

	"""
	Return paths to where different observation types are stored.

	The type of observatin requested is given by the input string `obs_type` 
	-- so far, have only coded in a path to high-res radiosondes (HRRS)
	"""

	data_dir_dict={'HRRS':'/data/c1/lneef/HRRS/'}
	
	return data_dir_dict[obs_type]
