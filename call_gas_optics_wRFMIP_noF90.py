##########################################################################################
#! /usr/bin/env python
#
# This script reads in the RFMIP data and calls the RRTMGP gas-optics kernels.
#
##########################################################################################
import urllib.request
import numpy as np
from cffi import FFI
from load_kdist_noF90 import load_kdist_noF90
import xarray as xr

##########################################################################################
def construct_arg4_fficdef(info):
  # Constructs the CFFI-compatible arguments for ffi.cdef()
  # Wants a dictionary with type, dims which might be strings or integers
  # If there are no dims prefix the name with *
  res = info["ctype"] + " "
  if "dims" in info.keys():
      res += info['name']
      for d in info["dims"]:
          res += "[{}]".format(d)
  else:
    res += "*" +info['name']
  return(res)

# The function defintion is a little preamble and defintions for all the arguments
def construct_fficdef(kernel_name, args, return_type="void" ):
    fdef = '{} {}('.format(return_type, kernel_name)
    # All the args but the last have trailing comma, space
    for a in args[:-1]:
        fdef += "{}, ".format(construct_arg4_fficdef(a))
    # The last argument needs a closing paren+semicolon and no comma
    return(fdef + "{});".format(construct_arg4_fficdef(args[-1])) )
##########################################################################################
def get_col_dry(vmr_h2o, p_lev, latitude = None):
	helmert1 = 9.80665
	helmert2 = 0.02586

	# Dimensions
	ncol = vmr_h2o[:,0].size
	nlay = vmr_h2o[0,:].size

	# Adjust gravity by latitude?
	if (np.any(latitude)): g0 = helmert1 - helmert2 * np.cos(2.0*pi*latitude/180.)
	else:                  g0 = np.full((ncol), grav, dtype=np.double)

	col_dry = np.zeros((ncol,nlay),dtype=np.double)
	for ilay in range(0,nlay):
		# Layer thickness [Pa]
		dp = np.abs(p_lev[:,ilay]-p_lev[:,ilay+1])
		# Mass of air [grams]
		fact  = 1. / (1. + vmr_h2o[:,ilay])
		m_air = 1000.*(m_dry + m_h2o*vmr_h2o[:,ilay])*fact
		# [molec/cm^2]
		col_dry[:,ilay] = 10.*dp*avogad*fact/(m_air*100.*g0)
	return col_dry

##########################################################################################
# Physical constants (from rte-rrtmgp/rrtmgp/mo_gas_optics_rrtmgp.F90)
# Molecular weight of water [kg/mol]
m_h2o  = 0.018016
# Molecular weight of dry air [kg/mol]
m_dry  = 0.028964
# Gravity at Earth's surface [m/s2]
grav   = 9.80665
# Avogadro's number [molec/mol]
avogad = 6.02214076
# pi
pi     = np.arccos(-1.)

##########################################################################################
##########################################################################################
# Which gases to use?
gases       = ["h2o",        "co2",              "o3",   "n2o",             "ch4",       "o2"]
gases_rfmip = ["water_vapor","carbon_dioxide_GM","ozone","nitrous_oxide_GM","methane_GM","oxygen_GM"]
ngas_req    = len(gases)

# Location of rte-rrtmgp 
#rte_rrtmgp_dir = "/home/dswales/Projects/radiation-nn/rrtmgp-nn-training/rte-rrtmgp/"
rte_rrtmgp_dir = "/scratch2/BMC/ome/Dustin.Swales/radiation-nn/rte-rrtmgp/"
file_kdistLW   = rte_rrtmgp_dir + "rrtmgp/data/rrtmgp-data-lw-g256-2018-12-04.nc"
file_kdistSW   = rte_rrtmgp_dir + "rrtmgp/data/rrtmgp-data-sw-g224-2018-12-04.nc"

# Download RFMIP profiles
conds_file     = "/scratch2/BMC/ome/Dustin.Swales/radiation-nn/multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
conds_url      = "http://aims3.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6/RFMIP/UColorado/UColorado-RFMIP-1-2/" + \
                 "atmos/fx/multiple/none/v20190401/" + conds_file
#conds_file     = "multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
#urllib.request.urlretrieve(conds_url, conds_file)

# Open mo_gas_optics_kernels library
ffi = FFI()
lib = ffi.dlopen("libs/mo_gas_optics_kernels.so")

# Load k-distribution files
print_info = False
output_to_ctypes = True
kdistLW = load_kdist_noF90(ffi, file_kdistLW, gases, print_info, output_to_ctypes)
kdistSW = load_kdist_noF90(ffi, file_kdistSW, gases, print_info, output_to_ctypes)

# Dimensions (flat)
ntemp          = kdistSW['ntempref'][0]
npres          = kdistSW['npressref'][0]
nflav          = kdistSW['nflavors'][0]
ngpt           = kdistSW['ngpt'][0]
nband          = kdistSW['nband'][0]
neta           = kdistSW['nmixfrac'][0]
ncontlower     = kdistSW['ncontlower_red'][0]
ncontupper     = kdistSW['ncontupper_red'][0]
nminorabslower = kdistSW['nminorabslower_red'][0]
nminorabsupper = kdistSW['nminorabsupper_red'][0]

# Load RFMIP data
data_RFMIP = xr.open_dataset(conds_file,concat_characters=True,decode_cf=True)

# Dimensions (flat)
ncol = data_RFMIP.lon.size
nlay = data_RFMIP.pres_layer[0,:].size
nlev = data_RFMIP.pres_level[0,:].size

#
# This section prepares the RFMIP data for use by the rrtmgp-kernels.
#
# Assume only using single RFMIP experiment for the time being, will expand later.
irfmip_expt = 0

# Create array of volume-mixing-ratios (vmr), populate with RFMIP data
vmr = np.zeros((ncol, nlay, ngas_req), dtype=np.double)
vmr[:,:,0] = data_RFMIP.water_vapor.values[irfmip_expt,:,:]
vmr[:,:,1] = data_RFMIP.carbon_dioxide_GM.values[irfmip_expt]
vmr[:,:,2] = data_RFMIP.ozone.values[irfmip_expt,:,:]
vmr[:,:,3] = data_RFMIP.nitrous_oxide_GM.values[irfmip_expt]
vmr[:,:,4] = data_RFMIP.methane_GM.values[irfmip_expt]
vmr[:,:,5] = data_RFMIP.oxygen_GM.values[irfmip_expt]

# Compute dry air column amounts [molec/cm^2]
col_dry = get_col_dry(vmr[:,:,0], data_RFMIP.pres_level.values, latitude=data_RFMIP.lat.values)

# Compute column gas amounts [molec/cm^2]
col_gas = np.zeros((ncol, nlay, ngas_req+1), dtype=np.double)
col_gas[:,:,0] = col_dry
col_gas[:,:,1] = vmr[:,:,0] * col_dry
col_gas[:,:,2] = vmr[:,:,1] * col_dry
col_gas[:,:,3] = vmr[:,:,2] * col_dry
col_gas[:,:,4] = vmr[:,:,3] * col_dry
col_gas[:,:,5] = vmr[:,:,4] * col_dry
col_gas[:,:,6] = vmr[:,:,5] * col_dry

# Convert to c-types for F90 interfacing. *Note* The K-distribution data is already cast 
# into the appropriate types for the kernels.

# RFMIP inputs
c_ncol    = ffi.new("int *", ncol) 
c_nlay    = ffi.new("int *", nlay)
c_play    = ffi.new("double ["+str(ncol)+" ]["+str(nlay)+"]",    \
                    data_RFMIP.pres_layer.values[0:ncol,:].tolist())
c_tlay    = ffi.new("double ["+str(ncol)+" ]["+str(nlay)+"]",    \
                    data_RFMIP.temp_layer.values[irfmip_expt,0:ncol,:].tolist())
c_col_gas = ffi.new("double ["+str(ncol)+" ]["+str(nlay)+"]["+str(ngas_req+1)+"]", \
                    col_gas[0:ncol,:,:].tolist())
c_col_dry = ffi.new("double ["+str(ncol)+" ]["+str(nlay)+"]", \
                    col_dry.tolist())

# Outputs from kernels
c_jtemp       = ffi.new("int ["+str(ncol)+" ]["+str(nlay)+"]")
c_jpress      = ffi.new("int ["+str(ncol)+" ]["+str(nlay)+"]")
c_tropo       = ffi.new("int ["+str(ncol)+" ]["+str(nlay)+"]")
c_jeta        = ffi.new("int [2]["          +str(nflav)+"]["+str(ncol)+"]["+str(nlay)+"]")
c_col_mix     = ffi.new("double [2]["       +str(nflav)+"]["+str(ncol)+"]["+str(nlay)+"]")
c_fmajor      = ffi.new("double [2][2][2][" +str(nflav)+"]["+str(ncol)+"]["+str(nlay)+"]")
c_fminor      = ffi.new("double [2][2]["    +str(nflav)+"]["+str(ncol)+"]["+str(nlay)+"]")
c_tau_sw      = ffi.new("double ["+str(ngpt)+"]["+str(nlay)+"]["+str(ncol)+"]")
c_tau_sw_rayl = ffi.new("double ["+str(ngpt)+"]["+str(nlay)+"]["+str(ncol)+"]")
c_tau         = ffi.new("double ["+str(ncol)+"]["+str(nlay)+"]["+str(ngpt)+"]")
c_ssa         = ffi.new("double ["+str(ncol)+"]["+str(nlay)+"]["+str(ngpt)+"]")
c_g           = ffi.new("double ["+str(ncol)+"]["+str(nlay)+"]["+str(ngpt)+"]")

##########################################################################################
#
# Create dictionaries for kernel interfacing
#
##########################################################################################
#
args_interpolation = \
        [{"name":"ncol",                            "ctype":"int"},                                        \
         {"name":"nlay",                            "ctype":"int"},                                        \
         {"name":"ngas",                            "ctype":"int"},                                        \
         {"name":"nflav",                           "ctype":"int"},                                        \
         {"name":"neta",                            "ctype":"int"},                                        \
         {"name":"npres",                           "ctype":"int"},                                        \
         {"name":"ntemp",                           "ctype":"int"},                                        \
         {"name":"flavor",                          "ctype":"int",    "dims":[2, nflav]},                  \
         {"name":"press_ref_log",                   "ctype":"double", "dims":[npres]},                     \
         {"name":"temp_ref",                        "ctype":"double", "dims":[ntemp]},                     \
         {"name":"press_ref_log_delta",             "ctype":"double"},                                     \
         {"name":"temp_ref_min",                    "ctype":"double"},                                     \
         {"name":"temp_ref_delta",                  "ctype":"double"},                                     \
         {"name":"press_ref_trop_log",              "ctype":"double"},                                     \
         {"name":"vmr_ref",                         "ctype":"double", "dims":[2, ngas_req+1, ntemp]},      \
         {"name":"play",                            "ctype":"double", "dims":[ncol, nlay]},                \
         {"name":"tlay",                            "ctype":"double", "dims":[ncol, nlay]},                \
         {"name":"col_gas",                         "ctype":"double", "dims":[ncol, nlay, ngas_req+1]},    \
         {"name":"jtemp",                           "ctype":"int",    "dims":[ncol, nlay]},                \
         {"name":"fmajor",                          "ctype":"double", "dims":[2, 2, 2, nflav, ncol, nlay]},\
         {"name":"fminor",                          "ctype":"double", "dims":[2, 2, nflav, ncol, nlay]},   \
         {"name":"col_mix",                         "ctype":"double", "dims":[2, nflav, ncol, nlay]},      \
         {"name":"tropo",                           "ctype":"int",    "dims":[ncol, nlay]},                \
         {"name":"jeta",                            "ctype":"int",    "dims":[2, nflav, ncol, nlay]},      \
         {"name":"jpress",                          "ctype":"int",    "dims":[ncol, nlay]}]
args_compute_tau_absorption = \
        [{"name":"ncol",                            "ctype":"int"},                                        \
         {"name":"nlay",                            "ctype":"int"},                                        \
         {"name":"nbnd",                            "ctype":"int"},                                        \
         {"name":"ngpt",                            "ctype":"int"},                                        \
         {"name":"ngas",                            "ctype":"int"},                                        \
         {"name":"nflav",                           "ctype":"int"},                                        \
         {"name":"neta",                            "ctype":"int"},                                        \
         {"name":"npres",                           "ctype":"int"},                                        \
         {"name":"ntemp",                           "ctype":"int"},                                        \
         {"name":"nminorlower",                     "ctype":"int"},                                        \
         {"name":"nminorklower",                    "ctype":"int"},                                        \
         {"name":"nminorupper",                     "ctype":"int"},                                        \
         {"name":"nminorkupper",                    "ctype":"int"},                                        \
         {"name":"idx_h2o",                         "ctype":"int"},                                        \
         {"name":"gpoint_flavor",                   "ctype":"int",    "dims":[2, ngpt]},                   \
         {"name":"band_lims_gpt",                   "ctype":"int",    "dims":[2, nband]},                  \
         {"name":"kmajor",                          "ctype":"double", "dims":[ngpt, neta, npres+1, ntemp]},\
         {"name":"kminor_lower",                    "ctype":"double", "dims":[ncontlower,neta,ntemp]},     \
         {"name":"kminor_upper",                    "ctype":"double", "dims":[ncontupper,neta,ntemp]},     \
         {"name":"minor_limits_gpt_lower",          "ctype":"int",    "dims":[2, nminorabslower]},         \
         {"name":"minor_limits_gpt_upper",          "ctype":"int",    "dims":[2, nminorabsupper]},         \
         {"name":"minor_scales_with_density_lower", "ctype":"int",    "dims":[nminorabslower]},            \
         {"name":"minor_scales_with_density_upper", "ctype":"int",    "dims":[nminorabsupper]},            \
         {"name":"scale_by_complement_lower",       "ctype":"int",    "dims":[nminorabslower]},            \
         {"name":"scale_by_complement_upper",       "ctype":"int",    "dims":[nminorabsupper]},            \
         {"name":"idx_minor_lower",                 "ctype":"int",    "dims":[nminorabslower]},            \
         {"name":"idx_minor_upper",                 "ctype":"int",    "dims":[nminorabsupper]},            \
         {"name":"idx_minor_scaling_lower",         "ctype":"int",    "dims":[nminorabslower]},            \
         {"name":"idx_minor_scaling_upper",         "ctype":"int",    "dims":[nminorabsupper]},            \
         {"name":"kminor_start_lower",              "ctype":"int",    "dims":[nminorabslower]},            \
         {"name":"kminor_start_upper",              "ctype":"int",    "dims":[nminorabsupper]},            \
         {"name":"tropo",                           "ctype":"int",    "dims":[ncol, nlay]},                \
         {"name":"col_mix",                         "ctype":"double", "dims":[2, nflav, ncol, nlay]},      \
         {"name":"fmajor",                          "ctype":"double", "dims":[2, 2, 2, nflav, ncol, nlay]},\
         {"name":"fminor",                          "ctype":"double", "dims":[2, 2, nflav, ncol, nlay]},   \
         {"name":"play",                            "ctype":"double", "dims":[ncol, nlay]},                \
         {"name":"tlay",                            "ctype":"double", "dims":[ncol, nlay]},                \
         {"name":"col_gas",                         "ctype":"double", "dims":[ncol, nlay, ngas_req+1]},    \
         {"name":"jeta",                            "ctype":"int",    "dims":[2, nflav, ncol, nlay]},      \
         {"name":"jtemp",                           "ctype":"int",    "dims":[ncol, nlay]},                \
         {"name":"jpress",                          "ctype":"int",    "dims":[ncol, nlay]},                \
         {"name":"tau",                             "ctype":"double", "dims":[ngpt,nlay,ncol]}]
args_compute_tau_rayleigh = \
        [{"name":"ncol",                            "ctype":"int"},                                        \
         {"name":"nlay",                            "ctype":"int"},                                        \
         {"name":"nbnd",                            "ctype":"int"},                                        \
         {"name":"ngpt",                            "ctype":"int"},                                        \
         {"name":"ngas",                            "ctype":"int"},                                        \
         {"name":"nflav",                           "ctype":"int"},                                        \
         {"name":"neta",                            "ctype":"int"},                                        \
         {"name":"npres",                           "ctype":"int"},                                        \
         {"name":"ntemp",                           "ctype":"int"},                                        \
         {"name":"gpoint_flavor",                   "ctype":"int",    "dims":[2, ngpt]},                   \
         {"name":"band_lims_gpt",                   "ctype":"int",    "dims":[2, nband]},                  \
         {"name":"krayl",                           "ctype":"double", "dims":[ngpt,neta,ntemp,2]},         \
         {"name":"idx_h2o",                         "ctype":"int"},                                        \
         {"name":"col_dry",                         "ctype":"double", "dims":[ncol, nlay]},                \
         {"name":"col_gas",                         "ctype":"double", "dims":[ncol, nlay, ngas_req+1]},    \
         {"name":"fminor",                          "ctype":"double", "dims":[2, 2, nflav, ncol, nlay]},   \
         {"name":"jeta",                            "ctype":"int",    "dims":[2, nflav, ncol, nlay]},      \
         {"name":"tropo",                           "ctype":"int",    "dims":[ncol, nlay]},                \
         {"name":"jtemp",                           "ctype":"int",    "dims":[ncol, nlay]},                \
         {"name":"tau_rayleigh",                    "ctype":"double", "dims":[ngpt, nlay, ncol]}]
args_combine_and_reorder_2str = \
        [{"name":"ncol",                            "ctype":"int"},                                        \
         {"name":"nlay",                            "ctype":"int"},                                        \
         {"name":"ngpt",                            "ctype":"int"},                                        \
         {"name":"tau_abs",                         "ctype":"double", "dims":[ngpt,nlay,ncol]},            \
         {"name":"tau_rayleigh",                    "ctype":"double", "dims":[ngpt,nlay,ncol]},            \
         {"name":"tau",                             "ctype":"double", "dims":[ncol,nlay,ngpt]},            \
         {"name":"ssa",                             "ctype":"double", "dims":[ncol,nlay,ngpt]},            \
         {"name":"g",                               "ctype":"double", "dims":[ncol,nlay,ngpt]}]


##########################################################################################
#
# Call gas-optics kernels...
#
########################################################################################## 

#
# Interpolation
#
ffi.cdef(construct_fficdef("interpolation",args_interpolation), override=True)
lib.interpolation(                                                            \
        c_ncol,                                                               \
        c_nlay,                                                               \
        kdistSW['ngas_req'],                                                  \
	kdistSW['nflavors'],                                                  \
        kdistSW['nmixfrac'],                                                  \
        kdistSW['npressref'],                                                 \
	kdistSW['ntempref'],                                                  \
        kdistSW['flavors'],                                                   \
        kdistSW['press_ref_log'],                                             \
	kdistSW['temp_ref'],                                                  \
        kdistSW['press_ref_log_delta'],                                       \
        kdistSW['temp_ref_min'],                                              \
	kdistSW['temp_ref_delta'],                                            \
        kdistSW['press_ref_trop_log'],                                        \
	kdistSW['vmr_ref_red'],                                               \
        c_play,                                                               \
        c_tlay,                                                               \
        c_col_gas,                                                            \
        c_jtemp,                                                              \
        c_fmajor,                                                             \
	c_fminor,                                                             \
        c_col_mix,                                                            \
        c_tropo,                                                              \
        c_jeta,                                                               \
        c_jpress)

#
# Compute_tau_absorption
#
ffi.cdef(construct_fficdef("compute_tau_absorption",args_compute_tau_absorption), override=True)
lib.compute_tau_absorption(                                                   \
	c_ncol,                                                               \
	c_nlay,                                                               \
	kdistSW['nband'],                                                     \
	kdistSW['ngpt'],                                                      \
	kdistSW['ngas_req'],                                                  \
	kdistSW['nflavors'],                                                  \
	kdistSW['nmixfrac'],                                                  \
	kdistSW['npressref'],                                                 \
	kdistSW['ntempref'],                                                  \
	kdistSW['nminorabslower_red'],                                        \
	kdistSW['ncontlower_red'],                                            \
	kdistSW['nminorabsupper_red'],                                        \
	kdistSW['ncontupper_red'],                                            \
	kdistSW['idx_h2o'],                                                   \
	kdistSW['gpoint_flavor'],                                             \
	kdistSW['bnd_limits_gpt'],                                            \
	kdistSW['kmajor'],                                                    \
	kdistSW['kminor_lower_red'],                                          \
	kdistSW['kminor_upper_red'],                                          \
	kdistSW['minor_limits_gpt_lower_red'],                                \
	kdistSW['minor_limits_gpt_upper_red'],                                \
	kdistSW['minor_scales_with_density_lower_red'],                       \
	kdistSW['minor_scales_with_density_upper_red'],                       \
	kdistSW['scale_by_complement_lower_red'],                             \
	kdistSW['scale_by_complement_upper_red'],                             \
	kdistSW['idx_minor_lower'],                                           \
	kdistSW['idx_minor_upper'],                                           \
	kdistSW['idx_minor_scaling_lower'],                                   \
	kdistSW['idx_minor_scaling_upper'],                                   \
	kdistSW['kminor_start_lower_red'],                                    \
	kdistSW['kminor_start_upper_red'],                                    \
	c_tropo,                                                              \
        c_col_mix,                                                            \
        c_fmajor,                                                             \
        c_fminor,                                                             \
        c_play,                                                               \
        c_tlay,                                                               \
        c_col_gas,                                                            \
        c_jeta,                                                               \
        c_jtemp,                                                              \
        c_jpress,                                                             \
        c_tau_sw)

#
# Compute_tau_rayleigh
#
ffi.cdef(construct_fficdef("compute_tau_rayleigh",args_compute_tau_rayleigh), override=True)
lib.compute_tau_rayleigh(                                                     \
	c_ncol,                                                               \
	c_nlay,                                                               \
	kdistSW['nband'],                                                     \
	kdistSW['ngpt'],                                                      \
	kdistSW['ngas_req'],                                                  \
	kdistSW['nflavors'],                                                  \
	kdistSW['nmixfrac'],                                                  \
	kdistSW['npressref'],                                                 \
	kdistSW['ntempref'],                                                  \
	kdistSW['gpoint_flavor'],                                             \
	kdistSW['bnd_limits_gpt'],                                            \
	kdistSW['krayl'],                                                     \
	kdistSW['idx_h2o'],                                                   \
	c_col_dry,                                                            \
	c_col_gas,                                                            \
	c_fminor,                                                             \
	c_jeta,                                                               \
	c_tropo,                                                              \
	c_jtemp,                                                              \
	c_tau_sw_rayl)

#
# Combine_and_reorder
#
ffi.cdef(construct_fficdef("combine_and_reorder_2str",args_combine_and_reorder_2str), override=True)
lib.combine_and_reorder_2str(                                                 \
        c_ncol,                                                               \
        c_nlay,                                                               \
        kdistSW['ngpt'],                                                      \
        c_tau_sw,                                                             \
        c_tau_sw_rayl,                                                        \
        c_tau, c_ssa, c_g)
