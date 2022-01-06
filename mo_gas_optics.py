##########################################################################################
#! /usr/bin/env python
#
# This script calls the RRTMGP gas-optics kernels.
#
##########################################################################################
import numpy as np
from cffi import FFI

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

##########################################################################################
# The function defintion is a little preamble and defintions for all the arguments
def construct_fficdef(kernel_name, args, return_type="void" ):
    fdef = '{} {}('.format(return_type, kernel_name)
    # All the args but the last have trailing comma, space
    for a in args[:-1]:
        fdef += "{}, ".format(construct_arg4_fficdef(a))
    # The last argument needs a closing paren+semicolon and no comma
    return(fdef + "{});".format(construct_arg4_fficdef(args[-1])) )

##########################################################################################
def construct_ffinew(kernel_name, args, return_type="void" ):
	fdef = '{} {}('.format(return_type, kernel_name)
	for a in args[:-1]:
		fdef += "{}, ".format(construct_arg4_fficdef(a))

##########################################################################################
def gas_optics_sw(kdistSW, p_lay, t_lay, col_gas, col_dry, do_twostream):
	# Open RRTMGP library
	ffi = FFI()
	lib = ffi.dlopen("libs/mo_gas_optics_kernels.so")

	# Dimensions
	# Forcing data
	ngas           = len(col_gas[:,0,0])-1
	ncol           = len(t_lay[0,:])
	nlay           = len(t_lay[:,0])
	# K-distributuion
	ntemp          = kdistSW['ntempref'][0]
	npres          = kdistSW['npressref'][0]
	nflav          = kdistSW['nflavors'][0]
	ngpt           = kdistSW['ngpt'][0]
	nband          = kdistSW['nband'][0]
	neta           = kdistSW['nmixfrac'][0]
	ncontlower     = kdistSW['ncontlower'][0]
	ncontupper     = kdistSW['ncontupper'][0]
	nminorabslower = kdistSW['nminorabslower'][0]
	nminorabsupper = kdistSW['nminorabsupper'][0]

	args_interpolation = \
        	[{"name":"ncol",                            "ctype":"int"},                                        \
         	 {"name":"nlay",                            "ctype":"int"},                                        \
         	 {"name":"ngas",                            "ctype":"int"},                                        \
         	 {"name":"nflav",                           "ctype":"int"},                                        \
        	 {"name":"neta",                            "ctype":"int"},                                        \
         	 {"name":"npres",                           "ctype":"int"},                                        \
         	 {"name":"ntemp",                           "ctype":"int"},                                        \
         	 {"name":"flavor",                          "ctype":"int",    "dims":[nflav, 2]},                  \
         	 {"name":"press_ref_log",                   "ctype":"double", "dims":[npres]},                     \
         	 {"name":"temp_ref",                        "ctype":"double", "dims":[ntemp]},                     \
         	 {"name":"press_ref_log_delta",             "ctype":"double"},                                     \
         	 {"name":"temp_ref_min",                    "ctype":"double"},                                     \
         	 {"name":"temp_ref_delta",                  "ctype":"double"},                                     \
         	 {"name":"press_ref_trop_log",              "ctype":"double"},                                     \
         	 {"name":"vmr_ref",                         "ctype":"double", "dims":[ntemp, ngas+1, 2]},          \
         	 {"name":"play",                            "ctype":"double", "dims":[nlay, ncol]},                \
         	 {"name":"tlay",                            "ctype":"double", "dims":[nlay, ncol]},                \
         	 {"name":"col_gas",                         "ctype":"double", "dims":[ngas+1, nlay, ncol]},        \
         	 {"name":"jtemp",                           "ctype":"int",    "dims":[nlay, ncol]},                \
         	 {"name":"fmajor",                          "ctype":"double", "dims":[nflav, nlay, ncol, 2, 2, 2]},\
         	 {"name":"fminor",                          "ctype":"double", "dims":[nflav, nlay, ncol, 2, 2]},   \
         	 {"name":"col_mix",                         "ctype":"double", "dims":[nflav, nlay, ncol, 2]},      \
         	 {"name":"tropo",                           "ctype":"int",    "dims":[nlay, ncol]},                \
         	 {"name":"jeta",                            "ctype":"int",    "dims":[nflav, nlay, ncol, 2]},      \
         	 {"name":"jpress",                          "ctype":"int",    "dims":[nlay, ncol]}]
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
         	 {"name":"gpoint_flavor",                   "ctype":"int",    "dims":[ngpt, 2]},                   \
         	 {"name":"band_lims_gpt",                   "ctype":"int",    "dims":[nband, 2]},                  \
         	 {"name":"kmajor",                          "ctype":"double", "dims":[ngpt, npres+1, neta, ntemp]},\
         	 {"name":"kminor_lower",                    "ctype":"double", "dims":[ncontlower,neta,ntemp]},     \
         	 {"name":"kminor_upper",                    "ctype":"double", "dims":[ncontupper,neta,ntemp]},     \
         	 {"name":"minor_limits_gpt_lower",          "ctype":"int",    "dims":[nminorabslower, 2]},         \
         	 {"name":"minor_limits_gpt_upper",          "ctype":"int",    "dims":[nminorabsupper, 2]},         \
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
         	 {"name":"tropo",                           "ctype":"int",    "dims":[nlay, ncol]},                \
         	 {"name":"col_mix",                         "ctype":"double", "dims":[nflav, nlay, ncol, 2]},      \
         	 {"name":"fmajor",                          "ctype":"double", "dims":[nflav, nlay, ncol, 2, 2, 2]},\
         	 {"name":"fminor",                          "ctype":"double", "dims":[nflav, nlay, ncol, 2, 2]},   \
         	 {"name":"play",                            "ctype":"double", "dims":[nlay, ncol]},                \
         	 {"name":"tlay",                            "ctype":"double", "dims":[nlay, ncol]},                \
         	 {"name":"col_gas",                         "ctype":"double", "dims":[ngas+1, nlay, ncol]},        \
         	 {"name":"jeta",                            "ctype":"int",    "dims":[nflav, nlay, ncol, 2]},      \
         	 {"name":"jtemp",                           "ctype":"int",    "dims":[nlay, ncol]},                \
         	 {"name":"jpress",                          "ctype":"int",    "dims":[nlay, ncol]},                \
         	 {"name":"tau",                             "ctype":"double", "dims":[ncol,nlay,ngpt]}]
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
        	 {"name":"gpoint_flavor",                   "ctype":"int",    "dims":[ngpt, 2]},                   \
        	 {"name":"band_lims_gpt",                   "ctype":"int",    "dims":[nband, 2]},                  \
        	 {"name":"krayl",                           "ctype":"double", "dims":[2,ngpt,neta,ntemp]},         \
        	 {"name":"idx_h2o",                         "ctype":"int"},                                        \
        	 {"name":"col_dry",                         "ctype":"double", "dims":[nlay, ncol]},                \
        	 {"name":"col_gas",                         "ctype":"double", "dims":[ngas+1, nlay, ncol]},        \
        	 {"name":"fminor",                          "ctype":"double", "dims":[nflav, nlay, ncol, 2, 2]},   \
        	 {"name":"jeta",                            "ctype":"int",    "dims":[nflav, nlay, ncol, 2]},      \
        	 {"name":"tropo",                           "ctype":"int",    "dims":[nlay, ncol]},                \
        	 {"name":"jtemp",                           "ctype":"int",    "dims":[nlay, ncol]},                \
        	 {"name":"tau_rayleigh",                    "ctype":"double", "dims":[ncol, nlay, ngpt]}]

	##########################################################################################
	#
	# Call gas-optics kernels...
	#
	########################################################################################## 

	# Dimensions
	c_ncol    = ffi.new("int *", ncol)
	c_nlay    = ffi.new("int *", nlay)
	c_ngas    = ffi.new("int *", ngas)
	# Arrays
	c_play    = ffi.new("double ["+str(nlay)+" ]["+str(ncol)+"]", p_lay.tolist())
	c_tlay    = ffi.new("double ["+str(nlay)+" ]["+str(ncol)+"]", t_lay.tolist())
	c_col_dry = ffi.new("double ["+str(nlay)+" ]["+str(ncol)+"]", col_dry.tolist())
	c_col_gas = ffi.new("double ["+str(ngas+1)+" ]["+str(nlay)+"]["+str(ncol)+"]", col_gas.tolist())        

	#
	# Interpolation
	#
	c_jtemp       = ffi.new("int    [" + str(nlay)  + "][" + str(ncol) + "]")
	c_jpress      = ffi.new("int    [" + str(nlay)  + "][" + str(ncol) + "]")
	c_tropo       = ffi.new("int    [" + str(nlay)  + "][" + str(ncol) + "]")
	c_jeta        = ffi.new("int    [" + str(nflav) + "][" + str(nlay) + "][" + str(ncol) + "][2]")
	c_col_mix     = ffi.new("double [" + str(nflav) + "][" + str(nlay) + "][" + str(ncol) + "][2]")
	c_fmajor      = ffi.new("double [" + str(nflav) + "][" + str(nlay) + "][" + str(ncol) + "][2][2][2]")
	c_fminor      = ffi.new("double [" + str(nflav) + "][" + str(nlay) + "][" + str(ncol) + "][2][2]")
	ffi.cdef(construct_fficdef("interpolation",args_interpolation), override=True)
	lib.interpolation(                                                            \
	        c_ncol,                                                               \
        	c_nlay,                                                               \
        	kdistSW['ngas'],                                                      \
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
		kdistSW['vmr_ref'],                                                   \
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
	c_tau_sw      = ffi.new("double ["+str(ncol)+"]["+str(nlay)+"]["+str(ngpt)+"]")
	ffi.cdef(construct_fficdef("compute_tau_absorption",args_compute_tau_absorption), override=True)
	lib.compute_tau_absorption(                                                   \
        	c_ncol,                                                               \
        	c_nlay,                                                               \
		kdistSW['nband'],                                                     \
		kdistSW['ngpt'],                                                      \
		kdistSW['ngas'],                                                      \
		kdistSW['nflavors'],                                                  \
		kdistSW['nmixfrac'],                                                  \
		kdistSW['npressref'],                                                 \
		kdistSW['ntempref'],                                                  \
		kdistSW['nminorabslower'],                                            \
		kdistSW['ncontlower'],                                                \
		kdistSW['nminorabsupper'],                                            \
		kdistSW['ncontupper'],                                                \
		kdistSW['idx_h2o'],                                                   \
		kdistSW['gpoint_flavor'],                                             \
		kdistSW['bnd_limits_gpt'],                                            \
		kdistSW['kmajor'],                                                    \
		kdistSW['kminor_lower'],                                              \
		kdistSW['kminor_upper'],                                              \
		kdistSW['minor_limits_gpt_lower'],                                    \
		kdistSW['minor_limits_gpt_upper'],                                    \
		kdistSW['minor_scales_with_density_lower'],                           \
		kdistSW['minor_scales_with_density_upper'],                           \
		kdistSW['scale_by_complement_lower'],                                 \
		kdistSW['scale_by_complement_upper'],                                 \
		kdistSW['idx_minor_lower'],                                           \
		kdistSW['idx_minor_upper'],                                           \
		kdistSW['idx_minor_scaling_lower'],                                   \
		kdistSW['idx_minor_scaling_upper'],                                   \
		kdistSW['kminor_start_lower'],                                        \
		kdistSW['kminor_start_upper'],                                        \
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
	c_tau_sw_rayl = ffi.new("double ["+str(ncol)+"]["+str(nlay)+"]["+str(ngpt)+"]")
	ffi.cdef(construct_fficdef("compute_tau_rayleigh",args_compute_tau_rayleigh), override=True)
	lib.compute_tau_rayleigh(                                                     \
        	c_ncol,                                                               \
        	c_nlay,                                                               \
		kdistSW['nband'],                                                     \
		kdistSW['ngpt'],                                                      \
		kdistSW['ngas'],                                                      \
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
	# Combine optical depths from gas absorption and Rayleigh scattering
	# Also, convert c-types
	#
	tau = np.empty((ncol,nlay,ngpt),dtype=np.double)
	if (do_twostream):
		ssa = np.empty((ncol,nlay,ngpt),dtype=np.double)
		g   = np.zeros((ncol,nlay,ngpt),dtype=np.double)
	for icol in range(0,ncol):
		for ilay in range(0,nlay):
			for igpt in range(0,ngpt):
				tau[icol,ilay,igpt] = c_tau_sw[icol][ilay][igpt] + \
						      c_tau_sw_rayl[icol][ilay][igpt]
				if (do_twostream):
					if (tau[icol,ilay,igpt] > 0):
						ssa[icol,ilay,igpt] = c_tau_sw_rayl[icol][ilay][igpt]/ \
								      tau[icol,ilay,igpt]
					else:
						ssa[icol,ilay,igpt] = 0.
	#
	# Create output structure
	#
	dataOUT = {}
	dataOUT["tau"] = tau
	if (do_twostream): 
		dataOUT["ssa"] = ssa
		dataOUT["g"]   = g

	return(dataOUT)
