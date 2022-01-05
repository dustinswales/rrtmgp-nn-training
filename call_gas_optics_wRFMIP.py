##########################################################################################
#! /usr/bin/env python
#
# This script reads in the RFMIP data and calls the RRTMGP gas-optics kernels.
#
##########################################################################################
import urllib.request
import numpy as np
from cffi import FFI
from load_kdist import load_kdist
import xarray as xr
from read_rfmip import read_rfmip
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

def construct_ffinew(kernel_name, args, return_type="void" ):
	fdef = '{} {}('.format(return_type, kernel_name)
	for a in args[:-1]:
		fdef += "{}, ".format(construct_arg4_fficdef(a))

##########################################################################################
##########################################################################################
# Which gases to use? Provide key for GP/forcing-dataset gasenames 
# - "chemical_name" is used by RRTMGP k-distribution data files
gases =  {"chemical_name":["h2o", "co2", "o3", "n2o", "co", "ch4", "o2", "n2", "ccl4",   \
                           "cfc11", "cfc12", "cfc22", "hfc143a", "hfc125", "hfc23",      \
                           "hfc32", "hfc134a", "cf4", "no2"]}

# Location of rte-rrtmgp 
#rte_rrtmgp_dir = "/home/dswales/Projects/radiation-nn/rrtmgp-nn-training/rte-rrtmgp/"
rte_rrtmgp_dir = "/scratch2/BMC/ome/Dustin.Swales/radiation-nn/rte-rrtmgp/"
file_kdistLW   = rte_rrtmgp_dir + "rrtmgp/data/rrtmgp-data-lw-g256-2018-12-04.nc"
file_kdistSW   = rte_rrtmgp_dir + "rrtmgp/data/rrtmgp-data-sw-g224-2018-12-04.nc"

# Download RFMIP profiles
conds_file     = "/scratch2/BMC/ome/Dustin.Swales/radiation-nn/data/multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
conds_url      = "http://aims3.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6/RFMIP/UColorado/UColorado-RFMIP-1-2/" + \
                 "atmos/fx/multiple/none/v20190401/" + conds_file
#conds_file     = "multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
#urllib.request.urlretrieve(conds_url, conds_file)

# Open mo_gas_optics_kernels library
ffi = FFI()
lib = ffi.dlopen("libs/mo_gas_optics_kernels.so")

# Load k-distribution files
print_info = True
kdistLW = load_kdist(ffi, file_kdistLW, gases["chemical_name"], print_info)
kdistSW = load_kdist(ffi, file_kdistSW, gases["chemical_name"], print_info)

# Load RFMIP data
irfmip_expt = 0
rfmip_data = read_rfmip(ffi, conds_file,gases["chemical_name"],irfmip_expt)

##########################################################################################
#
# Create dictionaries for kernel interfacing
#
##########################################################################################
# Dimensions
ngas           = len(gases["chemical_name"])
ncol           = rfmip_data["ncol"][0]
nlay           = rfmip_data["nlay"][0]
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
#args_combine_and_reorder_2str = \
#        [{"name":"ncol",                            "ctype":"int"},                                        \
#         {"name":"nlay",                            "ctype":"int"},                                        \
#         {"name":"ngpt",                            "ctype":"int"},                                        \
#         {"name":"tau_abs",                         "ctype":"double", "dims":[ncol,nlay,ngpt]},            \
#         {"name":"tau_rayleigh",                    "ctype":"double", "dims":[ncol,nlay,ngpt]},            \
#         {"name":"tau",                             "ctype":"double", "dims":[ngpt,nlay,ncol]},            \
#         {"name":"ssa",                             "ctype":"double", "dims":[ngpt,nlay,ncol]},            \
#         {"name":"g",                               "ctype":"double", "dims":[ngpt,nlay,ncol]}]

##########################################################################################
#
# Call gas-optics kernels...
#
########################################################################################## 

#
# Interpolation
#
c_jtemp       = ffi.new("int ["   +str(nlay) +"]["+str(ncol)+"]")
c_jpress      = ffi.new("int ["   +str(nlay) +"]["+str(ncol)+"]")
c_tropo       = ffi.new("int ["   +str(nlay) +"]["+str(ncol)+"]")
c_jeta        = ffi.new("int ["   +str(nflav)+"]["+str(nlay)+"]["+str(ncol)+"][2]")
c_col_mix     = ffi.new("double ["+str(nflav)+"]["+str(nlay)+"]["+str(ncol)+"][2]")
c_fmajor      = ffi.new("double ["+str(nflav)+"]["+str(nlay)+"]["+str(ncol)+"][2][2][2]")
c_fminor      = ffi.new("double ["+str(nflav)+"]["+str(nlay)+"]["+str(ncol)+"][2][2]")
ffi.cdef(construct_fficdef("interpolation",args_interpolation), override=True)
lib.interpolation(                                                            \
        rfmip_data["ncol"],                                                   \
        rfmip_data["nlay"],                                                   \
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
        rfmip_data["play"],                                                   \
        rfmip_data["tlay"],                                                   \
        rfmip_data["col_gas"],                                                \
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
        rfmip_data["ncol"],                                                   \
        rfmip_data["nlay"],                                                   \
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
        rfmip_data["play"],                                                   \
        rfmip_data["tlay"],                                                   \
        rfmip_data["col_gas"],                                                \
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
        rfmip_data["ncol"],                                                   \
        rfmip_data["nlay"],                                                   \
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
	rfmip_data["col_dry"],                                                \
	rfmip_data["col_gas"],                                                \
	c_fminor,                                                             \
	c_jeta,                                                               \
	c_tropo,                                                              \
	c_jtemp,                                                              \
	c_tau_sw_rayl)

#
# Combine_and_reorder
#
#c_tau         = ffi.new("double ["+str(ngpt)+"]["+str(nlay)+"]["+str(ncol)+"]")
#c_ssa         = ffi.new("double ["+str(ngpt)+"]["+str(nlay)+"]["+str(ncol)+"]")
#c_g           = ffi.new("double ["+str(ngpt)+"]["+str(nlay)+"]["+str(ncol)+"]")
#ffi.cdef(construct_fficdef("combine_and_reorder_2str",args_combine_and_reorder_2str), override=True)
#lib.combine_and_reorder_2str(                                                 \
#        rfmip_data["ncol"],                                                   \
#        rfmip_data["nlay"],                                                   \
#        kdistSW['ngpt'],                                                      \
#        c_tau_sw,                                                             \
#        c_tau_sw_rayl,                                                        \
#        c_tau, c_ssa, c_g)

##for ii in range(0,ncol):
#ii=0
#for il in range(0,nlay):
#  for ij in range(0,ngpt):
#    print(rfmip_data["play"][il][0],c_tau[ij][il][ii])


