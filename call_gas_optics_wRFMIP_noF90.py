##########################################################################################
#! /usr/bin/env python
#
# This script reads in the RFMIP data and calls the RRTMGP gas-optics kernels.
#
##########################################################################################
import sys
import os
import urllib.request
import numpy as np
import mo_gas_optics_kernels
from cffi import FFI
from load_kdist_noF90 import load_kdist_noF90
import xarray as xr
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
def cdef_interpolation(ffi, ncol, nlay, ntemp, npres, nflav, ngas):
	ffi.cdef("void interpolation(                                                     \
        int *ncol, int *nlay, int *ngas, int *nflav, int *neta, int *npres, int *ntemp,   \
        int flavor[2]["+str(nflav)+"],                                                    \
        double press_ref_log["+str(npres)+"],                                             \
        double temp_ref["+str(ntemp)+"],                                                  \
        double *press_ref_log_delta,                                                      \
        double *temp_ref_min,                                                             \
        double *temp_ref_delta,                                                           \
        double *press_ref_trop_log,                                                       \
        double vmr_ref[2]["+str(ngas+1)+"]["+str(ntemp)+"],                               \
        double play["+str(ncol)+"]["+str(nlay)+"],                                        \
        double tlay["+str(ncol)+"]["+str(nlay)+"],                                        \
        double col_gas["+str(ncol)+"]["+str(nlay)+"]["+str(ngas+1)+"],                    \
        int jtemp["+str(ncol)+"]["+str(nlay)+"],                                          \
        double fmajor[2][2][2]["+str(nflav)+"]["+str(ncol)+"]["+str(nlay)+"],             \
        double fminor[2][2]["+str(nflav)+"]["+str(ncol)+"]["+str(nlay)+"],                \
        double col_mix[2]["+str(nflav)+"]["+str(ncol)+"]["+str(nlay)+"],                  \
        int tropo["+str(ncol)+"]["+str(nlay)+"],                                          \
        int jeta[2]["+str(nflav)+"]["+str(ncol)+"]["+str(nlay)+"],                        \
        int jpress["+str(ncol)+"]["+str(nlay)+"]);", override=True)
        
##########################################################################################
def cdef_compute_tau_absorption(ffi, ngpt, nband, neta, npressref, ntemp, ncontlower,     \
                                ncontupper, nminorabslower, nminorabsupper, nflav, ngas,  \
                                ncol, nlay):
        ffi.cdef("void compute_tau_absorption(                                            \
        int *ncol, int *nlay, int *nbnd, int *ngpt, int *ngas, int *nflav, int *neta,     \
        int *npres, int *ntemp, int *nminorlower, int *nminorklower, int *nminorupper,    \
        int *nminorkupper, int *idx_h2o,                                                  \
        int gpoint_flavor[2]["+str(ngpt)+"],                                              \
        int band_lims_gpt[2]["+str(nband)+"],                                             \
        double kmajor["+str(ngpt)+"]["+str(neta)+"]["+str(npressref+1)+"]["+str(ntemp)+"],\
        double kminor_lower["+str(ncontlower)+"]["+str(neta)+"]["+str(ntemp)+"],          \
        double kminor_upper["+str(ncontupper)+"]["+str(neta)+"]["+str(ntemp)+"],          \
        int minor_limits_gpt_lower[2]["+str(nminorabslower)+"],                           \
        int minor_limits_gpt_upper[2]["+str(nminorabsupper)+"],                           \
        int minor_scales_with_density_lower["+str(nminorabslower)+"],                     \
        int minor_scales_with_density_upper["+str(nminorabsupper)+"],                     \
        int scale_by_complement_lower["+str(nminorabslower)+"],                           \
        int scale_by_complement_upper["+str(nminorabsupper)+"],                           \
        int idx_minor_lower["+str(nminorabslower)+"],                                     \
        int idx_minor_upper["+str(nminorabsupper)+"],                                     \
        int idx_minor_scaling_lower["+str(nminorabslower)+"],                             \
        int idx_minor_scaling_upper["+str(nminorabsupper)+"],                             \
        int kminor_start_lower["+str(nminorabslower)+"],                                  \
        int kminor_start_upper["+str(nminorabsupper)+"],                                  \
        int tropo["+str(ncol)+"]["+str(nlay)+"],                                          \
        double col_mix[2]["+str(nflav)+"]["+str(ncol)+"]["+str(nlay)+"],                  \
        double fmajor[2][2][2]["+str(nflav)+"]["+str(ncol)+"]["+str(nlay)+"],             \
        double fminor[2][2]["+str(nflav)+"]["+str(ncol)+"]["+str(nlay)+"],                \
        double play["+str(ncol)+"]["+str(nlay)+"],                                        \
        double tlay["+str(ncol)+"]["+str(nlay)+"],                                        \
        double col_gas["+str(ncol)+"]["+str(nlay)+"]["+str(ngas+1)+"],                    \
        int jeta[2]["+str(nflav)+"]["+str(ncol)+"]["+str(nlay)+"],                        \
        int jtemp["+str(ncol)+"]["+str(nlay)+"],                                          \
        int jpress["+str(ncol)+"]["+str(nlay)+"],                                         \
        double tau["+str(ngpt)+"]["+str(nlay)+"]["+str(ncol)+"] );", override=True)
        
##########################################################################################
def cdef_compute_tau_rayleigh(ffi, ngpt, nband, neta, ngas, ncol, nlay, nlfav, ntemp):
        ffi.cdef("void compute_tau_rayleigh                                               \
        int *ncol, int *nlay, int *nbnd, int *ngpt, int *ngas, int *nflav, int *neta,     \
        int *npres, int *ntemp,                                                           \
        int gpoint_flavor[2]["+str(ngpt)+"],                                              \
        int band_lims_gpt[2]["+str(nband)+"],                                             \
        double krayl["+str(ngpt)+"]["+str(neta)+"]["+str(ntemp)+"][2],                    \
		int *idx_h2o,                                                                     \
        double col_dry["+str(ncol)+"]["+str(nlay)+"],                                     \
        double col_gas["+str(ncol)+"]["+str(nlay)+"]["+str(ngas+1)+"],                    \
        double fminor[2][2]["+str(nflav)+"]["+str(ncol)+"]["+str(nlay)+"],                \
        int jeta[2]["+str(nflav)+"]["+str(ncol)+"]["+str(nlay)+"],                        \
        int tropo["+str(ncol)+"]["+str(nlay)+"],                                          \
        int jtemp["+str(ncol)+"]["+str(nlay)+"],                                          \
        double tau_rayleigh["+str(ngpt)+"]["+str(nlay)+"]["+str(ncol)+"] );",             \
        override=True)

##########################################################################################
def cdef_combine_and_reorder(ffi):
        return

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
pi    = np.arccos(-1.)

##########################################################################################
##########################################################################################
# Which gases to use?
gases       = ["h2o",        "co2",              "o3",   "n2o",             "ch4",       "o2"]
gases_rfmip = ["water_vapor","carbon_dioxide_GM","ozone","nitrous_oxide_GM","methane_GM","oxygen_GM"]
ngas_req    = len(gases)

# Location of rte-rrtmgp 
rte_rrtmgp_dir = "/home/dswales/Projects/radiation-nn/rrtmgp-nn-training/rte-rrtmgp/"
#rte_rrtmgp_dir = "/scratch2/BMC/ome/Dustin.Swales/radiation-nn/rte-rrtmgp/"
file_kdistLW   = rte_rrtmgp_dir + "rrtmgp/data/rrtmgp-data-lw-g256-2018-12-04.nc"
file_kdistSW   = rte_rrtmgp_dir + "rrtmgp/data/rrtmgp-data-sw-g224-2018-12-04.nc"

# Download RFMIP profiles
conds_file     = "/scratch2/BMC/ome/Dustin.Swales/radiation-nn/multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
conds_url      = "http://aims3.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6/RFMIP/UColorado/UColorado-RFMIP-1-2/" + \
                 "atmos/fx/multiple/none/v20190401/" + conds_file
conds_file     = "multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
#urllib.request.urlretrieve(conds_url, conds_file)

# Open mo_gas_optics_kernels library
ffi = FFI()
lib = ffi.dlopen("libs/mo_gas_optics_kernels.so")

# Load k-distribution files
print_info = True
output_to_ctypes = True
kdistLW = load_kdist_noF90(ffi, file_kdistLW, gases, print_info, output_to_ctypes)
kdistSW = load_kdist_noF90(ffi, file_kdistSW, gases, print_info, output_to_ctypes)

# Load RFMIP data
data_RFMIP = xr.open_dataset(conds_file,concat_characters=True,decode_cf=True)

# Dimensions (flat)
ncol_rfmip = data_RFMIP.lon.size
nlay_rfmip = data_RFMIP.pres_layer[0,:].size
nlev_rfmip = data_RFMIP.pres_level[0,:].size

print("RFMIP dims: ",ncol_rfmip, nlay_rfmip, nlev_rfmip)

#
# This section prepares the RFMIP data for use by the rrtmgp-kernels.
#
# Assume only using single RFMIP experiment for the time being, will expand later.
irfmip_expt = 0

# Create array of volume-mixing-ratios (vmr), populate with RFMIP data
#ngases = kdistLW['c_ngas_req'][0]
vmr = np.zeros((ncol_rfmip, nlay_rfmip, ngas_req), dtype=np.double)
vmr[:,:,0] = data_RFMIP.water_vapor.values[irfmip_expt,:,:]
vmr[:,:,1] = data_RFMIP.carbon_dioxide_GM.values[irfmip_expt]
vmr[:,:,2] = data_RFMIP.ozone.values[irfmip_expt,:,:]
vmr[:,:,3] = data_RFMIP.nitrous_oxide_GM.values[irfmip_expt]
vmr[:,:,4] = data_RFMIP.methane_GM.values[irfmip_expt]
vmr[:,:,5] = data_RFMIP.oxygen_GM.values[irfmip_expt]

# Compute dry air column amounts [molec/cm^2]
col_dry = get_col_dry(vmr[:,:,0], data_RFMIP.pres_level.values, latitude=data_RFMIP.lat.values)


# Compute column gas amounts [molec/cm^2]
col_gas = np.zeros((ncol_rfmip, nlay_rfmip, ngas_req+1), dtype=np.double)
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
nblock       = ncol_rfmip
c_nblock     = ffi.new("int *", nblock) 
c_nlay_rfmip = ffi.new("int *", nlay_rfmip)
c_play_rfmip = ffi.new("double ["+str(nblock)+" ]["+str(nlay_rfmip)+"]",    \
                       data_RFMIP.pres_layer.values[0:nblock,:].tolist())
c_tlay_rfmip = ffi.new("double ["+str(nblock)+" ]["+str(nlay_rfmip)+"]",    \
                       data_RFMIP.temp_layer.values[irfmip_expt,0:nblock,:].tolist())
c_col_gas    = ffi.new("double ["+str(nblock)+" ]["+str(nlay_rfmip)+"][ " + \
	               str(ngas_req+1)+"]", col_gas[0:nblock,:,:].tolist())
c_col_dry    = ffi.new("double ["+str(nblock)+" ]["+str(nlay_rfmip)+"]", col_dry.tolist())

# Outputs from kernels
c_jtemp   = ffi.new("int ["+str(nblock)+" ]["+str(nlay_rfmip)+"]")
c_jpress  = ffi.new("int ["+str(nblock)+" ]["+str(nlay_rfmip)+"]")
c_tropo   = ffi.new("int ["+str(nblock)+" ]["+str(nlay_rfmip)+"]")
c_jeta    = ffi.new("int [2]["          + str(kdistSW['nflavors'][0]) + "][" + \
	            str(nblock)+"]["+str(nlay_rfmip)+"]")
c_col_mix = ffi.new("double [2]["       + str(kdistSW['nflavors'][0]) + "][" + \
	            str(nblock)+"]["+str(nlay_rfmip)+"]")
c_fmajor  = ffi.new("double [2][2][2][" + str(kdistSW['nflavors'][0]) + "][" + \
                    str(nblock)+"]["+str(nlay_rfmip)+"]")
c_fminor  = ffi.new("double [2][2]["    + str(kdistSW['nflavors'][0]) + "][" + \
                    str(nblock)+"]["+str(nlay_rfmip)+"]")
c_tau_sw  = ffi.new("double [" + str(kdistSW['ngpt'][0]) + "][" + str(nlay_rfmip) + \
	            "][" + str(nblock) + "]")
c_tau_sw_rayl = ffi.new("double [" + str(kdistSW['ngpt'][0]) + "][" + str(nlay_rfmip) + \
	            "][" + str(nblock) + "]")
##########################################################################################
#
# Call interpolation ...
#
##########################################################################################
cdef_interpolation(ffi, nblock, nlay_rfmip, kdistSW['ntempref'][0],                      \
	kdistSW['npressref'][0], kdistSW['nflavors'][0], ngas_req)

lib.interpolation( c_nblock, c_nlay_rfmip, kdistSW['ngas_req'],                          \
	kdistSW['nflavors'], kdistSW['nmixfrac'], kdistSW['npressref'],                  \
	kdistSW['ntempref'], kdistSW['flavors'], kdistSW['press_ref_log'],               \
	kdistSW['temp_ref'], kdistSW['press_ref_log_delta'], kdistSW['temp_ref_min'],    \
	kdistSW['temp_ref_delta'], kdistSW['press_ref_trop_log'],                        \
	kdistSW['vmr_ref_red'], c_play_rfmip, c_tlay_rfmip, c_col_gas, c_jtemp, c_fmajor,\
	c_fminor, c_col_mix, c_tropo, c_jeta, c_jpress)

##########################################################################################
#
# Call compute_tau_absorption ...
#
##########################################################################################
cdef_compute_tau_absorption(ffi, kdistSW['ngpt'][0], kdistSW['nband'][0],                \
	kdistSW['nmixfrac'][0], kdistSW['npressref'][0], kdistSW['ntempref'][0],         \
	kdistSW['ncontlower_red'][0], kdistSW['ncontupper_red'][0],                      \
	kdistSW['nminorabslower_red'][0], kdistSW['nminorabsupper_red'][0],              \
	kdistSW['nflavors'][0], ngas_req, nblock, nlay_rfmip)

lib.compute_tau_absorption(                                                              \
	c_nblock,                                                                        \
	c_nlay_rfmip,                                                                    \
	kdistSW['nband'],                                                                \
	kdistSW['ngpt'],                                                                 \
	kdistSW['ngas_req'],                                                             \
	kdistSW['nflavors'],                                                             \
	kdistSW['nmixfrac'],                                                             \
	kdistSW['npressref'],                                                            \
	kdistSW['ntempref'],                                                             \
	kdistSW['nminorabslower_red'],                                                   \
	kdistSW['ncontlower_red'],                                                       \
	kdistSW['nminorabsupper_red'],                                                   \
	kdistSW['ncontupper_red'],                                                       \
	kdistSW['idx_h2o'],                                                              \
	kdistSW['gpoint_flavor'],                                                        \
	kdistSW['bnd_limits_gpt'],                                                       \
	kdistSW['kmajor'],                                                               \
	kdistSW['kminor_lower_red'],                                                     \
	kdistSW['kminor_upper_red'],                                                     \
	kdistSW['minor_limits_gpt_lower_red'],                                           \
	kdistSW['minor_limits_gpt_upper_red'],                                           \
	kdistSW['minor_scales_with_density_lower_red'],                                  \
	kdistSW['minor_scales_with_density_upper_red'],                                  \
	kdistSW['scale_by_complement_lower_red'],                                        \
	kdistSW['scale_by_complement_upper_red'],                                        \
	kdistSW['idx_minor_lower'],                                                      \
	kdistSW['idx_minor_upper'],                                                      \
	kdistSW['idx_minor_scaling_lower'],                                              \
	kdistSW['idx_minor_scaling_upper'],                                              \
	kdistSW['kminor_start_lower_red'],                                               \
	kdistSW['kminor_start_upper_red'],                                               \
	c_tropo, c_col_mix, c_fmajor, c_fminor, c_play_rfmip, c_tlay_rfmip, c_col_gas,   \
	c_jeta, c_jtemp, c_jpress, c_tau_sw)
##########################################################################################
#
# Call compute_tau_rayleigh
#
##########################################################################################
cdef_compute_tau_rayleigh(ffi, kdistSW['ngpt'][0], kdistSW['nband'][0],              \
	kdistSW['nmixfrac'][0], ngas_req, nblock, nlay_rfmip, kdistSW['nflavors'][0],    \
	kdistSW['ntempref'][0])

lib.compute_tau_rayleigh(                                                            \
	c_nblock,                                                                        \
	c_nlay_rfmip,                                                                    \
	kdistSW['nband'],                                                                \
	kdistSW['ngpt'],                                                                 \
	kdistSW['ngas_req'],                                                             \
	kdistSW['nflavors'],                                                             \
	kdistSW['nmixfrac'],                                                             \
	kdistSW['npressref'],                                                            \
	kdistSW['ntempref'],                                                             \
	kdistSW['gpoint_flavor'],                                                        \
	kdistSW['bnd_limits_gpt'],                                                       \
	kdistSW['krayl'],                                                                \
	kdistSW['idx_h2o'],                                                              \
	c_col_dry,                                                                       \
	c_col_gas,                                                                       \
	c_fminor,                                                                        \
	c_jeta,                                                                          \
	c_tropo,                                                                         \
	c_jtemp,                                                                         \
	c_tau_sw_rayl)

##########################################################################################
#
# Call combine_and_reorder
#
##########################################################################################
