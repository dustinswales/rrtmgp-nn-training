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
	else:                  g0 = np.full((ncol), grav, dtype=float)

	col_dry = np.zeros((ncol,nlay),dtype=float)
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
pi    = np.arccos(-1.)

##########################################################################################
##########################################################################################
# Which gases to use?
gases    = ["h2o","co2","o3","n2o","ch4","o2"]
ngas_req = len(gases)

gases_rfmip = ["water_vapor","carbon_dioxide_GM","ozone","nitrous_oxide_GM","methane_GM","oxygen_GM"]

# Location of rte-rrtmgp 
rte_rrtmgp_dir = "/home/dswales/Projects/radiation-nn/rrtmgp-nn-training/rte-rrtmgp/"
file_kdistLW   = rte_rrtmgp_dir + "rrtmgp/data/rrtmgp-data-lw-g256-2018-12-04.nc"
file_kdistSW   = rte_rrtmgp_dir + "rrtmgp/data/rrtmgp-data-sw-g224-2018-12-04.nc"

# Download RFMIP profile
conds_file     = "multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
conds_url      = "http://aims3.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6/RFMIP/UColorado/UColorado-RFMIP-1-2/" + \
                 "atmos/fx/multiple/none/v20190401/" + conds_file
urllib.request.urlretrieve(conds_url, conds_file)

# Open mo_gas_optics_init library
ffi = FFI()

# Load k-distribution files
print_info = False
output_to_ctypes = True
kdistLW = load_kdist_noF90(ffi, file_kdistLW, gases, print_info, output_to_ctypes)
#kdistSW = load_kdist_noF90(ffi, file_kdistSW, gases, print_info, output_to_ctypes)

# Load RFMIP data
data_RFMIP = xr.open_dataset(conds_file,concat_characters=True,decode_cf=True)
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
vmr = np.zeros((ncol_rfmip, nlay_rfmip, ngas_req), dtype=float)
vmr[:,:,0] = data_RFMIP.water_vapor.values[irfmip_expt,:,:]
vmr[:,:,1] = data_RFMIP.carbon_dioxide_GM.values[irfmip_expt]
vmr[:,:,2] = data_RFMIP.ozone.values[irfmip_expt,:,:]
vmr[:,:,3] = data_RFMIP.nitrous_oxide_GM.values[irfmip_expt]
vmr[:,:,4] = data_RFMIP.methane_GM.values[irfmip_expt]
vmr[:,:,5] = data_RFMIP.oxygen_GM.values[irfmip_expt]

# Compute dry air column amounts [molec/cm^2]
col_dry = get_col_dry(vmr[:,:,0], data_RFMIP.pres_level.values, latitude=data_RFMIP.lat.values)

# Compute column gas amounts [molec/cm^2]
col_gas = np.zeros((ncol_rfmip, nlay_rfmip, ngas_req+1), dtype=float)
col_gas[:,:,0] = col_dry
col_gas[:,:,1] = vmr[:,:,0] * col_dry
col_gas[:,:,2] = vmr[:,:,1] * col_dry
col_gas[:,:,3] = vmr[:,:,2] * col_dry
col_gas[:,:,4] = vmr[:,:,3] * col_dry
col_gas[:,:,5] = vmr[:,:,4] * col_dry
col_gas[:,:,6] = vmr[:,:,5] * col_dry

#
# ---- calculate gas optical depths ----
#













	
	
	
	
	
	
	