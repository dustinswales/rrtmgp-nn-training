##########################################################################################
#! /usr/bin/env python
#
# This script reads in the RFMIP data and calls the RRTMGP gas-optics kernels.
#
##########################################################################################
from load_kdist    import load_kdist
from read_rfmip    import read_rfmip
from mo_gas_optics import gas_optics_rrtmgp
import numpy as np
import xarray as xr
##########################################################################################

fileOUT = "/scratch2/BMC/ome/Dustin.Swales/radiation-nn/testout.nc"

# Which gases to use? (chemical_name)
gases = ["h2o", "co2", "o3", "n2o", "co", "ch4", "o2", "n2", "ccl4", "cfc11", "cfc12",   \
         "cfc22", "hfc143a", "hfc125", "hfc23", "hfc32", "hfc134a", "cf4", "no2"]

# Location of rte-rrtmgp k-distribution files
#file_kdistLW = "rte-rrtmgp/rrtmgp/data/rrtmgp-data-lw-g128-210809.nc"
#file_kdistSW = "rte-rrtmgp/rrtmgp/data/rrtmgp-data-sw-g112-210809.nc"
file_kdistLW = "rte-rrtmgp/rrtmgp/data/rrtmgp-data-lw-g256-2018-12-04.nc"
file_kdistSW = "rte-rrtmgp/rrtmgp/data/rrtmgp-data-sw-g224-2018-12-04.nc"

# File containing RFMIP profiles
conds_file  = "/scratch2/BMC/ome/Dustin.Swales/radiation-nn/data/multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
nrfmip_expt = 3

# Load k-distribution files
kdistLW = load_kdist(file_kdistLW, gases, True)
kdistSW = load_kdist(file_kdistSW, gases, True)
ngptLW  = kdistLW["ngpt"][0]
ngptSW  = kdistSW["ngpt"][0]
nbndLW  = kdistLW["nband"][0]
nbndSW  = kdistSW["nband"][0]

# For each RFMIP experiment...
init = True
exp = []
for irfmip_expt in range(0,nrfmip_expt):
	# Load RFMIP data.
	data  = read_rfmip(conds_file, gases, irfmip_expt)
	exp.append(data["exp"])
	print("Reading in RFMIP experiment ",data["exp"])

	if (init):
		init  = False
		ncol  = len(data["t_lay"][0,:])
		nlay  = len(data["t_lay"][:,0])
		tauLW = np.empty((ncol,nlay,ngptLW,nrfmip_expt),dtype=np.double)
		tauSW = np.empty((ncol,nlay,ngptSW,nrfmip_expt),dtype=np.double)
		ssaSW = np.empty((ncol,nlay,ngptSW,nrfmip_expt),dtype=np.double)
		gSW   = np.empty((ncol,nlay,ngptSW,nrfmip_expt),dtype=np.double)
		play  = np.empty((ncol,nlay,nrfmip_expt),dtype=np.double)
		tlay  = np.empty((ncol,nlay,nrfmip_expt),dtype=np.double)

	# Call RRTMGP SW gas-optics.
	print("   Calling RRTMGP SW kernels...")
	optical_props_sw = gas_optics_rrtmgp(kdistSW, data["p_lay"], data["t_lay"],      \
		data["col_gas"], data["col_dry"], data["t_lev"], data["t_sfc"], do_twostream=True)
	tauSW[:,:,:,irfmip_expt] = np.array(optical_props_sw['tau'])
	gSW[:,:,:,irfmip_expt]   = optical_props_sw['g']
	ssaSW[:,:,:,irfmip_expt] = optical_props_sw['ssa']

	# Call RRTMGP LW gas-optics
	print("   Calling RRTMGP LW kernels...")
	optical_props_lw = gas_optics_rrtmgp(kdistLW, data["p_lay"], data["t_lay"],      \
		data["col_gas"], data["col_dry"], data["t_lev"], data["t_sfc"], do_twostream=False)
	tauLW[:,:,:,irfmip_expt] = optical_props_lw['tau']

	#
	for icol in range(0,ncol):
		play[icol,:,irfmip_expt] = data["p_lay"][:,0]
		tlay[icol,:,irfmip_expt] = data["t_lay"][:,0]
# Write to output
tauLWout = xr.Dataset({"tau_lw": (("site", "layer", "gptLW", "exp"),tauLW)})
tauSWout = xr.Dataset({"tau_sw": (("site", "layer", "gptSW", "exp"),tauSW)})
ssaSWout = xr.Dataset({"ssa_sw": (("site", "layer", "gptSW", "exp"),ssaSW)})
gSWout   = xr.Dataset({"g_sw":   (("site", "layer", "gptSW", "exp"),gSW)})
expout   = xr.Dataset({"exp":    ((                          "exp"),exp)})
playout  = xr.Dataset({"p_lay":  (("site", "layer",          "exp"),play)})
tlayout  = xr.Dataset({"t_lay":  (("site", "layer",          "exp"),tlay)})
xr.merge([playout,tlayout,expout,tauLWout,tauSWout,ssaSWout,gSWout]).to_netcdf(fileOUT)

