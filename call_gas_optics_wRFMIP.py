##########################################################################################
#! /usr/bin/env python
#
# This script reads in the RFMIP data and calls the RRTMGP gas-optics kernels.
#
##########################################################################################
from load_kdist    import load_kdist
from read_rfmip    import read_rfmip
from mo_gas_optics import gas_optics_rrtmgp
##########################################################################################

# Which gases to use? (chemical_name)
gases = ["h2o", "co2", "o3", "n2o", "co", "ch4", "o2", "n2", "ccl4", "cfc11", "cfc12",   \
         "cfc22", "hfc143a", "hfc125", "hfc23", "hfc32", "hfc134a", "cf4", "no2"]

# Location of rte-rrtmgp k-distribution files
file_kdistLW = "rte-rrtmgp/rrtmgp/data/rrtmgp-data-lw-g128-210809.nc"
file_kdistSW = "rte-rrtmgp/rrtmgp/data/rrtmgp-data-sw-g112-210809.nc"

# File containing RFMIP profiles
conds_file  = "/scratch2/BMC/ome/Dustin.Swales/radiation-nn/data/multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
nrfmip_expt = 1

# Load k-distribution files
kdistLW    = load_kdist(file_kdistLW, gases, True)
kdistSW    = load_kdist(file_kdistSW, gases, True)

# For each RFMIP experiment...
for irfmip_expt in range(0,nrfmip_expt):
	# Load RFMIP data.
	data  = read_rfmip(conds_file, gases, irfmip_expt)

	# Call RRTMGP SW gas-optics.
	optical_props_sw = gas_optics_rrtmgp(kdistSW, data["p_lay"], data["t_lay"],      \
		data["col_gas"], data["col_dry"], data["t_lev"], data["t_sfc"], do_twostream=True)

	# Call RRTMGP LW gas-optics
	optical_props_lw = gas_optics_rrtmgp(kdistLW, data["p_lay"], data["t_lay"],      \
		data["col_gas"], data["col_dry"], data["t_lev"], data["t_sfc"], do_twostream=False)
