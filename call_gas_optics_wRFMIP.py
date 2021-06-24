###############################################################################################
#! /usr/bin/env python
#
# This script reads in the RFMIP data and calls the RRTMGP gas-optics kernels.
#
###############################################################################################
import sys
import os
import urllib.request
import netCDF4
import numpy as np
import argparse
import mo_gas_optics_kernels
from cffi import FFI

# Location of shared library file
rrtmgp_lib="/home/dswales/Projects/radiation-nn/rrtmgp-nn-training/rte-rrtmgp/build/mo_gas_optics_kernels.so"

# Open mo_gas_optics_kernels
ffi = mo_gas_optics_kernels.init()
lib = ffi.dlopen(rrtmgp_lib)

# Download RFMIP profiles
rte_rrtmgp_dir = "/home/dswales/Projects/radiation-nn/rrtmgp-nn-training/rte-rrtmgp/"
rfmip_dir      = os.path.join(rte_rrtmgp_dir, "examples", "rfmip-clear-sky")
conds_file     = "multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
conds_url      = "http://aims3.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6/RFMIP/UColorado/UColorado-RFMIP-1-2/" + \
                 "atmos/fx/multiple/none/v20190401/" + conds_file
print("Downloading RFMIP input file: "+conds_file)
urllib.request.urlretrieve(conds_url, conds_file)

# Read in data

# Read in k-distribution data

# Call zero array
# Call Interpolation
# Call compute_tau_absorption
# Call Combine and reorder