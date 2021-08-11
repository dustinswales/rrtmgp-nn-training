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
from load_kdist import load_kdist
from load_kdist_noF90 import load_kdist_noF90


##########################################################################################
##########################################################################################
# Which gases to use?
gases           = ["h2o","co2","o3","n2o","ch4","o2"]

# Location of rte-rrtmgp 
rte_rrtmgp_dir = "/home/dswales/Projects/radiation-nn/rrtmgp-nn-training/rte-rrtmgp/"
file_kdistLW   = rte_rrtmgp_dir + "rrtmgp/data/rrtmgp-data-lw-g256-2018-12-04.nc"
file_kdistSW   = rte_rrtmgp_dir + "rrtmgp/data/rrtmgp-data-sw-g224-2018-12-04.nc"

# Download RFMIP profile
#conds_file     = "multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
#conds_url      = "http://aims3.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6/RFMIP/UColorado/UColorado-RFMIP-1-2/" + \
#                 "atmos/fx/multiple/none/v20190401/" + conds_file
#urllib.request.urlretrieve(conds_url, conds_file)

# Open mo_gas_optics_init library
ffi = FFI()

# Load k-distribution files
print_info = False
output_to_ctypes = True
kdistLW = load_kdist_noF90(ffi, file_kdistLW, gases, print_info, output_to_ctypes)
kdistSW = load_kdist_noF90(ffi, file_kdistSW, gases, print_info, output_to_ctypes)
