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


##########################################################################################
##########################################################################################
# Which gases to use?
gases           = ["h2o","co2","o3","n2o","ch4","o2"]

# Location of rte-rrtmgp 
rte_rrtmgp_dir = "/home/dswales/Projects/radiation-nn/rrtmgp-nn-training/rte-rrtmgp/"
file_kdistLW   = rte_rrtmgp_dir + "rrtmgp/data/rrtmgp-data-lw-g256-2018-12-04.nc"
file_kdistSW   = rte_rrtmgp_dir + "rrtmgp/data/rrtmgp-data-sw-g224-2018-12-04.nc"

# Open mo_gas_optics_init library
ffi = FFI()

# Load k-distribution files
print_info = True
kdistLW = load_kdist(ffi, file_kdistLW, gases, print_info)
kdistSW = load_kdist(ffi, file_kdistSW, gases, print_info)
