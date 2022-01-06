#!/bin/bash

ml purge
ml intel/18.0.5.274
ml netcdf/4.7.0
ml

export FC=ifort
export FCFLAGS="-m64 -O3 -g -traceback -heap-arrays -assume realloc_lhs -extend-source 132"
export FCFLAGSS="-shared -fpic"
export NFHOME=/apps/netcdf/4.7.0/intel/18.0.5.274
export NCHOME=${NFHOME}
export RRTMGP_ROOT=/home/Dustin.Swales/Projects/radiation-nn/rrtmgp-nn-training/rte-rrtmgp

# RRTMGP libraries
#cd rte-rrtmgp
#make clean
#make libs
#cd ../

# gas-optics kernels shared library
mkdir -p rte-rrtmgp/build/libss
cd rte-rrtmgp/build/libss
${FC} ${FCFLAGS} ${FCFLAGSS} -c ${RRTMGP_ROOT}/rte/mo_rte_kind.F90
${FC} ${FCFLAGS} ${FCFLAGSS} -c ${RRTMGP_ROOT}/rte/mo_rte_util_array.F90
${FC} ${FCFLAGS} ${FCFLAGSS} mo_rte_kind.o mo_rte_util_array.o  ${RRTMGP_ROOT}/rrtmgp/kernels/mo_gas_optics_kernels.F90 -o mo_gas_optics_kernels.so
cd ../../../
