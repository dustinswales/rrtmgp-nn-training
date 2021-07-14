#!/bin/bash

ml purge
ml intel/20.0.2
ml netcdf/4.7.4
ml

export FC=ifort
export NFHOME=/apps/spack2/opt/spack/linux-centos7-x86_64/intel-20.0.2/netcdf-4.7.4/
export NCHOME=${NFHOME}
export FCFLAGS="-m64 -O3 -g -traceback -heap-arrays -assume realloc_lhs -extend-source 132 -shared -fpic"
export RRTMGP_ROOT=/home/dswales/Projects/radiation-nn/rrtmgp-nn-training/rte-rrtmgp

${FC} ${FCFLAGS} -I${RRTMGP_ROOT}/build/ ${RRTMGP_ROOT}/rrtmgp/kernels/mo_gas_optics_kernels.F90 -o mo_gas_optics_kernels.so
${FC} ${FCFLAGS} -I${RRTMGP_ROOT}/build/ mo_gas_optics_init.F90 -o mo_gas_optics_init.so
