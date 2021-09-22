#!/bin/bash

ml purge
ml intel/2020.2
ml netcdf/4.6.1
ml

export FC=ifort
export NFHOME=/apps/netcdf/4.6.1/intel/16.1.150/
export NCHOME=${NFHOME}
export FCFLAGS="-O3 -g -traceback -shared -fpic -check bounds"
#export RRTMGP_ROOT=/home/dswales/Projects/radiation-nn/rrtmgp-nn-training/rte-rrtmgp
export RRTMGP_ROOT=/scratch2/BMC/ome/Dustin.Swales/radiation-nn/rte-rrtmgp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dswales/Projects/radiation-nn/rrtmgp-nn-training/libs

mkdir -p libs
cd libs
cp ${RRTMGP_ROOT}/build/*.mod .
cp ${RRTMGP_ROOT}/build/*.o .

${FC} ${FCFLAGS} mo_rte_kind.o mo_rte_util_array.o     ${RRTMGP_ROOT}/rrtmgp/kernels/mo_gas_optics_kernels.F90 -o mo_gas_optics_kernels.so

cd ..
