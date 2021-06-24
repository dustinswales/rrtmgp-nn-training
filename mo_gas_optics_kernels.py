###############################################################################################
#! /usr/bin/env python
#
# C-signatures for RRTMGP Fortan subroutines calling with CFFI
# rte-rrtmgp/rrtmgp/kernels/mo_gas_optics_kernels.F90
#
##############################################################################################
from cffi import FFI

def init():
	ffi = FFI()
	##########################################################################################
	# interpolation
	##########################################################################################
	ffi.cdef("void interpolation(                                                            \
   	   int ncol, int nlay, int ngas, int nflav, int neta, int npres, int ntemp, int flavor[],\
       double press_ref_log, double temp_ref[], double press_ref_log_delta,                  \
       double temp_ref_min, double temp_ref_delta, double press_ref_trop_log,                \
       double vmr_ref[], double play[], double tlay[], double col_gas[], int jtemp[],        \
       double fmajor[], double fminor[], double col_mix[], bool tropo[], int jeta[],         \
   	   int jpress[]);", override=True)
	##########################################################################################
	# compute_tau_absorption
	##########################################################################################
	ffi.cdef("void compute_tau_absorption(                                                   \
       int ncol, int nlay, int nbnd, int ngpt, int ngas, int nflav, int neta, int npres,     \
       int ntemp, int nminorlower, int nminorklower, int nminorupper, int nminorkupper,      \
       int idx_h2o, int gpoint_flavor[], int band_lims_gpt[], double kmajor[],               \
       double kminor_lower[], double kminor_upper[], int minor_limits_gpt_lower[],           \
       int minor_limits_gpt_upper[], bool minor_scales_with_density_lower[],                 \
       bool minor_scales_with_density_upper[], bool scale_by_complement_lower[],             \
       bool scale_by_complement_upper[], int idx_minor_lower[], int idx_minor_upper[],       \
       int idx_minor_scaling_lower[], int idx_minor_scaling_upper[],                         \
       int kminor_start_lower[], int kminor_start_upper[], bool tropo[], double col_mix[],   \
       double fmajor[], double fminor[], double play[], double tlay[], double col_gas[],     \
       int jeta[], int jtemp[], int jpress[], double tau[]);", override=True)
	##########################################################################################
	# compute_tau_rayleigh
	##########################################################################################
	ffi.cdef("void compute_tau_rayleigh(                                                     \
       int ncol, int nlay, int nbnd, int ngpt, int ngas, int nflav, int neta, int npres,     \
       int ntemp, int gpoint_flavor[], int band_lims_gpt[], double krayl[], int idx_h2o,     \
       double col_dry[], double col_gas[], double fminor[], int jeta[], bool tropo[],        \
       int jtemp[], double tau_rayleigh[]);", override=True)
	##########################################################################################
	# gas_optical_depths_minor
	##########################################################################################
	ffi.cdef("void gas_optical_depths_minor(                                                 \
       int ncol, int nlay, int ngpt, int ngas, int nflav, int ntemp, int neta, int nminor,   \
       int nminork, int idx_h2o, int gpt_flv[], double kminor[], int minor_limits_gpt[],     \
       bool minor_scales_with_density[], bool scale_by_complement[], int kminor_start[],     \
       int idx_minor[], int idx_minor_scaling[], double play[], double tlay[],               \
       double col_gas[],  double fminor[], int jeta[], int layer_limits[], int jtemp[],      \
       double tau[]);", override=True)
	##########################################################################################
	# gas_optical_depths_major
	##########################################################################################
	ffi.cdef("void gas_optical_depths_major(                                                 \
       int ncol, int nlay, int nband, int ngpt, int nflav, int neta, int npres, int ntemp,   \
       int gpoint_flavor[], int band_lims_gpt[], double kmajor[], double col_mix[],          \
       double fmajor[], int jeta[], bool tropo[], int jtemp[], int jpress[], double tau[]);",\
       override=True)
	##########################################################################################
	# compute_Planck_source
	##########################################################################################
	ffi.cdef("void compute_Planck_source(                                                    \
       int ncol, int nlay, int nbnd, int ngpt, int nflav, int neta, int npres, int ntemp,    \
       int nPlanckTemp, double tlay[], double tlev[], double tsfc[], int sfc_lay,            \
       double fmajor[], int jeta[], bool tropo[], int jtemp[], int jpress[],                 \
       int gpoint_bands[], int band_lims_gpt[], double pfracin[], double temp_ref_min,       \
       double totplnk_delta, double totplnk[], int gpoint_flavor[], double sfc_src[],        \
       double lay_src[], double lev_src_inc[], double lev_src_dec[],                         \
       double sfc_source_Jac[]);", override=True)
	##########################################################################################
	# combine_and_reorder_2str
	##########################################################################################
	ffi.cdef("void combine_and_reorder_2str(                                                 \
       int ncol, int nlay, int ngpt, double tau_abs[], double tau_rayleigh[], double tau[],  \
       double ssa[], double g[]);", override=True)
	##########################################################################################
	# combine_and_reorder_nstr
	##########################################################################################
	ffi.cdef("void combine_and_reorder_nstr(                                                 \
       int ncol, int nlay, int ngpt, int nmom, double tau_abs[], double tau_rayleigh[],      \
       double tau[], double ssa[], double p[]);", override=True)

	return ffi