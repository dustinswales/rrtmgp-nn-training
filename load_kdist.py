##########################################################################################
#! /usr/bin/env python
#
##########################################################################################
import sys
import os
import urllib.request
import numpy as np
from cffi import FFI
import xarray as xr
from cdef_mo_gas_optics_init import cdef_reduce_minor_arrays,cdef_create_idx_minor,      \
	cdef_create_idx_minor_scaling,cdef_create_key_species_reduce,cdef_get_nflavors,  \
	cdef_create_flavor,cdef_create_gpoint_flavor

##########################################################################################
def xarray_string_to_bytes(string_length,array):
	return str.encode(str(np.array2string(array)[2:string_length-2]).ljust(string_length,' '))
	
##########################################################################################
def load_kdist(ffi, file_kdist, gases, print_info):

		# Open mo_gas_optics_init library
		lib_init = ffi.dlopen("libs/mo_gas_optics_init.so")

		requested_gases = xr.DataArray(gases, dims=["ngas_req"])

		# Load k-distribution data
		kdist = xr.open_dataset(file_kdist,concat_characters=True,decode_cf=True)

		# Longwave or Shortwave file?
		kdist_keys = list(kdist.keys())
		doLW = 'plank_fraction' in list(kdist.keys())
		doSW = 'rayl_lower'     in list(kdist.keys())

		# Dimensions (flat)
		strlen              = 32
		ngas_req            = len(requested_gases)
		nmixfrac            = kdist.mixing_fraction.size
		ncontlower          = kdist.contributors_lower.size
		ncontupper          = kdist.contributors_upper.size
		nmajorabs           = kdist.gas_names.size
		nminorabs           = kdist.gas_minor.size
		nminorabslower      = kdist.minor_gases_lower.size
		nminorabsupper      = kdist.minor_gases_upper.size
		npressref           = kdist.press_ref.size
		ntemp               = kdist.temp_ref.size
		npressiref          = npressref + 1
		ngpt                = kdist.kmajor[0,0,0,:].size
		nband               = kdist.bnd_limits_wavenumber[:,0].size
		natmlayer           = kdist.key_species[0,:,0].size
		npair               = kdist.key_species[0,0,:,].size
		nabsorber_ext       = kdist.vmr_ref[0,:,0].size
		
		# Scalars (flat)
		press_ref_trop      = kdist.press_ref_trop.values
		press_ref_trop_log  = np.log(kdist.press_ref_trop).values
		# minimum, maximum reference temperature, pressure -- assumes low-to-high ordering
		# for T, high-to-low ordering for p
		temp_ref_min        = kdist.temp_ref[0].values
		temp_ref_max        = kdist.temp_ref[ntemp-1].values
		press_ref_min       = kdist.press_ref[npressref-1].values
		press_ref_max       = kdist.press_ref[0].values
		
		# create press_ref_log, temp_ref_delta
		press_ref_log_delta = (np.log(press_ref_min)-np.log(press_ref_max))/(npressref-1)
		temp_ref_delta      = (temp_ref_max-temp_ref_min)/(ntemp-1)
		
		# Scheme specific (LW/SW) fields
		if (doLW):
			ntemp_Planck  = kdist.temperature_Planck.size
			nfit_coeffs   = kdist.optimal_angle_fit[0,:].size
			# Temperature steps for Planck function interpolation
			#   Assumes that temperature minimum and max are the same for the absorption 
			#   coefficient grid and the Planck grid and the Planck grid is equally spaced.
			totplnk_delta = (temp_ref_max-temp_ref_min) / (ntemp_Planck-1)
		
		if (doSW):
			absorption_coefficient_ref_P = kdist.absorption_coefficient_ref_P.values
			absorption_coefficient_ref_T = kdist.absorption_coefficient_ref_T.values
			tsi_default                  = kdist.tsi_default.values
			mg_default                   = kdist.mg_default.values
			sb_default                   = kdist.sb_default.values

		if (print_info):
			print("#################################################################################")
			print("Dimensions (IN/full): ")
			print("   ngas_req                     = ",ngas_req)
			print("   strlen                       = ",strlen)
			print("   ntemp                        = ",ntemp)
			print("   nmixfrac                     = ",nmixfrac)
			print("   npressref                    = ",npressref)
			print("   npressiref                   = ",npressiref)
			print("   ngpt                         = ",ngpt)
			print("   nband                        = ",nband)
			print("   natmlayer                    = ",natmlayer)
			print("   npair                        = ",npair)
			print("   nabsorber_ext                = ",nabsorber_ext)
			if(doLW):
				print("  *ntemp_Planck                 = ",ntemp_Planck)
				print("  *nfit_coeffs                  = ",nfit_coeffs)
			print("   nmajorabs                    = ",nmajorabs)
			print("   nminorabs                    = ",nminorabs)
			print("   ncontlower                   = ",ncontlower)
			print("   ncontupper                   = ",ncontupper)
			print("   nminorabslower               = ",nminorabslower)
			print("   nminorabsupper               = ",nminorabsupper)
			print("Scalars: ")
			print("   temp_ref_min                 = ",temp_ref_min)
			print("   temp_ref_max                 = ",temp_ref_max)
			print("   press_ref_min                = ",press_ref_min)
			print("   press_ref_max                = ",press_ref_max)
			print("   press_ref_trop               = ",press_ref_trop)
			print("   press_ref_trop_log           = ",press_ref_trop_log)
			print("   press_ref_log_delta          = ",press_ref_log_delta)
			print("   temp_ref_delta               = ",temp_ref_delta)
			if (doLW):
				print("  *totplnk_delta                = ",totplnk_delta)
			if (doSW):
				print("  *absorption_coefficient_ref_P = ",absorption_coefficient_ref_P)
				print("  *absorption_coefficient_ref_T = ",absorption_coefficient_ref_T)
				print("  *tsi_default                  = ",tsi_default)
				print("  *mg_default                   = ",mg_default)
				print("  *sb_default                   = ",sb_default)
		
		##########################################################################################
		#
		# Using the "requested_gases", reduce k-distribution data to only the gases needed...
		#
		##########################################################################################
		
		# Which major absorbing gases to include?
		gas_is_present_major = np.zeros((nmajorabs,), dtype=int)
		count = 0
		for im in range(0,nmajorabs):
			for igas in range(0,ngas_req):
				if requested_gases.values[igas] in (kdist.gas_names.values[im]).decode("utf-8"):
					gas_is_present_major[im] = 1
					count = count + 1
					break
				if (count == ngas_req): break
		# Which minor absorbing gases, in lower reference atmosphere, to include?
		ncontlower_red       = 0
		nminorabslower_red   = 0
		gas_is_present_lower = np.zeros((nminorabslower,), dtype=int)
		for im in range(0,nminorabslower):
			for igas in range(0,ngas_req):
				if requested_gases.values[igas] in (kdist.minor_gases_lower.values[im]).decode("utf-8"):
					ncontlower_red   = ncontlower_red   + \
						(kdist.minor_limits_gpt_lower.values[im,1]-kdist.minor_limits_gpt_lower.values[im,0]+1)
					nminorabslower_red = nminorabslower_red + 1
					gas_is_present_lower[im] = 1
					break
		
		# Which minor absorbing gases, in upper reference atmosphere, to include?
		ncontupper_red       = 0
		nminorabsupper_red   =  0
		gas_is_present_upper = np.zeros((nminorabsupper,), dtype=int)
		for im in range(0,nminorabsupper):
			for igas in range(0,ngas_req):
				if requested_gases.values[igas] in (kdist.minor_gases_upper.values[im]).decode("utf-8"):
					ncontupper_red   = ncontupper_red   + \
						(kdist.minor_limits_gpt_upper.values[im,1]-kdist.minor_limits_gpt_upper.values[im,0]+1)
					gas_is_present_upper[im] = 1
					nminorabsupper_red = nminorabsupper_red + 1
					break
		
		# Reduce volume mixing ratios for reference atmosphere...
		#  - Gas 0 is used in single-key species method, set to 1.0 (col_dry)
		vmr_ref_red        = np.zeros((natmlayer,ngas_req+1,ntemp))
		vmr_ref_red[:,0,:] = (np.reshape(kdist.vmr_ref[:,0,:].values,[natmlayer,ntemp])).tolist()
		gas_count          = 0
		for igas in range(0,nmajorabs):
			if (gas_is_present_major[igas]):
				gas_count = gas_count + 1
				vmr_ref_red[:,gas_count,:] = (np.reshape(kdist.vmr_ref[:,igas,:].values,[natmlayer,ntemp])).tolist()
		
		#
		# Create gpoint-to-band indexing array
		#
		gpt2band = np.zeros((ngpt),dtype=int)
		for im in range(0,nband):
			gpt2band[(kdist.bnd_limits_gpt[im,0].values)-1:(kdist.bnd_limits_gpt[im,1].values)] = im+1
		
		#
		# Need to encode/decode character data.
		#
		# Character arrays first need to be flattened to string_length*dimension, then define
		# c-types. In the F90 routines these are encoded/decoded.
		requested_gases_as_bytes = str.encode(str(requested_gases[0].values).ljust(strlen,' '))
		for x in range(1,ngas_req):
			requested_gases_as_bytes = requested_gases_as_bytes + str.encode(str(requested_gases[x].values).ljust(strlen,' '))
		#
		gas_names_major_as_bytes = xarray_string_to_bytes(strlen,kdist.gas_names.values[0])
		for x in range(1,nmajorabs):
			gas_names_major_as_bytes = gas_names_major_as_bytes   + xarray_string_to_bytes(strlen,kdist.gas_names.values[x] )
		#
		gas_minor_as_bytes        = xarray_string_to_bytes(strlen,kdist.gas_minor.values[0])
		identifier_minor_as_bytes = xarray_string_to_bytes(strlen,kdist.identifier_minor.values[0])
		for x in range(1,nminorabs):
			gas_minor_as_bytes        = gas_minor_as_bytes        + xarray_string_to_bytes(strlen,kdist.gas_minor.values[x] )
			identifier_minor_as_bytes = identifier_minor_as_bytes + xarray_string_to_bytes(strlen,kdist.identifier_minor.values[x])
		#
		minor_gases_lower_as_bytes = xarray_string_to_bytes(strlen,kdist.minor_gases_lower.values[0])
		scaling_gas_lower_as_bytes = xarray_string_to_bytes(strlen,kdist.scaling_gas_lower.values[0])
		for x in range(1,nminorabslower):
			minor_gases_lower_as_bytes = minor_gases_lower_as_bytes + xarray_string_to_bytes(strlen,kdist.minor_gases_lower.values[x])
			scaling_gas_lower_as_bytes = scaling_gas_lower_as_bytes + xarray_string_to_bytes(strlen,kdist.scaling_gas_lower.values[x])
		#
		minor_gases_upper_as_bytes = xarray_string_to_bytes(strlen,kdist.minor_gases_upper.values[0])
		scaling_gas_upper_as_bytes = xarray_string_to_bytes(strlen,kdist.scaling_gas_upper.values[0])
		for x in range(1,nminorabsupper):
			minor_gases_upper_as_bytes = minor_gases_upper_as_bytes + xarray_string_to_bytes(strlen,kdist.minor_gases_upper.values[x])
			scaling_gas_upper_as_bytes = scaling_gas_upper_as_bytes + xarray_string_to_bytes(strlen,kdist.scaling_gas_upper.values[x])
		
		if (print_info):
			print("Dimensions (OUT/reduced): ")
			print("   ncontlower_red               = ",ncontlower_red)
			print("   ncontupper_red               = ",ncontupper_red)
			print("   nminorabslower_red           = ",nminorabslower_red)
			print("   nminorabsupper_red           = ",nminorabsupper_red)
		
		##########################################################################################
		#
		# Copy data from k-distribution file into c-types...
		#
		##########################################################################################
		#
		# Dimensions/scalars
		#
		c_strlen              = ffi.new("int *",    strlen)
		c_ngas_req            = ffi.new("int *",    ngas_req) 
		c_ntemp               = ffi.new("int *",    ntemp) 
		c_nmixfrac            = ffi.new("int *",    nmixfrac) 
		c_ncontlower          = ffi.new("int *",    ncontlower)
		c_ncontupper          = ffi.new("int *",    ncontupper)
		c_nmajorabs           = ffi.new("int *",    nmajorabs)
		c_nminorabs           = ffi.new("int *",    nminorabs)
		c_nminorabslower      = ffi.new("int *",    nminorabslower)
		c_nminorabsupper      = ffi.new("int *",    nminorabsupper)
		c_npressref           = ffi.new("int *",    npressref)
		c_npressiref          = ffi.new("int *",    npressiref)
		c_ngpt                = ffi.new("int *",    ngpt)
		c_nband               = ffi.new("int *",    nband)
		c_natmlayer           = ffi.new("int *",    natmlayer)
		c_npair               = ffi.new("int *",    npair)
		c_nabsorber_ext       = ffi.new("int *",    nabsorber_ext)
		c_nflavors            = ffi.new("int *",    0)
		c_press_ref_trop      = ffi.new("double *", press_ref_trop)
		c_press_ref_trop_log  = ffi.new("double *", press_ref_trop_log)
		c_temp_ref_min        = ffi.new("double *", temp_ref_min)
		c_temp_ref_max        = ffi.new("double *", temp_ref_max)
		c_press_ref_min       = ffi.new("double *", press_ref_min)
		c_press_ref_max       = ffi.new("double *", press_ref_max)
		c_press_ref_log_delta = ffi.new("double *", press_ref_log_delta)
		c_temp_ref_delta      = ffi.new("double *", temp_ref_delta)
		if (doLW):
			c_nfit_coeffs   = ffi.new("int *",    nfit_coeffs)
			c_ntemp_Planck  = ffi.new("int *",    ntemp_Planck)
			c_totplnk_delta = ffi.new("double *", totplnk_delta) 
		if (doSW):
			c_absorption_coefficient_ref_P = ffi.new("double *", absorption_coefficient_ref_P)
			c_absorption_coefficient_ref_T = ffi.new("double *", absorption_coefficient_ref_T)
			c_tsi_default                  = ffi.new("double *", tsi_default)
			c_mg_default                   = ffi.new("double *", mg_default)
			c_sb_default                   = ffi.new("double *", sb_default)
		#
		# Arrays
		# allocate c-type, initialize with data from xarray, reshape for F90 interface (if necessary)
		#
		c_kmajor = ffi.new("double [" +                                                          \
			str(ngpt)           + "][" +                                                     \
			str(nmixfrac)       + "][" +                                                     \
			str(npressiref)     + "][" +                                                     \
			str(ntemp)          + "]",                                                       \
			(np.reshape(kdist.kmajor.values,[ngpt,nmixfrac,npressiref,ntemp])).tolist())
		c_kminor_lower = ffi.new("double [" +                                                    \
			str(ncontlower)     + "][" +                                                     \
			str(nmixfrac)       + "][" +                                                     \
			str(ntemp)          + "]",                                                       \
			(np.reshape(kdist.kminor_lower.values,[ncontlower,nmixfrac,ntemp])).tolist())
		c_kminor_upper = ffi.new("double [" +                                                    \
			str(ncontupper)     + "][" +                                                     \
			str(nmixfrac)       + "][" +                                                     \
			str(ntemp)          + "]",                                                       \
			(np.reshape(kdist.kminor_upper.values,[ncontupper,nmixfrac,ntemp])).tolist())
		c_key_species  = ffi.new("int ["   +                                                     \
			str(npair)          + "][" +                                                     \
			str(natmlayer)      + "][" +                                                     \
			str(nband)          + "]",                                                       \
			(np.reshape(kdist.key_species.values,[npair,natmlayer,nband])).tolist())
		c_vmr_ref = ffi.new("double [" +                                                         \
			str(natmlayer)      + "][" +                                                     \
			str(nabsorber_ext)  + "][" +                                                     \
			str(ntemp)          + "]",                                                       \
			(np.reshape(kdist.vmr_ref.values,[natmlayer,nabsorber_ext,ntemp])).tolist())
		c_minor_limits_gpt_lower          = ffi.new("int [2][" + str(nminorabslower) + "]",      \
			(np.reshape(kdist.minor_limits_gpt_lower.values,[2,nminorabslower])).tolist())
		c_minor_scales_with_density_lower = ffi.new("int ["    + str(nminorabslower) + "]",      \
			(kdist.minor_scales_with_density_lower.values).tolist())
		c_scale_by_complement_lower       = ffi.new("int ["    + str(nminorabslower) + "]",      \
			(kdist.scale_by_complement_lower.values).tolist())
		c_kminor_start_lower              = ffi.new("int ["    + str(nminorabslower) + "]",      \
			(kdist.kminor_start_lower.values).tolist())
		c_minor_limits_gpt_upper          = ffi.new("int [2][" + str(nminorabsupper) + "]",      \
			(np.reshape(kdist.minor_limits_gpt_upper.values,[2,nminorabsupper])).tolist())
		c_minor_scales_with_density_upper = ffi.new("int ["    + str(nminorabsupper) + "]",      \
			(kdist.minor_scales_with_density_upper.values).tolist())
		c_scale_by_complement_upper       = ffi.new("int ["    + str(nminorabsupper) + "]",      \
			(kdist.scale_by_complement_upper.values).tolist())
		c_kminor_start_upper              = ffi.new("int ["    + str(nminorabsupper) + "]",      \
			(kdist.kminor_start_upper.values).tolist())
		c_press_ref                       = ffi.new("double [" + str(npressref)      + "]",      \
			(kdist.press_ref.values).tolist())
		c_temp_ref                        = ffi.new("double [" + str(ntemp)          + "]",      \
			(kdist.temp_ref.values).tolist())
		c_press_ref_log                   = ffi.new("double [" + str(npressref)      + "]",      \
			(np.log(kdist.press_ref.values)).tolist())
		c_gpt2band                        = ffi.new("int ["    + str(ngpt)           + "]",      \
			gpt2band.tolist())
		c_gas_is_present_major            = ffi.new("int ["    + str(nmajorabs)      + "]" ,     \
			gas_is_present_major.tolist())
		c_gas_is_present_lower            = ffi.new("int ["    + str(nminorabslower) + "]",      \
			gas_is_present_lower.tolist())
		c_gas_is_present_upper            = ffi.new("int ["    + str(nminorabsupper) + "]",      \
			gas_is_present_upper.tolist())
		c_vmr_ref_red                     = ffi.new("double [" +                                 \
			str(natmlayer)      + "][" +                                                     \
			str(ngas_req+1)     + "][" +                                                     \
			str(ntemp)          + "]",                                                       \
			vmr_ref_red.tolist())
		c_requested_gases                 = ffi.new("char []", requested_gases_as_bytes)
		c_gas_names_major                 = ffi.new("char []", gas_names_major_as_bytes)
		c_gas_minor                       = ffi.new("char []", gas_minor_as_bytes)
		c_identifier_minor                = ffi.new("char []", identifier_minor_as_bytes)
		c_minor_gases_lower               = ffi.new("char []", minor_gases_lower_as_bytes)
		c_scaling_gas_lower               = ffi.new("char []", scaling_gas_lower_as_bytes)
		c_minor_gases_upper               = ffi.new("char []", minor_gases_upper_as_bytes)
		c_scaling_gas_upper               = ffi.new("char []", scaling_gas_upper_as_bytes)
		
		# Planck function tables (LW only)
		if (doLW):
			c_totplnk = ffi.new("double [" +                                                 \
				str(ntemp_Planck)   + "][" +                                             \
				str(nband)          + "]",                                               \
				(np.reshape(kdist.totplnk.values,[ntemp_Planck,nband])).tolist())
			c_planck_frac       = ffi.new("double [" +                                       \
				str(ngpt)           + "][" +                                             \
				str(nmixfrac)       + "][" +                                             \
				str(npressiref)     + "][" +                                             \
				str(ntemp)          + "]",                                               \
				(np.reshape(kdist.plank_fraction.values,[ngpt,nmixfrac,npressiref,ntemp])).tolist())
			c_optimal_angle_fit = ffi.new("double [" +                                       \
				str(nfit_coeffs)    + "][" +                                             \
				str(nband)          + "]",                                               \
				(np.reshape(kdist.optimal_angle_fit.values,[nfit_coeffs,nband])).tolist())
		
		# Rayleigh scattering contributions tables (SW only)
		if (doSW):
			c_rayl_lower = ffi.new("double [" +                                              \
				str(ngpt)           + "][" +                                             \
				str(nmixfrac)       + "][" +                                             \
				str(ntemp)          + "]",                                               \
				(np.reshape(kdist.rayl_lower.values,[ngpt,nmixfrac,ntemp])).tolist())
			c_rayl_upper = ffi.new("double [" +                                              \
				str(ngpt)           + "][" +                                             \
				str(nmixfrac)       + "][" +                                             \
				str(ntemp)          + "]",                                               \
				(np.reshape(kdist.rayl_upper.values,[ngpt,nmixfrac,ntemp])).tolist())
			c_solar_source_facular = ffi.new("double [" + str(ngpt)+ "]",                    \
				(kdist.solar_source_facular.values).tolist())
			c_solar_source_sunspot = ffi.new("double [" + str(ngpt)+ "]",                    \
				(kdist.solar_source_sunspot.values).tolist())
			c_solar_source_quiet = ffi.new("double ["    + str(ngpt)+ "]",                   \
				(kdist.solar_source_quiet.values).tolist())
		
		#
		# Define c-types for arrays that will be "reduced" to only the requested_gases
		#  - The gas array reduction uses the external-library "lib_init"
		# 
		c_key_species_red = ffi.new("int [" +                                                    \
			str(npair)          + "][" +                                                     \
			str(natmlayer)      + "][" +                                                     \
			str(nband)          + "]")
		c_kminor_lower_red = ffi.new("double [" +                                                \
			str(ncontlower_red) + "][" +                                                     \
			str(nmixfrac)       + "][" +                                                     \
			str(ntemp)          + "]")
		c_kminor_upper_red = ffi.new("double [" +                                                \
			str(ncontupper_red) + "][" +                                                     \
			str(nmixfrac)       + "][" +                                                     \
			str(ntemp)          + "]")
		c_key_species_list = ffi.new("int ["    +                                                \
			str(npair)          + "][" +                                                     \
			str(nband*2)         +"]")
		c_minor_limits_gpt_lower_red          = ffi.new("int [2][" + str(nminorabslower_red) +"]") 
		c_minor_limits_gpt_upper_red          = ffi.new("int [2][" + str(nminorabsupper_red) +"]") 
		c_gpoint_flavor                       = ffi.new("int [2][" + str(ngpt)               +"]")
		c_kminor_start_lower_red              = ffi.new("int ["    + str(nminorabslower_red) +"]")
		c_kminor_start_upper_red              = ffi.new("int ["    + str(nminorabsupper_red) +"]")
		c_minor_scales_with_density_lower_red = ffi.new("int ["    + str(nminorabslower_red) +"]")
		c_minor_scales_with_density_upper_red = ffi.new("int ["    + str(nminorabsupper_red) +"]")
		c_scale_by_complement_lower_red       = ffi.new("int ["    + str(nminorabslower_red) +"]")
		c_scale_by_complement_upper_red       = ffi.new("int ["    + str(nminorabsupper_red) +"]")
		c_idx_minor_lower                     = ffi.new("int ["    + str(nminorabslower_red) +"]")
		c_idx_minor_upper                     = ffi.new("int ["    + str(nminorabsupper_red) +"]")
		c_idx_minor_scaling_lower             = ffi.new("int ["    + str(nminorabslower_red) +"]")
		c_idx_minor_scaling_upper             = ffi.new("int ["    + str(nminorabsupper_red) +"]")
		c_key_species_present_init            = ffi.new("int ["    + str(nmajorabs)          +"]")
		c_is_key                              = ffi.new("int ["    + str(nmajorabs)          +"]")
		c_minor_gases_lower_red               = ffi.new("char []", strlen*nminorabslower_red)
		c_minor_gases_upper_red               = ffi.new("char []", strlen*nminorabsupper_red)
		c_scaling_gas_lower_red               = ffi.new("char []", strlen*nminorabslower_red)
		c_scaling_gas_upper_red               = ffi.new("char []", strlen*nminorabsupper_red)
		c_ncontlower_red                      = ffi.new("int *",   ncontlower_red)
		c_ncontupper_red                      = ffi.new("int *",   ncontupper_red)
		c_nminorabslower_red                  = ffi.new("int *",   nminorabslower_red)
		c_nminorabsupper_red                  = ffi.new("int *",   nminorabsupper_red)
		
		##########################################################################################
		#
		# Call F90 routines ....
		#
		##########################################################################################
		cdef_reduce_minor_arrays(ffi, strlen, ngas_req, nminorabslower, nminorabs, ncontlower,   \
                                         ntemp, nmixfrac, ncontlower_red, nminorabslower_red)
		lib_init.reduce_minor_arrays(c_strlen,c_ngas_req, c_ntemp, c_nmixfrac, c_ncontlower,     \
                                             c_nminorabs, c_nminorabslower, c_ncontlower_red,            \
                                             c_nminorabslower_red, c_kminor_lower, c_requested_gases,    \
                                             c_gas_minor, c_identifier_minor, c_minor_gases_lower,       \
                                             c_minor_limits_gpt_lower, c_gas_is_present_lower,           \
                                             c_minor_scales_with_density_lower,                          \
                                             c_scaling_gas_lower, c_scale_by_complement_lower,           \
                                             c_kminor_start_lower, c_kminor_lower_red,                   \
                                             c_minor_gases_lower_red, c_minor_limits_gpt_lower_red,      \
                                             c_minor_scales_with_density_lower_red,                      \
                                             c_scaling_gas_lower_red, c_scale_by_complement_lower_red,   \
                                             c_kminor_start_lower_red)
									 
		cdef_reduce_minor_arrays(ffi, strlen, ngas_req, nminorabsupper, nminorabs, ncontupper,   \
                                         ntemp, nmixfrac, ncontupper_red, nminorabsupper_red)
		lib_init.reduce_minor_arrays(c_strlen,c_ngas_req, c_ntemp, c_nmixfrac, c_ncontupper,     \
                                             c_nminorabs, c_nminorabsupper, c_ncontupper_red,            \
                                             c_nminorabsupper_red, c_kminor_upper, c_requested_gases,    \
                                             c_gas_minor, c_identifier_minor, c_minor_gases_upper,       \
                                             c_minor_limits_gpt_upper, c_gas_is_present_upper,           \
                                             c_minor_scales_with_density_upper,                          \
                                             c_scaling_gas_upper, c_scale_by_complement_upper,           \
                                             c_kminor_start_upper, c_kminor_upper_red,                   \
                                             c_minor_gases_upper_red, c_minor_limits_gpt_upper_red,      \
                                             c_minor_scales_with_density_upper_red,                      \
                                             c_scaling_gas_upper_red, c_scale_by_complement_upper_red,   \
                                             c_kminor_start_upper_red)
		
		cdef_create_idx_minor(ffi, strlen, ngas_req, nminorabs, nminorabslower_red)
		lib_init.create_idx_minor(c_strlen, c_ngas_req, c_nminorabs, c_nminorabslower_red,       \
                                          c_requested_gases, c_gas_minor, c_identifier_minor,            \
                                          c_minor_gases_lower_red, c_idx_minor_lower)
                
		cdef_create_idx_minor(ffi, strlen, ngas_req, nminorabs, nminorabsupper_red)
		lib_init.create_idx_minor(c_strlen, c_ngas_req, c_nminorabs, c_nminorabsupper_red,       \
                                          c_requested_gases, c_gas_minor, c_identifier_minor,            \
                                          c_minor_gases_upper_red, c_idx_minor_upper)
		
		cdef_create_idx_minor_scaling(ffi, strlen, ngas_req, nminorabslower_red)
		lib_init.create_idx_minor_scaling(c_strlen, c_ngas_req, c_nminorabslower_red,            \
                                                  c_requested_gases, c_scaling_gas_lower_red,            \
                                                  c_idx_minor_scaling_lower)

		cdef_create_idx_minor_scaling(ffi, strlen, ngas_req, nminorabsupper_red)
		lib_init.create_idx_minor_scaling(c_strlen, c_ngas_req, c_nminorabsupper_red,            \
                                                  c_requested_gases, c_scaling_gas_upper_red,            \
                                                  c_idx_minor_scaling_upper)

		cdef_create_key_species_reduce(ffi, strlen, nmajorabs, ngas_req, npair, natmlayer, nband)
		lib_init.create_key_species_reduce(c_strlen, c_nband, c_natmlayer, c_npair, c_nmajorabs, \
                                                   c_ngas_req, c_gas_names_major, c_requested_gases,     \
                                                   c_key_species, c_key_species_red,                     \
                                                   c_key_species_present_init)
		
		cdef_get_nflavors(ffi, npair, natmlayer, nband)
		lib_init.get_nflavors(c_npair, c_natmlayer, c_nband, c_key_species_red,                  \
                                      c_key_species_list, c_nflavors)
		
		c_flavors = ffi.new("int [" + str(npair) + "][" + str(c_nflavors[0]) + "]")
		cdef_create_flavor(ffi, npair, nband, c_nflavors[0], nmajorabs)
		lib_init.create_flavor(c_nband, c_npair, c_nflavors, c_nmajorabs, c_key_species_list,    \
                                       c_flavors, c_is_key)

		cdef_create_gpoint_flavor(ffi, ngpt, nband, npair, natmlayer, c_nflavors[0])
		lib_init.create_gpoint_flavor(c_npair, c_natmlayer, c_nband, c_ngpt, c_nflavors,         \
                                              c_key_species_red, c_flavors, c_gpt2band, c_gpoint_flavor)

		##########################################################################################
		#
		# Create structure containing all k-distribution data
		#
		##########################################################################################
		kdistOUT = {'c_strlen': c_strlen,                                                        \
                            'c_ngas_req': c_ngas_req,                                                    \
                            'c_ntemp': c_ntemp,                                                          \
                            'c_nmixfrac': c_nmixfrac,                                                    \
                            'c_ncontlower': c_ncontlower,                                                \
                            'c_ncontupper': c_ncontupper,                                                \
                            'c_ncontupper': c_ncontupper,                                                \
                            'c_nminorabs': c_nminorabs,                                                  \
                            'c_nminorabslower': c_nminorabslower,                                        \
                            'c_nminorabsupper': c_nminorabsupper,                                        \
                            'c_npressref': c_npressref,                                                  \
                            'c_npressiref': c_npressiref,                                                \
                            'c_ngpt': c_ngpt,                                                            \
                            'c_nband': c_nband,                                                          \
                            'c_natmlayer': c_natmlayer,                                                  \
                            'c_npair': c_npair,                                                          \
                            'c_nabsorber_ext': c_nabsorber_ext,                                          \
                            'c_nflavors': c_nflavors,                                                    \
                            'c_press_ref_trop': c_press_ref_trop,                                        \
                            'c_press_ref_trop_log': c_press_ref_trop_log,                                \
                            'c_temp_ref_min': c_temp_ref_min,                                            \
                            'c_temp_ref_max': c_temp_ref_max,                                            \
                            'c_press_ref_min': c_press_ref_min,                                          \
                            'c_press_ref_max': c_press_ref_max,                                          \
                            'c_press_ref_log_delta': c_press_ref_log_delta,                              \
                            'c_temp_ref_delta': c_temp_ref_delta,                                        \
                            'c_kmajor': c_kmajor,                                                        \
                            'c_kminor_lower': c_kminor_lower,                                            \
                            'c_kminor_upper': c_kminor_upper,                                            \
                            'c_key_species': c_key_species,                                              \
                            'c_vmr_ref': c_vmr_ref,                                                      \
                            'c_minor_limits_gpt_lower': c_minor_limits_gpt_lower,                        \
                            'c_minor_scales_with_density_lower': c_minor_scales_with_density_lower,      \
                            'c_scale_by_complement_lower': c_scale_by_complement_lower,                  \
                            'c_kminor_start_lower': c_kminor_start_lower,                                \
                            'c_minor_limits_gpt_upper': c_minor_limits_gpt_upper,                        \
                            'c_minor_scales_with_density_upper': c_minor_scales_with_density_upper,      \
                            'c_scale_by_complement_upper': c_scale_by_complement_upper,                  \
                            'c_kminor_start_upper': c_kminor_start_upper,                                \
                            'c_press_ref': c_press_ref,                                                  \
                            'c_temp_ref': c_temp_ref,                                                    \
                            'c_press_ref_log': c_press_ref_log,                                          \
                            'c_gpt2band': c_gpt2band,                                                    \
                            'c_gas_is_present_major': c_gas_is_present_major,                            \
                            'c_gas_is_present_lower': c_gas_is_present_lower,                            \
                            'c_gas_is_present_upper': c_gas_is_present_upper,                            \
                            'c_vmr_ref_red': c_vmr_ref_red,                                              \
                            'c_requested_gases': c_requested_gases,                                      \
                            'c_gas_names_major': c_gas_names_major,                                      \
                            'c_gas_minor': c_gas_minor,                                                  \
                            'c_identifier_minor': c_identifier_minor,                                    \
                            'c_minor_gases_lower': c_minor_gases_lower,                                  \
                            'c_scaling_gas_lower': c_scaling_gas_lower,                                  \
                            'c_minor_gases_upper': c_minor_gases_upper,                                  \
                            'c_scaling_gas_upper': c_scaling_gas_upper,                                  \
                            'c_key_species_red': c_key_species_red,                                      \
                            'c_kminor_lower_red': c_kminor_lower_red,                                    \
                            'c_kminor_upper_red': c_kminor_upper_red,                                    \
                            'c_key_species_list': c_key_species_list,                                    \
                            'c_minor_limits_gpt_lower_red': c_minor_limits_gpt_lower_red,                \
                            'c_minor_limits_gpt_upper_red': c_minor_limits_gpt_upper_red,                \
                            'c_gpoint_flavor': c_gpoint_flavor,                                          \
                            'c_kminor_start_lower_red': c_kminor_start_lower_red,                        \
                            'c_kminor_start_upper_red': c_kminor_start_upper_red,                        \
                            'c_minor_scales_with_density_lower_red': c_minor_scales_with_density_lower_red,\
                            'c_minor_scales_with_density_upper_red': c_minor_scales_with_density_upper_red,\
                            'c_scale_by_complement_lower_red': c_scale_by_complement_lower_red,          \
                            'c_scale_by_complement_upper_red': c_scale_by_complement_upper_red,          \
                            'c_idx_minor_lower': c_idx_minor_lower,                                      \
                            'c_idx_minor_upper': c_idx_minor_upper,                                      \
                            'c_idx_minor_scaling_lower': c_idx_minor_scaling_lower,                      \
                            'c_idx_minor_scaling_upper': c_idx_minor_scaling_upper,                      \
                            'c_key_species_present_init': c_key_species_present_init,                    \
                            'c_is_key': c_is_key,                                                        \
                            'c_minor_gases_lower_red': c_minor_gases_lower_red,                          \
                            'c_minor_gases_upper_red': c_minor_gases_upper_red,                          \
                            'c_scaling_gas_lower_red': c_scaling_gas_lower_red,                          \
                            'c_scaling_gas_upper_red': c_scaling_gas_upper_red,                          \
                            'c_ncontlower_red': c_ncontlower_red,                                        \
                            'c_ncontupper_red': c_ncontupper_red,                                        \
                            'c_nminorabslower_red': c_nminorabslower_red,                                \
                            'c_nminorabsupper_red': c_nminorabsupper_red,                                \
                            'c_flavors': c_flavors}
		if (doLW):
			kdistOUT['c_nfit_coeffs']                  = c_nfit_coeffs
			kdistOUT['c_ntemp_Planck']                 = c_ntemp_Planck
			kdistOUT['c_totplnk_delta']                = c_totplnk_delta
			kdistOUT['c_totplnk']                      = c_totplnk
			kdistOUT['c_planck_frac']                  = c_planck_frac
			kdistOUT['c_optimal_angle_fit']            = c_optimal_angle_fit
		if (doSW):
			kdistOUT['c_absorption_coefficient_ref_P'] = c_absorption_coefficient_ref_P
			kdistOUT['c_absorption_coefficient_ref_T'] = c_absorption_coefficient_ref_T
			kdistOUT['c_tsi_default']                  = c_tsi_default
			kdistOUT['c_mg_default']                   = c_mg_default
			kdistOUT['c_sb_default']                   = c_sb_default
			kdistOUT['c_rayl_lower']                   = c_rayl_lower
			kdistOUT['c_rayl_upper']                   = c_rayl_upper
			kdistOUT['c_solar_source_facular']         = c_solar_source_facular
			kdistOUT['c_solar_source_sunspot']         = c_solar_source_sunspot
			kdistOUT['c_solar_source_quiet']           = c_solar_source_quiet
		return kdistOUT
