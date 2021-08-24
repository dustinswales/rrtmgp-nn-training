##########################################################################################
#! /usr/bin/env python
#
##########################################################################################
import sys
import os
import numpy as np
import xarray as xr

##########################################################################################
##########################################################################################
def string_loc_in_array(str2find, strarray):
	for istr in range(0,len(strarray)):
		# Force string-array into target string length. 
		if (str(strarray[istr]).ljust(len(str2find),' ') == str2find):
			strindex = istr
			break
		else:
			strindex = -1
	return strindex

##########################################################################################
##########################################################################################
def create_idx_minor(nminorabs_red, minor_gases_red, ngas_req, requested_gases,  \
	identifier_minor, nidentifier_minor, gas_minor):
	idx_minor = np.zeros(nminorabs_red,dtype=int) - 1
	for iminorgas in range(0,nminorabs_red):
		idx_mnr = -1
		if (minor_gases_red[iminorgas] in identifier_minor):
			idx_mnr              = string_loc_in_array(minor_gases_red[iminorgas], identifier_minor)
			idx_minor[iminorgas] = string_loc_in_array(gas_minor[idx_mnr],         requested_gases)
	return idx_minor
	
##########################################################################################
##########################################################################################
def create_idx_minor_scaling(nminorabs_red, scaling_gas_red, ngas_req, requested_gases):
	idx_minor_scaling = np.zeros(nminorabs_red,dtype=int) - 1
	for iminorgas in range(0,nminorabs_red):
		idx_minor_scaling[iminorgas] = string_loc_in_array(scaling_gas_red[iminorgas],   \
			requested_gases )
	return idx_minor_scaling
	
##########################################################################################
##########################################################################################
def reduce_minor(nminorabs, nminorabs_red, ncont_red, ntemp, nmixfrac, gas_is_present,   \
	minor_gases_atm, scaling_gas_atm, minor_scales_with_density_atm,                     \
	scale_by_complement_atm, kminor_start_atm, minor_limits_gpt_atm, kminor_atm):
	
	minor_gases_atm_red               = []
	scaling_gas_atm_red               = []
	minor_scales_with_density_atm_red = []
	scale_by_complement_atm_red       = []
	for igas in range(0,nminorabs):
		if (gas_is_present[igas]):
			minor_gases_atm_red.append(minor_gases_atm[igas])
			scaling_gas_atm_red.append(scaling_gas_atm[igas])
			minor_scales_with_density_atm_red.append(minor_scales_with_density_atm[igas])
			scale_by_complement_atm_red.append(scale_by_complement_atm[igas])
	if (nminorabs_red == nminorabs):
		kminor_start_atm_red     = kminor_start_atm
		minor_limits_gpt_atm_red = minor_limits_gpt_atm
		kminor_atm_red           = kminor_atm
	else:
		kminor_start_atm_red     = np.zeros(nminorabs_red,              dtype=int)
		minor_limits_gpt_atm_red = np.zeros((nminorabs_red, 2),         dtype=int)
		kminor_atm_red           = np.zeros((ntemp,nmixfrac,ncont_red), dtype=np.double)
		icnt   = -1
		n_elim = 0
		for x in range(0, nminorabs):
			ng = minor_limits_gpt_atm[x,1] - minor_limits_gpt_atm[x,0] + 1
			if (gas_is_present[x]):
				icnt = icnt + 1
				minor_limits_gpt_atm_red[icnt,:]= minor_limits_gpt_atm[x,:]
				kminor_start_atm_red[icnt]      = kminor_start_atm[x] - n_elim
				for ij in range(0,ng):
					kminor_atm_red[:,:,kminor_start_atm_red[icnt]+ij-1] = \
						kminor_atm[:,:,kminor_start_atm[x]+ij-1]
			else:
				n_elim = n_elim + ng
	return [minor_gases_atm_red, scaling_gas_atm_red, minor_scales_with_density_atm_red, \
		    scale_by_complement_atm_red, kminor_start_atm_red, minor_limits_gpt_atm_red, \
			kminor_atm_red];

##########################################################################################
##########################################################################################	
def search_for_gases(ngas1, gas1, ngas2, gas2, limits_gpt = None):
	ncontatm_red      = 0
	nminorabsatm_red  = 0
	gas_is_present    = np.empty([ngas1],dtype=bool)
	gas_is_present[:] = False
	count             = 0
	for igas1 in range(0,ngas1):
		for igas2 in range(0,ngas2):
			if gas2[igas2] in gas1[igas1]:
				gas_is_present[igas1] = True
				count = count + 1
				if (np.any(limits_gpt)):
					ncontatm_red     = ncontatm_red + (limits_gpt[igas1,1]-limits_gpt[igas1,0]+1)
					nminorabsatm_red = nminorabsatm_red + 1
				break
			if (count == ngas2): break
	return [ncontatm_red, nminorabsatm_red, gas_is_present];

##########################################################################################
##########################################################################################
def load_kdist_noF90(ffi, file_kdist, gases, print_info, output_ctypes):

		# Load k-distribution data
		kdist = xr.open_dataset(file_kdist,concat_characters=True,decode_cf=True)		

		# Longwave or Shortwave file?
		kdist_keys = list(kdist.keys())
		doLW = 'plank_fraction' in list(kdist.keys())
		doSW = 'rayl_lower'     in list(kdist.keys())

		# Dimensions (flat)
		ngas_req            = len(gases)
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
		
		##################################################################################
		#
		# Using the "requested_gases", reduce k-distribution data to only the gases needed
		#
		##################################################################################

		# Local string arrays (Xarray strings are encoded, decoded below, only used by Py)
		requested_gases   = []
		gas_names         = []
		gas_minor         = []
		identifier_minor  = []
		minor_gases_lower = []
		scaling_gas_lower = []
		minor_gases_upper = []
		scaling_gas_upper = []
		for igas in range(0,ngas_req):
			requested_gases.append(gases[igas])
		for igas in range(0,nmajorabs):
			gas_names.append(kdist.gas_names.values[igas].decode())
		for igas in range(0,nminorabs):
			gas_minor.append(kdist.gas_minor.values[igas].decode())
			identifier_minor.append(kdist.identifier_minor.values[igas].decode())
		for igas in range(0,nminorabslower):
			minor_gases_lower.append(kdist.minor_gases_lower.values[igas].decode())
			scaling_gas_lower.append(kdist.scaling_gas_lower.values[igas].decode())
		for igas in range(0,nminorabsupper): 
			minor_gases_upper.append(kdist.minor_gases_upper.values[igas].decode())
			scaling_gas_upper.append(kdist.scaling_gas_upper.values[igas].decode())

		#
		# Which major absorbing gases to include?
		#
		majorgas = search_for_gases(nmajorabs, gas_names, ngas_req, requested_gases)
		gas_is_present_major = majorgas[2]				
				
		#
		# Which minor absorbing gases, in lower reference atmosphere, to include?
		#
		minorgas_lower = search_for_gases(nminorabslower, minor_gases_lower, ngas_req,   \
			requested_gases, limits_gpt = kdist.minor_limits_gpt_lower.values)
		ncontlower_red       = minorgas_lower[0]
		nminorabslower_red   = minorgas_lower[1]
		gas_is_present_lower = minorgas_lower[2]

		#
		# Which minor absorbing gases, in upper reference atmosphere, to include?
		#
		minorgas_upper = search_for_gases(nminorabsupper, minor_gases_upper, ngas_req,   \
			requested_gases, limits_gpt = kdist.minor_limits_gpt_upper.values)
		ncontupper_red       = minorgas_upper[0]
		nminorabsupper_red   = minorgas_upper[1]
		gas_is_present_upper = minorgas_upper[2]
		
		#
		# Reduce volume mixing ratios for reference atmosphere...
		#  - Gas 0 is used in single-key species method, set to 1.0 (col_dry)
		#
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
		
		if (print_info):
			print("Dimensions (OUT/reduced): ")
			print("   ncontlower_red               = ",ncontlower_red)
			print("   ncontupper_red               = ",ncontupper_red)
			print("   nminorabslower_red           = ",nminorabslower_red)
			print("   nminorabsupper_red           = ",nminorabsupper_red)

		#
        # Call reduce minor (lower)
        #
		reduce_minor_lower = reduce_minor(nminorabslower, nminorabslower_red,            \
			ncontlower_red, ntemp, nmixfrac, gas_is_present_lower,                       \
			minor_gases_lower, scaling_gas_lower,                                        \
			kdist.minor_scales_with_density_lower.values,                                \
			kdist.scale_by_complement_lower.values, kdist.kminor_start_lower.values,     \
			kdist.minor_limits_gpt_lower.values, kdist.kminor_lower.values)
		minor_gases_lower_red               = reduce_minor_lower[0]
		scaling_gas_lower_red               = reduce_minor_lower[1]
		minor_scales_with_density_lower_red = reduce_minor_lower[2]
		scale_by_complement_lower_red       = reduce_minor_lower[3]
		kminor_start_lower_red              = reduce_minor_lower[4]
		minor_limits_gpt_lower_red          = reduce_minor_lower[5]
		kminor_lower_red                    = reduce_minor_lower[6]
					           
		#
        # Call reduce minor (upper)
        #
		reduce_minor_upper = reduce_minor(nminorabsupper, nminorabsupper_red,            \
			ncontupper_red, ntemp, nmixfrac, gas_is_present_upper,                       \
			minor_gases_upper, scaling_gas_upper,                                        \
			kdist.minor_scales_with_density_upper.values,                                \
			kdist.scale_by_complement_upper.values, kdist.kminor_start_upper.values,     \
			kdist.minor_limits_gpt_upper.values, kdist.kminor_upper.values)
		minor_gases_upper_red               = reduce_minor_upper[0]
		scaling_gas_upper_red               = reduce_minor_upper[1]
		minor_scales_with_density_upper_red = reduce_minor_upper[2]
		scale_by_complement_upper_red       = reduce_minor_upper[3]
		kminor_start_upper_red              = reduce_minor_upper[4]
		minor_limits_gpt_upper_red          = reduce_minor_upper[5]
		kminor_upper_red                    = reduce_minor_upper[6]

		#
		# Call create_idx_minor
		#
		idx_minor_lower = create_idx_minor(nminorabslower_red, minor_gases_lower_red,    \
			ngas_req, requested_gases, identifier_minor, nminorabs, gas_minor)

		idx_minor_upper = create_idx_minor(nminorabsupper_red, minor_gases_upper_red,    \
			ngas_req, requested_gases,  identifier_minor, nminorabs, gas_minor)
		#print("idx_minor_lower: ",idx_minor_lower)
		#print("idx_minor_upper: ",idx_minor_upper)
		
		#
		# Call create_idx_minor_scaling
		#
		idx_minor_scaling_lower = create_idx_minor_scaling(nminorabslower_red,           \
			scaling_gas_lower_red, ngas_req, requested_gases)

		idx_minor_scaling_upper = create_idx_minor_scaling(nminorabsupper_red,           \
			scaling_gas_upper_red, ngas_req, requested_gases)
		#print("idx_minor_scaling_lower: ",idx_minor_scaling_lower)
		#print("idx_minor_scaling_upper: ",idx_minor_scaling_upper)
		
		#
		# Reduce "key_species"
		#
		key_species_present_init    = np.empty(nmajorabs,              dtype=bool)
		key_species_red             = np.empty((nband,natmlayer,npair),dtype=int)
		key_species_present_init[:] = True
		key_species_red[:,:,:]      = 0
		for ipair in range(0,npair):
			for iatm in range(0,natmlayer):
				for iband in range(0,nband):
					ki = kdist.key_species.values[iband,iatm,ipair]
					if (ki != 0):
						key_species_red[iband,iatm,ipair] = string_loc_in_array(  \
							gas_names[ki-1], requested_gases)+1
						if (key_species_red[iband,iatm,ipair] == -1): 
							key_species_present_init[ki] = False
					else:
						key_species_red[iband,iatm,ipair] = kdist.key_species.values[iband,iatm,ipair]

		#
		# Create "flavor" list. flavor is index into gas_names
		#
		key_species_list = np.empty((nband*2,npair),dtype=int)
		flavor_count     = 0
		for iband in range(0,nband):
			for iatm in range(0,natmlayer):
				key_species_list[flavor_count,:] = key_species_red[iband,iatm,:]
				if (key_species_red[iband,iatm,0]==key_species_red[iband,iatm,1]==0):
					key_species_list[flavor_count,0] = 2
					key_species_list[flavor_count,1] = 2
				flavor_count = flavor_count + 1

		# How many unique "flavors" in list? only keep those.
		is_unique    = np.empty((nband*2),dtype=bool)
		is_unique[:] = False
		u1           = key_species_list[0,0]
		u2           = key_species_list[0,1]
		is_unique[0] = True		
		for iflav in range(0,flavor_count):
			new_flavor = False
			xii = key_species_list[iflav,0] == u1
			yii = key_species_list[iflav,1] == u2
			xi = np.any(xii)
			yi = np.any(yii)
			if (not xi):
				new_flavor = True			
			else:
				if (not yi):
					new_flavor = True
			if (yi and xi):
				if (not np.any(yii[xii])):
					new_flavor = True
			if (new_flavor):
				u1 = np.append(u1,key_species_list[iflav,0])
				u2 = np.append(u2,key_species_list[iflav,1])
				is_unique[iflav] = True		
		flavor = np.stack((u1,u2))
		nunique_flavors = np.sum(is_unique)		

		#
		# Create "gpoint flavor" list
		#
		gpoint_flavor = np.ones((natmlayer,ngpt),dtype=int)
		for igpt in range(0,ngpt):
			for iatm in range(0,natmlayer):
				temparry = key_species_red[gpt2band[igpt]-1,iatm,:]
				if (np.all(temparry==0)): temparry=[2,2] 
				for iflav in range(0,nunique_flavors):
					if (temparry[0] == flavor[0,iflav] and temparry[1] == flavor[1,iflav]):
						gpoint_flavor[iatm,igpt] = iflav+1
						break

		#
		# Which species are key in one or more bands?
		#
		is_key    = np.empty(ngas_req, dtype=bool)
		is_key[:] = False
		for iflav in range(0,nunique_flavors):
			for iatm in range(0,natmlayer):
				if (flavor[iatm,iflav] != 0): is_key[flavor[iatm,iflav]-1] = True

		##################################################################################
		#
		# Copy data from k-distribution file into c-types...
		#
		##################################################################################
		#
		# Dimensions/scalars
		#
		c_ngas_req            = ffi.new("int *",    ngas_req) 
		c_ntempref            = ffi.new("int *",    ntemp) 
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
		c_nflavors            = ffi.new("int *",    nunique_flavors)
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
			str(ngpt)           + "][" +                                                         \
			str(nmixfrac)       + "][" +                                                         \
			str(npressiref)     + "][" +                                                         \
			str(ntemp)          + "]",                                                           \
			(np.reshape(kdist.kmajor.values,[ngpt,nmixfrac,npressiref,ntemp])).tolist())
		c_kminor_lower = ffi.new("double [" +                                                    \
			str(ncontlower)     + "][" +                                                         \
			str(nmixfrac)       + "][" +                                                         \
			str(ntemp)          + "]",                                                           \
			(np.reshape(kdist.kminor_lower.values,[ncontlower,nmixfrac,ntemp])).tolist())
		c_kminor_upper = ffi.new("double [" +                                                    \
			str(ncontupper)     + "][" +                                                         \
			str(nmixfrac)       + "][" +                                                         \
			str(ntemp)          + "]",                                                           \
			(np.reshape(kdist.kminor_upper.values,[ncontupper,nmixfrac,ntemp])).tolist())
		c_key_species  = ffi.new("int ["   +                                                     \
			str(npair)          + "][" +                                                         \
			str(natmlayer)      + "][" +                                                         \
			str(nband)          + "]",                                                           \
			(np.reshape(kdist.key_species.values,[npair,natmlayer,nband])).tolist())
		c_vmr_ref = ffi.new("double [" +                                                         \
			str(natmlayer)      + "][" +                                                         \
			str(nabsorber_ext)  + "][" +                                                         \
			str(ntemp)          + "]",                                                           \
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
			str(natmlayer)      + "][" +                                                         \
			str(ngas_req+1)     + "][" +                                                         \
			str(ntemp)          + "]",                                                           \
			vmr_ref_red.tolist())
		# Planck function tables (LW only)
		if (doLW):
			c_totplnk = ffi.new("double [" +                                                     \
				str(ntemp_Planck)   + "][" +                                                     \
				str(nband)          + "]",                                                       \
				(np.reshape(kdist.totplnk.values,[ntemp_Planck,nband])).tolist())
			c_planck_frac       = ffi.new("double [" +                                           \
				str(ngpt)           + "][" +                                                     \
				str(nmixfrac)       + "][" +                                                     \
				str(npressiref)     + "][" +                                                     \
				str(ntemp)          + "]",                                                       \
				(np.reshape(kdist.plank_fraction.values,[ngpt,nmixfrac,npressiref,ntemp])).tolist())
			c_optimal_angle_fit = ffi.new("double [" +                                           \
				str(nfit_coeffs)    + "][" +                                                     \
				str(nband)          + "]",                                                       \
				(np.reshape(kdist.optimal_angle_fit.values,[nfit_coeffs,nband])).tolist())
		
		# Rayleigh scattering contributions tables (SW only)
		if (doSW):
			c_rayl_lower = ffi.new("double [" +                                                  \
				str(ngpt)           + "][" +                                                     \
				str(nmixfrac)       + "][" +                                                     \
				str(ntemp)          + "]",                                                       \
				(np.reshape(kdist.rayl_lower.values,[ngpt,nmixfrac,ntemp])).tolist())
			c_rayl_upper = ffi.new("double [" +                                                  \
				str(ngpt)           + "][" +                                                     \
				str(nmixfrac)       + "][" +                                                     \
				str(ntemp)          + "]",                                                       \
				(np.reshape(kdist.rayl_upper.values,[ngpt,nmixfrac,ntemp])).tolist())
			c_solar_source_facular = ffi.new("double [" + str(ngpt)+ "]",                        \
				(kdist.solar_source_facular.values).tolist())
			c_solar_source_sunspot = ffi.new("double [" + str(ngpt)+ "]",                        \
				(kdist.solar_source_sunspot.values).tolist())
			c_solar_source_quiet = ffi.new("double ["    + str(ngpt)+ "]",                       \
				(kdist.solar_source_quiet.values).tolist())
		
		#
		# Define c-types for arrays that will be "reduced" to only the requested_gases
		# 
		c_key_species_red = ffi.new("int [" +                                                    \
			str(npair)          + "][" +                                                         \
			str(natmlayer)      + "][" +                                                         \
			str(nband)          + "]", \
		(np.reshape(key_species_red,[npair,natmlayer,nband])).tolist())
		c_kminor_lower_red = ffi.new("double [" +                                                \
			str(ncontlower_red) + "][" +                                                         \
			str(nmixfrac)       + "][" +                                                         \
			str(ntemp)          + "]", \
			(np.reshape(kminor_lower_red,[ncontlower_red,nmixfrac,ntemp])).tolist())			
		c_kminor_upper_red = ffi.new("double [" +                                                \
			str(ncontupper_red) + "][" +                                                         \
			str(nmixfrac)       + "][" +                                                         \
			str(ntemp)          + "]",\
			(np.reshape(kminor_upper_red,[ncontupper_red,nmixfrac,ntemp])).tolist())			
		c_key_species_list = ffi.new("int ["    +                                                \
			str(npair)          + "][" +                                                         \
			str(nband*2)         +"]",\
			(np.reshape(key_species_list,[npair,nband*2])).tolist())
		c_minor_limits_gpt_lower_red          = ffi.new("int [2][" + str(nminorabslower_red) +"]",\
			(np.reshape(minor_limits_gpt_lower_red,[2,nminorabslower_red])).tolist())		
		c_minor_limits_gpt_upper_red          = ffi.new("int [2][" + str(nminorabsupper_red) +"]",\
			(np.reshape(minor_limits_gpt_upper_red,[2,nminorabsupper_red])).tolist())		
		c_flavors                             = ffi.new("int [2][" + str(nunique_flavors)    +"]",\
			flavor.tolist())
		c_gpoint_flavor                       = ffi.new("int [2][" + str(ngpt)               +"]",\
			gpoint_flavor.tolist())			
		c_kminor_start_lower_red              = ffi.new("int ["    + str(nminorabslower_red) +"]",\
			kminor_start_lower_red.tolist())
		c_kminor_start_upper_red              = ffi.new("int ["    + str(nminorabsupper_red) +"]",\
			kminor_start_upper_red.tolist())
		c_idx_minor_lower                     = ffi.new("int ["    + str(nminorabslower_red) +"]",\
			idx_minor_lower.tolist())
		c_idx_minor_upper                     = ffi.new("int ["    + str(nminorabsupper_red) +"]",\
			idx_minor_upper.tolist())
		c_idx_minor_scaling_lower             = ffi.new("int ["    + str(nminorabslower_red) +"]",\
			idx_minor_scaling_lower.tolist())
		c_idx_minor_scaling_upper             = ffi.new("int ["    + str(nminorabsupper_red) +"]",\
			idx_minor_scaling_upper.tolist())
		c_key_species_present_init            = ffi.new("int ["    + str(nmajorabs)          +"]",\
			key_species_present_init.tolist())
		c_minor_scales_with_density_lower_red = ffi.new("int ["    + str(nminorabslower_red) +"]",\
			(minor_scales_with_density_lower_red[:]))
		c_minor_scales_with_density_upper_red = ffi.new("int ["    + str(nminorabsupper_red) +"]",\
			(minor_scales_with_density_upper_red[:]))
		c_scale_by_complement_lower_red       = ffi.new("int ["    + str(nminorabslower_red) +"]",\
			(scale_by_complement_lower_red[:]))
		c_scale_by_complement_upper_red       = ffi.new("int ["    + str(nminorabsupper_red) +"]",\
			(scale_by_complement_upper_red[:]))
		c_ncontlower_red                      = ffi.new("int *",   ncontlower_red)
		c_ncontupper_red                      = ffi.new("int *",   ncontupper_red)
		c_nminorabslower_red                  = ffi.new("int *",   nminorabslower_red)
		c_nminorabsupper_red                  = ffi.new("int *",   nminorabsupper_red)
		c_is_key                              = ffi.new("int ["    + str(ngas_req)          +"]",\
			is_key.tolist())

		##########################################################################################
		#
		# Create structure containing all k-distribution data
		#
		##########################################################################################
		if (output_ctypes):		
			kdistOUT = {'c_ngas_req': c_ngas_req,                                                    \
                        'c_ntempref': c_ntempref,                                                    \
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
		else:
			kdistOUT = {}	
		return kdistOUT
