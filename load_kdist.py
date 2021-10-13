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
def create_idx_minor(minor_gases_red, requested_gases, identifier_minor, gas_minor):
	idx_minor = np.ones(len(minor_gases_red), dtype=int)
	for iminorgas in range(0,len(minor_gases_red)):
		if (minor_gases_red[iminorgas] in identifier_minor):
                        idx_mnr = identifier_minor.index(minor_gases_red[iminorgas])
                        idx_minor[iminorgas] = requested_gases.index(gas_minor[idx_mnr])
	return idx_minor

##########################################################################################
##########################################################################################
def create_idx_minor_scaling(scaling_gas, requested_gases):
	idx_minor_scaling = np.full(len(scaling_gas), -1, dtype=int)
	for igas in range(0,len(scaling_gas)):
		if scaling_gas[igas]:
			idx_minor_scaling[igas] = requested_gases.index(scaling_gas[igas])
	return idx_minor_scaling

##########################################################################################
##########################################################################################
def reduce_minor(nminorabs, nminorabs_red, ncont_red, ntemp, nmixfrac, gas_is_present,
		 minor_gases, scaling_gas, minor_scales_with_density, scale_by_complement, \
		 kminor_start, minor_limits_gpt, kminor):

	minor_gases_red               = []
	scaling_gas_red               = []
	minor_scales_with_density_red = []
	scale_by_complement_red       = []
	kminor_start_red              = np.zeros(nminorabs_red,                dtype=int)
	minor_limits_gpt_red          = np.zeros((nminorabs_red, 2),           dtype=int)
	kminor_red                    = np.zeros((ntemp, nmixfrac, ncont_red), dtype=np.double)
	icnt   = -1
	n_elim = 0
	for igas in range(0, nminorabs):
		ng = minor_limits_gpt[igas, 1] - minor_limits_gpt[igas, 0] + 1
		if (gas_is_present[igas]):
			minor_gases_red.append(minor_gases[igas])
			scaling_gas_red.append(scaling_gas[igas])
			minor_scales_with_density_red.append(minor_scales_with_density[igas])
			scale_by_complement_red.append(scale_by_complement[igas])
			icnt = icnt + 1
			minor_limits_gpt_red[icnt, :] = minor_limits_gpt[igas, :]
			kminor_start_red[icnt]        = kminor_start[igas] - n_elim
			for ij in range(0, ng):
				kminor_red[:, :, kminor_start_red[icnt]+ij-1] = \
					kminor[:, :, kminor_start[igas]+ij-1]
		else:
			n_elim = n_elim + ng
	return [minor_gases_red, scaling_gas_red, minor_scales_with_density_red, \
		scale_by_complement_red, kminor_start_red, minor_limits_gpt_red, \
		kminor_red];

##########################################################################################
##########################################################################################	
def search_for_gases(gas1, gas2, limits_gpt):
	ncontatm_red     = 0
	nminorabsatm_red = 0
	gas_is_present   = np.full([len(gas1)], False, dtype=bool)
	for igas1 in range(0,len(gas1)):
		for igas2 in range(0,len(gas2)):
			if gas1[igas1].startswith(gas2[igas2].strip()):
				gas_is_present[igas1] = True
				ncontatm_red          = ncontatm_red + (limits_gpt[igas1,1]-limits_gpt[igas1,0]+1)
				nminorabsatm_red      = nminorabsatm_red + 1
	return [ncontatm_red, nminorabsatm_red, gas_is_present];

##########################################################################################
##########################################################################################
def load_kdist(ffi, file_kdist, gases, print_info):

		# Load k-distribution data
		kdist = xr.open_dataset(file_kdist,concat_characters=True,decode_cf=True)		

		# Longwave or Shortwave file?
		kdist_keys = list(kdist.keys())
		doLW = 'plank_fraction' in list(kdist.keys())
		doSW = 'rayl_lower'     in list(kdist.keys())

		# Dimensions (flat)
		ngas                = len(gases)
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
		# minimum, maximum reference temperature, pressure -- assumes low-to-high ordering
		# for T, high-to-low ordering for p
		temp_ref_min        = kdist.temp_ref[0].values
		temp_ref_max        = kdist.temp_ref[ntemp-1].values
		press_ref_min       = kdist.press_ref[npressref-1].values
		press_ref_max       = kdist.press_ref[0].values
		
		# create press_ref_log, temp_ref_delta
		press_ref_log_delta = (np.log(press_ref_min)-np.log(press_ref_max))/(npressref-1)
		temp_ref_delta      = (temp_ref_max-temp_ref_min)/(ntemp-1)
		
		
		if (print_info):
			print("#################################################################################")
			print("Dimensions (IN/full): ")
			print("   ngas                         = ",ngas)
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
				print("  *ntemp_Planck                 = ",kdist.temperature_Planck.size)
				print("  *nfit_coeffs                  = ",kdist.optimal_angle_fit[0,:].size)
			print("   nmajorabs                    = ",nmajorabs)
			print("   nminorabs                    = ",nminorabs)
			print("   ncontlower                   = ",ncontlower)
			print("   ncontupper                   = ",ncontupper)
			print("   nminorabslower               = ",nminorabslower)
			print("   nminorabsupper               = ",nminorabsupper)
		
		##################################################################################
		#
		# Using the "requested_gases", reduce k-distribution data to only the gases needed
		#
		##################################################################################

		# Local string arrays (Xarray strings are encoded, locals defined/decoded below)
		requested_gases   = []
		gas_names         = []
		gas_minor         = []
		identifier_minor  = []
		minor_gases_lower = []
		scaling_gas_lower = []
		minor_gases_upper = []
		scaling_gas_upper = []
		for igas in range(0,ngas):
			requested_gases.append(gases[igas])
		for igas in range(0,nmajorabs):
			gas_names.append(kdist.gas_names.values[igas].decode())
		for igas in range(0,nminorabs):
			gas_minor.append(kdist.gas_minor.values[igas].decode().strip())
			identifier_minor.append(kdist.identifier_minor.values[igas].decode().strip())
		for igas in range(0,nminorabslower):
			minor_gases_lower.append(kdist.minor_gases_lower.values[igas].decode().strip())
			scaling_gas_lower.append(kdist.scaling_gas_lower.values[igas].decode().strip())
		for igas in range(0,nminorabsupper): 
			minor_gases_upper.append(kdist.minor_gases_upper.values[igas].decode().strip())
			scaling_gas_upper.append(kdist.scaling_gas_upper.values[igas].decode().strip())
		
		#
		# Which minor absorbing gases, in lower reference atmosphere, to include?
		#
		minorgas_lower = search_for_gases(minor_gases_lower, requested_gases, kdist.minor_limits_gpt_lower.values)
		ncontlower_red       = minorgas_lower[0]
		nminorabslower_red   = minorgas_lower[1]
		gas_is_present_lower = minorgas_lower[2]

		#
		# Which minor absorbing gases, in upper reference atmosphere, to include?
		#
		minorgas_upper = search_for_gases(minor_gases_upper, requested_gases, kdist.minor_limits_gpt_upper.values)
		ncontupper_red       = minorgas_upper[0]
		nminorabsupper_red   = minorgas_upper[1]
		gas_is_present_upper = minorgas_upper[2]
		
		#
		# Reduce volume mixing ratios for reference atmosphere...
		#  - Gas 0 is used in single-key species method, set to 1.0 (col_dry)
		#
		vmr_ref_red        = np.zeros((natmlayer,ngas+1,ntemp))
		vmr_ref_red[:,0,:] = (np.reshape(kdist.vmr_ref[:,0,:].values,[natmlayer,ntemp])).tolist()
		gas_count          = 0
		for igas in range(0,nmajorabs):
			if (gas_names[igas].strip() in requested_gases):
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
		reduce_minor_lower = reduce_minor(nminorabslower, nminorabslower_red,                \
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
		reduce_minor_upper = reduce_minor(nminorabsupper, nminorabsupper_red,                \
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
		idx_minor_lower = create_idx_minor(minor_gases_lower_red, requested_gases, identifier_minor, gas_minor) + 1
		idx_minor_upper = create_idx_minor(minor_gases_upper_red, requested_gases, identifier_minor, gas_minor) + 1
		
		#
		# Call create_idx_minor_scaling
		#

		idx_minor_scaling_lower = create_idx_minor_scaling(scaling_gas_lower_red, requested_gases) + 1
		idx_minor_scaling_upper = create_idx_minor_scaling(scaling_gas_upper_red, requested_gases) + 1

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
						key_species_red[iband,iatm,ipair] = requested_gases.index(gas_names[ki-1].strip())+1
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
		is_key    = np.empty(ngas, dtype=bool)
		is_key[:] = False
		for iflav in range(0,nunique_flavors):
			for iatm in range(0,natmlayer):
				if (flavor[iatm,iflav] != 0): is_key[flavor[iatm,iflav]-1] = True
				
		# Store index into gas-array for water-vapor
		idx_h2o = requested_gases.index("h2o")

		##################################################################################
		#
		# Copy data from k-distribution file into c-types...
		#
		##################################################################################
		kdistOUT = {}
		# Data used by both the longwave and shortwave schemes.
		c_var_dict = [{"name":"ngas",                   "ctype":"int",    "init":ngas},\
			      {"name":"ntempref",               "ctype":"int",    "init":ntemp},\
			      {"name":"nmixfrac",               "ctype":"int",    "init":nmixfrac}, \
			      {"name":"nmajorabs",              "ctype":"int",    "init":nmajorabs},\
			      {"name":"nminorabs",              "ctype":"int",    "init":nminorabs},\
			      {"name":"npressref",              "ctype":"int",    "init":npressref},\
			      {"name":"npressiref",             "ctype":"int",    "init":npressiref},\
			      {"name":"ngpt",                   "ctype":"int",    "init":ngpt},\
			      {"name":"nband",                  "ctype":"int",    "init":nband},\
			      {"name":"natmlayer",              "ctype":"int",    "init":natmlayer},\
			      {"name":"npair",                  "ctype":"int",    "init":npair},\
			      {"name":"nabsorber_ext",          "ctype":"int",    "init":nabsorber_ext},\
			      {"name":"nflavors",               "ctype":"int",    "init":nunique_flavors},\
			      {"name":"idx_h2o",                "ctype":"int",    "init":idx_h2o},\
			      {"name":"press_ref_trop",         "ctype":"double", "init":kdist.press_ref_trop.values},\
			      {"name":"press_ref_trop_log",     "ctype":"double", "init":np.log(kdist.press_ref_trop).values},\
			      {"name":"temp_ref_min",           "ctype":"double", "init":temp_ref_min},\
			      {"name":"temp_ref_max",           "ctype":"double", "init":temp_ref_max},\
			      {"name":"press_ref_min",          "ctype":"double", "init":press_ref_min},\
			      {"name":"press_ref_max",          "ctype":"double", "init":press_ref_max},\
			      {"name":"press_ref_log_delta",    "ctype":"double", "init":press_ref_log_delta},\
			      {"name":"temp_ref_delta",         "ctype":"double", "init":temp_ref_delta},\
			      {"name":"ncontlower",             "ctype":"int",    "init":ncontlower_red},\
			      {"name":"ncontupper",             "ctype":"int",    "init":ncontupper_red},\
			      {"name":"nminorabslower",         "ctype":"int",    "init":nminorabslower_red},\
			      {"name":"nminorabsupper",         "ctype":"int",    "init":nminorabsupper_red},\
			      {"name":"kmajor",                 "ctype":"double", "dims":[ngpt,nmixfrac,npressiref,ntemp],\
			       "init": np.reshape(kdist.kmajor.values,[ngpt,nmixfrac,npressiref,ntemp]).tolist()},\
			      {"name":"bnd_limits_gpt",         "ctype":"int",    "dims":[2,nband],\
			       "init": np.reshape(kdist.bnd_limits_gpt.values,[2,nband]).tolist()},\
			      {"name":"press_ref",              "ctype":"double", "dims":[npressref],\
			       "init": kdist.press_ref.values.tolist()},\
			      {"name":"temp_ref",               "ctype":"double", "dims":[ntemp],\
			       "init": kdist.temp_ref.values.tolist()},\
			      {"name":"press_ref_log",          "ctype":"double", "dims":[npressref],\
			       "init": np.log(kdist.press_ref.values).tolist()},\
			      {"name":"gpt2band",               "ctype":"int",    "dims":[ngpt],\
			       "init": gpt2band.tolist()},\
			      {"name":"vmr_ref",                "ctype":"double", "dims":[natmlayer,ngas+1,ntemp],\
			       "init": vmr_ref_red.tolist()},\
			      {"name":"key_species",            "ctype":"int",    "dims":[npair,natmlayer,nband],\
			       "init": np.reshape(key_species_red,[npair,natmlayer,nband]).tolist()},\
			      {"name":"kminor_lower",           "ctype":"double", "dims":[ncontlower_red,nmixfrac,ntemp],\
			       "init": np.reshape(kminor_lower_red,[ncontlower_red,nmixfrac,ntemp]).tolist()},\
			      {"name":"kminor_upper",           "ctype":"double", "dims":[ncontupper_red,nmixfrac,ntemp],\
			       "init": np.reshape(kminor_upper_red,[ncontupper_red,nmixfrac,ntemp]).tolist()},\
			      {"name":"key_species_list",       "ctype":"int",    "dims":[npair,nband*2],\
			       "init": np.reshape(key_species_list,[npair,nband*2]).tolist()},\
			      {"name":"minor_limits_gpt_lower", "ctype":"int",    "dims":[2,nminorabslower_red],\
			       "init": np.reshape(minor_limits_gpt_lower_red,[2,nminorabslower_red]).tolist()},\
			      {"name":"minor_limits_gpt_upper", "ctype":"int",    "dims":[2,nminorabsupper_red],\
			       "init": np.reshape(minor_limits_gpt_upper_red,[2,nminorabsupper_red]).tolist()},\
			      {"name":"flavors",                "ctype":"int",    "dims":[2,nunique_flavors],\
			       "init": flavor.tolist()},\
			      {"name":"gpoint_flavor",          "ctype":"int",    "dims":[2,ngpt],\
			       "init": gpoint_flavor.tolist()},\
			      {"name":"kminor_start_lower",              "ctype":"int",    "dims":[nminorabslower_red],\
			       "init": kminor_start_lower_red.tolist()},\
			      {"name":"kminor_start_upper",              "ctype":"int",    "dims":[nminorabsupper_red],\
			       "init": kminor_start_upper_red.tolist()},\
			      {"name":"idx_minor_lower",                 "ctype":"int",    "dims":[nminorabslower_red],\
			       "init": idx_minor_lower.tolist()},\
			      {"name":"idx_minor_upper",                 "ctype":"int",    "dims":[nminorabsupper_red],\
                               "init": idx_minor_upper.tolist()},\
			      {"name":"idx_minor_scaling_lower",         "ctype":"int",    "dims":[nminorabslower_red],\
                               "init": idx_minor_scaling_lower.tolist()},\
			      {"name":"idx_minor_scaling_upper",         "ctype":"int",    "dims":[nminorabsupper_red],\
                               "init": idx_minor_scaling_upper.tolist()},\
			      {"name":"key_species_present_init",        "ctype":"int",    "dims":[nmajorabs],\
                               "init": key_species_present_init.tolist()},\
			      {"name":"minor_scales_with_density_lower", "ctype":"int",    "dims":[nminorabslower_red],\
                               "init": minor_scales_with_density_lower_red[:]},\
			      {"name":"minor_scales_with_density_upper", "ctype":"int",    "dims":[nminorabsupper_red],\
                               "init": minor_scales_with_density_upper_red[:]},\
			      {"name":"scale_by_complement_lower",       "ctype":"int",    "dims":[nminorabslower_red],\
                               "init": scale_by_complement_lower_red[:]},\
			      {"name":"scale_by_complement_upper",       "ctype":"int",    "dims":[nminorabsupper_red],\
			       "init": scale_by_complement_upper_red[:]},\
			      {"name":"is_key",                          "ctype":"int",    "dims":[ngas],\
			       "init":is_key.tolist()}]
		# Data only used by the RRTMGP longwave scheme
		if (doLW):
			ntemp_Planck  = kdist.temperature_Planck.size
			nfit_coeffs   = kdist.optimal_angle_fit[0,:].size
			# Temperature steps for Planck function interpolation
			#   Assumes that temperature minimum and max are the same for the absorption 
			#   coefficient grid and the Planck grid and the Planck grid is equally spaced.
			totplnk_delta = (temp_ref_max-temp_ref_min) / (kdist.temperature_Planck.size-1)			
			c_varLW_dict =[{"name":"nfit_coeffs",       "ctype":"int",    "init": nfit_coeffs},\
				       {"name":"ntemp_Planck",      "ctype":"int",    "init": ntemp_Planck},\
				       {"name":"totplnk_delta",     "ctype":"double", "init": totplnk_delta},\
				       {"name":"totplnk",           "ctype":"double", "dims":[ntemp_Planck,nband],\
				        "init": np.reshape(kdist.totplnk.values,[ntemp_Planck,nband]).tolist()},\
				       {"name":"planck_frac",       "ctype":"double", "dims":[ngpt,nmixfrac,npressiref,ntemp],\
				        "init": np.reshape(kdist.plank_fraction.values,[ngpt,nmixfrac,npressiref,ntemp]).tolist()},\
				       {"name":"optimal_angle_fit", "ctype":"double", "dims":[nfit_coeffs,nband],\
				        "init": np.reshape(kdist.optimal_angle_fit.values,[nfit_coeffs,nband]).tolist() }]
			c_var_dict.extend(c_varLW_dict)
		# Data only used by the RRTMGP shortwave scheme
		if (doSW):
			rayl = np.empty((ngpt,nmixfrac,ntemp,2),dtype=np.double)
			rayl[:,:,:,0] =  np.reshape(kdist.rayl_lower.values,[ngpt,nmixfrac,ntemp])
			rayl[:,:,:,1] =  np.reshape(kdist.rayl_upper.values,[ngpt,nmixfrac,ntemp])
			c_varSW_dict = [{"name":"absorption_coefficient_ref_P", "ctype":"double",\
					 "init": kdist.absorption_coefficient_ref_P.values},\
					{"name":"absorption_coefficient_ref_T", "ctype":"double",\
					 "init": kdist.absorption_coefficient_ref_T.values},\
					{"name":"tsi_default",                  "ctype":"double",\
					 "init": kdist.tsi_default.values},\
					{"name":"mg_default",                   "ctype":"double",\
					 "init": kdist.mg_default.values},\
					{"name":"sb_default",                   "ctype":"double",\
					 "init": kdist.sb_default.values},\
					{"name":"krayl",                        "ctype":"double", "dims":[ngpt,nmixfrac,ntemp,2],\
					 "init":rayl.tolist()},\
					{"name":"solar_source_facular",         "ctype":"double", "dims":[ngpt],\
                                         "init": kdist.solar_source_facular.values.tolist()},\
					{"name":"solar_source_sunspot",         "ctype":"double", "dims":[ngpt],\
                                         "init": kdist.solar_source_sunspot.values.tolist()},\
					{"name":"solar_source_quiet",           "ctype":"double", "dims":[ngpt],\
                                         "init": kdist.solar_source_quiet.values.tolist()}]
			c_var_dict.extend(c_varSW_dict)

		# Define ctypes for k-distribution data.
		for ivars in range(0,len(c_var_dict)):
			# Build first argument to ffi.new. (e.g. double [2][4] OR int *)
			res =  c_var_dict[ivars]["ctype"]+" "
			if "dims" in c_var_dict[ivars].keys():
				for d in c_var_dict[ivars]["dims"]:
          				res += "[{}]".format(d)
			else:
				res += "*"
			# Second arguement is value to initialize ctype to.
			kdistOUT[c_var_dict[ivars]["name"]] = ffi.new(res,c_var_dict[ivars]["init"])

		return kdistOUT
