##########################################################################################
#! /usr/bin/env python 
#
# This script reads in the RFMIP data, convert
#

import numpy as np
import xarray as xr
##########################################################################################
def get_col_dry(vmr_h2o, p_lev, latitude = None):
        helmert1 = 9.80665
        helmert2 = 0.02586
        pi       = np.arccos(-1.)
        # Molecular weight of water [kg/mol]
        m_h2o  = 0.018016
        # Molecular weight of dry air [kg/mol]
        m_dry  = 0.028964
        # Gravity at Earth's surface [m/s2]
        grav   = 9.80665
        # Avogadro's number [molec/mol] 
        avogad = 6.02214076e23

        # Adjust gravity by latitude? 
        if (np.any(latitude)): g0 = helmert1 - helmert2 * np.cos(2.0*pi*latitude/180.)
        else:                  g0 = np.full((vmr_h2o[:,0].size), grav, dtype=np.double)

        col_dry = np.zeros((vmr_h2o[:,0].size,vmr_h2o[0,:].size),dtype=np.double)
        for ilay in range(0,vmr_h2o[0,:].size):
            # Layer thickness [Pa] 
            dp = np.abs(p_lev[:,ilay]-p_lev[:,ilay+1])
            # Mass of air [grams]
            fact  = 1. / (1. + vmr_h2o[:,ilay])
            m_air = (m_dry + m_h2o*vmr_h2o[:,ilay])*fact
            # [molec/cm^2]
            col_dry[:,ilay] = 10.*dp*avogad*fact/(1000.*m_air*100.*g0)
        return col_dry

##########################################################################################
def read_rfmip(input_file, gases, irfmip_expt):
	ngas = len(gases)

	# Load data
	data = xr.open_dataset(input_file,concat_characters=True,decode_cf=True)
	print("Reading RFMIP data for: ",data["expt_label"].values[irfmip_expt])
	ncol = data["lon"].size
	nlay = data["pres_layer"][0,:].size
	nlev = data["pres_level"][0,:].size

    	# This dictionary contains the mapping between RFMIP naming conventions and their chemical
    	# names, used by RRTMGP.
	RFMIP_gases = [ {"name":"water_vapor",             "chemical_name":"h2o"},\
			{"name":"ozone",                   "chemical_name":"o3"},\
			{"name":"oxygen_GM",               "chemical_name":"o2"},\
			{"name":"carbon_dioxide_GM",       "chemical_name":"co2"},\
			{"name":"nitrous_oxide_GM",        "chemical_name":"n2o"},\
			{"name":"methane_GM",              "chemical_name":"ch4"},\
			{"name":"carbon_monoxide_GM",      "chemical_name":"co"},\
			{"name":"nitrogen_GM",             "chemical_name":"n2"},\
			{"name":"carbon_tetrachloride_GM", "chemical_name":"ccl4"},\
			{"name":"cfc11_GM",                "chemical_name":"cfc11"},\
			{"name":"cfc12_GM",                "chemical_name":"cfc12"},\
			{"name":"hcfc22_GM",               "chemical_name":"cfc22"},\
			{"name":"hfc22_GM",                "chemical_name":"hfc22"},\
			{"name":"hfc143a_GM",              "chemical_name":"hfc143a"},\
			{"name":"hfc125_GM",               "chemical_name":"hfc125"},\
			{"name":"hfc23_GM",                "chemical_name":"hfc23"},\
			{"name":"hfc32_GM",                "chemical_name":"hfc32"},\
			{"name":"hfc134a_GM",              "chemical_name":"hfc134a"},\
			{"name":"cf4_GM",                  "chemical_name":"cf4"},\
			{"name":"hfc125_GM",               "chemical_name":"no2"}]

    	# Create array of volume-mixing-ratios
	vmr = np.zeros((ncol, nlay, ngas), dtype=np.double)
	for gas in RFMIP_gases:
		if (gas["chemical_name"] in gases):
			# Store output in same order as requested gas list
			gas2gas = gases.index(gas["chemical_name"])
			if (np.ndim(data[gas["name"]]) == 3):
				vmr[:,:,gas2gas] = data[gas["name"]][irfmip_expt,:,:].values
			if (np.ndim(data[gas["name"]]) == 1):
				vmr[:,:,gas2gas] = data[gas["name"]][irfmip_expt].values
			# Scale data
			scale_factor = np.double(data[gas["name"]].attrs["units"])
			vmr[:,:,gas2gas] = scale_factor*vmr[:,:,gas2gas]

    	# Compute column gas amounts [molec/cm^2]
	col_gas = np.zeros((ngas+1, nlay, ncol), dtype=np.double)
	col_dry = get_col_dry(vmr[:,:,gases.index("h2o")], data["pres_level"].values)
	for icol in range(0,ncol):
		col_gas[0,:,icol] = col_dry[icol,:]
	for igas in range(1,ngas):
		for icol in range(0,ncol):
			for ilay in range(0,nlay):
				col_gas[igas,ilay,icol] = vmr[icol,ilay,igas-1] * col_dry[icol,ilay]

	# Create output dictionary.
	dataOUT = {}
	dataOUT["col_gas"] = col_gas
	dataOUT["col_dry"] = col_gas[:,:,0]
	dataOUT["p_lay"]   = np.transpose(data["pres_layer"].values)
	dataOUT["t_lay"]   = np.transpose(data["temp_layer"][irfmip_expt,0:ncol,:].values)
	return(dataOUT)
