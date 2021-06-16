###############################################################################################
#! /usr/bin/env python
#
# This script adds perturbations to the RFMIP atmospheric profiles.
#
# The perturbations are applied to the RFMIP profiles following Veerman et al (2021) 
# (https://doi.org/10.1098/rsta.2020.0095)
#
# The RFMIP files contain several forcings to perturb:
# 1  - "Present day (PD)"
# 2  - "Pre-industrial (PI) greenhouse gas concentrations"
# 3  - "4xCO2"
# 4  - "\"future\""
# 5  - "0.5xCO2"
# 6  - "2xCO2"
# 7  - "3xCO2"
# 8  - "8xCO2"
# 9  - "PI CO2"
# 10 - "PI CH4"
# 11 - "PI N2O"
# 12 - "PI O3"
# 13 - "PI HCs"
# 14 - "+4K"
# 15 - "+4K, const. RH"
# 16 - "PI all"
# 17 - "\"future\" all"
# 18 - "LGM"
#
###############################################################################################
import sys
import os
import urllib.request
import netCDF4
import numpy as np
import argparse

###############################################################################################
###############################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("npb",       help="Number of perturbations to perform", type=int)
parser.add_argument("--verbose", help="Print more info from script?", type=int,default=0)
args    = parser.parse_args()
nperts  = args.npb
verbose = args.verbose

# RFMIP experiments to include?
experiments = ["Present day (PD)", "Pre-industrial (PI) greenhouse gas concentrations"]

###############################################################################################
#
# No changes below
#
###############################################################################################

# Parameters from Veerman2021 ()  
c1_q  = 0.75
c2_o3 = 0.75
c3_T  = 5.

# Which fields in the RFMIP file need to read in? 
vars_to_perturb= ["pres_level","temp_level","ozone","water_vapor"]
vars_to_recomp = ["pres_layer","temp_layer"]

# Download RFMIP profiles
rte_rrtmgp_dir = "home/dswales/Projects/radiation-nn/rte-rrtmgp/"
rfmip_dir      = os.path.join(rte_rrtmgp_dir, "examples", "rfmip-clear-sky")
conds_file     = "multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
conds_url      = "http://aims3.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6/RFMIP/UColorado/UColorado-RFMIP-1-2/" + \
                 "atmos/fx/multiple/none/v20190401/" + conds_file
if (verbose): print("Downloading RFMIP input file: "+conds_file)
urllib.request.urlretrieve(conds_url, conds_file)

# Output file
conds_file_pert = "multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none_Nperts"+str(nperts).zfill(5)+".nc"

print("###############################################################################################")
print("Downloading:   "+conds_file)
print("Adding         " +str(nperts)+ " perturbed RFMIP profiles")
print("Output:        "+conds_file_pert) 
print("###############################################################################################")

# Get number of RFMIP sites and complete list of experiments available
dataset = netCDF4.Dataset(conds_file)
experiments_ALL = dataset.variables["expt_label"][:]
for name, dimension in dataset.dimensions.items():
    if(name=="site"): nsites = dimension.size
dataset.close()

# Subset the full list of experiments. Only use experiments requested
nexp  = len(experiments)
iexp  = np.zeros(nexp)
count = 0
for ij in range(0,len(experiments_ALL)):
    if (experiments_ALL[ij] in experiments):
        iexp[count]=ij
        count =count +1

###############################################################################################
#
# Create perturbed RFMIP profiles
# Copy all fields to new file. Only for experiments requested. Perturb h20,o3,p,T.
#
###############################################################################################
with netCDF4.Dataset(conds_file) as src, netCDF4.Dataset(conds_file_pert, "w") as dst:
    #
    # Copy Global attributes
    #
    dst.setncatts(src.__dict__)
    #
    # Setup dimensions
    #
    for name, dimension in src.dimensions.items():
        if (name == "expt"):
            dst.createDimension("expt",nexp)
        elif (name == "site"):
            dst.createDimension("site",nsites*(nperts+1))
        else:
            dst.createDimension(name, dimension.size)
    ###########################################################################################
    #
    # Loop over variables in file...
    #
    # Copy and subset (by experiment) all fields except for the fields that are to be perturbed.
    #
    ###########################################################################################
    for name, variable in src.variables.items():
        ########################################################################################
        #
        # FIELDS TO COPY
        #
        ########################################################################################
        if name not in vars_to_perturb and name not in vars_to_recomp:
            if (verbose): print("Copying  " + name + " ...")
            x = dst.createVariable(name, variable.datatype, variable.dimensions)
            dst[name].setncatts(src[name].__dict__)
            # Subsetting by RFMIP experiment
            if ("expt" in variable.dimensions):
                if (len(variable.dimensions) == 1):
                    for ij in range(0,nexp):
                        dst[name][ij] = src[name][iexp[ij]]
                else:
                    for ij in range(0,nexp):
                        dst[name][ij,ip*nsites:ip*nsites+nsites] = src[name][iexp[ij],:]
            # Fields NOT dependent on RFMIP experiment
            else:
                if (len(variable.dimensions) == 1):
                    for ip in range(0,nperts+1):
                        dst[name][ip*nsites:ip*nsites+nsites] = src[name][:]
        ########################################################################################
        #
        # FIELDS TO PERTURB
        #
        ######################################################################################## 
        if name in vars_to_perturb:
            x = dst.createVariable(name, variable.datatype, variable.dimensions)
            dst[name].setncatts(src[name].__dict__)
            ####################################################################################
            # Ozone
            #################################################################################### 
            if (name =="ozone"):
                for ij in range(0,nexp):
                    for ip in range(0,nperts+1):
                        if (ip == 0):
                            dst[name][ij,ip*nsites:ip*nsites+nsites,:] = src[name][iexp[ij],:,:]
                        else:
                            if (verbose): print("Perturbing " + name + " at perturbation " + str(ip).zfill(5) +" for "+experiments[ij])
                            r2 = np.random.uniform(-1,1)
                            for ik in range(0,len(src[name][iexp[ij],1,:])):
                                dst[name][ij,ip*nsites:ip*nsites+nsites,ik] = \
                                    src[name][iexp[ij],:,ik]*(1+c2_o3*r2)
            #################################################################################### 
            # Water-vapor
            ####################################################################################
            if (name =="water_vapor"):
                for ij in range(0,nexp):
                    for ip in range(0,nperts+1):
                        if (ip == 0):
                            dst[name][ij,ip*nsites:ip*nsites+nsites,:] = src[name][iexp[ij],:,:]
                        else:
                            if (verbose): print("Perturbing " + name + " at perturbation " + str(ip).zfill(5) +" for "+experiments[ij])
                            r1 = np.random.uniform(-1,1)
                            for ik in range(0,len(src[name][iexp[ij],1,:])):
                                dst[name][ij,ip*nsites:ip*nsites+nsites,ik] = \
                                    src[name][iexp[ij],:,ik]*(1+c1_q*r1)        
            ####################################################################################
            # Pressure at layer-interfaces
            #################################################################################### 
            if (name =="pres_level"):
                for ip in range(0,nperts+1):
                    if (ip == 0):
                        dst[name][ip*nsites:ip*nsites+nsites,:] = src[name][:,:]
                    else:
                        if (verbose): print("Perturbing " + name + " at perturbation " +str(ip).zfill(5))
                        if (name =="pres_level"):
                            r4 = np.random.uniform(0.05,0.95)
                            dst[name][ip*nsites:ip*nsites+nsites,0] = src[name][:,0]
                            for ik in range(1,len(src[name][1,:])-1):
                                dst[name][ip*nsites:ip*nsites+nsites,ik] = \
                                    (src[name][:,ik+1] - src[name][:,ik-1])*r4+src[name][:,ik-1]
                            dst[name][ip*nsites:ip*nsites+nsites,len(src[name][1,:])-1] = \
                                src[name][:,len(src[name][1,:])-1]
            ####################################################################################
            # Temperature at layer-interfaces
            #################################################################################### 
            if (name =="temp_level"):
               for ij in range(0,nexp):
                    for ip in range(0,nperts+1):
                        if (ip == 0):
                            dst[name][ij,ip*nsites:ip*nsites+nsites,:] = src[name][iexp[ij],:,:]
                        else:
                            if (verbose): print("Perturbing " + name + " at perturbation " + str(ip).zfill(5) +" for "+experiments[ij])
                            r3 = np.random.uniform(-1,1)
                            for ik in range(0,len(src[name][iexp[ij],1,:])):
                                dst[name][ij,ip*nsites:ip*nsites+nsites,ik] = \
                                    src[name][iexp[ij],:,ik]+(c3_T*r3)

        #######################################################################################
        #
        # FIELDS DEPENDENT ON PERTURBED FIELDS
        # (create now, compute later after interface variables have been perturbed)
        #
        #######################################################################################
        elif name in vars_to_recomp:
            if (verbose): print("Recomputing " + name + " ...")
            x = dst.createVariable(name, variable.datatype, variable.dimensions)
            dst[name].setncatts(src[name].__dict__)
    
    ########################################################################################## 
    #
    # Loop over variables in file and recompute dependencies (e.g layer-temerature/pressure)
    #
    ########################################################################################## 
    for name, variable in src.variables.items():
        if name in vars_to_recomp:
            ##################################################################################
            # Pressure at layer-centers
            ##################################################################################
            if (name =="pres_layer"):
                if (verbose): print("Recomputing " + name)
                dst[name][:,:] = (dst["pres_level"][:,1:len(dst[name][1,:])+1] + \
                                  dst["pres_level"][:,0:len(dst[name][1,:])])  / 2.
            ##################################################################################
            # Temperature at layer-centers
            ##################################################################################
            if (name == "temp_layer"):
                if (verbose): print("Recomputing " + name)
                dst[name][:,:,:] = (dst["temp_level"][:,:,1:len(dst[name][1,1,:])+1] + \
                                    dst["temp_level"][:,:,0:len(dst[name][1,1,:])])  / 2.
