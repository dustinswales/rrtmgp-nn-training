##########################################################################################
#! /usr/bin/env python
#
##########################################################################################
from cffi import FFI

##########################################################################################
def cdef_reduce_minor_arrays(ffi, strlen, ngas, nminorabsatm, nminorabs, ncontatm, ntemp,\
                             nmixfrac, ncontatm_red, ngas_atm_red):
    ffi.cdef("void reduce_minor_arrays(                                                  \
        int *strlen,                                                                     \
        int *ngas_req,                                                                   \
        int *ntemp,                                                                      \
        int *nmixfrac,                                                                   \
        int *ncont,                                                                      \
        int *nminorabs,                                                                  \
        int *nminorabsatm,                                                               \
        int *ncontatm_red,                                                               \
        int *ngas_atm_red,                                                               \
        double kminor_atm["+str(ncontatm)+"]["+str(nmixfrac)+"]["+str(ntemp)+"],         \
        char requested_gasesIN["+str(strlen*ngas)+"],                                    \
        char gas_minorIN["+str(strlen*nminorabs)+"],                                     \
        char identifier_minorIN["+str(strlen*nminorabs)+"],                              \
        char minor_gases_atmIN["+str(strlen*nminorabsatm)+"],                            \
        int minor_limits_gpt_atm[2]["+str(nminorabsatm)+"],                              \
        int gas_is_present["+str(strlen*nminorabsatm)+"],                                \
        int minor_scales_with_density_atm["+str(nminorabsatm)+"],                        \
        char scaling_gas_atmIN["+str(strlen*nminorabsatm)+"],                            \
        int scale_by_complement_atm["+str(nminorabsatm)+"],                              \
        int kminor_start_atm["+str(nminorabsatm)+"],                                     \
        double kminor_atm_red["+str(ncontatm_red)+"]["+str(nmixfrac)+"]["+str(ntemp)+"], \
        char minor_gases_atm_red["+str(strlen*ngas_atm_red)+"],                          \
        int minor_limits_gpt_atm_red[2]["+str(ngas_atm_red)+"],                          \
        int minor_scales_with_density_atm_red["+str(ngas_atm_red)+"],                    \
        char scaling_gas_atm_red["+str(strlen*ngas_atm_red)+"],                          \
        int scale_by_complement_atm_red["+str(ngas_atm_red)+"],                          \
        int kminor_start_atm_red["+str(ngas_atm_red)+"] );", override=True)
#    return ffi

def cdef_create_idx_minor(ffi, strlen, ngas, nminorabs, num_gas_atm):
    ffi.cdef("void create_idx_minor(                                                     \
        int *strlen,                                                                     \
        int *ngas_req,                                                                   \
        int *nminorabs,                                                                  \
        int *ngas_atm_red,                                                               \
        char requested_gasesIN["+str(strlen*ngas)+"],                                    \
        char gas_minorIN["+str(strlen*nminorabs)+"],                                     \
        char identifier_minorIN["+str(strlen*nminorabs)+"],                              \
        char minor_gases_atm_redIN["+str(strlen*num_gas_atm)+"],                         \
        int idx_minor_atm["+str(num_gas_atm)+"] );", override=True)
#    return ffi

def cdef_create_idx_minor_scaling(ffi, strlen, ngas, num_gas_atm):
        ffi.cdef("void create_idx_minor_scaling(                                         \
        int *strlen,                                                                     \
        int *ngas_req,                                                                   \
        int *ngas_atm_red,                                                               \
        char requested_gasesIN["+str(strlen*ngas)+"],                                    \
        char scaling_gas_atm_redIN["+str(strlen*num_gas_atm)+"],                         \
        int idx_minor_scaling_atm["+str(num_gas_atm)+"]);", override=True) 
#    return ffi
    
def cdef_create_key_species_reduce(ffi, strlen, nmajorabs, ngas_req, npair, natmlayer, nband):
        ffi.cdef("void create_key_species_reduce(                                        \
        int *strlen,                                                                     \
        int *nband,                                                                      \
        int *natmlayer,                                                                  \
        int *npair,                                                                      \
        int *ngas,                                                                       \
        int *ngas_req,                                                                   \
        char gas_names_majorIN["+str(strlen*nmajorabs)+"],                               \
        char requested_gasesIN["+str(strlen*ngas_req)+"],                                \
        int key_species["+str(npair)+"]["+str(natmlayer)+"]["+str(nband)+"],             \
        int key_species_red["+str(npair)+"]["+str(natmlayer)+"]["+str(nband)+"],         \
        int key_species_present_init["+str(strlen*nmajorabs)+"]) ;", override=True) 
#    return ffi
        
def cdef_get_nflavors(ffi, npair, natmlayer, nband):
        ffi.cdef("void get_nflavors(                                                     \
        int *npair,                                                                      \
        int *natmlayer,                                                                  \
        int *nband,                                                                      \
        int key_species_red["+str(npair)+"]["+str(natmlayer)+"]["+str(nband)+"],         \
        int key_species_list["+str(npair)+"]["+str(nband*2)+"],                          \
        int *nflavor) ;",override=True) 
#    return ffi
        
def cdef_create_flavor(ffi, npair, nband, nflavor, nmajorabs):
        ffi.cdef("void create_flavor(                                                    \
        int *nband,                                                                      \
        int *npair,                                                                      \
        int *nflavor,                                                                    \
        int *nmajorabs,                                                                  \
        int key_species_list["+str(npair)+"]["+str(nband*2)+"],                          \
        int flavor["+str(npair)+"]["+str(nflavor)+"],                                    \
        int is_key["+str(nmajorabs)+"]) ;", override=True) 
#    return ffi

def cdef_create_gpoint_flavor(ffi, ngpt, nband, npair, natmlayer, nflavor):
        ffi.cdef("void create_gpoint_flavor(                                             \
        int *npair,                                                                      \
        int *natmlayer,                                                                  \
        int *nband,                                                                      \
        int *ngpt,                                                                       \
        int *nflavor,                                                                    \
        int key_species_red["+str(npair)+"]["+str(natmlayer)+"]["+str(nband)+"],         \
        int flavor["+str(npair)+"]["+str(nflavor)+"],                                    \
        int gpt2band["+str(ngpt)+"],                                                     \
        int gpoint_flavor[2]["+str(ngpt)+"]) ;", override=True)
#    return ffi