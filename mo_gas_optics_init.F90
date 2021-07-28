module mo_gas_optics_init
  use mo_rte_kind,           only: wp, wl	
  use mo_rrtmgp_util_string, only: string_in_array, string_loc_in_array
  implicit none
  public
contains

  ! --------------------------------------------------------------------------------------
  !
  ! --------------------------------------------------------------------------------------
  pure function rewrite_key_species_pair(key_species_pair)
    ! (0,0) becomes (2,2) -- because absorption coefficients for these g-points will be 0.
    integer, dimension(2) :: rewrite_key_species_pair
    integer, dimension(2), intent(in) :: key_species_pair
    rewrite_key_species_pair = key_species_pair
    if (all(key_species_pair(:).eq.(/0,0/))) then
      rewrite_key_species_pair(:) = (/2,2/)
    end if
  end function

  ! --------------------------------------------------------------------------------------
  ! true is key_species_pair exists in key_species_list
  ! --------------------------------------------------------------------------------------
  pure function key_species_pair_exists(key_species_list, key_species_pair)
    logical                             :: key_species_pair_exists
    integer, dimension(:,:), intent(in) :: key_species_list
    integer, dimension(2),   intent(in) :: key_species_pair
    integer :: i
    do i=1,size(key_species_list,dim=2)
      if (all(key_species_list(:,i).eq.key_species_pair(:))) then
        key_species_pair_exists = .true.
        return
      end if
    end do
    key_species_pair_exists = .false.
  end function key_species_pair_exists

  ! --------------------------------------------------------------------------------------
  ! returns flavor index; -1 if not found
  ! --------------------------------------------------------------------------------------
  pure function key_species_pair2flavor(flavor, key_species_pair)
    integer :: key_species_pair2flavor
    integer, dimension(:,:), intent(in) :: flavor
    integer, dimension(2), intent(in) :: key_species_pair
    integer :: iflav
    do iflav=1,size(flavor,2)
      if (all(key_species_pair(:).eq.flavor(:,iflav))) then
        key_species_pair2flavor = iflav
        return
      end if
    end do
    key_species_pair2flavor = -1
  end function key_species_pair2flavor
  
  subroutine decode_string_array(n,string_length,stringIN,stringOUT)
    integer, intent(in) :: n, string_length
    character,dimension(string_length*n),intent(in) :: stringIN
    character(len=string_length),dimension(n),intent(out) :: stringOUT
    integer :: ij, ik, count
    
    count=1
    do ij=1,n
       do ik=1,string_length
          stringOUT(ij)(ik:ik) = stringIN(count)
          count = count + 1
       enddo
    enddo
  end subroutine decode_string_array
  
  subroutine encode_string_array(n,string_length,stringIN,stringOUT)
    integer, intent(in) :: n, string_length
    character(len=string_length),dimension(n),intent(in) :: stringIN
    character, dimension(string_length*n),intent(out) :: stringOUT
    integer :: ij, ik, count
    
    count = 1
    do ij=1,n
       do ik=1,string_length
          stringOUT(count) = stringIN(ij)(ik:ik)
          count = count + 1
       enddo
    enddo
  end subroutine encode_string_array

  ! --------------------------------------------------------------------------------------
  !
  ! --------------------------------------------------------------------------------------
  subroutine reduce_minor_arrays(strlen, ngas_req, ntemp, nmixfrac, ncont, nminorabs,       &
       nminorabsatm, ncontatm_red, ngas_atm_red, kminor_atm, requested_gasesIN, gas_minorIN,&
       identifier_minorIN, minor_gases_atmIN, minor_limits_gpt_atm, gas_is_present,         &
       minor_scales_with_density_atm, scaling_gas_atmIN, scale_by_complement_atm,           &
       kminor_start_atm,                                                                    &  	
       kminor_atm_red, minor_gases_atm_redOUT, minor_limits_gpt_atm_red,                    &
       minor_scales_with_density_atm_red, scaling_gas_atm_redOUT,                           &
       scale_by_complement_atm_red, kminor_start_atm_red) bind(C, name="reduce_minor_arrays")

    ! Inputs
    integer,intent(in) :: strlen,ngas_req,ntemp,nmixfrac,ncont,nminorabs,nminorabsatm,ncontatm_red,ngas_atm_red
    real(wp),dimension(ncont,nmixfrac,ntemp), intent(in) :: kminor_atm
    integer,dimension(2,nminorabsatm),        intent(in) :: minor_limits_gpt_atm
    logical(wl),dimension(nminorabsatm),      intent(in) :: minor_scales_with_density_atm
    logical(wl),dimension(nminorabsatm),      intent(in) :: scale_by_complement_atm
    integer, dimension(nminorabsatm),         intent(in) :: kminor_start_atm
    character,dimension(strlen*ngas_req),     intent(in) :: requested_gasesIN
    character,dimension(strlen*nminorabs),    intent(in) :: gas_minorIN
    character,dimension(strlen*nminorabs),    intent(in) :: identifier_minorIN
    character,dimension(strlen*nminorabsatm), intent(in) :: minor_gases_atmIN
    character,dimension(strlen*nminorabsatm), intent(in) :: scaling_gas_atmIN
    logical(wl), dimension(nminorabsatm),     intent(in) :: gas_is_present

    
    ! Outputs
    real(wp),    dimension(ncontatm_red, nmixfrac, ntemp), intent(out) :: kminor_atm_red
    integer,     dimension(2,ngas_atm_red),      intent(out) :: minor_limits_gpt_atm_red
    integer,     dimension(ngas_atm_red),        intent(out) :: kminor_start_atm_red    
    logical(wl), dimension(ngas_atm_red),        intent(out) :: minor_scales_with_density_atm_red
    logical(wl), dimension(ngas_atm_red),        intent(out) :: scale_by_complement_atm_red
    character,   dimension(strlen*ngas_atm_red), intent(out) :: minor_gases_atm_redOUT
    character,   dimension(strlen*ngas_atm_red), intent(out) :: scaling_gas_atm_redOUT
    
    ! Local variables
    integer :: i, j, ks, idx_mnr, tot_g, icnt, n_elim, ng
    integer, dimension(ngas_atm_red) :: indexes
    character(len=strlen), dimension(ngas_req)     :: requested_gases
    character(len=strlen), dimension(nminorabs)    :: gas_minor, identifier_minor
    character(len=strlen), dimension(nminorabsatm) :: minor_gases_atm,scaling_gas_atm
    character(len=strlen), dimension(ngas_atm_red) :: minor_gases_atm_red
    character(len=strlen), dimension(ngas_atm_red) :: scaling_gas_atm_red

    ! Decode character array inputs
    call decode_string_array(ngas_req,     strlen, requested_gasesIN,  requested_gases)
    call decode_string_array(nminorabs,    strlen, gas_minorIN,        gas_minor)
    call decode_string_array(nminorabs,    strlen, identifier_minorIN, identifier_minor)
    call decode_string_array(nminorabsatm, strlen, minor_gases_atmIN,  minor_gases_atm)
    call decode_string_array(nminorabsatm, strlen, scaling_gas_atmIN,  scaling_gas_atm)
    
    
    if ((ngas_atm_red .eq. nminorabsatm)) then
       ! Character data not allowed in OpenACC regions?
       minor_gases_atm_red         = minor_gases_atm
       scaling_gas_atm_red         = scaling_gas_atm
       kminor_atm_red              = kminor_atm
       minor_limits_gpt_atm_red    = minor_limits_gpt_atm
       minor_scales_with_density_atm_red = minor_scales_with_density_atm
       scale_by_complement_atm_red = scale_by_complement_atm
       kminor_start_atm_red        = kminor_start_atm
    else
       ! Find the integer indexes for the gases that are present
       indexes = pack([(i, i = 1, size(minor_gases_atm))], mask=gas_is_present)
       
       minor_gases_atm_red               = minor_gases_atm(indexes)
       scaling_gas_atm_red               = scaling_gas_atm(indexes)
       minor_scales_with_density_atm_red = minor_scales_with_density_atm(indexes)
       scale_by_complement_atm_red       = scale_by_complement_atm(indexes)
       kminor_start_atm_red              = kminor_start_atm(indexes)
       
       icnt = 0
       n_elim = 0
       do i = 1, nminorabsatm
          ng = minor_limits_gpt_atm(2,i)-minor_limits_gpt_atm(1,i)+1
          if(gas_is_present(i)) then
             icnt = icnt + 1
             minor_limits_gpt_atm_red(1:2,icnt) = minor_limits_gpt_atm(1:2,i)
             kminor_start_atm_red(icnt) = kminor_start_atm(i)-n_elim
             ks = kminor_start_atm_red(icnt)
             do j = 1, ng
                kminor_atm_red(kminor_start_atm_red(icnt)+j-1,:,:) = &
                     kminor_atm(kminor_start_atm(i)+j-1,:,:)
             enddo
          else
             n_elim = n_elim + ng
          endif
       enddo
    endif
    
    ! Encode character array outputs
    call encode_string_array(ngas_atm_red, strlen, minor_gases_atm_red, minor_gases_atm_redOUT)    
    call encode_string_array(ngas_atm_red, strlen, scaling_gas_atm_red, scaling_gas_atm_redOUT)
    
  end subroutine reduce_minor_arrays
  
  ! --------------------------------------------------------------------------------------
  ! create index list for extracting col_gas needed for minor gas optical depth calculations
  ! --------------------------------------------------------------------------------------
  subroutine create_idx_minor(strlen, ngas_req, nminorabs, num_gas_atm, requested_gasesIN, &
       gas_minorIN, identifier_minorIN, minor_gases_atm_redIN, idx_minor_atm) bind(C, name="create_idx_minor")
    integer,intent(in) :: strlen, ngas_req, nminorabs, num_gas_atm
    character, dimension(strlen*ngas_req),     intent(in) :: requested_gasesIN
    character, dimension(strlen*nminorabs),    intent(in) :: gas_minorIN
    character, dimension(strlen*nminorabs),    intent(in) :: identifier_minorIN	
    character, dimension(strlen*num_gas_atm),  intent(in) :: minor_gases_atm_redIN
    integer,   dimension(num_gas_atm),         intent(out) :: idx_minor_atm
    ! local
    integer :: imnr, idx_mnr
    character(len=strlen), dimension(ngas_req)    :: requested_gases
    character(len=strlen), dimension(nminorabs)   :: gas_minor, identifier_minor
    character(len=strlen), dimension(num_gas_atm) :: minor_gases_atm
 
    ! Decode character array inputs
    call decode_string_array(ngas_req,    strlen, requested_gasesIN,     requested_gases)
    call decode_string_array(nminorabs,   strlen, gas_minorIN,           gas_minor)
    call decode_string_array(nminorabs,   strlen, identifier_minorIN,    identifier_minor)
    call decode_string_array(num_gas_atm, strlen, minor_gases_atm_redIN, minor_gases_atm)
    
    do imnr = 1, size(minor_gases_atm,dim=1) ! loop over minor absorbers in each band
       ! Find identifying string for minor species in list of possible identifiers (e.g. h2o_slf)
       idx_mnr     = string_loc_in_array(minor_gases_atm(imnr), identifier_minor)
       ! Find name of gas associated with minor species identifier (e.g. h2o)
       idx_minor_atm(imnr) = string_loc_in_array(gas_minor(idx_mnr),    requested_gases)
    enddo
  end subroutine create_idx_minor
  
  ! --------------------------------------------------------------------------------------
  ! create index for special treatment in density scaling of minor gases
  ! --------------------------------------------------------------------------------------
  !subroutine create_idx_minor_scaling(gas_names, &
  !  scaling_gas_atm, idx_minor_scaling_atm) bind(C, name="create_idx_minor_scaling")
  subroutine create_idx_minor_scaling(strlen, ngas_req, num_gas_atm, requested_gasesIN,   &
       scaling_gas_atm_redIN, idx_minor_scaling_atm) bind(C, name="create_idx_minor_scaling")
    integer,intent(in) :: strlen, ngas_req, num_gas_atm
    character, dimension(strlen*ngas_req),     intent(in) :: requested_gasesIN
    character, intent(in) :: scaling_gas_atm_redIN
    integer, dimension(num_gas_atm), intent(out) :: idx_minor_scaling_atm  
    integer :: imnr
    character(len=strlen), dimension(num_gas_atm) :: scaling_gas_atm_red
    character(len=strlen), dimension(ngas_req)    :: requested_gases
    
    ! Decode character array inputs
    call decode_string_array(ngas_req,    strlen, requested_gasesIN,     requested_gases)
    call decode_string_array(num_gas_atm, strlen, scaling_gas_atm_redIN, scaling_gas_atm_red)
    
    do imnr = 1, size(scaling_gas_atm_red,dim=1) ! loop over minor absorbers in each band
       ! This will be -1 if there's no interacting gas
       idx_minor_scaling_atm(imnr) = string_loc_in_array(scaling_gas_atm_red(imnr), requested_gases)
    enddo
  end subroutine create_idx_minor_scaling
  
  ! --------------------------------------------------------------------------------------
  !
  ! --------------------------------------------------------------------------------------  
  subroutine create_key_species_reduce(strlen, nband, natmlayer, npair, ngas, ngas_red,  &
  	gas_namesIN, requested_gasesIN, key_species, key_species_red, key_species_present_init)&
  	bind(C, name="create_key_species_reduce")
	! Inputs
	integer, intent(in)  :: strlen, nband, natmlayer, npair, ngas, ngas_red
    character, dimension(strlen*ngas),           intent(in)  :: gas_namesIN
    character, dimension(strlen*ngas_red),       intent(in)  :: requested_gasesIN
    integer,   dimension(npair,natmlayer,nband), intent(in)  :: key_species
	! Outputs
    integer,   dimension(npair,natmlayer,nband), intent(out) :: key_species_red
    logical,   dimension(ngas),                  intent(out) :: key_species_present_init
    ! Local
    character(len=strlen),dimension(ngas) :: gas_names
    character(len=strlen),dimension(ngas_red) :: requested_gases
    integer :: ip, ia, it

	! Decode character array inputs
    call decode_string_array(ngas,     strlen, gas_namesIN,       gas_names)
    call decode_string_array(ngas_red, strlen, requested_gasesIN, requested_gases)

    key_species_present_init = .true.
    do ip = 1, npair
      do ia = 1, natmlayer
        do it = 1, nband
          if (key_species(ip,ia,it) .ne. 0) then
            key_species_red(ip,ia,it) = string_loc_in_array(gas_names(key_species(ip,ia,it)),requested_gases)
            if (key_species_red(ip,ia,it) .eq. -1) key_species_present_init(key_species(ip,ia,it)) = .false.
          else
            key_species_red(ip,ia,it) = key_species(ip,ia,it)
          endif
        enddo
      end do
    enddo
  end subroutine create_key_species_reduce  
  
  ! --------------------------------------------------------------------------------------
  ! create flavor list --
  !   an unordered array of extent (2,:) containing all possible pairs of key species
  !   used in either upper or lower atmos
  ! --------------------------------------------------------------------------------------
  subroutine get_nflavors(npair, natmlayer, nband, key_species, key_species_list,        &
       nflavor) bind(C, name="get_nflavors")
    integer, intent(in) :: npair,natmlayer,nband
    integer, dimension(npair,natmlayer,nband), intent(in) :: key_species  
    integer, dimension(npair,nband*2),intent(out) :: key_species_list
    integer, intent(inout) :: nflavor
    
    integer :: ibnd, iatm, i
    ! prepare list of key_species
    i = 1
    do ibnd=1,nband
       do iatm=1,natmlayer
          key_species_list(:,i) = key_species(:,iatm,ibnd)
          i = i + 1
       end do
    end do
    ! rewrite single key_species pairs
    do i=1,nband*2
       key_species_list(:,i) = rewrite_key_species_pair(key_species_list(:,i))
    end do
    ! count unique key species pairs
    nflavor = 0
    do i=1,nband*2
       if (.not.key_species_pair_exists(key_species_list(:,1:i-1),key_species_list(:,i))) then
          nflavor = nflavor + 1
       end if
    end do
  end subroutine get_nflavors
  
  subroutine create_flavor(nband, npair, nflavor, nmajorabs, key_species_list, flavor,   &
  	is_key)  bind(C, name="create_flavor")
    integer,     intent(in) :: nband, npair, nflavor, nmajorabs
    integer,     intent(in),  dimension(npair,nband*2) :: key_species_list
    integer,     intent(out), dimension(npair,nflavor) :: flavor
	logical(wl), intent(out), dimension(nmajorabs)     :: is_key
    integer :: i,j,iflavor
    
    iflavor = 1
    do i=1,nband*2
       if (.not.key_species_pair_exists(key_species_list(:,1:i-1),key_species_list(:,i))) then		
          flavor(:,iflavor) = key_species_list(:,i)
          iflavor = iflavor + 1
       end if
    end do
    
    ! Which species are key in one or more bands? (flavor is an index into kdist.gas_names)
	is_key(:) = .false.
    do i=1,2
    	do j=1,nflavor
    		if (flavor(i,j) .ne. 0) is_key(flavor(i,j)) = .true.
    	enddo
    enddo

  end subroutine create_flavor

  ! --------------------------------------------------------------------------------------
  ! create gpoint_flavor list
  !   a map pointing from each g-point to the corresponding entry in the "flavor list"
  ! --------------------------------------------------------------------------------------
  subroutine create_gpoint_flavor(npair, natmlayer, nband, ngpt, nflavor, key_species,   &
  	flavor, gpt2band, gpoint_flavor) bind(C, name="create_gpoint_flavor")
	integer, intent(in) :: nband, npair, nflavor, ngpt, natmlayer
    integer, dimension(npair,natmlayer,nband), intent(in) :: key_species
    integer, dimension(ngpt), intent(in) :: gpt2band
    integer, dimension(npair,nflavor), intent(in) :: flavor
    integer, dimension(2,ngpt), intent(out) :: gpoint_flavor
    integer :: igpt, iatm

    do igpt=1,ngpt
      do iatm=1,natmlayer
        gpoint_flavor(iatm,igpt) = key_species_pair2flavor( &
          flavor, &
          rewrite_key_species_pair(key_species(:,iatm,gpt2band(igpt))) &
        )
      end do
    end do
  end subroutine create_gpoint_flavor  
  ! --------------------------------------------------------------------------------------
  ! --------------------------------------------------------------------------------------
end module mo_gas_optics_init
