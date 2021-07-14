module mo_gas_optics_init
  use mo_rte_kind,           only: wp, wl	
  use mo_rrtmgp_util_string, only: string_in_array, string_loc_in_array
  implicit none
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

  ! --------------------------------------------------------------------------------------
  !
  ! --------------------------------------------------------------------------------------
  subroutine reduce_minor_arrays(available_gases, &
                           gas_names, &
                           gas_minor,identifier_minor,&
                           kminor_atm, &
                           minor_gases_atm, &
                           minor_limits_gpt_atm, &
                           minor_scales_with_density_atm, &
                           scaling_gas_atm, &
                           scale_by_complement_atm, &
                           kminor_start_atm, &
                           kminor_atm_red, &
                           minor_gases_atm_red, &
                           minor_limits_gpt_atm_red, &
                           minor_scales_with_density_atm_red, &
                           scaling_gas_atm_red, &
                           scale_by_complement_atm_red, &
                           kminor_start_atm_red) bind(C, name="reduce_minor_arrays")

    character(len=*), dimension(:),     intent(in) :: available_gases
    character(len=*), dimension(:),     intent(in) :: gas_names
    real(wp),         dimension(:,:,:), intent(in) :: kminor_atm
    character(len=*), dimension(:),     intent(in) :: gas_minor, &
                                                      identifier_minor
    character(len=*), dimension(:),     intent(in) :: minor_gases_atm
    integer,          dimension(:,:),   intent(in) :: minor_limits_gpt_atm
    logical(wl),      dimension(:),     intent(in) :: minor_scales_with_density_atm
    character(len=*), dimension(:),     intent(in) :: scaling_gas_atm
    logical(wl),      dimension(:),     intent(in) :: scale_by_complement_atm
    integer,          dimension(:),     intent(in) :: kminor_start_atm
    real(wp),         dimension(:,:,:), allocatable, &
                                        intent(out) :: kminor_atm_red
    character(len=*), dimension(:), allocatable, &
                                        intent(out) :: minor_gases_atm_red
    integer,          dimension(:,:), allocatable, &
                                        intent(out) :: minor_limits_gpt_atm_red
    logical(wl),      dimension(:),    allocatable, &
                                        intent(out) ::minor_scales_with_density_atm_red
    character(len=*), dimension(:), allocatable, &
                                        intent(out) ::scaling_gas_atm_red
    logical(wl),      dimension(:), allocatable, intent(out) :: &
                                                scale_by_complement_atm_red
    integer,          dimension(:), allocatable, intent(out) :: &
                                                kminor_start_atm_red

    ! Local variables
    integer :: i, j, ks
    integer :: idx_mnr, nm, tot_g, red_nm
    integer :: icnt, n_elim, ng
    logical, dimension(:), allocatable :: gas_is_present
    integer, dimension(:), allocatable :: indexes

    nm = size(minor_gases_atm)
    tot_g=0
    allocate(gas_is_present(nm))
    do i = 1, size(minor_gases_atm)
      idx_mnr = string_loc_in_array(minor_gases_atm(i), identifier_minor)
      gas_is_present(i) = string_in_array(gas_minor(idx_mnr),available_gases)
      if(gas_is_present(i)) then
        tot_g = tot_g + (minor_limits_gpt_atm(2,i)-minor_limits_gpt_atm(1,i)+1)
      endif
    enddo
    red_nm = count(gas_is_present)

    allocate(minor_gases_atm_red              (red_nm),&
             minor_scales_with_density_atm_red(red_nm), &
             scaling_gas_atm_red              (red_nm), &
             scale_by_complement_atm_red      (red_nm), &
             kminor_start_atm_red             (red_nm))
    allocate(minor_limits_gpt_atm_red(2, red_nm))
    allocate(kminor_atm_red(tot_g, size(kminor_atm,2), size(kminor_atm,3)))

    if ((red_nm .eq. nm)) then
      ! Character data not allowed in OpenACC regions?
      minor_gases_atm_red         = minor_gases_atm
      scaling_gas_atm_red         = scaling_gas_atm
      kminor_atm_red              = kminor_atm
      minor_limits_gpt_atm_red    = minor_limits_gpt_atm
      minor_scales_with_density_atm_red = minor_scales_with_density_atm
      scale_by_complement_atm_red = scale_by_complement_atm
      kminor_start_atm_red        = kminor_start_atm
    else
      allocate(indexes(red_nm))
      ! Find the integer indexes for the gases that are present
      indexes = pack([(i, i = 1, size(minor_gases_atm))], mask=gas_is_present)

      minor_gases_atm_red  = minor_gases_atm        (indexes)
      scaling_gas_atm_red  = scaling_gas_atm        (indexes)
      minor_scales_with_density_atm_red = &
                             minor_scales_with_density_atm(indexes)
      scale_by_complement_atm_red = &
                             scale_by_complement_atm(indexes)
      kminor_start_atm_red = kminor_start_atm       (indexes)

      icnt = 0
      n_elim = 0
      do i = 1, nm
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
  end subroutine reduce_minor_arrays
  
  ! --------------------------------------------------------------------------------------
  ! create index list for extracting col_gas needed for minor gas optical depth calculations
  ! --------------------------------------------------------------------------------------
  subroutine create_idx_minor(gas_names, &
    gas_minor, identifier_minor, minor_gases_atm, idx_minor_atm) bind(C, name="create_idx_minor")
    character(len=*), dimension(:), intent(in) :: gas_names
    character(len=*), dimension(:), intent(in) :: &
                                                  gas_minor, &
                                                  identifier_minor
    character(len=*), dimension(:), intent(in) :: minor_gases_atm
    integer, dimension(:), allocatable, &
                                   intent(out) :: idx_minor_atm

    ! local
    integer :: imnr
    integer :: idx_mnr
    allocate(idx_minor_atm(size(minor_gases_atm,dim=1)))
    do imnr = 1, size(minor_gases_atm,dim=1) ! loop over minor absorbers in each band
          ! Find identifying string for minor species in list of possible identifiers (e.g. h2o_slf)
          idx_mnr     = string_loc_in_array(minor_gases_atm(imnr), identifier_minor)
          ! Find name of gas associated with minor species identifier (e.g. h2o)
          idx_minor_atm(imnr) = string_loc_in_array(gas_minor(idx_mnr),    gas_names)
    enddo
  end subroutine create_idx_minor  

  ! --------------------------------------------------------------------------------------
  ! create index for special treatment in density scaling of minor gases
  ! --------------------------------------------------------------------------------------
  subroutine create_idx_minor_scaling(gas_names, &
    scaling_gas_atm, idx_minor_scaling_atm) bind(C, name="create_idx_minor_scaling")
    character(len=*), dimension(:), intent(in) :: gas_names
    character(len=*), dimension(:), intent(in) :: scaling_gas_atm
    integer, dimension(:), allocatable, &
                                   intent(out) :: idx_minor_scaling_atm

    ! local
    integer :: imnr
    allocate(idx_minor_scaling_atm(size(scaling_gas_atm,dim=1)))
    do imnr = 1, size(scaling_gas_atm,dim=1) ! loop over minor absorbers in each band
          ! This will be -1 if there's no interacting gas
          idx_minor_scaling_atm(imnr) = string_loc_in_array(scaling_gas_atm(imnr), gas_names)
    enddo
  end subroutine create_idx_minor_scaling  
  
  ! --------------------------------------------------------------------------------------
  !
  ! --------------------------------------------------------------------------------------  
  subroutine create_key_species_reduce(gas_names,gas_names_red, &
    key_species,key_species_red,key_species_present_init) bind(C, name="create_key_species_reduce")
    character(len=*), &
              dimension(:),       intent(in) :: gas_names
    character(len=*), &
              dimension(:),       intent(in) :: gas_names_red
    integer,  dimension(:,:,:),   intent(in) :: key_species
    integer,  dimension(:,:,:), allocatable, intent(out) :: key_species_red

    logical, dimension(:), allocatable, intent(out) :: key_species_present_init
    integer :: ip, ia, it, np, na, nt

    np = size(key_species,dim=1)
    na = size(key_species,dim=2)
    nt = size(key_species,dim=3)
    allocate(key_species_red(size(key_species,dim=1), &
                             size(key_species,dim=2), &
                             size(key_species,dim=3)))
    allocate(key_species_present_init(size(gas_names)))
    key_species_present_init = .true.

    do ip = 1, np
      do ia = 1, na
        do it = 1, nt
          if (key_species(ip,ia,it) .ne. 0) then
            key_species_red(ip,ia,it) = string_loc_in_array(gas_names(key_species(ip,ia,it)),gas_names_red)
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
  subroutine create_flavor(key_species, flavor) bind(C, name="create_flavor")
    integer, dimension(:,:,:), intent(in) :: key_species
    integer, dimension(:,:), allocatable, intent(out) :: flavor
    integer, dimension(2,size(key_species,3)*2) :: key_species_list

    integer :: ibnd, iatm, i, iflavor
    ! prepare list of key_species
    i = 1
    do ibnd=1,size(key_species,3)
      do iatm=1,size(key_species,1)
        key_species_list(:,i) = key_species(:,iatm,ibnd)
        i = i + 1
      end do
    end do
    ! rewrite single key_species pairs
    do i=1,size(key_species_list,2)
        key_species_list(:,i) = rewrite_key_species_pair(key_species_list(:,i))
    end do
    ! count unique key species pairs
    iflavor = 0
    do i=1,size(key_species_list,2)
      if (.not.key_species_pair_exists(key_species_list(:,1:i-1),key_species_list(:,i))) then
        iflavor = iflavor + 1
      end if
    end do
    ! fill flavors
    allocate(flavor(2,iflavor))
    iflavor = 0
    do i=1,size(key_species_list,2)
      if (.not.key_species_pair_exists(key_species_list(:,1:i-1),key_species_list(:,i))) then
        iflavor = iflavor + 1
        flavor(:,iflavor) = key_species_list(:,i)
      end if
    end do
  end subroutine create_flavor  
  
  ! --------------------------------------------------------------------------------------
  ! create gpoint_flavor list
  !   a map pointing from each g-point to the corresponding entry in the "flavor list"
  ! --------------------------------------------------------------------------------------
  subroutine create_gpoint_flavor(key_species, gpt2band, flavor, gpoint_flavor) bind(C, name="create_gpoint_flavor")
    integer, dimension(:,:,:), intent(in) :: key_species
    integer, dimension(:), intent(in) :: gpt2band
    integer, dimension(:,:), intent(in) :: flavor
    integer, dimension(:,:), intent(out), allocatable :: gpoint_flavor
    integer :: ngpt, igpt, iatm
    ngpt = size(gpt2band)
    allocate(gpoint_flavor(2,ngpt))
    do igpt=1,ngpt
      do iatm=1,2
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
