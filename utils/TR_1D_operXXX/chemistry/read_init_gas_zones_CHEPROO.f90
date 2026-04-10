subroutine read_init_gas_zones_CHEPROO(this,unit,gas_zones,reactive_zones)
    use chemistry_Lagr_m
    implicit none
    class(chemistry_c) :: this
    integer(kind=4), intent(in) :: unit !> file
    type(gas_type_c), intent(out), allocatable :: gas_zones(:)
    type(reactive_zone_c), intent(inout), allocatable, optional :: reactive_zones(:)
    
    integer(kind=4) :: i,j,k,ngtype,igtype,nrwtype,gas_ind,num_gas_zones,num_gases_loc,num_gases_var,num_gases_cst
    integer(kind=4), allocatable :: ind_gases(:)
    character(len=256) :: str,constrain,label
    real(kind=8) :: guess,conc,temp,part_press,vol
    type(gas_c) :: gas
    type(gas_phase_c) :: gas_phase
    type(reactive_zone_c) :: react_zone
    logical :: flag

    read(unit,*) ngtype !> number of gas zones
    
    allocate(gas_zones(ngtype))
    allocate(ind_gases(this%chem_syst%gas_phase%num_species)) !> chapuza

    num_gas_zones=0 !> counter gas zones
    !num_gases_glob=1 !> counter gases in chemical system
    
    do
        read(unit,*) igtype, temp, vol
        if (igtype<1 .or. igtype>ngtype) error stop
        read(unit,*) str
        if (str=='*') then
            if (num_gas_zones<ngtype) then
                continue
            else
                exit
            end if
        else if (index(str,'gas')/=0) then
            num_gas_zones=num_gas_zones+1
            num_gases_loc=0 !> counter gases in this zone
            num_gases_var=0
            num_gases_cst=0
            do
                read(unit,*) gas%name
                if (gas%name=='*') then
                    call react_zone%set_chem_syst_react_zone(this%chem_syst)
                    call react_zone%gas_phase%allocate_gases(num_gases_loc)
                    call react_zone%gas_phase%set_num_var_act_species_phase(num_gases_var)
                    call react_zone%gas_phase%set_num_cst_act_species_phase(num_gases_cst)
                    call gas_zones(igtype)%gas_chem%set_reactive_zone(react_zone)
                    call gas_zones(igtype)%gas_chem%set_temp(temp+273.15) !> Kelvin
                    call gas_zones(igtype)%gas_chem%set_volume(vol)
                    call gas_zones(igtype)%gas_chem%allocate_partial_pressures()
                    call gas_zones(igtype)%gas_chem%allocate_conc_gases()
                    call gas_zones(igtype)%gas_chem%allocate_log_act_coeffs_gases()
                    call gas_zones(igtype)%gas_chem%allocate_var_act_species_indices(num_gases_var)
                    call gas_zones(igtype)%gas_chem%allocate_cst_act_species_indices(num_gases_cst)
                    call gas_zones(igtype)%gas_chem%set_indices_gases()
                    exit
                else
                    call this%chem_syst%gas_phase%is_gas_in_gas_phase(gas,flag,gas_ind)
                    if (flag==.true.) then
                        num_gases_loc=num_gases_loc+1
                        ind_gases(num_gases_loc)=gas_ind
                        if (this%chem_syst%gas_phase%gases(gas_ind)%cst_act_flag==.true.) then
                            num_gases_cst=num_gases_cst+1
                        else
                            num_gases_var=num_gases_var+1
                        end if
                    else
                        error stop
                    end if
                end if
            end do
            if (num_gas_zones==ngtype) exit
        else
            exit
        end if
    end do
    rewind(unit)
    num_gas_zones=0 !> counter gas zones
    !num_gases_glob=1 !> counter gases in chemical system
    do
        read(unit,*) label
        if (label=='INITIAL GAS ZONES') then
            read(unit,*) ngtype !> number of gas zones
            do i=1,ngtype
                read(unit,*) igtype, temp
                read(unit,*) str
                !call gas_zones(igtype)%gas_chem%set_temp(temp)
                num_gas_zones=num_gas_zones+1
                !num_gases_loc=0 !> counter gases in this zone
                do j=1,gas_zones(igtype)%gas_chem%reactive_zone%gas_phase%num_species
                    read(unit,*) gas%name, part_press
                    gas_zones(igtype)%gas_chem%reactive_zone%gas_phase%gases(j)=this%chem_syst%gas_phase%gases(ind_gases(j)) !> we set gas
                    gas_zones(igtype)%gas_chem%activities(j)=part_press !> activities are partial pressures
                    !gas_zones(igtype)%gas_chem%concentrations(j)=conc !> concentrations are moles
                end do
                call gas_zones(igtype)%gas_chem%compute_conc_gases_ideal() !> we assume gases are ideal
                call gas_zones(igtype)%gas_chem%compute_pressure()
                !call gas_zones(igtype)%gas_chem%compute_vol_gas()
                call gas_zones(igtype)%gas_chem%compute_log_act_coeffs_gases()
                read(unit,*) str
            end do
        else if (label=='end') then
            backspace(unit)
            exit
        else
            continue
        end if
    end do
    if (present(reactive_zones)) then
        if (.not. allocated(reactive_zones)) then
            allocate(reactive_zones(ngtype))
        end if
        do i=1,size(reactive_zones)
            reactive_zones(i)%gas_phase=gas_zones(i)%gas_chem%reactive_zone%gas_phase
        end do
    end if
end subroutine