subroutine read_init_min_zones_CHEPROO(this,unit,init_min_zones,reactive_zones)
    use chemistry_Lagr_m
    implicit none
    class(chemistry_c) :: this
    integer(kind=4), intent(in) :: unit !> file
    type(solid_type_c), intent(out), allocatable :: init_min_zones(:)
    type(reactive_zone_c), intent(inout), allocatable, optional :: reactive_zones(:)
    
    integer(kind=4) :: i,j,k,nmtype,nrwtype,icon,num_min_zones,num_mins_rz,num_mins_glob,num_mins_loc,ind_rz,imtype,num_rz,min_ind,num_mins_var,num_mins_cst
    integer(kind=4), allocatable :: init_min_zones_indices(:),min_indices(:,:)
    character(len=256) :: str,constrain,label,min_name
    real(kind=8) :: guess,c_tot,temp,vol_frac,react_surf,conc 
    logical :: min_flag
    type(mineral_c) :: mineral
    type(reactive_zone_c) :: react_zone
    type(reactive_zone_c), allocatable :: react_zones(:)
    

    allocate(react_zones(this%chem_syst%num_minerals)) !> chapuza

    read(unit,*) nmtype
    
    allocate(min_indices(this%chem_syst%num_minerals,nmtype))
    
    call react_zone%set_chem_syst_react_zone(this%chem_syst)
    call react_zone%allocate_minerals_react_zone(this%chem_syst%num_minerals)
    call react_zone%allocate_non_flowing_species()
    call react_zone%set_num_solids()
    call react_zone%set_minerals_react_zone()
    
    if (nmtype>0) then
        num_rz=nmtype
        allocate(init_min_zones(nmtype))
        num_min_zones=0 !> counter mineral zones
        do
            read(unit,*) imtype, temp !> index of mineral zone, temperature
            if (imtype<1 .or. imtype>nmtype) error stop
                read(unit,*) str
                if (str=='*') then
                    if (num_min_zones<nmtype) then
                        continue
                    else
                        exit
                    end if
                else if (index(str,'mineral')/=0) then
                    num_min_zones=num_min_zones+1
                    num_mins_loc=0 !> counter minerals in this zone
                    num_mins_cst=0
                    num_mins_var=0
                    do
                        read(unit,*) mineral%name
                        if (mineral%name=='*') then
                            !call react_zone%set_chem_syst_react_zone(this%chem_syst)
                            !call react_zone%allocate_minerals_react_zone(num_mins_loc)
                            !call react_zone%allocate_non_flowing_species()
                            !call react_zone%set_num_solids()
                            !call react_zone%set_num_mins_cst_act(num_mins_cst)
                            !call react_zone%set_num_mins_var_act(num_mins_var)
                            !> aqui habria que comprobar si se estan repitiendo zonas reactivas
                            call init_min_zones(imtype)%solid_chem%set_reactive_zone(react_zone)
                            call init_min_zones(imtype)%solid_chem%allocate_vol_fracts()
                            call init_min_zones(imtype)%solid_chem%allocate_react_surfaces()
                            call init_min_zones(imtype)%solid_chem%allocate_conc_solids()
                            call init_min_zones(imtype)%solid_chem%allocate_activities()
                            call init_min_zones(imtype)%solid_chem%set_temp(temp)
                            call init_min_zones(imtype)%solid_chem%allocate_var_act_species_indices(react_zone%num_minerals_var_act)
                            call init_min_zones(imtype)%solid_chem%allocate_cst_act_species_indices(react_zone%num_minerals_cst_act)
                            call init_min_zones(imtype)%solid_chem%set_indices_solids()
                            exit
                        else
                            call this%chem_syst%is_mineral_in_chem_syst(mineral,min_flag,min_ind)
                            if (min_flag==.true.) then
                                num_mins_loc=num_mins_loc+1
                                min_indices(num_mins_loc,num_min_zones)=min_ind
                                if (this%chem_syst%minerals(min_ind)%mineral%cst_act_flag==.true.) then
                                    num_mins_cst=num_mins_cst+1
                                else
                                    num_mins_var=num_mins_var+1
                                end if
                            else
                                error stop
                            end if
                        end if
                    end do
                else
                    exit
                end if
            !end do
            if (num_min_zones==nmtype) exit
        end do
        rewind(unit)
    !> Second iteration
        num_min_zones=0 !> counter mineral zones
        do
            read(unit,*) label
            if (label=='INITIAL MINERAL ZONES') then
                read(unit,*) nmtype !> number of mineral zones
                do
                    read(unit,*) imtype, temp
                    if (imtype<1 .or. imtype>nmtype) error stop
                    read(unit,*) str
                    if (index(str,'mineral')/=0) then
                        num_min_zones=num_min_zones+1
                        num_mins_loc=0 !> counter minerals in this zone
                        do
                            read(unit,*) mineral%name, vol_frac, react_surf !> react_surf: [m^2 min./m^3 rock]
                            if (mineral%name=='*') then
                                !call init_min_zones(imtype)%solid_chem%react_zones(imtype)%set_eq_reactions()
                                !call init_min_zones(imtype)%solid_chem%react_zones(imtype)%set_stoich_mat_react_zone()
                                exit
                            else
                                num_mins_loc=num_mins_loc+1
                                init_min_zones(imtype)%solid_chem%reactive_zone%minerals(min_indices(num_mins_loc,num_min_zones))=this%chem_syst%minerals(min_indices(num_mins_loc,num_min_zones)) !> we set mineral
                                init_min_zones(imtype)%solid_chem%vol_fracts(min_indices(num_mins_loc,num_min_zones))=vol_frac !> we set volumetric fraction
                                init_min_zones(imtype)%solid_chem%react_surfaces(min_indices(num_mins_loc,num_min_zones))=react_surf !> we set reactive surface
                                init_min_zones(imtype)%solid_chem%concentrations(min_indices(num_mins_loc,num_min_zones))=1d0 !> we assume minerals are pure phases
                                init_min_zones(imtype)%solid_chem%activities(min_indices(num_mins_loc,num_min_zones))=1d0 !> we assume minerals are pure phases
                            end if
                        end do
                        if (num_min_zones==nmtype) then
                            exit
                        end if
                    else
                        exit
                    end if
                end do
                exit
            else
                continue
            end if
        end do
        if (present(reactive_zones)) then
            allocate(reactive_zones(nmtype))
            do i=1,nmtype
                reactive_zones(i)=init_min_zones(i)%solid_chem%reactive_zone
            end do
        end if
    end if
end subroutine