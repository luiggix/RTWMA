subroutine read_init_cat_exch_zones_CHEPROO(this,unit,init_cat_exch_zones,reactive_zones)
    use chemistry_Lagr_m
    use Gaines_Thomas_m
    implicit none
    class(chemistry_c) :: this
    integer(kind=4), intent(in) :: unit !> file
    type(solid_type_c), intent(out), allocatable :: init_cat_exch_zones(:)
    type(reactive_zone_c), intent(inout), allocatable, optional :: reactive_zones(:)
    
    integer(kind=4) :: i,j,k,ndtype,idtype,nrwtype,icon,num_ads_zones,num_mins_glob,num_mins_loc
    integer(kind=4), allocatable :: init_min_zones_indices(:)
    character(len=256) :: str,constrain,label
    real(kind=8) :: c_int,c_ext,spec_sorb_surf,CEC
    logical :: min_zone_flag
    type(mineral_c) :: mineral
    type(reactive_zone_c) :: react_zone
    type(Gaines_Thomas_c), target :: Gaines_Thomas

    read(unit,*) ndtype !> number of surface adsorption zones
    
    allocate(init_cat_exch_zones(ndtype))

!> Vamos a asumir que los complejos de superficie están en el mismo orden que en el sistema quimico
    num_ads_zones=0 !> counter adsorption zones
    do
        read(unit,*) idtype
        if (idtype<0 .or. idtype>ndtype) error stop
        read(unit,*) str, c_int
        read(unit,*) str, c_ext
        read(unit,*) str, spec_sorb_surf
        read(unit,*) str, CEC
        call react_zone%set_chem_syst_react_zone(this%chem_syst)
        call react_zone%set_cat_exch_zone()
        !call react_zone%cat_exch_zone%set_CEC(CEC)
        react_zone%cat_exch_zone%convention=>Gaines_Thomas !> by default
        call react_zone%set_non_flowing_species()
        call react_zone%set_eq_reactions()
        call react_zone%set_num_solids()
        call react_zone%set_stoich_mat_react_zone()
        call init_cat_exch_zones(idtype)%solid_chem%set_reactive_zone(react_zone)
        !call init_cat_exch_zones(idtype)%solid_chem%set_CEC(CEC)
        call init_cat_exch_zones(idtype)%solid_chem%allocate_conc_solids()
        !call init_cat_exch_zones(idtype)%solid_chem%allocate_conc_comp_solids(1) !> chapuza
        init_cat_exch_zones(idtype)%solid_chem%CEC=CEC !> falta un set
        call init_cat_exch_zones(idtype)%solid_chem%allocate_equivalents()
        call init_cat_exch_zones(idtype)%solid_chem%allocate_log_act_coeffs_solid_chem()
        !allocate(init_cat_exch_zones(idtype)%solid_chem%log_act_coeffs(init_cat_exch_zones(idtype)%solid_chem%reactive_zone%num_solids))
        init_cat_exch_zones(idtype)%solid_chem%log_act_coeffs=0d0 !> chapuza
        init_cat_exch_zones(idtype)%solid_chem%log_Jacobian_act_coeffs=0d0 !> chapuza
        !init_cat_exch_zones(idtype)%solid_chem%log_act_coeffs=-log10(init_cat_exch_zones(idtype)%solid_chem%concentrations) !> chapuza
        !call init_cat_exch_zones(idtype)%solid_chem%reactive_zone%cat_exch_zone%compute_log_act_coeffs_ads_cats(init_cat_exch_zones(idtype)%solid_chem%log_act_coeffs(2:init_cat_exch_zones(idtype)%solid_chem%reactive_zone%cat_exch_zone%num_surf_compl))
        call init_cat_exch_zones(idtype)%solid_chem%allocate_activities()
        init_cat_exch_zones(idtype)%solid_chem%concentrations(1)=1d-16 !> 'x-', chapuza
        !call init_cat_exch_zones(idtype)%solid_chem%compute_activities_ads_cats() !> chapuza
        !init_cat_exch_zones(idtype)%solid_chem%activities(init_cat_exch_zones(idtype)%solid_chem%reactive_zone%num_solids)=1d0-       
        num_ads_zones=num_ads_zones+1
        if (num_ads_zones==ndtype) exit
    end do
    if (present(reactive_zones)) then
        allocate(reactive_zones(ndtype))
        do i=1,ndtype
            reactive_zones(i)=init_cat_exch_zones(i)%solid_chem%reactive_zone
        end do
    end if
end subroutine