!!> Reads initial target waters and their associated target solids (if present)
!!> We assume file has already been opened
!subroutine read_target_waters_init_bis(this,unit,init_water_types,bd_water_types,init_sol_types,niter,CV_flag)
!    use chemistry_Lagr_m
!    implicit none
!    class(chemistry_c) :: this
!    integer(kind=4), intent(in) :: unit !> file
!    class(water_type_c), intent(in) :: init_water_types(:)
!    class(water_type_c), intent(in) :: bd_water_types(:)
!    class(solid_type_c), intent(in) :: init_sol_types(:) !> includes mineral and adsorption zones (in that order)
!    !real(kind=8), intent(in) :: tolerance !> for linear system solver
!    !real(kind=8), intent(in) :: rel_tolerance !> for Delta_c1
!    !real(kind=8), intent(in) :: control_factor
!    !integer(kind=4), intent(in) :: niter_max
!    integer(kind=4), intent(out) :: niter !> number of iterations
!    logical, intent(out) :: CV_flag !> TRUE if converges, FALSE otherwise
!    
!    integer(kind=4) :: ind,i,j,k,m,niwtype,num_tar_wat,tar_wat_ind,iwtype,istype,nstype,nbwtype,bwtype,mix_wat_ind
!    real(kind=8), allocatable :: conc_init_species(:,:),u_init(:,:),c1_init(:,:),c2aq_init(:,:),gamma_1(:),gamma_2aq(:)
!    character(len=256) :: label
!    logical :: flag_comp,flag_surf
!    
!    niwtype=size(init_water_types)
!    nbwtype=size(bd_water_types)
!    nstype=size(init_sol_types)
!    
!    flag_comp=.true.
!    !if (num_surf_compl>0) then
!    !>    flag_surf=.true.
!    !else
!    !>    flag_surf=.false.
!    !end if
!    
!    do
!        read(unit,*) label
!        if (label=='end') then
!            exit
!        else if (label=='INITIAL TARGET WATERS') then
!            read(unit,*) num_tar_wat
!            if (num_tar_wat<0) error stop
!            call this%allocate_target_waters(num_tar_wat)
!            call this%allocate_target_solids(num_tar_wat) !> we assume bijection with target waters
!            call this%allocate_ext_waters()
!            if (nstype>0 .and. nbwtype>0) then
!                read(unit,*) tar_wat_ind, bwtype, istype
!            !> El if de abajo deberia estar en una subrutina
!                if (tar_wat_ind<1 .or. tar_wat_ind>this%num_target_waters) then
!                    error stop
!                else if (bwtype<1 .or. bwtype>nbwtype) then
!                    error stop
!                else if (istype<1 .or. istype>nstype) then
!                    error stop
!                else
!                    this%target_waters_init(tar_wat_ind)=bd_water_types(bwtype)%aq_chem
!                    call this%target_waters_init(tar_wat_ind)%set_chem_syst_aq_chem(this%chem_syst) !> chapuza
!                    call this%target_waters_init(tar_wat_ind)%set_aq_phase(this%chem_syst%aq_phase) !> chapuza
!                    this%target_solids_init(tar_wat_ind)=init_sol_types(istype)%solid_chem
!                    !call this%target_waters_init(tar_wat_ind)%set_solid_chemistry(init_min_zones(imtype))
!                    !print *, init_sol_types(istype)%solid_chem%vol_fracts
!                    call this%target_solids_init(tar_wat_ind)%set_reactive_zone(init_sol_types(istype)%solid_chem%reactive_zone)
!                    !call this%target_solids(tar_wat_ind)%set_reactive_zone(this%reactive_zones(1)) !> autentica chapuza ad hoc
!                    !call this%target_solids(tar_wat_ind)%allocate_concentrations()
!                    !print *, this%target_solids(tar_wat_ind)%reactive_zone%Se
!                    call this%target_waters_init(tar_wat_ind)%set_solid_chemistry(this%target_solids_init(tar_wat_ind))
!                    call this%target_waters_init(tar_wat_ind)%set_speciation_alg_dimensions(flag_comp)
!                    call this%target_waters_init(tar_wat_ind)%compute_speciation_alg_arrays()
!                    call this%target_waters_init(tar_wat_ind)%set_prim_species_indices()
!                    call this%target_waters_init(tar_wat_ind)%set_sec_var_act_species_indices()
!                    call this%target_waters_init(tar_wat_ind)%compute_U_SkT_prod()
!                    !call this%target_waters_init(tar_wat_ind)%compute_c2nc_aq_from_c1_aq_expl() !> rezaei
!                    !call this%target_waters_init(tar_wat_ind)%compute_c2nc_from_c1_aq_Picard(niter,CV_flag) !> rezaei
!                    !call this%target_waters_init(tar_wat_ind)%compute_activities_aq_var_act_species()
!                    !call this%target_waters_init(tar_wat_ind)%compute_pH()
!                    !call this%target_waters_init(tar_wat_ind)%compute_act_water()
!                    !call this%target_waters_init(tar_wat_ind)%compute_salinity()
!                    !call this%target_waters_init(tar_wat_ind)%compute_alkalinity()
!                    !call this%target_waters_init(tar_wat_ind)%compute_molarities()
!                    !call this%target_waters_init(tar_wat_ind)%compute_c_nc_from_u_Newton(tolerance,rel_tolerance,control_factor,niter_max,niter,CV_flag)
!                    call this%target_waters_init(tar_wat_ind)%compute_conc_comp_aq() !> rezaei
!                    call this%target_waters_init(tar_wat_ind)%allocate_reaction_rates()
!                    
!                    call this%ext_waters(tar_wat_ind)%set_aq_phase(this%target_waters_init(tar_wat_ind)%aq_phase)
!                    call this%ext_waters(tar_wat_ind)%set_solid_chemistry(this%target_waters_init(tar_wat_ind)%solid_chemistry)
!                    call this%ext_waters(tar_wat_ind)%set_speciation_alg(this%target_waters_init(tar_wat_ind)%speciation_alg)
!                    call this%ext_waters(tar_wat_ind)%allocate_conc_aq_species()
!                    call this%ext_waters(tar_wat_ind)%allocate_conc_comp()
!                    call this%ext_waters(tar_wat_ind)%set_conc_aq_species()
!                    call this%ext_waters(tar_wat_ind)%compute_conc_comp_aq()
!                end if
!            end if
!            if (nstype>0) then
!            !> El do de abajo deberia estar en una subrutina
!                do i=2,this%num_target_waters !> chapuza
!                    read(unit,*) tar_wat_ind, iwtype, istype
!                    if (tar_wat_ind<1 .or. tar_wat_ind>this%num_target_waters) then
!                        error stop
!                    else if (iwtype<1 .or. iwtype>niwtype) then
!                        error stop
!                    else if (istype<1 .or. istype>nstype) then
!                        error stop
!                    else
!                        this%target_waters_init(tar_wat_ind)=init_water_types(iwtype)%aq_chem
!                        call this%target_waters_init(tar_wat_ind)%set_chem_syst_aq_chem(this%chem_syst)
!                        call this%target_waters_init(tar_wat_ind)%set_aq_phase(this%chem_syst%aq_phase)
!                        this%target_solids_init(tar_wat_ind)=init_sol_types(istype)%solid_chem
!                        !call this%target_waters_init(tar_wat_ind)%set_solid_chemistry(init_min_zones(imtype))
!                        !print *, init_sol_types(istype)%reactive_zone%Se
!                        call this%target_solids_init(tar_wat_ind)%set_reactive_zone(init_sol_types(istype)%solid_chem%reactive_zone)
!                        !call this%target_solids(tar_wat_ind)%set_reactive_zone(this%reactive_zones(1)) !> autentica chapuza ad hoc
!                        !call this%target_solids(tar_wat_ind)%allocate_concentrations()
!                        !print *, this%target_solids(tar_wat_ind)%reactive_zone%Se
!                        call this%target_waters_init(tar_wat_ind)%set_solid_chemistry(this%target_solids_init(tar_wat_ind))
!                        call this%target_waters_init(tar_wat_ind)%set_speciation_alg_dimensions(flag_comp)
!                        call this%target_waters_init(tar_wat_ind)%compute_speciation_alg_arrays()
!                        call this%target_waters_init(tar_wat_ind)%set_prim_species_indices()
!                        call this%target_waters_init(tar_wat_ind)%set_sec_var_act_species_indices()
!                        call this%target_waters_init(tar_wat_ind)%compute_U_SkT_prod()
!                        !call this%target_waters_init(tar_wat_ind)%compute_c2nc_aq_from_c1_aq_expl() !> rezaei
!                        !call this%target_waters_init(tar_wat_ind)%compute_c2nc_from_c1_aq_Picard(niter,CV_flag) !> rezaei
!                        !call this%target_waters_init(tar_wat_ind)%compute_activities_aq_var_act_species()
!                        !call this%target_waters_init(tar_wat_ind)%compute_pH()
!                        !call this%target_waters_init(tar_wat_ind)%compute_act_water()
!                        !call this%target_waters_init(tar_wat_ind)%compute_salinity()
!                        !call this%target_waters_init(tar_wat_ind)%compute_alkalinity()
!                        !call this%target_waters_init(tar_wat_ind)%compute_molarities()
!                        !call this%target_waters_init(tar_wat_ind)%compute_c_nc_from_u_Newton(tolerance,rel_tolerance,control_factor,niter_max,niter,CV_flag)
!                        call this%target_waters_init(tar_wat_ind)%compute_conc_comp_aq()
!                        call this%target_waters_init(tar_wat_ind)%allocate_reaction_rates()
!                        call this%ext_waters(tar_wat_ind)%set_aq_phase(this%target_waters_init(tar_wat_ind)%aq_phase)
!                        call this%ext_waters(tar_wat_ind)%set_solid_chemistry(this%target_waters_init(tar_wat_ind)%solid_chemistry)
!                        call this%ext_waters(tar_wat_ind)%set_speciation_alg(this%target_waters_init(tar_wat_ind)%speciation_alg)
!                        call this%ext_waters(tar_wat_ind)%allocate_conc_aq_species()
!                        call this%ext_waters(tar_wat_ind)%allocate_conc_comp()
!                        call this%ext_waters(tar_wat_ind)%set_conc_aq_species()
!                        call this%ext_waters(tar_wat_ind)%compute_conc_comp_aq()
!                    end if
!                end do
!            end if
!            !> El if de abajo deberia estar en una subrutina
!            if (nstype>0 .and. nbwtype>0) then
!            !    read(unit,*) tar_wat_ind, bwtype, istype
!            !    if (tar_wat_ind<1 .or. tar_wat_ind>this%num_target_waters) then
!            !        error stop
!            !    else if (bwtype<1 .or. bwtype>nbwtype) then
!            !        error stop
!            !    else if (istype<1 .or. istype>nstype) then
!            !        error stop
!            !    else
!            !        this%target_waters_init(tar_wat_ind)=bd_water_types(bwtype)%aq_chem
!            !        call this%target_waters_init(tar_wat_ind)%set_chem_syst_aq_chem(this%chem_syst) !> chapuza
!            !        call this%target_waters_init(tar_wat_ind)%set_aq_phase(this%chem_syst%aq_phase) !> chapuza
!            !        !call this%target_waters_init(tar_wat_ind)%set_aq_phase(bd_water_types(bwtype)%aq_chem%aq_phase)
!            !        this%target_solids_init(tar_wat_ind)=init_sol_types(istype)%solid_chem
!            !        !call this%target_waters_init(tar_wat_ind)%set_solid_chemistry(init_min_zones(imtype))
!            !        !print *, init_sol_types(istype)%reactive_zone%Se
!            !        call this%target_solids_init(tar_wat_ind)%set_reactive_zone(init_sol_types(istype)%solid_chem%reactive_zone)
!            !        !call this%target_solids(tar_wat_ind)%set_reactive_zone(this%reactive_zones(1)) !> autentica chapuza ad hoc
!            !        !call this%target_solids(tar_wat_ind)%allocate_concentrations()
!            !        !print *, this%target_solids(tar_wat_ind)%reactive_zone%Se
!            !        call this%target_waters_init(tar_wat_ind)%set_solid_chemistry(this%target_solids_init(tar_wat_ind))
!            !        call this%target_waters_init(tar_wat_ind)%set_speciation_alg_dimensions(flag_comp)
!            !        call this%target_waters_init(tar_wat_ind)%compute_speciation_alg_arrays()
!            !        call this%target_waters_init(tar_wat_ind)%set_prim_species_indices()
!            !        call this%target_waters_init(tar_wat_ind)%set_sec_var_act_species_indices()
!            !        call this%target_waters_init(tar_wat_ind)%compute_U_SkT_prod()
!            !        !call this%target_waters_init(tar_wat_ind)%compute_c2nc_aq_from_c1_aq_expl() !> rezaei
!            !        !call this%target_waters_init(tar_wat_ind)%compute_c2nc_from_c1_aq_Picard(niter,CV_flag) !> rezaei
!            !        !call this%target_waters_init(tar_wat_ind)%compute_activities_aq_var_act_species()
!            !        !call this%target_waters_init(tar_wat_ind)%compute_pH()
!            !        !call this%target_waters_init(tar_wat_ind)%compute_act_water()
!            !        !call this%target_waters_init(tar_wat_ind)%compute_salinity()
!            !        !call this%target_waters_init(tar_wat_ind)%compute_alkalinity()
!            !        !call this%target_waters_init(tar_wat_ind)%compute_molarities()
!            !        !call this%target_waters_init(tar_wat_ind)%compute_c_nc_from_u_Newton(tolerance,rel_tolerance,control_factor,niter_max,niter,CV_flag)
!            !        call this%target_waters_init(tar_wat_ind)%compute_conc_comp_aq()
!            !        call this%target_waters_init(tar_wat_ind)%allocate_reaction_rates()
!            !        
!            !        call this%ext_waters(tar_wat_ind)%set_aq_phase(this%target_waters_init(tar_wat_ind)%aq_phase)
!            !        call this%ext_waters(tar_wat_ind)%set_solid_chemistry(this%target_waters_init(tar_wat_ind)%solid_chemistry)
!            !        call this%ext_waters(tar_wat_ind)%set_speciation_alg(this%target_waters_init(tar_wat_ind)%speciation_alg)
!            !        call this%ext_waters(tar_wat_ind)%allocate_conc_aq_species()
!            !        call this%ext_waters(tar_wat_ind)%allocate_conc_comp()
!            !        call this%ext_waters(tar_wat_ind)%set_conc_aq_species()
!            !        call this%ext_waters(tar_wat_ind)%compute_conc_comp_aq()
!            !    end if
!                this%target_waters=this%target_waters_init
!                this%target_solids=this%target_solids_init
!                do i=1,this%num_target_waters
!                    call this%target_waters(i)%set_solid_chemistry(this%target_solids(i))
!                end do
!            end if
!            if (nstype==0) then
!                read(unit,*) tar_wat_ind, bwtype!, istype
!            !> El if de abajo deberia estar en una subrutina
!                if (tar_wat_ind<1 .or. tar_wat_ind>this%num_target_waters) then
!                    error stop
!                else if (bwtype<1 .or. bwtype>nbwtype) then
!                    error stop
!                !else if (istype<1 .or. istype>nstype) then
!                !    error stop
!                else
!                    this%target_waters_init(tar_wat_ind)=bd_water_types(bwtype)%aq_chem
!                    call this%target_waters_init(tar_wat_ind)%set_chem_syst_aq_chem(this%chem_syst) !> chapuza
!                    call this%target_waters_init(tar_wat_ind)%set_aq_phase(this%chem_syst%aq_phase) !> chapuza
!                    !this%target_solids_init(tar_wat_ind)=init_sol_types(istype)%solid_chem
!                    !call this%target_waters_init(tar_wat_ind)%set_solid_chemistry(init_min_zones(imtype))
!                    !print *, init_sol_types(istype)%solid_chem%vol_fracts
!                    !call this%target_solids_init(tar_wat_ind)%set_reactive_zone(init_sol_types(istype)%solid_chem%reactive_zone)
!                    !call this%target_solids(tar_wat_ind)%set_reactive_zone(this%reactive_zones(1)) !> autentica chapuza ad hoc
!                    !call this%target_solids(tar_wat_ind)%allocate_concentrations()
!                    !print *, this%target_solids(tar_wat_ind)%reactive_zone%Se
!                    !call this%target_waters_init(tar_wat_ind)%set_solid_chemistry(this%target_solids_init(tar_wat_ind))
!                    call this%target_waters_init(tar_wat_ind)%set_speciation_alg_dimensions(flag_comp)
!                    call this%target_waters_init(tar_wat_ind)%compute_speciation_alg_arrays()
!                    call this%target_waters_init(tar_wat_ind)%set_prim_species_indices()
!                    call this%target_waters_init(tar_wat_ind)%set_sec_var_act_species_indices()
!                    call this%target_waters_init(tar_wat_ind)%compute_U_SkT_prod()
!                    !call this%target_waters_init(tar_wat_ind)%compute_c2nc_aq_from_c1_aq_expl() !> rezaei
!                    !call this%target_waters_init(tar_wat_ind)%compute_c2nc_from_c1_aq_Picard(niter,CV_flag) !> rezaei
!                    !call this%target_waters_init(tar_wat_ind)%compute_activities_aq_var_act_species()
!                    !call this%target_waters_init(tar_wat_ind)%compute_pH()
!                    !call this%target_waters_init(tar_wat_ind)%compute_act_water()
!                    !call this%target_waters_init(tar_wat_ind)%compute_salinity()
!                    !call this%target_waters_init(tar_wat_ind)%compute_alkalinity()
!                    !call this%target_waters_init(tar_wat_ind)%compute_molarities()
!                    !call this%target_waters_init(tar_wat_ind)%compute_c_nc_from_u_Newton(tolerance,rel_tolerance,control_factor,niter_max,niter,CV_flag)
!                    call this%target_waters_init(tar_wat_ind)%compute_conc_comp_aq() !> rezaei
!                    call this%target_waters_init(tar_wat_ind)%allocate_reaction_rates()
!                    
!                    call this%ext_waters(tar_wat_ind)%set_aq_phase(this%target_waters_init(tar_wat_ind)%aq_phase)
!                    !call this%ext_waters(tar_wat_ind)%set_solid_chemistry(this%target_waters_init(tar_wat_ind)%solid_chemistry)
!                    call this%ext_waters(tar_wat_ind)%set_speciation_alg(this%target_waters_init(tar_wat_ind)%speciation_alg)
!                    call this%ext_waters(tar_wat_ind)%allocate_conc_aq_species()
!                    call this%ext_waters(tar_wat_ind)%allocate_conc_comp()
!                    call this%ext_waters(tar_wat_ind)%set_conc_aq_species()
!                    call this%ext_waters(tar_wat_ind)%compute_conc_comp_aq()
!                end if
!                do i=2,this%num_target_waters
!                    read(unit,*) tar_wat_ind, iwtype
!                    if (tar_wat_ind<1 .or. tar_wat_ind>this%num_target_waters) then
!                        error stop
!                    else if (iwtype<1 .or. iwtype>niwtype) then
!                        error stop
!                    else
!                        this%target_waters_init(tar_wat_ind)=init_water_types(iwtype)%aq_chem
!                        call this%target_waters_init(tar_wat_ind)%set_chem_syst_aq_chem(this%chem_syst)
!                        if (this%target_waters_init(tar_wat_ind)%chem_syst%num_eq_reacts_homog>0) then
!                            call this%target_waters_init(tar_wat_ind)%set_speciation_alg_dimensions(flag_comp)
!                            call this%target_waters_init(tar_wat_ind)%compute_speciation_alg_arrays()
!                            !call this%target_waters_init(tar_wat_ind)%set_conc_sec_var_act_species() !> initial guess c2nc
!                            if (allocated(this%target_waters_init(tar_wat_ind)%conc_comp)) then
!                                call this%target_waters_init(tar_wat_ind)%compute_c_nc_from_u_aq_Newton(niter,CV_flag)
!                            else
!                                call this%target_waters_init(tar_wat_ind)%compute_c2nc_from_c1_aq_Picard(niter,CV_flag)
!                                call this%target_waters_init(tar_wat_ind)%compute_conc_comp_aq()
!                            end if
!                            call this%target_waters_init(tar_wat_ind)%compute_U_SkT_prod()
!                        end if
!                        call this%target_waters_init(tar_wat_ind)%allocate_reaction_rates()
!                        call this%ext_waters(tar_wat_ind)%set_aq_phase(this%target_waters_init(tar_wat_ind)%aq_phase)
!                        call this%ext_waters(tar_wat_ind)%allocate_conc_aq_species()
!                        call this%ext_waters(tar_wat_ind)%allocate_conc_comp(this%target_waters_init(tar_wat_ind)%speciation_alg%num_prim_species)
!                        call this%ext_waters(tar_wat_ind)%compute_conc_comp_aq()
!                    end if
!                end do
!                this%target_waters=this%target_waters_init
!            end if
!        else
!            continue
!        end if
!    end do
!    !call this%set_reactive_zones()
!end subroutine