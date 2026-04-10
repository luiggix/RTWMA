!> This subroutine reads a water type from CHEPROO data input
subroutine read_wat_type_CHEPROO(this,n_p_aq,num_cstr,model,Jac_flag,unit,niter,CV_flag,surf_chem)
    use chem_type_m
    implicit none
    class(water_type_c) :: this
    integer(kind=4), intent(in) :: n_p_aq !> number of primary aqueous species
    integer(kind=4), intent(in) :: num_cstr !> number of constrains
    !integer(kind=4), intent(in) :: num_gas_cstr !> number of gas constrains
    integer(kind=4), intent(in) :: model !> activity coefficients model
    integer(kind=4), intent(in) :: Jac_flag !> 0: incremental coeffficinets, 1: analtical
    integer(kind=4), intent(in) :: unit !> file unit
    integer(kind=4), intent(out) :: niter !> number of iterations
    logical, intent(out) :: CV_flag !> TRUE if converges, FALSE otherwise
    class(solid_chemistry_c), intent(inout), optional :: surf_chem !> surface chemistry (if there are exchange reactions)
    !real(kind=8), intent(in), optional :: c1_surf !> chapuza
    !real(kind=8), intent(in), optional :: CEC !> chapuza
    !real(kind=8), intent(out), optional :: conc_exch(:) !> chapuza
    
    integer(kind=4) :: i,j,k,l,m,niwtype,nbwtype,nrwtype,gas_ind,min_ind,n_comp_aq,ind_cstr,n_gas_constr,n_aq_comp,icon,sp_ind
    integer(kind=4), allocatable :: icons(:),indices_constrains(:),gas_indices(:),n_icons(:),prim_indices(:)
    real(kind=8), allocatable :: ctots(:)
    character(len=256) :: prim_sp_name,label,aq_sp_name
    character(len=256), allocatable :: constrains(:)
    real(kind=8) :: temp,conc,ionic_act,guess,ctot
    logical :: flag_gas,flag_min,flag,flag_comp,flag_surf
    
    type(reactive_zone_c) :: react_zone
    type(gas_chemistry_c) :: gas_chem
    type(gas_c) :: gas
    type(mineral_c) :: mineral
    type(aq_species_c) :: aq_species
    type(species_c) :: constrain
    
    !do
    !    read(unit,*) label
    !    if (label=='INITIAL AND BOUNDARY WATER TYPES') then
    !        i=0 !> counter initial water types
    !        read(unit,*) model
    !        read(unit,*) this%num_init_wat_types, this%num_bd_wat_types, this%num_rech_wat_types
        if (n_p_aq<0 .or. n_p_aq>this%aq_chem%aq_phase%num_species) then
            error stop
        else if (num_cstr<0 .or. num_cstr>n_p_aq) then
            error stop
        else
            allocate(prim_indices(n_p_aq),icons(n_p_aq),ctots(n_p_aq),indices_constrains(num_cstr))
            if (this%aq_chem%chem_syst%gas_phase%num_species>0) then
                call react_zone%set_gas_phase(this%aq_chem%chem_syst%gas_phase)
                call gas_chem%set_reactive_zone(react_zone)
                call gas_chem%allocate_partial_pressures()
                call gas_chem%allocate_conc_gases()
                call gas_chem%allocate_log_act_coeffs_gases()
                call gas_chem%set_temp(this%aq_chem%temp)
                call gas_chem%set_volume(1d0) !> arbitrary
                call this%aq_chem%set_gas_chemistry(gas_chem)
            end if
        end if
        !call this%aq_chem%allocate_conc_comp(n_p_aq)
        !call this%aq_chem%allocate_log_act_coeffs()
        !call this%aq_chem%set_log_act_coeffs() 
        !call this%aq_chem%allocate_activities_aq_species()
        read(unit,*) label
        k=1 !> counter primary species
        l=1 !> counter constrains
        m=1 !> counter gas constrains
        if (index(label,'guess')/=0) then !> 'icon, guess, ctot, constrain'
            i=i+1
            allocate(n_icons(4))
            n_icons=0 !> number of each icon option
            do 
                read(unit,*) aq_species%name, icon, guess, ctot, constrain%name
                if (aq_species%name=='*') then
                    exit
                else
                    call this%aq_chem%aq_phase%is_species_in_aq_phase(aq_species,flag,sp_ind)
                    if (flag==.true.) then
                        prim_indices(k)=sp_ind
                        !> Chapuza
                        if (icon==1) then
                            n_icons(1)=n_icons(1)+1
                        else if (icon==2) then
                            n_icons(2)=n_icons(2)+1
                            n_aq_comp=n_aq_comp+1
                        else if (icon==3) then
                            n_icons(3)=n_icons(3)+1
                            if (aq_species%name=='h+') then
                                call this%aq_chem%aq_phase%set_ind_proton(sp_ind)
                                call this%aq_chem%set_pH(-log10(ctot))
                            end if
                        else if (icon==4) then
                            n_icons(4)=n_icons(4)+1
                            call this%aq_chem%chem_syst%is_eq_reaction_in_chem_syst(constrain%name,flag,ind_cstr)
                            if (flag==.true.) then
                                indices_constrains(l)=ind_cstr
                                l=l+1
                                if (ind_cstr>this%aq_chem%chem_syst%num_minerals_eq+this%aq_chem%aq_phase%num_aq_complexes+this%aq_chem%chem_syst%num_redox_eq_reacts) then
                                    this%aq_chem%gas_chemistry%activities(ind_cstr-this%aq_chem%chem_syst%num_minerals_eq-this%aq_chem%aq_phase%num_aq_complexes-this%aq_chem%chem_syst%num_redox_eq_reacts)=ctot
                                end if
                                !call this%aq_chem%chem_syst%gas_phase%is_gas_in_gas_phase(gas,flag,gas_ind)
                            else
                                error stop
                            end if
                        !else if (icon==5) then
                        !    n_icons(6)=n_icons(6)+1
                        else
                            error stop "icon option not implemented yet"
                        end if
                        this%aq_chem%concentrations(sp_ind)=guess
                        icons(sp_ind)=icon
                        ctots(sp_ind)=ctot
                        !constrains(sp_ind)=constrain
                        k=k+1 !> aqui hay que verificar dimension
                    else
                        error stop 
                    end if
                end if
            end do
            !call this%aq_chem%gas_chemistry%compute_pressure()
            !call this%aq_chem%gas_chemistry%compute_log_act_coeffs_gases()
        else
            error stop
        end if
    do i=1,this%aq_chem%aq_phase%num_species
        call this%aq_chem%aq_phase%aq_species(i)%params_act_coeff%compute_csts(this%aq_chem%aq_phase%aq_species(i)%valence,this%aq_chem%params_aq_sol,model)
    end do
!> We set speciation algebra
    call this%aq_chem%speciation_alg%set_flag_comp(.false.)
    if (this%aq_chem%chem_syst%cat_exch%num_surf_compl>0) then
        flag_surf=.true.
    else
        flag_surf=.false.
    end if
    call this%aq_chem%speciation_alg%set_flag_cat_exch(flag_surf)
    call this%aq_chem%speciation_alg%set_dimensions(this%aq_chem%chem_syst%num_species,this%aq_chem%chem_syst%num_eq_reacts,this%aq_chem%chem_syst%num_cst_act_species,this%aq_chem%aq_phase%num_species,this%aq_chem%aq_phase%num_species-this%aq_chem%aq_phase%wat_flag,this%aq_chem%chem_syst%num_min_kin_reacts)
    call this%aq_chem%speciation_alg%compute_arrays(this%aq_chem%chem_syst%Se,this%aq_chem%chem_syst%get_eq_csts(),this%aq_chem%CV_params%abs_tol)
            
    call this%aq_chem%set_prim_species_indices()
    call this%aq_chem%set_sec_var_act_species_indices()
    call this%aq_chem%aq_phase%set_ind_diss_solids()
    
    !call this%aq_chem%allocate_conc_comp_aq()
    !call this%aq_chem%allocate_log_act_coeffs_aq_chem()

!> Initial guess c2
    !call this%aq_chem%compute_c2_from_c1_aq_ideal() !> initial guess c2_aq
!> Initial guess activities
    !call this%aq_chem%compute_activities_aq_var_act_species() !> initial guess activities
    !call this%aq_chem%compute_act_water() !> initial guess activity water
!> We compute initial concentrations (in molalities)
    !> aqui podrias usar polimorifsmo
    if (sum(n_icons(2:4))>0) then
        if (present(surf_chem)) then
            if (Jac_flag==1) then
                call this%aq_chem%initialise_conc_anal_exch(icons,n_icons,indices_constrains,ctots,surf_chem,niter,CV_flag)
            else
                error stop
            end if
        else
            if (Jac_flag==0) then
                call this%aq_chem%initialise_conc_incr_coeff(icons,n_icons,indices_constrains,ctots,niter,CV_flag)
            else if (Jac_flag==1) then
                call this%aq_chem%initialise_conc_anal(icons,n_icons,indices_constrains,ctots,niter,CV_flag)
            else
                error stop
            end if
        end if
    end if
end subroutine