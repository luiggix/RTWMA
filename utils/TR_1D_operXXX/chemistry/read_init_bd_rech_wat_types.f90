subroutine read_init_bd_rech_wat_types_CHEPROO(this,unit,ind_wat_type,num_aq_prim_array,num_cstr_array,init_cat_exch_zones)
    use chemistry_Lagr_m
    implicit none
    class(chemistry_c) :: this
    integer(kind=4), intent(in) :: unit !> file unit
    integer(kind=4), intent(out), allocatable :: ind_wat_type(:)
    integer(kind=4), intent(out), allocatable :: num_aq_prim_array(:)
    integer(kind=4), intent(out), allocatable :: num_cstr_array(:)
    class(solid_type_c), intent(inout), optional :: init_cat_exch_zones(:)
    
    integer(kind=4) :: i,j,k,l,nwtype,icon,n_p_aq,gas_ind,min_ind,model,niter
    !integer(kind=4), allocatable :: num_aq_prim_array(:),num_cstr_array(:)
    character(len=256) :: prim_sp_name,constrain,label,name
    real(kind=8) :: guess,c_tot,temp,conc
    logical :: CV_flag,flag,flag_surf,flag_comp
    
    type(reactive_zone_c) :: react_zone
    type(gas_chemistry_c) :: gas_chem
    type(aq_species_c) :: aq_species
    type(mineral_c) :: mineral
    type(aq_phase_c) :: old_aq_phase
    
    
    read(unit,*) this%act_coeffs_model
    
    read(unit,*) this%num_wat_types
    
    !read(unit,*) this%num_init_wat_types, this%num_bd_wat_types, this%num_rech_wat_types
    
    call this%allocate_wat_types()
    !this%num_wat_types=this%num_init_wat_types+this%num_bd_wat_types+this%num_rech_wat_types !> total number of water types
    allocate(num_aq_prim_array(this%num_wat_types),num_cstr_array(this%num_wat_types),ind_wat_type(this%num_wat_types))
    
    num_aq_prim_array=0
    num_cstr_array=0
    
     do i=1,this%num_wat_types
        read(unit,*) j, temp !> we read index water type and temperature (in Celsius)
        if (j<1 .or. j>this%num_wat_types) error stop
        ind_wat_type(i)=j
        call this%wat_types(j)%aq_chem%set_chem_syst_aq_chem(this%chem_syst)
        call this%wat_types(j)%aq_chem%set_aq_phase(this%chem_syst%aq_phase) !> redundante
        call this%wat_types(j)%aq_chem%set_temp(temp+273.18) !> Kelvin
        call this%wat_types(j)%aq_chem%set_density()
        call this%wat_types(j)%aq_chem%allocate_conc_aq_species()
        call this%wat_types(j)%aq_chem%allocate_log_act_coeffs_aq_chem()
        call this%wat_types(j)%aq_chem%allocate_activities_aq_species()
        read(unit,"(A20)") name !> we read name of water type
        call this%wat_types(j)%set_chem_type_name(trim(name))
        read(unit,*) label
        if (index(label,'icon')/=0) then !> 'icon, guess, ctot, constrain'
            !allocate(n_icons(4))
            !n_icons=0 !> number of each icon option
            do 
                read(unit,*) aq_species%name, icon!, guess, ctot, constrain%name
                if (aq_species%name=='*') then
                    exit
                else
                    call this%wat_types(j)%aq_chem%aq_phase%is_species_in_aq_phase(aq_species,flag)
                    if (flag==.true.) then
                        num_aq_prim_array(j)=num_aq_prim_array(j)+1
                        if (icon==4) then
                            num_cstr_array(j)=num_cstr_array(j)+1
                        !if (icon==1) then
                        !    n_icons(1)=n_icons(1)+1
                        !else if (icon==2) then
                        !    n_icons(2)=n_icons(2)+1
                        !    !n_aq_comp=n_aq_comp+1
                        !else if (icon==3) then
                        !    n_icons(3)=n_icons(3)+1
                        !    if (aq_species%name=='h+') then
                        !        call this%aq_chem%aq_phase%set_ind_proton(sp_ind)
                        !        call this%aq_chem%set_pH(-log10(ctot))
                        !    end if
                        !else if (icon==4) then
                        !    n_icons(4)=n_icons(4)+1 !> number of constrains
                        !    !num_cstr=num_cstr+1
                        !    !call this%aq_chem%chem_syst%is_eq_reaction_in_chem_syst(constrain,flag,ind_cstr)
                        !    !if (flag==.true.) then
                        !    !    indices_constrains(l)=ind_cstr
                        !    !    l=l+1
                        !    !else
                        !    !    error stop
                        !    !end if
                        !!else if (icon==5) then
                        !!    n_icons(6)=n_icons(6)+1
                        else if (icon<1 .or. icon>4) then
                            error stop "icon option not implemented yet"
                        end if
                        !this%aq_chem%concentrations(sp_ind)=guess
                        !icons(sp_ind)=icon
                        !ctots(sp_ind)=ctot
                        !constrains(sp_ind)=constrain
                        k=k+1 !> aqui hay que verificar dimension
                    else
                        error stop 
                    end if
                end if
            end do
        else
            error stop
        end if
    end do
        !do i=1,this%num_init_wat_types
        !    read(unit,*) j, temp !> we read index water type and temperature (in Celsius)
        !    if (j<1 .or. j>this%num_init_wat_types) error stop
        !    ind_wat_type(i)=j
        !    call this%init_wat_types(j)%aq_chem%set_chem_syst_aq_chem(this%chem_syst)
        !    call this%init_wat_types(j)%aq_chem%set_aq_phase(this%chem_syst%aq_phase) !> redundante
        !    call this%init_wat_types(j)%aq_chem%set_temp(temp+273.15) !> Celsius to Kelvin
        !    call this%init_wat_types(j)%aq_chem%set_density()
        !    call this%init_wat_types(j)%aq_chem%allocate_conc_aq_species()
        !    call this%init_wat_types(j)%aq_chem%allocate_log_act_coeffs_aq_chem()
        !    call this%init_wat_types(j)%aq_chem%allocate_activities_aq_species()
        !    read(unit,"(A20)") name !> we read name of water type
        !    call this%init_wat_types(j)%set_chem_type_name(trim(name))
        !    read(unit,*) label
        !    if (index(label,'icon')/=0) then !> 'icon, guess, ctot, constrain'
        !        !allocate(n_icons(4))
        !        !n_icons=0 !> number of each icon option
        !        do 
        !            read(unit,*) aq_species%name, icon!, guess, ctot, constrain%name
        !            if (aq_species%name=='*') then
        !                exit
        !            else
        !                call this%init_wat_types(j)%aq_chem%aq_phase%is_species_in_aq_phase(aq_species,flag)
        !                if (flag==.true.) then
        !                    num_aq_prim_array(j)=num_aq_prim_array(j)+1
        !                    if (icon==4) then
        !                        num_cstr_array(j)=num_cstr_array(j)+1
        !                    !if (icon==1) then
        !                    !    n_icons(1)=n_icons(1)+1
        !                    !else if (icon==2) then
        !                    !    n_icons(2)=n_icons(2)+1
        !                    !    !n_aq_comp=n_aq_comp+1
        !                    !else if (icon==3) then
        !                    !    n_icons(3)=n_icons(3)+1
        !                    !    if (aq_species%name=='h+') then
        !                    !        call this%aq_chem%aq_phase%set_ind_proton(sp_ind)
        !                    !        call this%aq_chem%set_pH(-log10(ctot))
        !                    !    end if
        !                    !else if (icon==4) then
        !                    !    n_icons(4)=n_icons(4)+1 !> number of constrains
        !                    !    !num_cstr=num_cstr+1
        !                    !    !call this%aq_chem%chem_syst%is_eq_reaction_in_chem_syst(constrain,flag,ind_cstr)
        !                    !    !if (flag==.true.) then
        !                    !    !    indices_constrains(l)=ind_cstr
        !                    !    !    l=l+1
        !                    !    !else
        !                    !    !    error stop
        !                    !    !end if
        !                    !!else if (icon==5) then
        !                    !!    n_icons(6)=n_icons(6)+1
        !                    else if (icon<1 .or. icon>4) then
        !                        error stop "icon option not implemented yet"
        !                    end if
        !                    !this%aq_chem%concentrations(sp_ind)=guess
        !                    !icons(sp_ind)=icon
        !                    !ctots(sp_ind)=ctot
        !                    !constrains(sp_ind)=constrain
        !                    k=k+1 !> aqui hay que verificar dimension
        !                else
        !                    error stop 
        !                end if
        !            end if
        !        end do
        !    else
        !        error stop
        !    end if
        !    !j=j+1
        !    !call this%init_wat_types(j)%read_wat_type_CHEPROO(unit,niter,CV_flag)
        !    !if (i==this%num_init_wat_types) exit
        !end do
        !i=0 !> counter boundary water types
        !do i=1,this%num_bd_wat_types
        !    read(unit,*) j, temp !> we read index water type and temperature (in Celsius)
        !    if (j<1 .or. j>this%num_bd_wat_types) error stop
        !    !i=i+1
        !    ind_wat_type(this%num_init_wat_types+i)=j
        !    call this%bd_wat_types(j)%aq_chem%set_chem_syst_aq_chem(this%chem_syst)
        !    call this%bd_wat_types(j)%aq_chem%set_aq_phase(this%chem_syst%aq_phase)
        !    call this%bd_wat_types(j)%aq_chem%set_temp(temp+273.15) !> Celsius to Kelvin
        !    call this%bd_wat_types(j)%aq_chem%set_density()
        !    call this%bd_wat_types(j)%aq_chem%allocate_conc_aq_species()
        !    call this%bd_wat_types(j)%aq_chem%allocate_log_act_coeffs_aq_chem()
        !    call this%bd_wat_types(j)%aq_chem%allocate_activities_aq_species()
        !    read(unit,"(A20)") name !> we read name of water type
        !    call this%bd_wat_types(j)%set_chem_type_name(trim(name))
        !    read(unit,*) label
        !    if (index(label,'icon')/=0) then !> 'icon, guess, ctot, constrain'
        !        !allocate(n_icons(4))
        !        !n_icons=0 !> number of each icon option
        !        do 
        !            read(unit,*) aq_species%name, icon!, guess, ctot, constrain%name
        !            if (aq_species%name=='*') then
        !                exit
        !            else
        !                call this%bd_wat_types(j)%aq_chem%aq_phase%is_species_in_aq_phase(aq_species,flag)
        !                if (flag==.true.) then
        !                    num_aq_prim_array(this%num_init_wat_types+j)=num_aq_prim_array(this%num_init_wat_types+j)+1
        !                    if (icon==4) then
        !                        num_cstr_array(this%num_init_wat_types+j)=num_cstr_array(this%num_init_wat_types+j)+1
        !                    !if (icon==1) then
        !                    !    n_icons(1)=n_icons(1)+1
        !                    !else if (icon==2) then
        !                    !    n_icons(2)=n_icons(2)+1
        !                    !    !n_aq_comp=n_aq_comp+1
        !                    !else if (icon==3) then
        !                    !    n_icons(3)=n_icons(3)+1
        !                    !    if (aq_species%name=='h+') then
        !                    !        call this%aq_chem%aq_phase%set_ind_proton(sp_ind)
        !                    !        call this%aq_chem%set_pH(-log10(ctot))
        !                    !    end if
        !                    !else if (icon==4) then
        !                    !    n_icons(4)=n_icons(4)+1 !> number of constrains
        !                    !    !num_cstr=num_cstr+1
        !                    !    !call this%aq_chem%chem_syst%is_eq_reaction_in_chem_syst(constrain,flag,ind_cstr)
        !                    !    !if (flag==.true.) then
        !                    !    !    indices_constrains(l)=ind_cstr
        !                    !    !    l=l+1
        !                    !    !else
        !                    !    !    error stop
        !                    !    !end if
        !                    !!else if (icon==5) then
        !                    !!    n_icons(6)=n_icons(6)+1
        !                    else if (icon<1 .or. icon>4) then
        !                        error stop "icon option not implemented yet"
        !                    end if
        !                    !this%aq_chem%concentrations(sp_ind)=guess
        !                    !icons(sp_ind)=icon
        !                    !ctots(sp_ind)=ctot
        !                    !constrains(sp_ind)=constrain
        !                    k=k+1 !> aqui hay que verificar dimension
        !                else
        !                    error stop 
        !                end if
        !            end if
        !        end do
        !    else
        !        error stop
        !    end if
        !    !call this%bd_wat_types(j)%read_wat_type_CHEPROO(unit,niter,CV_flag)
        !    !j=j+1
        !    !if (i==this%num_this%bd_wat_types) exit
        !end do
    rewind(unit)
    do
        read(unit,*) label
        if (label=='INITIAL AND BOUNDARY WATER TYPES') then
            read(unit,*) model
            read(unit,*) this%num_wat_types
            do i=1,this%num_wat_types
                !i=i+1
                read(unit,*) j, temp !> we read index water type and temperature (in Celsius)
                read(unit,*) name
            !> Chapuza
                if (PRESENT(init_cat_exch_zones)) then
                    if (SIZE(init_cat_exch_zones)==1) then
                        call this%wat_types(j)%read_wat_type_CHEPROO(num_aq_prim_array(j),num_cstr_array(j),this%act_coeffs_model,this%Jac_flag,unit,niter,CV_flag,init_cat_exch_zones(1)%solid_chem)
                    end if
                else
                    call this%wat_types(j)%read_wat_type_CHEPROO(num_aq_prim_array(j),num_cstr_array(j),this%act_coeffs_model,this%Jac_flag,unit,niter,CV_flag)
                end if
            end do
            !read(unit,*) this%num_init_wat_types, this%num_bd_wat_types, this%num_rech_wat_types
            !do i=1,this%num_init_wat_types
            !    !i=i+1
            !    read(unit,*) j, temp !> we read index water type and temperature (in Celsius)
            !    read(unit,*) name
            !!> Chapuza
            !    if (PRESENT(init_cat_exch_zones)) then
            !        if (SIZE(init_cat_exch_zones)==1) then
            !            call this%init_wat_types(j)%read_wat_type_CHEPROO(num_aq_prim_array(j),num_cstr_array(j),this%act_coeffs_model,this%Jac_flag,unit,niter,CV_flag,init_cat_exch_zones(1)%solid_chem)
            !        end if
            !    else
            !        call this%init_wat_types(j)%read_wat_type_CHEPROO(num_aq_prim_array(j),num_cstr_array(j),this%act_coeffs_model,this%Jac_flag,unit,niter,CV_flag)
            !    end if
            !end do
            !do i=1,this%num_bd_wat_types
            !    read(unit,*) j, temp !> we read index water type and temperature (in Celsius)
            !    read(unit,*) name
            !!> Chapuza
            !    if (PRESENT(init_cat_exch_zones)) then
            !        if (SIZE(init_cat_exch_zones)==1) then
            !            call this%bd_wat_types(j)%read_wat_type_CHEPROO(num_aq_prim_array(j),num_cstr_array(j),this%act_coeffs_model,this%Jac_flag,unit,niter,CV_flag,init_cat_exch_zones(1)%solid_chem)
            !        end if
            !    else
            !        call this%bd_wat_types(j)%read_wat_type_CHEPROO(num_aq_prim_array(j),num_cstr_array(j),this%act_coeffs_model,this%Jac_flag,unit,niter,CV_flag)
            !    end if            
            !end do
            exit
         else
            continue
         end if
    end do
!> We eliminate constant activity species from component matrix and we rearrange aqueous species and equilibrium reactions
    call this%chem_syst%speciation_alg%set_flag_comp(.true.)
    if (this%chem_syst%cat_exch%num_surf_compl>0) then
        flag_surf=.true.
    else
        flag_surf=.false.
    end if
    call this%chem_syst%speciation_alg%set_flag_cat_exch(flag_surf)
    call this%chem_syst%speciation_alg%compute_num_prim_species(this%chem_syst%num_min_kin_reacts)
    call this%chem_syst%speciation_alg%compute_num_aq_sec_var_act_species()
    old_aq_phase=this%chem_syst%aq_phase !> chapuza
    call this%chem_syst%aq_phase%rearrange_aq_species()
    call this%chem_syst%aq_phase%set_indices_aq_phase()
    call this%chem_syst%rearrange_species()
    call this%chem_syst%compute_z2() !> chapuza
    
    call this%chem_syst%rearrange_eq_reacts()
    call this%chem_syst%set_stoich_mat()
    call this%chem_syst%speciation_alg%compute_arrays(this%chem_syst%Se,this%chem_syst%get_eq_csts(),this%CV_params%zero)
!> Chapuza
    do i=1,this%num_wat_types
        !> rearrange cocnentrations and activities
        call this%wat_types(i)%aq_chem%rearrange_state_vars(old_aq_phase)
    end do
    !do i=1,this%num_init_wat_types
    !    !> rearrange cocnentrations and activities
    !    call this%init_wat_types(i)%aq_chem%rearrange_state_vars(old_aq_phase)
    !end do
    !do i=1,this%num_bd_wat_types
    !    call this%bd_wat_types(i)%aq_chem%rearrange_state_vars(old_aq_phase)
    !end do
end subroutine