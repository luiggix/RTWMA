!> Reads 'master25_modif.dat' database
subroutine read_master25(this,path,unit)
    use chem_system_m
    implicit none
    class(chem_system_c) :: this !> chemical system object
    character(len=*), intent(in) :: path
    integer(kind=4), intent(in) :: unit !> file unit
     
    integer(kind=4) :: wat_ind,i,j,int_var,num_reactants,ind_pr_sp,ind_reacts,ind_mins,ind_gases,ind_exch_cats,ind_cst_act_sp,ind_var_act_sp,num_temp_data,ind_aq_compl,num_pr_aq_sp,num_aq_sp,num_aq_compl,num_cst_act_sp,num_var_act_sp,num_eq_reacts,num_reacts,num_mins,num_gases,num_exch_cats,counter_aq_Compl,counter_mins,counter_surf_compl,counter_gases,aq_sp_ind,min_ind,gas_ind,surf_compl_ind
    integer(kind=4) :: num_cst_act_gases,num_cst_act_mins,num_var_act_gases,num_var_act_mins
    integer(kind=4), allocatable :: react_indices(:),indices_exch_cats(:),aux_ind(:),indices_aq_sp(:)
    logical :: aq_sp_flag,min_flag,gas_flag,surf_compl_flag
    real(kind=8) :: unk,log_K,valence,mol_weight,diff_vol,mol_vol
    real(kind=8), allocatable :: global_stoich_coeffs(:),aux_coeffs(:)
    character(len=256) :: str,name,convention,filename
    character(len=256), allocatable :: str1(:),str2(:),species_str(:),global_stoich_names(:)
        
    type(aq_species_c) :: aq_species
    type(mineral_c) :: mineral
    type(gas_c) :: gas
    type(solid_c) :: surf_compl
    
    filename=trim(path)//'master25_modif.dat'

    open(unit,file=filename,status='old',action='read')
    
    allocate(react_indices(this%num_reacts),indices_aq_sp(this%aq_phase%num_species),indices_exch_cats(this%cat_exch%num_exch_cats))
!> First iteration of database
    num_aq_sp=0 !> number of aqueous species
    num_pr_aq_sp=0 !> number of primary aqueous species (CHEPROO)
    ind_aq_compl=0 !> index aqueous complexes
    num_aq_compl=0 !> number of aqueous complexes
    num_cst_act_sp=0 !> number of constant activity species
    num_var_act_sp=0 !> number of variable activity species
    num_cst_act_gases=0 !> number of constant activity gases
    num_var_act_gases=0 !> number of variable activity gases
    num_cst_act_mins=0 !> number of constant activity minerals
    num_var_act_mins=0 !> number of variable activity minerals
    num_eq_reacts=this%num_redox_eq_reacts !> number of equilibrium reactions
    num_reacts=this%num_redox_eq_reacts !> number of reactions
    ind_mins=0 !> index minerals
    num_mins=0 !> number of minerals
    ind_gases=0 !> index gases
    num_gases=0 !> number of gases
    ind_exch_cats=0 !> index exchangeable cations
    num_exch_cats=0 !> number of exchangeable cations
    wat_ind=0
    !> Temperature
    read(unit,*) str, num_temp_data
    !> Primary aqueous species (CHEPROO)
    do
        read(unit,*,iostat=int_var) aq_species%name, aq_species%params_act_coeff%ion_size_param, aq_species%valence!, aq_species%molecular_weight!, aq_species%params_act_coeff%b_TJ, aq_species%molecular_weight
        if (aq_species%name=='null') then
            exit
        else
            call this%aq_phase%is_species_in_aq_phase(aq_species,aq_sp_flag,aq_sp_ind)
            if (aq_sp_flag==.true.) then 
                backspace(unit) !> autentica chapuza
                read(unit,*,iostat=int_var) aq_species%name, aq_species%params_act_coeff%ion_size_param, aq_species%valence, mol_weight
                call aq_species%set_molecular_weight(mol_weight*1d-3)
                if (aq_species%name=='h2o') then
                    call aq_species%set_cst_act_flag(.true.)
                    call this%cst_act_species(1)%assign_species(aq_species)
                    num_cst_act_sp=num_cst_act_sp+1
                else
                    call aq_species%set_cst_act_flag(.false.)
                    if (aq_sp_ind>this%aq_phase%ind_wat .and. this%aq_phase%ind_wat>0) then !> chapuza
                        !aq_sp_ind=aq_sp_ind-1
                        call this%var_act_species(aq_sp_ind-1)%assign_species(aq_species)
                    else
                        call this%var_act_species(aq_sp_ind)%assign_species(aq_species)
                    end if
                    num_var_act_sp=num_var_act_sp+1
                end if
                call this%species(aq_sp_ind)%assign_species(aq_species)
                indices_aq_sp(num_aq_sp+1)=aq_sp_ind
                this%aq_phase%aq_species(aq_sp_ind)=aq_species
                num_aq_sp=num_aq_sp+1
                num_pr_aq_sp=num_pr_aq_sp+1
            else
                continue
            end if
        end if
    end do
    !> Aqueous complexes
    do
        read(unit,*,iostat=int_var) aq_species%name, num_reactants
        if (aq_species%name=='null') then
            exit
        else
            call this%aq_phase%is_species_in_aq_phase(aq_species,aq_sp_flag,aq_sp_ind)
            if (aq_sp_flag==.true.) then
                if (aq_species%name=='h2o') then
                    !wat_ind=aq_sp_ind
                    call aq_species%set_cst_act_flag(.true.)
                    num_cst_act_sp=num_cst_act_sp+1
                else
                    call aq_species%set_cst_act_flag(.false.)
                    num_var_act_sp=num_var_act_sp+1
                end if
            !> We assume all primary species are aqueous
                call this%eq_reacts(aq_sp_ind-num_pr_aq_sp+this%num_redox_eq_reacts)%allocate_reaction(num_reactants+1)
                call this%eq_reacts(aq_sp_ind-num_pr_aq_sp+this%num_redox_eq_reacts)%set_react_type(1)
                call this%eq_reacts(aq_sp_ind-num_pr_aq_sp+this%num_redox_eq_reacts)%set_react_name(aq_species%name)
                react_indices(aq_sp_ind-num_pr_aq_sp+this%num_redox_eq_reacts)=ind_aq_compl+1
                indices_aq_sp(num_aq_sp+1)=aq_sp_ind
                num_aq_compl=num_aq_compl+1
                num_aq_sp=num_aq_sp+1
            else
                continue
            end if
        end if
        ind_aq_compl=ind_aq_compl+1
    end do
    num_eq_reacts=num_eq_reacts+num_aq_compl
    num_reacts=num_reacts+num_aq_compl
    !> Minerals
    do
        read(unit,*,iostat=int_var) mineral%name, mol_vol, num_reactants
        if (mineral%name=='null') then
            exit
        else
            call this%is_mineral_in_chem_syst(mineral,min_flag,min_ind)
            if (min_flag==.true.) then
                !call mineral%mineral%assign_species(this%minerals(min_ind)%mineral)
                !call mineral%mineral%set_name(mineral%name)
                call this%minerals(min_ind)%mineral%set_mol_vol(mol_vol*1d-3)
                !call mineral%mineral%set_cst_act_flag(.true.)
                !call mineral%mineral%set_valence(0)
                !this%minerals(min_ind)=mineral
                call this%species(this%aq_phase%num_species+min_ind)%assign_species(this%minerals(min_ind)%mineral)
                if (this%minerals(min_ind)%mineral%cst_act_flag==.true.) then
                    call this%cst_act_species(num_cst_act_sp+min_ind)%assign_species(this%minerals(min_ind)%mineral)
                    num_cst_act_mins=num_cst_act_mins+1
                else
                    call this%var_act_species(num_var_act_sp+min_ind)%assign_species(this%minerals(min_ind)%mineral)
                    num_var_act_mins=num_var_act_mins+1
                end if
                if (min_ind>this%num_min_kin_reacts) then
                    call this%eq_reacts(num_eq_reacts+min_ind-this%num_min_kin_reacts)%allocate_reaction(num_reactants+1)
                    call this%eq_reacts(num_eq_reacts+min_ind-this%num_min_kin_reacts)%set_react_type(2)
                    call this%eq_reacts(num_eq_reacts+min_ind-this%num_min_kin_reacts)%set_react_name(mineral%name)
                    call this%eq_reacts(num_eq_reacts+min_ind-this%num_min_kin_reacts)%species(num_reactants+1)%assign_species(this%minerals(min_ind)%mineral)
                    this%eq_reacts(num_eq_reacts+min_ind-this%num_min_kin_reacts)%stoichiometry(num_reactants+1)=-1d0
                else
                    call this%min_kin_reacts(min_ind)%allocate_reaction(num_reactants+1)
                    call this%min_kin_reacts(min_ind)%set_react_type(2)
                    call this%min_kin_reacts(min_ind)%set_react_name(mineral%name)
                    call this%min_kin_reacts(min_ind)%species(num_reactants+1)%assign_species(this%minerals(min_ind)%mineral)
                    this%min_kin_reacts(min_ind)%stoichiometry(num_reactants+1)=-1d0
                end if
                react_indices(num_reacts+min_ind)=ind_mins+1
                num_mins=num_mins+1
            else
                continue
            end if
        end if
        ind_mins=ind_mins+1
    end do
    num_var_act_sp=num_var_act_sp+num_var_act_mins
    num_cst_act_sp=num_cst_act_sp+num_cst_act_mins
    num_eq_reacts=num_eq_reacts+this%num_minerals_eq
    num_reacts=num_reacts+num_mins
    !> Gases
    do
        read(unit,*,iostat=int_var) gas%name, mol_weight, diff_vol, num_reactants
        if (gas%name=='null') then
            exit
        else
            call this%gas_phase%is_gas_in_gas_phase(gas,gas_flag,gas_ind)
            if (gas_flag==.true.) then
                call this%gas_phase%gases(gas_ind)%set_molecular_weight(mol_weight*1d-3)
                !> falta set diff_vol
                num_gases=num_gases+1
                call this%species(this%aq_phase%num_species+this%num_minerals+gas_ind)%assign_species(gas)
                if (this%gas_phase%gases(gas_ind)%cst_act_flag==.false.) then
                    call this%var_act_species(num_var_act_sp+gas_ind)%assign_species(gas)
                    num_var_act_gases=num_var_act_gases+1
                else
                    call this%cst_act_species(num_cst_act_sp+gas_ind)%assign_species(gas)
                    num_cst_act_gases=num_cst_act_gases+1
                end if
                call this%eq_reacts(num_eq_reacts+gas_ind)%allocate_reaction(num_reactants+1)
                call this%eq_reacts(num_eq_reacts+gas_ind)%set_react_type(6)
                call this%eq_reacts(num_eq_reacts+gas_ind)%set_react_name(gas%name)
                call this%eq_reacts(num_eq_reacts+gas_ind)%species(num_reactants+1)%assign_species(this%gas_phase%gases(gas_ind))
                this%eq_reacts(num_eq_reacts+gas_ind)%stoichiometry(num_reactants+1)=-1d0
                react_indices(num_reacts+gas_ind)=ind_gases+1
            else
                continue
            end if
        end if
        ind_gases=ind_gases+1
    end do
    num_var_act_sp=num_var_act_sp+num_var_act_gases
    num_cst_act_sp=num_cst_act_sp+num_cst_act_gases
    num_eq_reacts=num_eq_reacts+num_gases
    num_reacts=num_reacts+num_gases
    !> Surface complexes
    if (this%cat_exch%num_surf_compl>0) then
        call this%var_act_species(num_var_act_sp+1)%set_name('x-')
        call this%var_act_species(num_var_act_sp+1)%set_cst_act_flag(.false.)
        call this%var_act_species(num_var_act_sp+1)%set_valence(-1)
        call this%species(this%num_species-this%cat_exch%num_surf_compl+1)%assign_species(this%var_act_species(num_var_act_sp+1))
    end if
    do
        read(unit,*,iostat=int_var) surf_compl%name, num_reactants
        if (surf_compl%name=='null') then
            exit
        else
            call this%cat_exch%is_surf_compl_in(surf_compl,surf_compl_flag,surf_compl_ind)
            if (surf_compl_flag==.true.) then
                num_exch_cats=num_exch_cats+1
                call this%var_act_species(num_var_act_sp+surf_compl_ind)%set_name(surf_compl%name)
                call this%var_act_species(num_var_act_sp+surf_compl_ind)%set_cst_act_flag(.false.)
                call this%var_act_species(num_var_act_sp+surf_compl_ind)%set_valence(0)
                call this%eq_reacts(num_eq_reacts+surf_compl_ind-1)%allocate_reaction(num_reactants+1)
                call this%eq_reacts(num_eq_reacts+surf_compl_ind-1)%set_react_type(3)
                react_indices(num_reacts+surf_compl_ind-1)=ind_exch_cats+1
                indices_exch_cats(num_exch_cats)=surf_compl_ind
            else
                continue
            end if
        end if
        ind_exch_cats=ind_exch_cats+1
    end do
    num_var_act_sp=num_var_act_sp+this%cat_exch%num_surf_compl
    num_eq_reacts=num_eq_reacts+num_exch_cats
    num_reacts=num_reacts+num_exch_cats
    
    rewind(unit)
    
!> Second iteration of database
    counter_aq_compl=1 !> counter aqueous complexes
    ind_aq_compl=0 !> index aqueous complexes 
    ind_cst_act_sp=0 !> index constant activity species
    ind_var_act_sp=num_pr_aq_sp-this%aq_phase%wat_flag !> index variable activity species
    ind_mins=0 !> index minerals
    counter_mins=1 !> counter minerals
    ind_gases=0 !> index gases
    counter_gases=1 !> counter gases
    ind_exch_cats=0 !> index exchangeable cations
    counter_surf_compl=1 !> counter surface complexes
    !> Temperature
    read(unit,*) str, num_temp_data
    !> Primary aqueous species
    do
        read(unit,*,iostat=int_var) name
        if (name=='null') exit
    end do
    !> Aqueous complexes
    if (num_aq_compl>0) then
        do
            call is_int_in_1D_array(counter_aq_compl,react_indices(this%num_redox_eq_reacts+1:this%num_redox_eq_reacts+num_aq_compl),aq_sp_flag,ind_reacts)
            if (aq_sp_flag==.true.) then
                ind_aq_compl=ind_aq_compl+1
                ind_reacts=ind_reacts+this%num_redox_eq_reacts
                read(unit,*,iostat=int_var) aq_species%name, num_reactants, ((this%eq_reacts(ind_reacts)%stoichiometry(j), this%eq_reacts(ind_reacts)%species(j)%name), j=1,num_reactants), log_K, aq_species%params_act_coeff%ion_size_param, aq_species%valence, mol_weight!, aq_species%params_act_coeff%b_TJ, aq_species%molecular_weight
                call aq_species%set_molecular_weight(mol_weight*1d-3)
                if (aq_species%name=='h2o') then
                    ind_cst_act_sp=ind_cst_act_sp+1
                    call aq_species%set_cst_act_flag(.true.)
                    call this%cst_act_species(1)%assign_species(aq_species)
                else
                    ind_var_act_sp=ind_var_act_sp+1
                    call aq_species%set_cst_act_flag(.false.)
                    if (indices_aq_sp(num_pr_aq_sp+ind_aq_compl)>this%aq_phase%ind_wat) then
                        call this%var_act_species(indices_aq_sp(num_pr_aq_sp+ind_aq_compl)-1)%assign_species(aq_species)
                    else
                        call this%var_act_species(indices_aq_sp(num_pr_aq_sp+ind_aq_compl))%assign_species(aq_species)
                    end if
                end if
                call this%species(indices_aq_sp(num_pr_aq_sp+ind_aq_compl))%assign_species(aq_species)
                call this%aq_phase%aq_species(indices_aq_sp(num_pr_aq_sp+ind_aq_compl))%assign_species(aq_species)
                call this%eq_reacts(ind_reacts)%set_eq_cst(10**(-log_K))
                call this%eq_reacts(ind_reacts)%set_single_species(aq_species,this%eq_reacts(ind_reacts)%num_species)
                this%eq_reacts(ind_reacts)%stoichiometry(this%eq_reacts(ind_reacts)%num_species)=-1d0
                call this%eq_reacts(ind_reacts)%change_sign_stoichiometry()
            else
                read(unit,*,iostat=int_var) name
                if (name=='null') exit
            end if
            counter_aq_compl=counter_aq_compl+1
        end do
    else
        do
            read(unit,*,iostat=int_var) name
            if (name=='null') exit
        end do
    end if
    !> Minerals
    if (num_mins>0) then
        do
            call is_int_in_1D_array(counter_mins,react_indices(this%num_redox_eq_reacts+num_aq_compl+1:this%num_redox_eq_reacts+num_aq_compl+num_mins),min_flag,ind_reacts)
            if (min_flag==.true. .AND. ind_reacts>this%num_min_kin_reacts) then
                ind_reacts=ind_reacts+this%num_redox_eq_reacts+num_aq_compl-this%num_min_kin_reacts
                read(unit,*,iostat=int_var) name, mol_vol, num_reactants, ((this%eq_reacts(ind_reacts)%stoichiometry(j), this%eq_reacts(ind_reacts)%species(j)%name), j=1,num_reactants), log_K
                call this%eq_reacts(ind_reacts)%set_eq_cst(10**(-log_K))
                !call this%eq_reacts(ind_reacts)%species(this%eq_reacts(ind_reacts)%num_species)%assign_species()
                !call this%eq_reacts(ind_reacts)%species(this%eq_reacts(ind_reacts)%num_species)%set_name(name)
                !call this%eq_reacts(ind_reacts)%species(this%eq_reacts(ind_reacts)%num_species)%set_valence(0)
                !call this%eq_reacts(ind_reacts)%species(this%eq_reacts(ind_reacts)%num_species)%set_cst_act_flag(.true.)
                !call this%eq_reacts(ind_reacts)%species(this%eq_reacts(ind_reacts)%num_species)%set_mol_vol(mol_vol*1d-3)
                !this%eq_reacts(ind_reacts)%stoichiometry(this%eq_reacts(ind_reacts)%num_species)=-1d0
                call this%eq_reacts(ind_reacts)%change_sign_stoichiometry()
                ind_mins=ind_mins+1
                if (this%eq_reacts(ind_reacts)%species(this%eq_reacts(ind_reacts)%num_species)%cst_act_flag==.true.) then
                    ind_cst_act_sp=ind_cst_act_sp+1
                else
                    ind_var_act_sp=ind_var_act_sp+1
                end if
            else if  (min_flag==.true.) then
                !ind_reacts=ind_reacts-this%num_minerals_eq
                read(unit,*,iostat=int_var) name, mol_vol, num_reactants, ((this%min_kin_reacts(ind_reacts)%stoichiometry(j), this%min_kin_reacts(ind_reacts)%species(j)%name), j=1,num_reactants), log_K
                call this%min_kin_reacts(ind_reacts)%set_eq_cst(10**(-log_K))
                !call this%min_kin_reacts(ind_reacts)%species(this%min_kin_reacts(ind_reacts)%num_species)%set_name(name)
                !call this%min_kin_reacts(ind_reacts)%species(this%min_kin_reacts(ind_reacts)%num_species)%set_valence(0)
                !call this%min_kin_reacts(ind_reacts)%species(this%min_kin_reacts(ind_reacts)%num_species)%set_cst_act_flag(.true.)
                !call this%min_kin_reacts(ind_reacts)%species(this%min_kin_reacts(ind_reacts)%num_species)%set_mol_vol(mol_vol*1d-3)
                !this%min_kin_reacts(ind_reacts)%stoichiometry(this%min_kin_reacts(ind_reacts)%num_species)=-1d0
                call this%min_kin_reacts(ind_reacts)%change_sign_stoichiometry()
                ind_mins=ind_mins+1
                if (this%min_kin_reacts(ind_reacts)%species(this%min_kin_reacts(ind_reacts)%num_species)%cst_act_flag==.true.) then
                    ind_cst_act_sp=ind_cst_act_sp+1
                else
                    ind_var_act_sp=ind_var_act_sp+1
                end if
            else
                read(unit,*,iostat=int_var) name
                if (name=='null') exit
            end if
            counter_mins=counter_mins+1
        end do
    else
        do
            read(unit,*,iostat=int_var) name
            if (name=='null') exit
        end do
    end if
    !> Gases
    if (num_gases>0) then
        do
            call is_int_in_1D_array(counter_gases,react_indices(this%num_redox_eq_reacts+num_aq_compl+num_mins+1:this%num_redox_eq_reacts+num_aq_compl+num_mins+num_gases),gas_flag,ind_reacts)
            if (gas_flag==.true.) then
                ind_reacts=ind_reacts+this%num_redox_eq_reacts+num_aq_compl+this%num_minerals_eq
                read(unit,*,iostat=int_var) name, mol_weight, diff_vol, num_reactants, ((this%eq_reacts(ind_reacts)%stoichiometry(j), this%eq_reacts(ind_reacts)%species(j)%name), j=1,num_reactants), log_K
                call this%eq_reacts(ind_reacts)%set_eq_cst(10**(-log_K))
                !call this%eq_reacts(ind_reacts)%species(this%eq_reacts(ind_reacts)%num_species)%set_name(name)
                !call this%eq_reacts(ind_reacts)%species(this%eq_reacts(ind_reacts)%num_species)%set_molecular_weight(mol_weight*1d-3)
                !call this%eq_reacts(ind_reacts)%species(this%eq_reacts(ind_reacts)%num_species)%set_valence(0)
                !call this%eq_reacts(ind_reacts)%species(this%eq_reacts(ind_reacts)%num_species)%set_cst_act_flag(.false.) !> "true" solo si gas tiene presion constante
                !this%eq_reacts(ind_reacts)%stoichiometry(this%eq_reacts(ind_reacts)%num_species)=-1d0
                call this%eq_reacts(ind_reacts)%change_sign_stoichiometry()
                !ind_var_act_sp=ind_var_act_sp+1
                ind_gases=ind_gases+1
                if (this%eq_reacts(ind_reacts)%species(this%eq_reacts(ind_reacts)%num_species)%cst_act_flag==.true.) then
                    ind_cst_act_sp=ind_cst_act_sp+1
                else
                    ind_var_act_sp=ind_var_act_sp+1
                end if
            else
                read(unit,*,iostat=int_var) name
                if (name=='null') exit
            end if
            counter_gases=counter_gases+1
        end do
    else
        do
            read(unit,*,iostat=int_var) name
            if (name=='null') exit
        end do
    end if
    !> Surface complexes
    if (num_exch_cats>0) then
        do
            call is_int_in_1D_array(counter_surf_compl,react_indices(num_aq_compl+num_mins+num_gases+1:num_aq_compl+num_mins+num_gases+num_exch_cats),surf_compl_flag,ind_reacts)
            if (surf_compl_flag==.true.) then
                ind_exch_cats=ind_exch_cats+1
                ind_reacts=ind_reacts+this%num_redox_eq_reacts+num_aq_compl+this%num_minerals_eq+num_gases
                read(unit,*,iostat=int_var) surf_compl%name, num_reactants, ((this%eq_reacts(ind_reacts)%stoichiometry(j), this%eq_reacts(ind_reacts)%species(j)%name), j=1,num_reactants), log_K, surf_compl%valence
                this%cat_exch%surf_compl(indices_exch_cats(ind_exch_cats))=surf_compl
                call this%species(this%num_species-this%cat_exch%num_surf_compl+indices_exch_cats(ind_exch_cats))%assign_species(surf_compl)
                call this%var_act_species(ind_var_act_sp+indices_exch_cats(ind_exch_cats))%assign_species(surf_compl)
                call this%eq_reacts(ind_reacts)%set_eq_cst(10**(-log_K))
                call this%eq_reacts(ind_reacts)%set_react_name(surf_compl%name)
                call this%eq_reacts(ind_reacts)%set_single_species(surf_compl,this%eq_reacts(ind_reacts)%num_species)
                this%eq_reacts(ind_reacts)%stoichiometry(this%eq_reacts(ind_reacts)%num_species)=-1d0
                call this%eq_reacts(ind_reacts)%change_sign_stoichiometry()
            else
                read(unit,*,iostat=int_var) name
                if (name=='null') exit
            end if
            counter_surf_compl=counter_surf_compl+1
        end do
        ind_var_act_sp=ind_var_act_sp+this%cat_exch%num_surf_compl
    else
        do
            read(unit,*,iostat=int_var) name
            if (name=='null') exit
        end do
    end if
    close(unit)
    call this%rearrange_species()
    call this%compute_z2() !> chapuza
end subroutine