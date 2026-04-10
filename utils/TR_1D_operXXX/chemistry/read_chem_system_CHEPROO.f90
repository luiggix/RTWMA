!> Lectura sistema quimico basada en CHEPROO
!> Suponemos que el archivo ya ha sido abierto
subroutine read_chem_system_CHEPROO(this,path_DB,unit)
    use chem_system_m
    implicit none
    class(chem_system_c) :: this !> chemical system object
    character(len=*), intent(in) :: path_DB
    integer(kind=4), intent(in) :: unit !> file unit
    
    real(kind=8), allocatable :: Sk(:,:),logK(:),gamma_1(:),gamma_2(:)
    integer(kind=4) :: exch_cat_valence,n_eq_homog,unit_master_25,unit_kinetics,unit_redox,i,j,num_sp,ind_var_act_sp,num_var_act_sp,num_cst_act_sp,k,num_aq_sp,num_sec_aq_sp,exch_cat_ind, n_min_kin, n_gas_kin,index,kin_react_type,n_r,num_mins,num_gases,num_surf_compl,num_exch_cats,n_eq,n_k,num_mins_eq_indices,n_redox,n_lin_kin,num_aq_compl,num_mins_eq,num_cst_act_gases,num_var_act_gases
    integer(kind=4) :: num_cst_act_mins,num_var_act_mins,n_redox_eq,n_redox_kin
    integer(kind=4), allocatable :: n_tar(:),mins_eq_indices(:),gases_eq_indices(:)
    real(kind=8) :: aux,conc,temp,SI
    character(len=256) :: str,str1,str2,str3,str4,str5,Monod_name,file_kin_params,label
    character(len=:), allocatable :: str_block_trim,str_trim,valence_str,exch_cat_val,exch_cat_name
    logical :: flag,eq_label,exch_cat_flag,cst_act_label
    
    character(len=256), allocatable :: aq_species_str(:),prim_species_str(:),cst_act_species_str(:),minerals_str(:),solid_species_str(:),kin_react_names(:)
    type(species_c) :: species
    type(species_c), allocatable :: surf_compl(:)
    type(aq_species_c) :: exch_cat
    type(aq_species_c), allocatable :: aq_species(:),exch_cats(:),prim_species(:)
    type(mineral_c) :: mineral
    type(mineral_c), allocatable :: mins(:)
    type(surface_c) :: cat_exch_obj
    type(gas_c) :: gas
    type(gas_c), allocatable :: gases(:)
    class(kin_params_c), pointer :: p_kin_params=>null()
    class(kin_reaction_c), pointer :: p_kin_react=>null()
    type(kin_reaction_ptr_c) :: kin_react_ptr
    class(kin_reaction_ptr_c), allocatable :: kin_reacts(:)
    type(eq_reaction_c) :: eq_react
    type(eq_reaction_c), allocatable :: eq_reacts(:)
    
    type(reaction_c), allocatable :: reacts(:)
    
    logical :: flag_comp,flag_surf
    
    num_sp=0
    num_aq_sp=0
    num_aq_compl=0
    num_cst_act_sp=0
    num_cst_act_gases=0
    num_var_act_gases=0
    num_cst_act_mins=0
    num_var_act_mins=0
    num_var_act_sp=0
    num_surf_compl=0
    num_exch_cats=0
    num_gases=0
    num_mins=0
    num_mins_eq=0
    n_eq=0
    n_eq_homog=0
    n_k=0
    n_min_kin=0
    n_gas_kin=0
    n_lin_kin=0
    n_redox_eq=0
    n_redox_kin=0
    do
        read(unit,*) label
        if (label=='end') then
            rewind(unit)
            exit
        else if (label=='PRIMARY AQUEOUS SPECIES') then
            do
                read(unit,*) str
                if (str=='*') then
                    exit
                else if (str=='h2o') then
                    num_cst_act_sp=num_cst_act_sp+1
                    this%aq_phase%wat_flag=1
                    this%aq_phase%ind_wat=num_aq_sp+1
                else
                    num_var_act_sp=num_var_act_sp+1
                end if
                num_aq_sp=num_aq_sp+1
                num_sp=num_sp+1
            end do
        else if (label=='AQUEOUS COMPLEXES') then
            do
                read(unit,*) str
                if (str=='*') then
                    exit
                else if (str=='h2o') then
                    num_cst_act_sp=num_cst_act_sp+1
                    this%aq_phase%wat_flag=1
                    this%aq_phase%ind_wat=num_aq_sp+1
                else
                    num_var_act_sp=num_var_act_sp+1
                end if
                num_aq_sp=num_aq_sp+1
                num_aq_compl=num_aq_compl+1
                num_sp=num_sp+1
                n_eq=n_eq+1
                n_eq_homog=n_eq_homog+1
            end do            
        else if (label=='MINERALS') then
            do
                read(unit,*) str, eq_label, cst_act_label
                if (str=='*') exit
                num_mins=num_mins+1
                num_sp=num_sp+1
                if (cst_act_label==.true.) then
                    num_cst_act_mins=num_cst_act_mins+1
                    num_cst_act_sp=num_cst_act_sp+1
                else
                    num_var_act_mins=num_var_act_mins+1
                    num_var_act_sp=num_var_act_sp+1
                end if
                if (eq_label==.true.) then
                    n_eq=n_eq+1
                    num_mins_eq=num_mins_eq+1
                else
                    n_k=n_k+1
                    n_min_kin=n_min_kin+1
                end if
            end do 
        else if (label=='GASES') then
            do
                read(unit,*) str, eq_label, cst_act_label
                if (str=='*') exit
                num_gases=num_gases+1
                num_sp=num_sp+1
                if (cst_act_label==.true.) then 
                    num_cst_act_gases=num_cst_act_gases+1
                    num_cst_act_sp=num_cst_act_sp+1
                else
                    num_var_act_gases=num_var_act_gases+1
                    num_var_act_sp=num_var_act_sp+1
                end if
                if (eq_label==.true.) then
                    n_eq=n_eq+1
                else
                    n_k=n_k+1
                    n_gas_kin=n_gas_kin+1
                end if
            end do 
        else if (label=='SURFACE COMPLEXES') then
            do
                read(unit,*) str
                if (str=='*') exit
                str_trim=trim(str)
                num_surf_compl=num_surf_compl+1
                num_var_act_sp=num_var_act_sp+1
                num_sp=num_sp+1
                exch_cat_ind=index(str_trim,'-')
                if (exch_cat_ind>0 .and. exch_cat_ind<len(str_trim)) then
                    num_exch_cats=num_exch_cats+1
                    n_eq=n_eq+1
                end if
            end do
        else if (label=='REDOX REACTIONS') then
            do
                read(unit,*) str, eq_label
                if (str=='*') exit
                if (eq_label==.true.) then
                    n_redox_eq=n_redox_eq+1
                    n_eq=n_eq+1
                else
                    n_redox_kin=n_redox_kin+1
                    n_k=n_k+1
                end if
            end do
        else 
            continue
        end if
    end do
    
    call this%aq_phase%allocate_aq_species(num_aq_sp)
    allocate(this%aq_phase%ind_diss_solids(this%aq_phase%num_species-this%aq_phase%wat_flag))
    call this%aq_phase%set_num_aq_complexes(num_aq_compl)
    call this%allocate_cst_act_species(num_cst_act_sp)
    call this%allocate_var_act_species(num_var_act_sp)
    call this%compute_num_species()
    call this%allocate_species()
    call this%allocate_minerals(num_mins)
    this%num_minerals_eq=num_mins_eq !> falta un set
    call this%gas_phase%allocate_gases(num_gases)
    call this%cat_exch%allocate_surf_compl(num_surf_compl)
    call this%cat_exch%allocate_exch_cats(num_exch_cats)
    this%num_eq_reacts_homog=n_eq_homog !> falta un set aqui
    this%num_redox_eq_reacts=n_redox_eq !> falta un set aqui
    call this%allocate_reacts(n_eq,n_k)
    call this%allocate_min_kin_reacts(n_min_kin)
    call this%allocate_redox_kin_reacts(n_redox_kin)
    call this%allocate_lin_kin_reacts(n_lin_kin)
    call this%set_num_reacts()
    
    call this%gas_phase%set_num_var_act_species_phase(num_var_act_gases)
    call this%gas_phase%set_num_cst_act_species_phase(num_cst_act_gases)
    
    call this%speciation_alg%set_flag_comp(.false.)
    if (num_surf_compl>0) then
        flag_surf=.true.
    else
        flag_surf=.false.
    end if
    call this%speciation_alg%set_flag_cat_exch(flag_surf)
    call this%speciation_alg%set_dimensions(this%num_species,this%num_eq_reacts,this%num_cst_act_species,this%aq_phase%num_species,this%aq_phase%num_species-this%aq_phase%wat_flag,this%num_min_kin_reacts)
    ind_var_act_sp=0
    do
        read(unit,*) label
        if (label=='end') then
            exit
        else if (label=='PRIMARY AQUEOUS SPECIES') then !> suponemos ordenadas en primarias y secundarias
            i=0 !> counter aqueous species
            do
                read(unit,*) str
                if (str=='*') exit
                i=i+1
                str_trim=trim(str)
                call this%aq_phase%aq_species(i)%set_name(str_trim)
                if (str_trim=='h2o') then
                    call this%aq_phase%aq_species(i)%set_cst_act_flag(.true.)
                    call this%cst_act_species(1)%assign_species(this%aq_phase%aq_species(i))
                else
                    ind_var_act_sp=ind_var_act_sp+1
                    call this%aq_phase%aq_species(i)%set_cst_act_flag(.false.)
                    call this%var_act_species(ind_var_act_sp)%assign_species(this%aq_phase%aq_species(i))
                end if
            end do
        else if (label=='AQUEOUS COMPLEXES') then
            do
                read(unit,*) str
                if (str=='*') exit
                i=i+1
                str_trim=trim(str)
                call this%aq_phase%aq_species(i)%set_name(str_trim)
                if (str_trim=='h2o') then
                    call this%aq_phase%aq_species(i)%set_cst_act_flag(.true.)
                    call this%cst_act_species(1)%assign_species(this%aq_phase%aq_species(i))
                else
                    ind_var_act_sp=ind_var_act_sp+1
                    call this%aq_phase%aq_species(i)%set_cst_act_flag(.false.)
                    call this%var_act_species(ind_var_act_sp)%assign_species(this%aq_phase%aq_species(i))
                end if
            end do             
        else if (label=='MINERALS') then
            allocate(mins_eq_indices(0))
            i=0 !> minerals equilibrium
            j=0 !> minerals kinetic
            do
                read(unit,*) str, eq_label, cst_act_label
                if (str=='*') exit
                str_trim=trim(str)
                if (eq_label==.true.) then
                    i=i+1
                    call this%minerals(this%num_min_kin_reacts+i)%set_phase_name(str_trim)
                    call this%minerals(this%num_min_kin_reacts+i)%mineral%set_name(str_trim)
                    call this%minerals(this%num_min_kin_reacts+i)%mineral%set_cst_act_flag(cst_act_label)
                    call this%minerals(this%num_min_kin_reacts+i)%mineral%set_valence(0)
                else
                    j=j+1
                    call this%minerals(j)%set_phase_name(str_trim)
                    call this%minerals(j)%mineral%set_name(str_trim)
                    call this%minerals(j)%mineral%set_cst_act_flag(cst_act_label)
                    call this%minerals(j)%mineral%set_valence(0)
                end if
            end do
        else if (label=='GASES') then
            allocate(gases_eq_indices(0))
            i=0 !> gases
            do
                read(unit,*) str, eq_label, cst_act_label
                if (str=='*') exit
                i=i+1
                str_trim=trim(str)
                call this%gas_phase%gases(i)%set_name(str_trim)
                call this%gas_phase%gases(i)%set_cst_act_flag(cst_act_label)
                call this%gas_phase%gases(i)%set_valence(0)
            end do
        else if (label=='SURFACE COMPLEXES') then
            i=0 !> surface complexes
            j=0 !> exchangeable cations
            do
                read(unit,*) str
                if (str=='*') exit
                i=i+1
                str_trim=trim(str)
                call this%cat_exch%surf_compl(i)%set_name(str_trim)
                call this%cat_exch%surf_compl(i)%set_cst_act_flag(.false.)
                exch_cat_ind=index(str_trim,'-')
                if (exch_cat_ind>0 .and. exch_cat_ind<len(str_trim)-1) then
                    j=j+1
                    if (exch_cat_ind>1) then
                        if (exch_cat_ind==2) then
                            exch_cat_valence=1
                            allocate(character(len=(len(str_trim)-exch_cat_ind+1)) :: exch_cat_name)
                            write(exch_cat_name(1:len(str_trim)-exch_cat_ind),"(A)") str_trim(exch_cat_ind+1:len(str_trim))
                            write(exch_cat_name(len(str_trim)-exch_cat_ind+1:len(str_trim)-exch_cat_ind+1),"(A1)") '+'
                        else
                            exch_cat_val=trim(str(2:exch_cat_ind-1))
                            read(exch_cat_val,*) exch_cat_valence
                            allocate(character(len=(len(str_trim)-exch_cat_ind+1+len(exch_cat_val))) :: exch_cat_name)
                            write(exch_cat_name(1:len(str_trim)-exch_cat_ind),"(A)") str_trim(exch_cat_ind+1:len(str_trim))
                            write(exch_cat_name(len(str_trim)-exch_cat_ind+1:len(str_trim)-exch_cat_ind+1),"(A1)") '+'
                            write(exch_cat_name(len(exch_cat_name)-len(exch_cat_val)+1:len(exch_cat_name)),"(A)") exch_cat_val
                        end if
                        
                    end if
                    call exch_cat%set_name(exch_cat_name)
                    call exch_cat%set_valence(exch_cat_valence)
                    call this%aq_phase%is_species_in_aq_phase(exch_cat,exch_cat_flag,exch_cat_ind)
                    if (exch_cat_flag==.true.) then
                        call this%cat_exch%exch_cats(j)%assign_species(this%aq_phase%aq_species(exch_cat_ind))
                        call this%cat_exch%exch_cats(j)%set_valence(exch_cat_valence) !> chapuza
                    else
                        error stop "Exchangeable cation is not in the chemical system"
                    end if
                    deallocate(exch_cat_name)
                end if
                
            end do
        else if (label=='REDOX REACTIONS') then
            i=0 !> redox kinetic reactions
            j=0 !> redox equilibrium reactions
            do
                read(unit,*) str, eq_label
                if (str=='*') exit
                str_trim=trim(str)
                if (eq_label==.false.) then
                    i=i+1
                    call this%redox_kin_reacts(i)%set_react_name(str_trim)
                    call this%redox_kin_reacts(i)%set_react_type(4)
                else
                    j=j+1
                    call this%eq_reacts(j)%set_react_name(str_trim)
                    call this%eq_reacts(j)%set_react_type(4)
                end if
            end do
        else 
            continue
        end if
    end do
    unit_master_25=2
    call this%read_master25(path_DB,unit_master_25)
    unit_kinetics=3
    if (this%num_min_kin_reacts>0) then
        call this%read_kinetics_DB(path_DB,unit_kinetics)
    end if
    unit_redox=4
    if (this%num_redox_kin_reacts>0 .or. this%num_redox_eq_reacts>0) then
        call this%read_redox_reacts(path_DB,unit_redox)
    end if
    call this%set_stoich_mat()
end subroutine