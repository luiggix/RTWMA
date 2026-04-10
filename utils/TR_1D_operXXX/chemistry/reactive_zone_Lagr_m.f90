!> Set of heterogeneous reactions in equilibrium and "non flowing" species that define these reactions
module reactive_zone_Lagr_m
    use speciation_algebra_m
    use time_discr_m
    use chem_system_m
    use surf_compl_m
    use array_ops_m
    use gas_zone_m
    implicit none
    save
    type, public :: reactive_zone_c
        integer(kind=4) :: num_non_flowing_species=0 !> number of non-flowing species
        type(species_c), allocatable :: non_flowing_species(:) !> non-flowing species
        integer(kind=4) :: num_minerals=0 !> number of minerals in equilibrium
        integer(kind=4) :: num_minerals_cst_act=0 !> number of minerals with constant activity
        integer(kind=4) :: num_minerals_var_act=0 !> number of minerals with variable activity
        type(mineral_c), allocatable :: minerals(:) !> minerals in equilibrium
        type(cat_exch_c) :: cat_exch_zone !> cation exchange zone
        integer(kind=4) :: num_solids=0 !> number of solids
        type(gas_phase_c) :: gas_phase !> gas phase
        real(kind=8), allocatable :: stoich_mat(:,:) !> stoichiometric matrix
        real(kind=8), allocatable :: stoich_mat_sol(:,:) !> solid stoichiometric matrix
        integer(kind=4) :: num_eq_reactions=0 !> number of equilibrium reactions
        type(eq_reaction_c), allocatable :: eq_reactions(:) !> equilibrium heterogeneous reactions
        class(chem_system_c), pointer :: chem_syst !>  (same chemical system as chemistry class)
    contains
    !> Set
        procedure, public :: set_minerals_react_zone
        procedure, public :: set_non_flowing_species
        procedure, public :: set_non_flowing_species_from_chem_syst
        procedure, public :: set_single_non_flowing_species
        procedure, public :: set_num_non_flowing_species
        procedure, public :: set_num_solids
        procedure, public :: set_chem_syst_react_zone
        procedure, public :: set_num_eq_reactions
        procedure, public :: set_cat_exch_zone
        procedure, public :: set_eq_reactions
        procedure, public :: set_stoich_mat_react_zone
        procedure, public :: set_stoich_mat_sol
        procedure, public :: set_num_mins_cst_act
        procedure, public :: set_num_mins_var_act
        procedure, public :: set_gas_phase
    !> Allocate/deallocate
        procedure, public :: allocate_non_flowing_species
        procedure, public :: allocate_minerals_react_zone
        procedure, public :: allocate_eq_reactions
        procedure, public :: deallocate_react_zone
    !> Update
        procedure, public :: update_num_eq_reacts
        procedure, public :: update_reactive_zone
        procedure, public :: update_eq_reactions
    !> Get
        procedure, public :: get_eq_csts_react_zone
    !> Compute
        procedure, public :: compute_num_species_react_zone
        procedure, public :: compute_num_cst_act_species_react_zone
        procedure, public :: compute_Delta_t_crit_reactive_zone
    !> Is
        procedure, public :: is_nf_species_in_react_zone
        procedure, public :: is_mineral_in_react_zone
    end type
!**************************************************************************************************
    interface
         subroutine set_stoich_mat_react_zone(this)
            import reactive_zone_c
            implicit none
            class(reactive_zone_c) :: this
         end subroutine
        
         subroutine set_stoich_mat_sol(this)
            import reactive_zone_c
            implicit none
            class(reactive_zone_c) :: this
        end subroutine
        
        !subroutine compute_rk_mat(this,conc,rk_mat)
        !>    import reactive_zone_c
        !>    implicit none
        !>    class(reactive_zone_c), intent(in) :: this
        !>    real(kind=8), intent(in) :: conc(:,:)
        !>    real(kind=8), intent(out) :: rk_mat(:,:)
        !end subroutine
        !
        !subroutine compute_rk_vec(this,conc,rk_vec)
        !>    import reactive_zone_c
        !>    implicit none
        !>    class(reactive_zone_c), intent(in) :: this
        !>    real(kind=8), intent(in) :: conc(:)
        !>    real(kind=8), intent(out) :: rk_vec(:)
        !end subroutine
        
        
        

        
       
        
        subroutine compute_Delta_t_crit_reactive_zone(this,B_mat,F_mat,Delta_t_crit)
            import reactive_zone_c
            import tridiag_matrix_c
            import diag_matrix_c
            implicit none
            class(reactive_zone_c) :: this
            class(tridiag_matrix_c), intent(in) :: B_mat
            class(diag_matrix_c), intent(in) :: F_mat
            real(kind=8), intent(out) :: Delta_t_crit
        end subroutine
        
        subroutine read_reactive_zone_Lagr(this,filename,line)
            import reactive_zone_c
            implicit none
            class(reactive_zone_c) :: this
            character(len=*), intent(in) :: filename !> nombre del archivo de entrada
            integer(kind=4), intent(in) :: line
        end subroutine
        
        subroutine write_reactive_zone(this)
            import reactive_zone_c
            implicit none
            class(reactive_zone_c) :: this
        end subroutine
        
       
        

        
       
        
        subroutine update_reactive_zone(this,old_nf_ind,new_react_zone)
            import reactive_zone_c
            implicit none
            class(reactive_zone_c) :: this
            integer(kind=4), intent(in) :: old_nf_ind(:)
            class(reactive_zone_c), intent(out) :: new_react_zone
        end subroutine
        
        subroutine update_eq_reactions(this,old_eq_reacts_ind)
            import reactive_zone_c
            implicit none
            class(reactive_zone_c) :: this
            integer(kind=4), intent(in) :: old_eq_reacts_ind(:)
        end subroutine
        
        subroutine compare_react_zones(react_zone_1,react_zone_2,flag)
            import reactive_zone_c
            implicit none
            class(reactive_zone_c), intent(in) :: react_zone_1
            class(reactive_zone_c), intent(in) :: react_zone_2
            logical, intent(out) :: flag !> true if same non flowing species, false otherwise
        end subroutine
    end interface
    
    
    
    contains
        subroutine set_num_non_flowing_species(this,num_non_flowing_species)
            implicit none
            class(reactive_zone_c) :: this
            integer(kind=4), intent(in) :: num_non_flowing_species
            this%num_non_flowing_species=num_non_flowing_species
        end subroutine
        
        subroutine set_num_solids(this,num_solids)
            implicit none
            class(reactive_zone_c) :: this
            integer(kind=4), intent(in), optional :: num_solids
            if (present(num_solids)) then
                this%num_solids=num_solids
            else
                this%num_solids=this%num_minerals+this%cat_exch_zone%num_surf_compl
            end if
        end subroutine
        
        subroutine set_num_mins_cst_act(this,num_mins_cst_act)
            implicit none
            class(reactive_zone_c) :: this
            integer(kind=4), intent(in) :: num_mins_cst_act
            this%num_minerals_cst_act=num_mins_cst_act
        end subroutine
        
        subroutine set_num_mins_var_act(this,num_mins_var_act)
            implicit none
            class(reactive_zone_c) :: this
            integer(kind=4), intent(in) :: num_mins_var_act
            this%num_minerals_var_act=num_mins_var_act
        end subroutine
        
        subroutine set_gas_phase(this,gas_phase)
            implicit none
            class(reactive_zone_c) :: this
            class(gas_phase_c), intent(in) :: gas_phase
            this%gas_phase=gas_phase
        end subroutine
        
        subroutine check_num_solids(this)
            implicit none 
            class(reactive_zone_c) :: this
            if (this%num_solids/=this%num_minerals+this%cat_exch_zone%num_surf_compl) then
                error stop "Wrong number of solids in reactive zone object"
            end if
        end subroutine
        
        subroutine allocate_non_flowing_species(this,num_non_flowing_species)
            implicit none
            class(reactive_zone_c) :: this
            integer(kind=4), intent(in), optional :: num_non_flowing_species
            if (present(num_non_flowing_species)) then
                call this%set_num_non_flowing_species(num_non_flowing_species)
            end if
            if (allocated(this%non_flowing_species)) then
                deallocate(this%non_flowing_species)
            end if
            allocate(this%non_flowing_species(this%num_non_flowing_species))
        end subroutine
        
        subroutine set_non_flowing_species_from_chem_syst(this,non_flowing_species_ind)
            implicit none
            class(reactive_zone_c) :: this
            integer(kind=4), intent(in) :: non_flowing_species_ind(:)
            integer(kind=4) :: i
            call this%set_num_non_flowing_species(size(non_flowing_species_ind))
            call this%allocate_non_flowing_species()
            do i=1,this%num_non_flowing_species
                call this%non_flowing_species(i)%assign_species(this%chem_syst%species(non_flowing_species_ind(i)))
            end do
        end subroutine
        !
        
        subroutine set_non_flowing_species(this,non_flowing_species)
            implicit none
            class(reactive_zone_c) :: this
            class(species_c), intent(in), optional :: non_flowing_species(:)
            
            integer(kind=4) :: i,j,k
            
            if (present(non_flowing_species)) then
                this%non_flowing_species=non_flowing_species
            else
                j=0 !> counter constant non flowing species
                k=0 !> counter variable non flowing species
                call this%allocate_non_flowing_species(this%num_minerals+this%cat_exch_zone%num_surf_compl+this%gas_phase%num_species)
            !> First: minerals
                do i=1,this%num_minerals
                    if (this%minerals(i)%mineral%cst_act_flag==.true.) then
                        j=j+1
                        call this%non_flowing_species(j)%assign_species(this%minerals(i)%mineral)
                    else
                        k=k+1
                        call this%non_flowing_species(this%num_minerals_cst_act+k)%assign_species(this%minerals(i)%mineral)
                    end if
                end do
            !> Second: surface complexes
                do i=1,this%cat_exch_zone%num_surf_compl
                    call this%non_flowing_species(this%num_minerals+i)%assign_species(this%cat_exch_zone%surf_compl(i))
                end do
                j=0
                k=0
            !> Third: gases
                do i=1,this%gas_phase%num_species
                    if (this%gas_phase%gases(i)%cst_act_flag==.true.) then
                        j=j+1
                        call this%non_flowing_species(this%num_solids+j)%assign_species(this%gas_phase%gases(i))
                    else
                        k=k+1
                        call this%non_flowing_species(this%num_solids+this%gas_phase%num_cst_act_species+k)%assign_species(this%gas_phase%gases(i))
                    end if
                end do
            end if
        end subroutine
        
        subroutine set_single_non_flowing_species(this,non_flowing_species_ind,chem_syst_ind)
            implicit none
            class(reactive_zone_c) :: this
            integer(kind=4), intent(in) :: non_flowing_species_ind !> index in reactive zone
            integer(kind=4), intent(in) :: chem_syst_ind !> index in chemical system
            if (non_flowing_species_ind<1 .or. non_flowing_species_ind>chem_syst_ind .or. chem_syst_ind>this%chem_syst%num_species) error stop
            call this%non_flowing_species(non_flowing_species_ind)%assign_species(this%chem_syst%species(chem_syst_ind))
        end subroutine
        
        subroutine set_chem_syst_react_zone(this,chem_syst)
            implicit none
            class(reactive_zone_c) :: this
            class(chem_system_c), intent(in), target :: chem_syst
            this%chem_syst=>chem_syst
        end subroutine
        
        function get_eq_csts_react_zone(this) result(K)
            implicit none
            class(reactive_zone_c), intent(in) :: this
            real(kind=8), allocatable :: K(:)
            
            integer(kind=4) :: i
          
            allocate(K(this%num_eq_reactions))
            do i=1,this%num_eq_reactions
                K(i)=this%eq_reactions(i)%eq_cst
            end do
           
        end function
        
        !subroutine get_minerals_eq(this)
        !>    implicit none
        !>    class(reactive_zone_c) :: this
        !>    
        !>    integer(kind=4) :: i
        !>    !do i=1,this%mineral_zone%num_minerals_eq
        !>    !>    this%non_flowing_species(i)%name=this%mineral_zone%minerals_eq(i)%name
        !>    !>    this%non_flowing_species(i)%cst_act_flag=.true.
        !>    !>    this%non_flowing_species(i)%valence=0
        !>    !end do
        !>    i=1 !> counter minerals reactive zone
        !>    j=1 !> counter equilibrium reactions chemical system
        !>    do
        !>        call this%chem_syst%eq_reacts(j)%is_species_in_react(this%non_flowing_species(i),flag,sp_ind)
        !>        !call this%chem_syst%is_eq_reaction_in_chem_syst(this%non_flowing_species(i),flag,eq_react_ind)
        !>        if (flag==.true.) then
        !>            this%eq_reactions(i)=this%chem_syst%eq_reacts(j)
        !>            if (i<this%num_eq_recactions) then
        !>                i=i+1
        !>                j=1
        !>            else
        !>                exit
        !>            end if
        !>            !call append_int_1D_array(eq_react_indices,eq_react_ind)
        !>        else if (j<this%chem_syst%num_eq_reacts) then
        !>            j=j+1
        !>        else
        !>            error stop "This equilibrium reaction is not in the chemical system"
        !>        end if
        !>    end do
        !end subroutine
       
        
        
        
      
       
        
        subroutine set_eq_reactions(this,eq_reactions_ind) 
            implicit none
            class(reactive_zone_c) :: this
            integer(kind=4), intent(in), optional :: eq_reactions_ind(:)
            
            integer(kind=4) :: i,l,j,k,eq_react_ind,n_eq,sp_ind 
            integer(kind=4), allocatable :: eq_react_indices(:)
            logical :: flag

            call this%allocate_eq_reactions()
            !this%eq_reactions(1:this%chem_syst%num_eq_reacts_homog)=this%chem_syst%eq_reacts(1:this%chem_syst%num_eq_reacts_homog)
            !allocate(eq_react_indices(0))
            if (present(eq_reactions_ind)) then
                !call this%allocate_eq_reactions(size(eq_reactions_ind))
                if (size(eq_reactions_ind)/=this%num_eq_reactions) error stop
                this%eq_reactions=this%chem_syst%eq_reacts(eq_reactions_ind)
            else
                !i=1 !> counter equilibrium reactions reactive zone
                !j=1 !> counter equilibrium reactions chemical system
                !!k=1 !> counter minerals
                !do k=1,this%num_minerals
                !>    call this%chem_syst%is_mineral_in_chem_syst(this%minerals(k),flag,min_ind)
                !>    if (flag==.true.) then
                !>        this%eq_reactions(i)=this%chem_syst%eq_reacts(min_ind)
                !>        i=i+1
                !>    end if
                !end do
                !this%eq_reactions(i:i+this%chem_syst%aq_phase%num_aq_complexes)=this%eq_reacts(this%chem_syst%num_minerals_eq+1:this%chem_syst%num_minerals_eq+this%chem_syst%aq_phase%num_aq_complexes)
                !do k=1,this%num_minerals
                !>    call this%chem_syst%is_mineral_in_chem_syst(this%minerals(k),flag,min_ind)
                !>    if (flag==.true.) then
                !>        this%eq_reactions(i)=this%chem_syst%eq_reacts(min_ind)
                !>        i=i+1
                !>    end if
                !end do
                !do k=1,this%num_minerals
                !>    call this%chem_syst%is_mineral_in_chem_syst(this%minerals(k),flag,min_ind)
                !>    if (flag==.true.) then
                !
                !>        this%eq_reactions(i)=this%chem_syst%eq_reacts(min_ind)
                !>        i=i+1
                !>    end if
                !end do
                i=1 !> counter mineral reactions reactive zone
                l=1 !> counter exchange & gas reactions reactive zone
                j=1 !> counter equilibrium reactions chemical system
                k=1 !> counter non flowing species
                this%eq_reactions(this%num_minerals_cst_act+this%gas_phase%num_cst_act_species+1:this%num_minerals+this%gas_phase%num_cst_act_species+this%chem_syst%num_redox_eq_reacts+this%chem_syst%aq_phase%num_aq_complexes)=this%chem_syst%eq_reacts(this%chem_syst%num_minerals_eq+this%chem_syst%gas_phase%num_cst_act_species+1:this%chem_syst%num_minerals_eq+this%chem_syst%gas_phase%num_cst_act_species+this%chem_syst%num_redox_eq_reacts+this%chem_syst%aq_phase%num_aq_complexes)
                do
                    if (this%non_flowing_species(k)%name=='x-') then !> chapuza
                        k=k+1
                    end if
                    call this%chem_syst%eq_reacts(j)%is_species_in_react(this%non_flowing_species(k),flag,sp_ind)
                    !call this%chem_syst%is_eq_reaction_in_chem_syst(this%non_flowing_species(i),flag,eq_react_ind)
                    if (flag==.true.) then
                        if (this%non_flowing_species(k)%cst_act_flag==.true.) then
                            this%eq_reactions(i)=this%chem_syst%eq_reacts(j)
                            i=i+1
                        else
                            this%eq_reactions(this%num_minerals_cst_act+this%gas_phase%num_cst_act_species+this%chem_syst%num_redox_eq_reacts+this%chem_syst%aq_phase%num_aq_complexes+l)=this%chem_syst%eq_reacts(j)
                            l=l+1
                        end if
                    !!> Chapuza
                    !>    if (k==this%num_minerals) then
                    !>        this%eq_reactions(k+1:k+this%chem_syst%aq_phase%num_aq_complexes)=this%chem_syst%eq_reacts(this%chem_syst%num_minerals_eq+1:this%chem_syst%num_minerals_eq+this%chem_syst%aq_phase%num_aq_complexes)
                    !>        i=i+this%chem_syst%aq_phase%num_aq_complexes
                    !>    end if
                        if (k<this%num_non_flowing_species) then
                            !i=i+1
                            k=k+1
                            j=1
                        else
                            exit
                        end if
                        !call append_int_1D_array(eq_react_indices,eq_react_ind)
                    else if (j<this%chem_syst%num_eq_reacts) then
                        j=j+1
                    else
                        error stop "This equilibrium reaction is not in the chemical system"
                    end if
                end do
                !this%eq_reactions(i+1:this%num_eq_reactions)=this%chem_syst%eq_reacts(this%chem_syst%num_eq_reacts-this%chem_syst%num_eq_reacts_homog+1:this%chem_syst%num_eq_reacts)
                !call this%chem_syst%is_water_dissoc_in_chem_syst(flag,eq_react_ind)
                !if (flag==.true.) then
                !>    call append_int_1D_array(eq_react_indices,eq_react_ind)
                !end if
                !call this%allocate_eq_reactions(size(eq_react_indices))
                !this%eq_reactions=this%chem_syst%eq_reacts(eq_react_indices)
            end if
        end subroutine
        
        !subroutine set_kin_reactions(this,kin_reactions)
        !>    implicit none
        !>    class(reactive_zone_c) :: this
        !>    class(kin_reaction_ptr_c), intent(in), optional :: kin_reactions(:)
        !>    if (present(kin_reactions)) then
        !>        this%kin_reactions=kin_reactions
        !>    else
        !>        this%kin_reactions=this%chem_syst%kin_reacts !> we assume all kinetic reactions take place in this reactive zone
        !>        !this%num_kin_reactions=size(this%kin_reactions)
        !>    end if
        !end subroutine
        
        !subroutine set_min_kin_reactions(this,min_kin_reactions)
        !>    implicit none
        !>    class(reactive_zone_c) :: this
        !>    class(kin_mineral_c), intent(in), optional :: min_kin_reactions(:)
        !>    if (present(min_kin_reactions)) then
        !>        this%min_kin_reactions=min_kin_reactions
        !>    else
        !>        this%min_kin_reactions=this%chem_syst%min_kin_reacts !> we assume all kinetic reactions take place in this reactive zone
        !>        this%num_min_kin_reactions=size(this%min_kin_reactions)
        !>    end if
        !end subroutine
        
        !subroutine compute_num_reactions(this)
        !>    implicit none
        !>    class(reactive_zone_c) :: this
        !>    !this%num_kin_reactions=size(this%kin_reactions)
        !>    !this%num_eq_reactions=size(this%eq_reactions)
        !>    this%num_reactions=this%chem_syst%num_kin_reacts+this%num_eq_reactions !> faltan reacciones homgeneas equilibrio
        !end subroutine
        
        subroutine set_minerals_react_zone(this,mineral_indices)
            implicit none
            class(reactive_zone_c) :: this
            integer(kind=4), intent(in), optional :: mineral_indices(:) !> in chemical system
            integer(kind=4) :: i
            if (present(mineral_indices)) then
                if (size(mineral_indices)>this%chem_syst%num_minerals) error stop
                this%num_minerals=size(mineral_indices)
                this%minerals=this%chem_syst%minerals(mineral_indices)
            else
                this%num_minerals=this%chem_syst%num_minerals_eq
                this%minerals=this%chem_syst%minerals(this%chem_syst%num_min_kin_reacts+1:this%chem_syst%num_minerals)
                do i=1,this%num_minerals
                    if (this%minerals(i)%mineral%cst_act_flag==.true.) then
                        this%num_minerals_cst_act=this%num_minerals_cst_act+1
                    else
                        this%num_minerals_var_act=this%num_minerals_var_act+1
                    end if
                end do
            end if
        end subroutine
        
        subroutine allocate_minerals_react_zone(this,num_minerals)
            implicit none
            class(reactive_zone_c) :: this
            integer(kind=4), intent(in), optional :: num_minerals
            if (present(num_minerals)) then
                if (num_minerals<0 .or. num_minerals>this%chem_syst%num_minerals) error stop
                this%num_minerals=num_minerals
            end if
            allocate(this%minerals(this%num_minerals))
        end subroutine
        
        !subroutine set_mineral_zone(this,mineral_zone)
        !>    implicit none
        !>    class(reactive_zone_c) :: this
        !>    class(mineral_zone_c), intent(in) :: mineral_zone
        !>    this%mineral_zone=mineral_zone
        !end subroutine
        
        subroutine set_num_eq_reactions(this,num_eq_reacts)
            implicit none
            class(reactive_zone_c) :: this
            integer(kind=4), intent(in) :: num_eq_reacts
            this%num_eq_reactions=num_eq_reacts
        end subroutine
        
        !subroutine set_num_kin_reactions(this,num_kin_reacts)
        !>    implicit none
        !>    class(reactive_zone_c) :: this
        !>    integer(kind=4), intent(in) :: num_kin_reacts
        !>    this%num_kin_reactions=num_kin_reacts
        !end subroutine
        
        subroutine update_num_eq_reacts(this,num_old_eq_reacts)
            implicit none
            class(reactive_zone_c) :: this
            integer(kind=4), intent(in) :: num_old_eq_reacts
            this%num_eq_reactions=this%num_eq_reactions-num_old_eq_reacts
        end subroutine
        
        subroutine allocate_eq_reactions(this)
            implicit none
            class(reactive_zone_c) :: this
            this%num_eq_reactions=this%num_minerals+this%cat_exch_zone%num_exch_cats+this%chem_syst%aq_phase%num_aq_complexes+this%gas_phase%num_species+this%chem_syst%num_redox_eq_reacts
            if (allocated(this%eq_reactions)) then
                deallocate(this%eq_reactions)
            end if
            allocate(this%eq_reactions(this%num_eq_reactions))
        end subroutine
        
        !subroutine allocate_kin_reactions(this,num_kin_reactions)
        !>    implicit none
        !>    class(reactive_zone_c) :: this
        !>    integer(kind=4), intent(in), optional :: num_kin_reactions
        !>    if (present(num_kin_reactions)) then
        !>        call this%set_num_kin_reactions(num_kin_reactions)
        !>    end if
        !>    allocate(this%kin_reactions(this%num_kin_reactions))
        !end subroutine
        
        subroutine is_nf_species_in_react_zone(this,nf_species,flag,nf_species_ind)
            implicit none
            class(reactive_zone_c), intent(in) :: this
            class(species_c), intent(in) :: nf_species
            logical, intent(out) :: flag
            integer(kind=4), intent(out) :: nf_species_ind
            
            integer(kind=4) :: i
            
            nf_species_ind=0
            flag=.false.
            do i=1,this%num_non_flowing_species
                if (nf_species%name==this%non_flowing_species(i)%name) then
                    flag=.true.
                    nf_species_ind=i
                    exit
                end if
            end do
        end subroutine
        
        function compute_num_species_react_zone(this) result(n_sp)
            implicit none
            class(reactive_zone_c), intent(in) :: this
            integer(kind=4) :: n_sp
            n_sp=this%chem_syst%aq_phase%num_species+this%num_non_flowing_species
        end function
        
        function compute_num_cst_act_species_react_zone(this) result(n_c)
            implicit none
            class(reactive_zone_c), intent(in) :: this
            integer(kind=4) :: n_c
            logical :: flag
            n_c=this%num_non_flowing_species    !> we assume:    1) all non-flowing species are pure minerals
            call this%chem_syst%aq_phase%is_water_in_aq_phase(flag)
            if (flag==.true.) then
                n_c=n_c+1 
            end if
        end function
        
        
        
        subroutine is_mineral_in_react_zone(this,mineral,flag,index)
            implicit none
            class(reactive_zone_c), intent(in) :: this
            class(mineral_c), intent(in) :: mineral
            logical, intent(out) :: flag
            integer(kind=4), intent(out), optional :: index !> in reactive zone minerals
            
            integer(kind=4) :: i
            
            flag=.false.
            if (present(index)) then
                index=0
            end if
            do i=1,this%num_minerals
                if (mineral%name==this%minerals(i)%name) then
                    flag=.true.
                    if (present(index)) then
                        index=i
                    end if
                    exit
                end if
            end do
        end subroutine
        
        !subroutine set_prim_species_indices(this,num_prim_species)
        !>    implicit none
        !>    class(reactive_zone_c) :: this
        !>    integer(kind=4), intent(in) :: num_prim_species
        !>    
        !>    integer(kind=4) :: i,j,k
        !>    logical :: flag
        !>    
        !>    allocate(this%prim_species_indices(num_prim_species))
        !>    
        !>    k=1 !> counter primary species
        !>    do i=1,this%num_eq_reactions
        !>        do j=1,num_prim_species
        !>            call this%eq_reactions(i)%is_species_in_react(this%chem_syst%aq_phase%aq_species(j),flag)
        !>            if (flag==.true.) then
        !>                this%prim_species_indices(k)=j
        !>                k=k+1
        !>            else
        !>                continue
        !>            end if
        !>        end do
        !>    end do
        !end subroutine
        
        subroutine set_cat_exch_zone(this,cat_exch_zone)
            implicit none
            class(reactive_zone_c) :: this
            class(cat_exch_c), intent(in), optional :: cat_exch_zone
            
            integer(kind=4) :: i
            logical :: flag
            
            if (present(cat_exch_zone)) then
                do i=1,cat_exch_zone%num_surf_compl
                    call this%chem_syst%cat_exch%is_surf_compl_in(cat_exch_zone%surf_compl(i),flag)
                    if (flag==.false.) then
                        error stop
                    end if
                end do
                this%cat_exch_zone=cat_exch_zone
            else
                this%cat_exch_zone=this%chem_syst%cat_exch
            end if
        end subroutine
        
        subroutine deallocate_react_zone(this)
            implicit none
            class(reactive_zone_c) :: this
            deallocate(this%minerals)
            deallocate(this%non_flowing_species)
        end subroutine
end module