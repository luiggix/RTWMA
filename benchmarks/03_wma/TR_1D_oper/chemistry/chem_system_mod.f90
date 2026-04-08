!> This class contains the chemical theory (like a "chemistry book"), that is, species, phases and reactions.
module chem_system_m
    use mineral_m
    use matrices_m
    use eq_reaction_m
    use lin_kin_reaction_m
    use redox_kin_reaction_m
    use kin_mineral_m
    use aq_phase_m
    use gas_phase_m
    use surf_compl_m
    use speciation_algebra_m
    use array_ops_m
    implicit none
    save
    type, public :: chem_system_c
        integer(kind=4) :: num_species !> number of species
        integer(kind=4) :: num_reacts !> number of reactions
        type(species_c), allocatable :: species(:)  !> species
        real(kind=8), allocatable :: z2(:) !> squared charges of species
        type(aq_phase_c) :: aq_phase !> aqueous phase (we assume only 1)
        type(gas_phase_c) :: gas_phase !> gas phase (we assume only 1)
        integer(kind=4) :: num_var_act_species !> number of variable activity species
        type(species_c), allocatable :: var_act_species(:) !> variable activity species
        integer(kind=4) :: num_minerals !> number of minerals
        integer(kind=4) :: num_minerals_eq !> number of minerals in equilibrium
        type(mineral_c), allocatable :: minerals(:) !> minerals
        integer(kind=4) :: num_cst_act_species !> number of constant activity species
        type(species_c), allocatable :: cst_act_species(:) !> constant activity species
        type(cat_exch_c) :: cat_exch !> cation exchange object
        real(kind=8), allocatable :: stoich_mat(:,:) !> global stoichiometric matrix (S)
        real(kind=8), allocatable :: Se(:,:) !> equilibrium stoichiometric matrix
        real(kind=8), allocatable :: Sk(:,:) !> kinetic stoichiometric matrix
        integer(kind=4) :: num_eq_reacts=0 !> number of equilibrium reactions
        integer(kind=4) :: num_eq_reacts_homog=0 !> number of homogeneous equilibrium reactions
        integer(kind=4) :: num_redox_eq_reacts=0 !> number of redox equilibrium reactions
        type(eq_reaction_c), allocatable :: eq_reacts(:) !> equilibrium reactions
        type(speciation_algebra_c) :: speciation_alg !> speciation algebra object
        integer(kind=4) :: num_kin_reacts=0 !> number of kinetic reactions
        type(kin_reaction_ptr_c), allocatable :: kin_reacts(:) !> kinetic reactions pointers (not used at the moment)
        integer(kind=4) :: num_lin_kin_reacts=0 !> number of linear kinetic reactions
        type(lin_kin_reaction_c), allocatable :: lin_kin_reacts(:) !> linear kinetic reactions
        integer(kind=4) :: num_min_kin_reacts=0 !> number of mineral kinetic reactions
        type(kin_mineral_c), allocatable :: min_kin_reacts(:) !> mineral kinetic reactions
        integer(kind=4) :: num_redox_kin_reacts=0 !> number of Monod reactions
        type(redox_kin_c), allocatable :: redox_kin_reacts(:) !> redox kinetic reactions
    contains
    !> Set
        procedure, public :: set_num_species
        procedure, public :: set_num_reacts
        procedure, public :: set_species
        procedure, public :: set_cat_exch_obj
        procedure, public :: set_num_cst_act_species
        procedure, public :: set_cst_act_species
        procedure, public :: set_var_act_species
        procedure, public :: set_num_var_act_species
        procedure, public :: set_redox_kin_reacts
        procedure, public :: set_minerals
        procedure, public :: set_eq_reacts
        procedure, public :: set_kin_reacts
        procedure, public :: set_num_lin_kin_reacts
        procedure, public :: set_num_min_kin_reacts
        procedure, public :: set_stoich_mat
    !> Allocate
        procedure, public :: allocate_species
        procedure, public :: allocate_cst_act_species
        procedure, public :: allocate_var_act_species
        procedure, public :: allocate_eq_reacts
        procedure, public :: allocate_reacts
        procedure, public :: allocate_redox_kin_reacts
        procedure, public :: allocate_minerals
        procedure, public :: allocate_min_kin_reacts
        procedure, public :: allocate_lin_kin_reacts
    !> Compute
        procedure, public :: compute_num_kin_reacts
        procedure, public :: compute_num_species
        procedure, public :: compute_z2
    !> Read
        procedure, public :: read_chem_system
        procedure, public :: read_chem_system_CHEPROO
        procedure, public :: read_chem_system_PFLOTRAN
        procedure, public :: read_master25
        procedure, public :: read_kinetics_DB
        procedure, public :: read_redox_reacts
        procedure, public :: read_PHREEQC_DB_opc1
        procedure, public :: read_PHREEQC_DB_opc2
    !> Is
        procedure, public :: is_mineral_in_chem_syst
        procedure, public :: is_eq_reaction_in_chem_syst
        procedure, public :: is_redox_kin_reaction_in_chem_syst
        procedure, public :: is_species_in_chem_syst
    !> Get
        procedure, public :: get_species_indices_react
        procedure, public :: get_eq_csts
    !> Rearrange
        procedure, public :: rearrange_eq_reacts
        procedure, public :: rearrange_species
    end type
!****************************************************************************************************************************************************
    interface
        subroutine read_chem_system(this,path,filename)
            import chem_system_c
            implicit none
            class(chem_system_c) :: this
            character(len=*), intent(in) :: path
            character(len=*), intent(in) :: filename
        end subroutine
        
        subroutine read_chem_system_CHEPROO(this,path_DB,unit)
            import chem_system_c
            implicit none
            class(chem_system_c) :: this
            character(len=*), intent(in) :: path_DB
            integer(kind=4), intent(in) :: unit
        end subroutine
        
        subroutine read_chem_system_PFLOTRAN(this,path,unit)
            import chem_system_c
            implicit none
            class(chem_system_c) :: this
            character(len=*), intent(in) :: path
            integer(kind=4), intent(in) :: unit
        end subroutine

        subroutine read_master25(this,path,unit)
            import chem_system_c
            import reaction_c
            implicit none
            class(chem_system_c) :: this
            character(len=*), intent(in) :: path
            integer(kind=4), intent(in) :: unit
        end subroutine
        
        subroutine read_kinetics_DB(this,path,unit)
            import chem_system_c
            implicit none
            class(chem_system_c) :: this
            character(len=*), intent(in) :: path
            integer(kind=4), intent(in) :: unit
        end subroutine
        
        subroutine read_redox_reacts(this,path,unit)
            import chem_system_c
            implicit none
            class(chem_system_c) :: this
            character(len=*), intent(in) :: path
            integer(kind=4), intent(in) :: unit
        end subroutine
        
        subroutine read_PHREEQC_DB_opc1(this,filename)
            import chem_system_c
            implicit none
            class(chem_system_c) :: this
            character(len=*), intent(in) :: filename
        end subroutine
        
        subroutine read_PHREEQC_DB_opc2(this,filename)
            import chem_system_c
            implicit none
            class(chem_system_c) :: this
            character(len=*), intent(in) :: filename
        end subroutine
        
        subroutine read_PFLOTRAN_DB(this,unit,filename)
        import chem_system_c
            implicit none
            class(chem_system_c) :: this
            integer(kind=4), intent(in) :: unit
            character(len=*), intent(in) :: filename
        end subroutine
        

        
      
       
     
       
    subroutine append_phase(this,phase)
        import chem_system_c
        import phase_c
        implicit none
        class(chem_system_c) :: this
        class(phase_c), intent(in) :: phase
    end subroutine
    
    subroutine append_reaction(this,reaction)
        import chem_system_c
        import reaction_c
        implicit none
        class(chem_system_c) :: this
        class(reaction_c), intent(in) :: reaction
    end subroutine
    
        subroutine set_stoich_mat(this)
            import chem_system_c
            implicit none
            class(chem_system_c) :: this
        end subroutine
      
    end interface
!****************************************************************************************************************************************************
    contains

        
        subroutine set_num_species(this,num_species)
            implicit none
            class(chem_system_c) :: this
            integer(kind=4), intent(in), optional :: num_species
            if (present(num_species)) then
                if (num_species<1) then
                    error stop "Number of species must be positive"
                else
                    this%num_species=num_species
                end if
            else
                this%num_species=size(this%species)
            end if
        end subroutine
        
        subroutine set_num_reacts(this,num_reacts)
            implicit none
            class(chem_system_c) :: this
            integer(kind=4), intent(in), optional :: num_reacts
            if (present(num_reacts)) then
                if (num_reacts>this%num_species) then
                    error stop "Number of reacts cannot be greater than number of species"
                else
                    this%num_reacts=num_reacts
                end if
            else
                this%num_reacts=this%num_eq_reacts+this%num_kin_reacts
            end if
        end subroutine
        
        subroutine allocate_species(this,num_species)
            implicit none
            class(chem_system_c) :: this
            integer(kind=4), intent(in), optional :: num_species
            if (present(num_species)) then
                this%num_species=num_species
            end if
            if (allocated(this%species)) then
                deallocate(this%species)
            end if
            allocate(this%species(this%num_species))
        end subroutine
        
        subroutine allocate_cst_act_species(this,num_cst_act_species)
            implicit none
            class(chem_system_c) :: this
            integer(kind=4), intent(in), optional :: num_cst_act_species
            if (present(num_cst_act_species)) then
                this%num_cst_act_species=num_cst_act_species                
            end if
            if (allocated(this%cst_act_species)) then
                deallocate(this%cst_act_species)
            end if
            allocate(this%cst_act_species(this%num_cst_act_species))
        end subroutine
        
        subroutine allocate_var_act_species(this,num_var_act_species)
            implicit none
            class(chem_system_c) :: this
            integer(kind=4), intent(in), optional :: num_var_act_species
            if (present(num_var_act_species)) then
                this%num_var_act_species=num_var_act_species                
            end if
            allocate(this%var_act_species(this%num_var_act_species))
        end subroutine
        
        subroutine set_species(this,species) !> sets species (first variable activity, then constant activity)
            implicit none
            class(chem_system_c) :: this
            class(species_c), intent(in), optional :: species(:)
            
            integer(kind=4) :: i
            
            if (present(species)) then
                if (allocated(this%species) .and. size(species)/=this%num_species) then
                    error stop "Wrong number of species"
                else
                    this%species=species
                    this%num_species=size(species)
                end if
            else if (allocated(this%cst_act_species) .and. allocated(this%var_act_species)) then
                call this%allocate_species()
                this%species(1:this%num_var_act_species)=this%var_act_species
                this%species(this%num_var_act_species+1:this%num_species)=this%cst_act_species
            end if
        end subroutine
        

        
        subroutine compute_num_species(this)
            implicit none
            class(chem_system_c) :: this
            this%num_species=this%num_var_act_species+this%num_cst_act_species
        end subroutine
        
        subroutine compute_num_kin_reacts(this)
            implicit none
            class(chem_system_c) :: this
            this%num_kin_reacts=this%num_lin_kin_reacts+this%num_min_kin_reacts+this%num_redox_kin_reacts
        end subroutine
        
        subroutine allocate_reacts(this,num_eq_reacts,num_kin_reacts)
            implicit none
            class(chem_system_c) :: this
            integer(kind=4), intent(in), optional :: num_eq_reacts,num_kin_reacts
            if (present(num_eq_reacts) .and. present(num_kin_reacts)) then
                this%num_eq_reacts=num_eq_reacts
                this%num_kin_reacts=num_kin_reacts
            else if (.not. present(num_eq_reacts) .and. .not. present(num_kin_reacts)) then
                continue
            else
                error stop "Both optional parameters must be present in subroutine 'allocate_reacts'"
            end if
            allocate(this%eq_reacts(this%num_eq_reacts))
            allocate(this%kin_reacts(this%num_kin_reacts))
        end subroutine
        
        subroutine allocate_eq_reacts(this,num_eq_reacts)
            implicit none
            class(chem_system_c) :: this
            integer(kind=4), intent(in), optional :: num_eq_reacts
            if (present(num_eq_reacts)) then
                this%num_eq_reacts=num_eq_reacts
            end if
            allocate(this%eq_reacts(this%num_eq_reacts))
        end subroutine
        
        subroutine allocate_redox_kin_reacts(this,num_redox_kin_reacts)
            implicit none
            class(chem_system_c) :: this
            integer(kind=4), intent(in), optional :: num_redox_kin_reacts
            if (present(num_redox_kin_reacts)) then
                this%num_redox_kin_reacts=num_redox_kin_reacts
            end if
            allocate(this%redox_kin_reacts(this%num_redox_kin_reacts))
        end subroutine
        
        subroutine allocate_lin_kin_reacts(this,num_lin_kin_reacts)
            implicit none
            class(chem_system_c) :: this
            integer(kind=4), intent(in), optional :: num_lin_kin_reacts
            if (present(num_lin_kin_reacts)) then
                this%num_lin_kin_reacts=num_lin_kin_reacts
            end if
            allocate(this%lin_kin_reacts(this%num_lin_kin_reacts))
        end subroutine
        
        subroutine allocate_minerals(this,num_minerals)
            implicit none
            class(chem_system_c) :: this
            integer(kind=4), intent(in), optional :: num_minerals
            if (present(num_minerals)) then
                this%num_minerals=num_minerals
            end if
            allocate(this%minerals(this%num_minerals))
        end subroutine
        
        subroutine allocate_min_kin_reacts(this,num_min_kin)
            implicit none
            class(chem_system_c) :: this
            integer(kind=4), intent(in), optional :: num_min_kin
            if (.not. allocated(this%minerals)) error stop
            if (present(num_min_kin)) then
                this%num_min_kin_reacts=num_min_kin
            end if
            allocate(this%min_kin_reacts(this%num_min_kin_reacts))
        end subroutine
        
        
        subroutine set_num_cst_act_species(this,num_cst_act_species)
            implicit none
            class(chem_system_c) :: this
            integer(kind=4), intent(in), optional :: num_cst_act_species
            if (present(num_cst_act_species)) then
                this%num_cst_act_species=num_cst_act_species
            else
                this%num_cst_act_species=size(this%cst_act_species)
            end if
        end subroutine
        
        subroutine set_cst_act_species(this,cst_act_species) !> first water, then minerals
            implicit none
            class(chem_system_c) :: this
            type(species_c), intent(in), optional :: cst_act_species(:)
            
            integer(kind=4) :: i,wat_ind,j
            type(aq_species_c) :: water
            logical :: wat_flag
            
            if (present(cst_act_species)) then
                if (size(cst_act_species)>this%num_species) then
                    error stop "Number of constant activity species cannot be greater than number of species"
                else
                    this%cst_act_species=cst_act_species
                end if
            else
                j=0 !> counter constant activity species
                call this%aq_phase%is_water_in_aq_phase(wat_flag)
                if (wat_flag==.true.) then
                    j=j+1
                    call this%cst_act_species(j)%assign_species(this%aq_phase%aq_species(this%aq_phase%ind_wat))
                end if
                do i=1,this%num_minerals
                    call this%cst_act_species(j+i)%assign_species(this%minerals(i)%mineral) !> we assume minerals are pure
                end do
                j=j+this%num_minerals
            end if
        end subroutine
        

        
        subroutine set_cat_exch_obj(this,cat_exch)
            implicit none
            class(chem_system_c) :: this
            class(cat_exch_c), intent(in) :: cat_exch
            this%cat_exch=cat_exch
        end subroutine
        
      
        
        subroutine set_num_var_act_species(this,num_var_act_species)
            implicit none
            class(chem_system_c) :: this
            integer(kind=4), intent(in), optional :: num_var_act_species
            if (present(num_var_act_species)) then
                this%num_var_act_species=num_var_act_species
            else if (allocated(this%var_act_species)) then
                this%num_var_act_species=size(this%var_act_species)
            else if (allocated(this%species) .and. allocated(this%cst_act_species)) then
                this%num_var_act_species=this%num_species-this%num_cst_act_species
            else
                error stop "Unable to compute the number of variable activity species"
            end if
        end subroutine
        
        subroutine set_var_act_species(this,var_act_species) !> first aqueous, then gases, then surface complexes
            implicit none
            class(chem_system_c) :: this
            type(species_c), intent(in), optional :: var_act_species(:)
            integer(kind=4) :: i,j,k
            if (present(var_act_species)) then
                if (size(var_act_species)>this%num_species) then
                    error stop "Number of variable activity species cannot be greater than number of species"
                else
                    this%var_act_species=var_act_species
                end if
            else
                call this%set_num_var_act_species()
                call this%allocate_var_act_species()
                i=1 !> counter aqueous species
                j=1 !> counter variable activity species
                do
                    if (i/=this%aq_phase%ind_wat) then
                        call this%var_act_species(j)%assign_species(this%aq_phase%aq_species(i))
                        if (j<this%num_var_act_species) then
                            j=j+1
                        else
                            exit
                        end if
                        if (i<this%aq_phase%num_species) then
                            i=i+1
                        else
                            exit
                        end if
                    else if (i<this%aq_phase%num_species) then
                        i=i+1
                    else
                        exit
                    end if
                end do
                do k=1,this%gas_phase%num_species
                    call this%var_act_species(i+k)%assign_species(this%gas_phase%gases(k))
                end do
                i=i+this%gas_phase%num_species
                do k=1,this%cat_exch%num_surf_compl
                    call this%var_act_species(i+k)%assign_species(this%cat_exch%surf_compl(k))
                end do
            end if
        end subroutine
        
        subroutine set_minerals(this,minerals)
            implicit none
            class(chem_system_c) :: this
            class(mineral_c), intent(in) :: minerals(:)
            this%minerals=minerals
            this%num_minerals=size(this%minerals) 
        end subroutine
        
     
        
        subroutine set_eq_reacts(this,eq_reacts)
            implicit none
            class(chem_system_c) :: this
            class(eq_reaction_c), intent(in) :: eq_reacts(:)
            if (allocated(this%eq_reacts)) then
                deallocate(this%eq_reacts)
            end if
            this%eq_reacts=eq_reacts
            this%num_eq_reacts=size(this%eq_reacts)
        end subroutine
        
        subroutine set_kin_reacts(this,kin_reacts)
            implicit none
            class(chem_system_c) :: this
            class(kin_reaction_ptr_c), intent(in) :: kin_reacts(:)
            if (allocated(this%kin_reacts)) then
                deallocate(this%kin_reacts)
            end if
            this%kin_reacts=kin_reacts
            this%num_kin_reacts=size(this%kin_reacts)
        end subroutine
        
       subroutine set_num_lin_kin_reacts(this,num_lin_kin_reacts)
            implicit none
            class(chem_system_c) :: this
            integer(kind=4), intent(in) :: num_lin_kin_reacts
            this%num_lin_kin_reacts=num_lin_kin_reacts
       end subroutine
       
       subroutine set_num_min_kin_reacts(this,num_min_kin_reacts)
            implicit none
            class(chem_system_c) :: this
            integer(kind=4), intent(in) :: num_min_kin_reacts
            this%num_min_kin_reacts=num_min_kin_reacts
       end subroutine
       
       subroutine set_redox_kin_reacts(this,redox_kin_reacts)
            implicit none
            class(chem_system_c) :: this
            class(redox_kin_c), intent(in) :: redox_kin_reacts(:)
            if (allocated(this%redox_kin_reacts)) then
                deallocate(this%redox_kin_reacts)
            end if
            this%redox_kin_reacts=redox_kin_reacts
            this%num_redox_kin_reacts=size(redox_kin_reacts)
        end subroutine
        
       
        
        
        
        subroutine is_mineral_in_chem_syst(this,mineral,flag,mineral_ind)
            implicit none
            class(chem_system_c), intent(in) :: this
            class(mineral_c), intent(in) :: mineral
            logical, intent(out) :: flag
            integer(kind=4), intent(out), optional :: mineral_ind
            
            integer(kind=4) :: i
            
            flag=.false.
            if (present(mineral_ind)) then
                mineral_ind=0
            end if
            do i=1,this%num_minerals
                if (mineral%name==this%minerals(i)%name) then
                    flag=.true.
                    if (present(mineral_ind)) then
                        mineral_ind=i
                    end if
                    exit
                end if
            end do
        end subroutine
        
        subroutine is_species_in_chem_syst(this,species,flag,species_ind)
            implicit none
            class(chem_system_c), intent(in) :: this
            class(species_c), intent(in) :: species
            logical, intent(out) :: flag !> TRUE if species belongs to chemical system
            integer(kind=4), intent(out), optional :: species_ind !> species index in 'species' attribute
            
            integer(kind=4) :: i
            
            flag=.false.
            if (present(species_ind)) then
                species_ind=0
            end if
            do i=1,this%num_species
                if (species%name==this%species(i)%name) then
                    flag=.true.
                    if (present(species_ind)) then
                        species_ind=i
                    end if
                    exit
                end if
            end do
        end subroutine
        
        
        
        subroutine is_eq_reaction_in_chem_syst(this,react_name,flag,react_ind)
            implicit none
            class(chem_system_c), intent(in) :: this
            character(len=*), intent(in) :: react_name
            logical, intent(out) :: flag
            integer(kind=4), intent(out), optional :: react_ind
            
            integer(kind=4) :: i,sp_ind
            integer(kind=4), allocatable :: sp_indices(:)
            logical :: sp_flag
            
            flag=.false.
            if (present(react_ind)) then
                react_ind=0
            end if
            do i=1,this%num_eq_reacts
                !call this%eq_reacts(i)%is_species_in_react(species,sp_flag)
                !if (sp_flag==.true.) then
                if (this%eq_reacts(i)%name==react_name) then
                    flag=.true.
                    if (present(react_ind)) then
                        react_ind=i
                    end if
                    exit
                end if
            end do
        end subroutine
        
        subroutine is_redox_kin_reaction_in_chem_syst(this,react_name,flag,react_ind)
            implicit none
            class(chem_system_c), intent(in) :: this
            character(len=*), intent(in) :: react_name
            logical, intent(out) :: flag
            integer(kind=4), intent(out), optional :: react_ind
            
            integer(kind=4) :: i,sp_ind
            integer(kind=4), allocatable :: sp_indices(:)
            logical :: sp_flag
            
            flag=.false.
            if (present(react_ind)) then
                react_ind=0
            end if
            do i=1,this%num_redox_kin_reacts
                !call this%eq_reacts(i)%is_species_in_react(species,sp_flag)
                !if (sp_flag==.true.) then
                if (this%redox_kin_reacts(i)%name==react_name) then
                    flag=.true.
                    if (present(react_ind)) then
                        react_ind=i
                    end if
                    exit
                end if
            end do
        end subroutine
        
        
        
        subroutine get_species_indices_react(this,react,species_indices) !> gets the indices in chemical system of reaction species
            implicit none
            class(chem_system_c), intent(in) :: this
            class(reaction_c), intent(in) :: react !> reaction
            integer(kind=4), intent(out), allocatable :: species_indices(:) !> indices of species in chemical system
            
            logical :: flag
            integer(kind=4) :: sp_ind,i
            
            allocate(species_indices(0))
            
            do i=1,react%num_species
                call this%is_species_in_chem_syst(react%species(i),flag,sp_ind)
                if (flag==.false.) then
                    error stop "This reaction does not belong to chemical system"
                else
                    call append_int_1D_array(species_indices,sp_ind)
                end if
            end do
        end subroutine
        
        function get_eq_csts(this) result(K) !> gets equilibrium constants of equilibrium reactions
            implicit none
            class(chem_system_c), intent(in) :: this
            real(kind=8), allocatable :: K(:)
            
            integer(kind=4) :: i
          
            allocate(K(this%num_eq_reacts))
            do i=1,this%num_eq_reacts
                K(i)=this%eq_reacts(i)%eq_cst
            end do
        end function
        
        subroutine rearrange_eq_reacts(this) !> first constant activity minerals, then constant activity gases, then aqueous complexes, then variable activity solids, finally variable activity gases
            implicit none
            class(chem_system_c) :: this
            
            integer(kind=4) :: i,ind_min,ind_aq,ind_gas,ind_surf,ind_cst_act,ind_var_act,ind_redox
            type(eq_reaction_c), allocatable :: aux_eq_reacts(:)
            
            aux_eq_reacts=this%eq_reacts
            deallocate(this%eq_reacts)
            allocate(this%eq_reacts(this%num_eq_reacts))
            ind_cst_act=1
            ind_var_act=this%num_cst_act_species-this%aq_phase%wat_flag+1
            ind_aq=this%num_cst_act_species-this%aq_phase%wat_flag+this%num_redox_eq_reacts+1
            ind_redox=this%num_cst_act_species-this%aq_phase%wat_flag+1
            ind_gas=ind_aq+this%aq_phase%num_aq_complexes
            ind_surf=ind_gas+this%gas_phase%num_species
            do i=1,this%num_eq_reacts
                if (aux_eq_reacts(i)%react_type==2) then !> mineral dissolution/precipitation
                    if (aux_eq_reacts(i)%species(aux_eq_reacts(i)%num_species)%cst_act_flag==.true.) then
                        this%eq_reacts(ind_cst_act)=aux_eq_reacts(i)
                        ind_cst_act=ind_cst_act+1
                    else
                        this%eq_reacts(this%aq_phase%num_aq_complexes+this%num_redox_eq_reacts+ind_var_act)=aux_eq_reacts(i)
                        ind_var_act=ind_var_act+1
                    end if
                else if (aux_eq_reacts(i)%react_type==1) then !> aqueous complex
                    this%eq_reacts(ind_aq)=aux_eq_reacts(i)
                    ind_aq=ind_aq+1
                else if (aux_eq_reacts(i)%react_type==6) then !> gas
                    if (aux_eq_reacts(i)%species(aux_eq_reacts(i)%num_species)%cst_act_flag==.true.) then
                        this%eq_reacts(ind_cst_act)=aux_eq_reacts(i)
                        ind_cst_act=ind_cst_act+1
                    else
                        this%eq_reacts(this%aq_phase%num_aq_complexes+this%num_redox_eq_reacts+ind_var_act)=aux_eq_reacts(i)
                        ind_var_act=ind_var_act+1
                    end if
                else if (aux_eq_reacts(i)%react_type==3) then !> cation exchange
                    this%eq_reacts(ind_var_act)=aux_eq_reacts(i)
                    ind_var_act=ind_var_act+1
                else if (aux_eq_reacts(i)%react_type==4) then !> redox
                    this%eq_reacts(ind_redox)=aux_eq_reacts(i)
                    ind_redox=ind_redox+1
                end if
            end do
        end subroutine
        
        subroutine rearrange_species(this)
            implicit none
            class(chem_system_c) :: this
            integer(kind=4) :: i,num_sp,num_aq_sec,num_var_act_sp,num_cst_act_sp
            num_sp=0 !> counter number of species
            num_var_act_sp=0 !> counter number of variable activity species
            num_cst_act_sp=this%aq_phase%wat_flag !> counter number of constant activity species
            if (this%speciation_alg%flag_comp==.false. .and. this%speciation_alg%flag_cat_exch==.true.) then
            !> first primary aqueous and free surfaces, then secondary aqueous, minerals, gases and secondary surface complexes 
                num_aq_sec=this%aq_phase%num_species-this%speciation_alg%num_aq_prim_species
                do i=1,this%speciation_alg%num_aq_prim_species
                    call this%species(i)%assign_species(this%aq_phase%aq_species(i))
                end do
                call this%species(this%speciation_alg%num_prim_species)%assign_species(this%cat_exch%surf_compl(1)) !> free surface
                num_sp=num_sp+this%speciation_alg%num_prim_species
                do i=1,num_aq_sec
                    call this%species(num_sp+i)%assign_species(this%aq_phase%aq_species(this%speciation_alg%num_aq_prim_species+i))
                end do
                num_sp=num_sp+num_aq_sec
                do i=1,this%num_minerals
                    call this%species(num_sp+i)%assign_species(this%minerals(i)%mineral)
                end do
                num_sp=num_sp+this%num_minerals
                do i=1,this%gas_phase%num_species
                    call this%species(num_sp+i)%assign_species(this%gas_phase%gases(i))
                end do
                num_sp=num_sp+this%gas_phase%num_species
                do i=1,this%cat_exch%num_exch_cats
                    call this%species(num_sp+i)%assign_species(this%cat_exch%surf_compl(1+i))
                end do
            else if (this%speciation_alg%flag_comp==.true. .and. this%speciation_alg%flag_cat_exch==.true.) then
            !> first primary aqueous and free surfaces, then secondary variable activity aqueous, solids and gases, then secondary constant activity aqueous, solids and gases
                do i=1,this%speciation_alg%num_aq_prim_species
                    call this%species(i)%assign_species(this%aq_phase%aq_species(i))
                end do
                call this%species(this%speciation_alg%num_prim_species)%assign_species(this%cat_exch%surf_compl(1))
                do i=1,this%speciation_alg%num_aq_sec_var_act_species
                    call this%species(this%speciation_alg%num_prim_species+i)%assign_species(this%aq_phase%aq_species(this%speciation_alg%num_aq_prim_species+i))
                end do
                num_var_act_sp=1+this%speciation_alg%num_aq_var_act_species
                do i=1,this%num_minerals
                    if (this%minerals(i)%mineral%cst_act_flag==.false.) then
                        call this%species(num_var_act_sp+1)%assign_species(this%minerals(i)%mineral)
                        num_var_act_sp=num_var_act_sp+1
                    else
                        call this%species(this%num_var_act_species+num_cst_act_sp+1)%assign_species(this%minerals(i)%mineral)
                        num_cst_act_sp=num_cst_act_sp+1
                    end if
                end do
                do i=1,this%cat_exch%num_exch_cats
                    if (this%cat_exch%surf_compl(1+i)%cst_act_flag==.false.) then
                        call this%species(num_var_act_sp+1)%assign_species(this%cat_exch%surf_compl(1+i))
                        num_var_act_sp=num_var_act_sp+1
                    else
                        call this%species(this%num_var_act_species+num_cst_act_sp+1)%assign_species(this%cat_exch%surf_compl(1+i))
                        num_cst_act_sp=num_cst_act_sp+1
                    end if
                end do
                do i=1,this%gas_phase%num_species
                    if (this%gas_phase%gases(i)%cst_act_flag==.false.) then
                        call this%species(num_var_act_sp+1)%assign_species(this%gas_phase%gases(i))
                        num_var_act_sp=num_var_act_sp+1
                    else
                        call this%species(this%num_var_act_species+num_cst_act_sp+1)%assign_species(this%gas_phase%gases(i))
                        num_cst_act_sp=num_cst_act_sp+1
                    end if
                end do
                do i=1,this%num_cst_act_species
                    this%species(this%num_var_act_species+i)=this%cst_act_species(i)
                end do
            else if (this%speciation_alg%flag_comp==.true. .and. this%speciation_alg%flag_cat_exch==.false.) then
                !call this%set_species()
                do i=1,this%speciation_alg%num_aq_var_act_species
                    call this%species(i)%assign_species(this%var_act_species(i))
                end do
                num_var_act_sp=num_var_act_sp+this%speciation_alg%num_aq_var_act_species
                if (this%aq_phase%wat_flag==1) then
                    call this%species(this%num_var_act_species+1)%assign_species(this%cst_act_species(1))
                end if
                do i=1,this%num_minerals
                    if (this%minerals(i)%mineral%cst_act_flag==.false.) then
                        call this%species(num_var_act_sp+1)%assign_species(this%minerals(i)%mineral)
                        num_var_act_sp=num_var_act_sp+1
                    else
                        call this%species(this%num_var_act_species+num_cst_act_sp+1)%assign_species(this%minerals(i)%mineral)
                        num_cst_act_sp=num_cst_act_sp+1
                    end if
                end do
                do i=1,this%gas_phase%num_species
                    if (this%gas_phase%gases(i)%cst_act_flag==.false.) then
                        call this%species(num_var_act_sp+1)%assign_species(this%gas_phase%gases(i))
                        num_var_act_sp=num_var_act_sp+1
                    else
                        call this%species(this%num_var_act_species+num_cst_act_sp+1)%assign_species(this%gas_phase%gases(i))
                        num_cst_act_sp=num_cst_act_sp+1
                    end if
                end do
            else
                continue
            end if
        end subroutine
        
        subroutine compute_z2(this) !> computes array squared species charges
            implicit none
            class(chem_system_c) :: this
            integer(kind=4) :: i
            if (.not. allocated(this%z2)) then
                allocate(this%z2(this%num_species))
            end if
            do i=1,this%num_species
                this%z2(i)=this%species(i)%valence**2
            end do
        end subroutine
        
end module 