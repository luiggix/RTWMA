!> Aqueous phase subclass:
!!   contains the aqueous species in a solution
!!   computes log_10 activity coefficients and its Jacobian
module aq_phase_m
    use phase_m
    use aq_species_m
    use vectors_m
    implicit none
    save
    type, public, extends(phase_c) :: aq_phase_c
        !real(kind=8), allocatable :: z2(:) !> squared charges of aqueous species
        integer(kind=4) :: num_aq_complexes=0 !> number of aqueous complexes
        type(aq_species_c), allocatable :: aq_species(:) !> aqueous species in solution
        integer(kind=4) :: wat_flag=0 !> 1 if water is present, 0 otherwise
        integer(kind=4) :: ind_wat=0 !> index of water in 'aq_species' array
        integer(kind=4), allocatable :: ind_diss_solids(:) !> indices of dissolved solids
        integer(kind=4) :: ind_proton=0 !> index 'H+' in 'aq_species' array
        integer(kind=4) :: ind_oh=0 !> index 'OH-' in 'aq_species' array
        integer(kind=4) :: ind_bicarb=0 !> index 'HCO3-' in 'aq_species' array
        integer(kind=4) :: ind_carb=0 !> index 'CO3-2' in 'aq_species' array
    contains
    !> Set
        procedure, public :: set_ind_proton
        procedure, public :: set_ind_wat
        procedure, public :: set_ind_diss_solids
        procedure, public :: set_indices_aq_phase
        procedure, public :: set_num_aq_complexes
        procedure, public :: set_aq_species
        procedure, public :: set_single_aq_species
    !> Allocate
        procedure, public :: allocate_aq_species
    !> Compute
        procedure, public :: compute_log_act_coeffs_aq_phase
        procedure, public :: compute_log_Jacobian_act_coeffs_aq_phase
        !procedure, public :: compute_d_log_gamma_d_c1
        !procedure, public :: compute_d_log_gamma_d_c2_aq
        !procedure, public :: compute_z2
    !> Rearrange
        procedure, public :: rearrange_aq_species
    !> Is
        procedure, public :: is_species_in_aq_phase
        procedure, public :: is_water_in_aq_phase

    end type
    
    interface
        subroutine compute_log_act_coeffs_aq_phase(this,ionic_act,params_aq_sol,log_act_coeffs)
            import aq_phase_c
            import params_aq_sol_t
            implicit none
            class(aq_phase_c) :: this
            real(kind=8), intent(in) :: ionic_act
            class(params_aq_sol_t), intent(in) :: params_aq_sol
            real(kind=8), intent(out) :: log_act_coeffs(:) !> must be allocated
        end subroutine
        
        subroutine compute_log_Jacobian_act_coeffs_aq_phase(this,out_prod,conc,log_Jacobian_act_coeffs)
            import aq_phase_c
            implicit none
            class(aq_phase_c) :: this
            !real(kind=8), intent(in) :: ionic_act
            real(kind=8), intent(in) :: out_prod(:,:) !> outer product between d_log_gamma_d_I and z^2
            !real(kind=8), intent(in) :: dc2_dc1(:,:) !> Jacobian of secondary concentrations with respect to primary cocnentrationa
            !real(kind=8), intent(in) :: log_act_coeffs(:)
            real(kind=8), intent(in) :: conc(:) !> concentration of species in a given target
            real(kind=8), intent(out) :: log_Jacobian_act_coeffs(:,:) !> must be allocated
        end subroutine
        
        !subroutine compute_d_log_gamma_d_I(this,ionic_act,params_aq_sol,d_log_gamma_d_I)
        !    import aq_phase_c
        !    import params_aq_sol_t
        !    implicit none
        !    class(aq_phase_c) :: this
        !    real(kind=8), intent(in) :: ionic_act
        !    class(params_aq_sol_t), intent(in) :: params_aq_sol
        !    real(kind=8), intent(out) :: d_log_gamma_d_I(:) !> must be allocated
        !end subroutine
        
        
    end interface
    
    contains
        
        subroutine set_num_aq_complexes(this,num_aq_complexes)
            implicit none
            class(aq_phase_c) :: this
            integer(kind=4), intent(in) :: num_aq_complexes
            this%num_aq_complexes=num_aq_complexes
        end subroutine
        
        subroutine set_ind_proton(this,ind_proton)
            implicit none
            class(aq_phase_c) :: this
            integer(kind=4), intent(in) :: ind_proton
            this%ind_proton=ind_proton
        end subroutine
        
        subroutine set_ind_wat(this,ind_wat)
            implicit none
            class(aq_phase_c) :: this
            integer(kind=4), intent(in) :: ind_wat
            this%ind_wat=ind_wat
        end subroutine
        
        subroutine allocate_aq_species(this,num_species)
            implicit none
            class(aq_phase_c) :: this
            integer(kind=4), intent(in), optional :: num_species
            if (present(num_species)) then
                this%num_species=num_species
            end if
            allocate(this%aq_species(this%num_species))
        end subroutine
        
        subroutine set_aq_species(this,aq_species)
            implicit none
            class(aq_phase_c) :: this
            class(aq_species_c), intent(in) :: aq_species(:)
            if (allocated(this%aq_species) .and. size(aq_species)>this%num_species) then
                error stop "Number of aqueous species cannot be greater than number of species"
            else
                this%aq_species=aq_species
                this%num_species=size(aq_species)
            end if
        end subroutine
        
        subroutine set_single_aq_species(this,aq_species,index)
            implicit none
            class(aq_phase_c) :: this
            class(aq_species_c), intent(in) :: aq_species
            integer(kind=4), intent(in) :: index
            if (.not. allocated(this%aq_species)) then
                error stop "Aqueous species must be allocated"
            else if (index>this%num_species) then
                error stop "Index is higher than number of aqueous species"
            else if (index<1) then
                error stop "Index must be strictly positive"
            else
                this%aq_species(index)=aq_species
            end if
        end subroutine
        
        subroutine rearrange_aq_species(this)
    !> Rearranges aqueous species in primary and secondary, respectively
            implicit none
            class(aq_phase_c) :: this
            
            type(aq_species_c), allocatable :: aux(:)
            integer(kind=4) :: i,j,k
            
            aux=this%aq_species
            deallocate(this%aq_species)
            call this%allocate_aq_species()
            j=1
            k=0
            do i=1,this%num_species
                if (aux(i)%cst_act_flag==.false.) then
                    call this%aq_species(j)%assign_species(aux(i))
                    if (j<this%num_species) then
                        j=j+1
                    end if
                else
                    call this%aq_species(this%num_species-k)%assign_species(aux(i))
                    if (aux(i)%name=='h2o') then
                        call this%set_ind_wat(this%num_species-k)
                    end if
                    k=k+1
                end if
            end do
            call this%set_ind_diss_solids()
            !call this%compute_z2()
        end subroutine
        
        subroutine append_aq_species(this,aq_species)
            implicit none
            class(aq_phase_c) :: this
            type(aq_species_c), intent(in) :: aq_species
            
            integer(kind=4) :: i
            type(aq_species_c), allocatable :: aux_array(:)
            
            aux_array=this%aq_species
            deallocate(this%aq_species)
            if (size(aux_array)>0) then
                call this%allocate_aq_species(size(aux_array)+1)
                do i=1,this%num_species-1
                    this%aq_species(i)=aux_array(i)
                end do
                this%aq_species(this%num_species)=aq_species
            else
                call this%allocate_aq_species(1)
                call this%set_aq_species([aq_species]) !> chapuza
            end if
        end subroutine
        
        subroutine is_species_in_aq_phase(this,aq_species,flag,aq_species_ind)
            implicit none
            class(aq_phase_c) :: this
            class(species_c), intent(in) :: aq_species
            logical, intent(out) :: flag
            integer(kind=4), intent(out), optional :: aq_species_ind
            
            integer(kind=4) :: i
            
            flag=.false.
            if (present(aq_species_ind)) then
                aq_species_ind=0
            end if
            do i=1,this%num_species
                if (aq_species%name==this%aq_species(i)%name) then
                    flag=.true.
                    if (aq_species%name=='h2o') then
                        this%ind_wat=i
                    end if
                    if (present(aq_species_ind)) then
                        aq_species_ind=i
                    end if
                    exit
                end if
            end do
        end subroutine
        
        subroutine is_water_in_aq_phase(this,flag)
            implicit none
            class(aq_phase_c) :: this
            logical, intent(out) :: flag
            
            type(aq_species_c) :: water
            
            water%name='h2o'
            call this%is_species_in_aq_phase(water,flag)
        end subroutine
        
        subroutine set_indices_aq_phase(this)
            implicit none
            class(aq_phase_c) :: this
            
            integer(kind=4) :: i
            
            do i=1,this%num_species
                if (this%aq_species(i)%name=='h2o') then
                    this%ind_wat=i
                else if (this%aq_species(i)%name=='h+') then
                    this%ind_proton=i
                else if (this%aq_species(i)%name=='oh-') then
                    this%ind_oh=i
                else if (this%aq_species(i)%name=='hco3-') then
                    this%ind_bicarb=i
                else if (this%aq_species(i)%name=='co3-2') then
                    this%ind_carb=i
                else
                    continue
                end if
            end do
        end subroutine
        
        subroutine set_ind_diss_solids(this)
            implicit none
            class(aq_phase_c) :: this
            
            integer(kind=4) :: i,j
            
            j=1
            do i=1,this%num_species
                if (this%aq_species(i)%name/='h2o') then
                    this%ind_diss_solids(j)=i
                    if (j<this%num_species-this%wat_flag) then
                        j=j+1
                    else
                        exit
                    end if
                end if
            end do
        end subroutine
        
        !subroutine compute_z2(this) !> computes array squared aqueous species charges
        !    implicit none
        !    class(aq_phase_c) :: this
        !    !real(kind=8), allocatable :: z2(:)
        !    integer(kind=4) :: i
        !    !allocate(this%z2(this%num_species))
        !    do i=1,this%num_species
        !        this%z2(i)=this%aq_species(i)%valence**2
        !    end do
        !end subroutine
        
        !subroutine compute_d_log_gamma_d_c1(this,d_log_gamma_d_I,dI_dc1,c1,d_log_gamma_d_c1)
        !    implicit none
        !    class(aq_phase_c), intent(in) :: this
        !    real(kind=8), intent(in) :: d_log_gamma_d_I(:)  !> Jacobian of log_10(primary activity coefficients) with respect to ionic activity
        !    real(kind=8), intent(in) :: dI_dc1(:) !> Jacobian of ionic activity with respect to primary cocnentrationa
        !    real(kind=8), intent(in) :: c1(:)  
        !    real(kind=8), intent(out) :: d_log_gamma_d_c1(:,:) 
        !    
        !    integer(kind=4) :: i,j,n_prim
        !    real(kind=8), allocatable :: out_prod(:,:)
        !    
        !    n_prim=size(c1)
        !                
        !    do i=1,this%num_species
        !        do j=1,n_prim
        !            d_log_gamma_d_c1(i,j)=d_log_gamma_d_I(i)*dI_dc1(j)*c1(j)*log(1d1)
        !        end do
        !    end do
        !end subroutine
        
        !subroutine compute_d_log_gamma_d_c2_aq(this,d_log_gamma_d_I,c2_aq,d_log_gamma_d_c2_aq)
        !    implicit none
        !    class(aq_phase_c), intent(in) :: this
        !    real(kind=8), intent(in) :: d_log_gamma_d_I(:)  
        !    real(kind=8), intent(in) :: c2_aq(:)  
        !    real(kind=8), intent(out) :: d_log_gamma_d_c2_aq(:,:) 
        !    
        !    integer(kind=4) :: j,n_p_aq
        !    
        !    n_p_aq=this%num_species-size(c2_aq)
        !    
        !    !z2=this%get_z2()
        !    !out_prod=outer_prod_vec(d_log_gamma_d_I,this%z2(n_prim+1:this%num_species))
        !    
        !    !do j=1,size(c2_aq)
        !    !    d_log_gamma_d_c2_aq(:,j)=out_prod(:,j)*c2_aq(j)
        !    !end do
        !    !d_log_gamma_d_c2_aq=d_log_gamma_d_c2_aq*log(1d1)/2d0
        !    do j=1,size(c2_aq)
        !        d_log_gamma_d_c2_aq(:,j)=5d-1*this%z2(n_p_aq+j)*c2_aq(j)*log(1d1)*d_log_gamma_d_I
        !    end do
        !end subroutine
end module