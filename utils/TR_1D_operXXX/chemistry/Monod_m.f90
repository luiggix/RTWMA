!> Redox kinetic reaction module
module redox_kin_reaction_m
    use kin_reaction_m
    use Monod_params_m
    use aq_phase_m
    implicit none
    save
    type, public, extends(kin_reaction_c) :: redox_kin_c
        type(Monod_params_c) :: params !> kinetic parameters Monod
    contains
    !> Set
        procedure, public :: set_Monod_params
    !> Compute
        procedure, public :: compute_drk_dc_Monod
        procedure, public :: compute_rk_Monod
    end type
    
    !type, public :: monod_type
    !>    PetscInt :: id
    !>    character(len=MAXWORDLENGTH) :: species_name
    !>    PetscReal :: half_saturation_constant
    !>    PetscReal :: threshold_concentration
    !>    type(monod_type), pointer :: next
    !>  end type monod_type
    
    interface
     
        subroutine compute_rk_Monod(this,conc,rk)
            import redox_kin_c
            implicit none
            class(redox_kin_c), intent(in) :: this
            real(kind=8), intent(in) :: conc(:)
            real(kind=8), intent(out) :: rk
        end subroutine

        
        subroutine compute_drk_dc_Monod(this,conc,rk,drk_dc)
            import redox_kin_c
            implicit none
            class(redox_kin_c), intent(in) :: this
            real(kind=8), intent(in) :: conc(:)
            real(kind=8), intent(in) :: rk
            real(kind=8), intent(out) :: drk_dc(:)
        end subroutine
        
      
    end interface
    
    contains

        
        subroutine set_Monod_params(this,Monod_params)
            implicit none
            class(redox_kin_c) :: this
            class(Monod_params_c), intent(in) :: Monod_params
            this%params=Monod_params
        end subroutine
        
        !subroutine set_indices_aq_phase_Monod(this,aq_phase)
        !!> First inhibitors, then TEAs, and finally DOC
        !    implicit none
        !    class(redox_kin_c) :: this
        !    class(aq_phase_c), intent(in) :: aq_phase
        !    
        !    integer(kind=4) :: i,aq_species_ind
        !    type(aq_species_c) :: DOC
        !    logical :: flag
        !    
        !    call DOC%set_name('ch2o(aq)')
        !    do i=1,this%params%n_inh
        !    !> chapuza (puede haber especies no acuosas)
        !        call aq_phase%is_species_in_aq_phase(this%params%inhibitors(i),flag,aq_species_ind)
        !        if (flag==.true.) then
        !            this%indices_aq_phase(i)=aq_species_ind
        !        end if
        !    end do
        !    do i=1,this%params%n_M
        !    !> chapuza (puede haber especies no acuosas)
        !        call aq_phase%is_species_in_aq_phase(this%params%TEAs(i),flag,aq_species_ind)
        !        if (flag==.true.) then
        !            this%indices_aq_phase(this%params%n_inh+i)=aq_species_ind
        !        end if
        !    end do
        !    call aq_phase%is_species_in_aq_phase(DOC,flag,aq_species_ind)
        !    if (flag==.true.) then
        !        this%indices_aq_phase(this%params%num_terms+1)=aq_species_ind
        !    end if
        !end subroutine
        
        !function get_Monod(this) result(Monod)
        !>    implicit none
        !>    class(kin_reaction_c), intent(in) :: this
        !>    type(redox_kin_c) :: Monod
        !>    select type (kin_params=>this%kin_params)
        !>    type is (redox_kin_c)
        !>        Monod=kin_params
        !>    end select
        !end function
        

end module