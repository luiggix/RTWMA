!> Gas phase subclass: contains gases in gas phase
module gas_phase_m
    use phase_m
    use gas_m
    implicit none
    save
    type, public, extends(phase_c) :: gas_phase_c
        type(gas_c), allocatable :: gases(:) !> gases
    contains
    !> Allocate
        procedure, public :: allocate_gases
    !> Is
        procedure, public :: is_gas_in_gas_phase
    !> Compute
        procedure, public :: compute_log_act_coeffs_gas_phase
        procedure, public :: compute_log_Jacobian_act_coeffs_gas_phase
    end type
    
!>PFLOTRAN:
    
  !type, public :: gas_species_type
  !>  PetscInt :: id
  !>  character(len=MAXWORDLENGTH) :: name
  !>  PetscReal :: itype
  !>  PetscReal :: molar_volume
  !>  PetscReal :: molar_weight
  !>  PetscBool :: print_me
  !>  type(database_rxn_type), pointer :: dbaserxn
  !>  type(gas_species_type), pointer :: next
  !end type gas_species_type
  !
  !type, public :: gas_type
  !
  !>  PetscInt :: ngas
  !>  PetscInt :: nactive_gas
  !>  PetscInt :: npassive_gas
  !
  !>  type(gas_species_type), pointer :: list
  !
  !>  !> gas species names
  !>  character(len=MAXWORDLENGTH), pointer :: active_names(:)
  !>  character(len=MAXWORDLENGTH), pointer :: passive_names(:)
  !>  PetscBool :: print_all
  !>  PetscBool :: print_concentration
  !>  PetscBool :: print_partial_pressure
  !>  PetscBool, pointer :: active_print_me(:)
  !>  PetscBool, pointer :: passive_print_me(:)
  !
  !>  PetscInt, pointer :: acteqspecid(:,:)   !> (0:ncomp in rxn)
  !>  PetscReal, pointer :: acteqstoich(:,:)
  !>  PetscInt, pointer :: acteqh2oid(:)       !> id of water, if present
  !>  PetscReal, pointer :: acteqh2ostoich(:)  !> stoichiometry of water, if present
  !>  PetscReal, pointer :: acteqlogK(:)
  !>  PetscReal, pointer :: acteqlogKcoef(:,:)
  !
  !>  PetscReal, pointer :: actmolarwt(:)
  !>  PetscReal, pointer :: pasmolarwt(:)
  !
  !>  PetscInt, pointer :: paseqspecid(:,:)   !> (0:ncomp in rxn)
  !>  PetscReal, pointer :: paseqstoich(:,:)
  !>  PetscInt, pointer :: paseqh2oid(:)       !> id of water, if present
  !>  PetscReal, pointer :: paseqh2ostoich(:)  !> stoichiometry of water, if present
  !>  PetscReal, pointer :: paseqlogK(:)
  !>  PetscReal, pointer :: paseqlogKcoef(:,:)
  !
  !end type gas_type
    
    interface
        subroutine compute_log_act_coeffs_gas_phase(this,ionic_act,log_act_coeffs)
            import gas_phase_c
            implicit none
            class(gas_phase_c) :: this
            real(kind=8), intent(in) :: ionic_act
            real(kind=8), intent(out) :: log_act_coeffs(:) !> must be allocated
        end subroutine
        
        subroutine compute_log_Jacobian_act_coeffs_gas_phase(this,ionic_act,log_act_coeffs,conc,log_Jacobian_act_coeffs)
            import gas_phase_c
            implicit none
            class(gas_phase_c) :: this
            real(kind=8), intent(in) :: ionic_act
            real(kind=8), intent(in) :: log_act_coeffs(:)
            real(kind=8), intent(in) :: conc(:) !> concentration of gas species in a given target
            real(kind=8), intent(out) :: log_Jacobian_act_coeffs(:,:) !> must be allocated
        end subroutine
    end interface
    
    contains

    
        subroutine allocate_gases(this,num_species)
            implicit none
            class(gas_phase_c) :: this
            integer(kind=4), intent(in), optional :: num_species
            if (present(num_species)) then
                this%num_species=num_species
            end if
            if (allocated(this%gases)) then
                deallocate(this%gases)
            end if
            allocate(this%gases(this%num_species))
        end subroutine
        
        subroutine is_gas_in_gas_phase(this,gas,flag,gas_ind) !> checks if gas belongs to gas phase
            implicit none
            class(gas_phase_c), intent(in) :: this
            class(gas_c), intent(in) :: gas !> gas
            logical, intent(out) :: flag !> TRUE if it belongs to gas phase, FALSE otherwise
            integer(kind=4), intent(out), optional :: gas_ind !> index gas in gas phase
            
            integer(kind=4) :: i
            
            flag=.false.
            if (present(gas_ind)) then
                gas_ind=0
            end if
            do i=1,this%num_species
                if (gas%name==this%gases(i)%name) then
                    flag=.true.
                    if (present(gas_ind)) then
                        gas_ind=i
                    end if
                    exit
                end if
            end do
        end subroutine
end module