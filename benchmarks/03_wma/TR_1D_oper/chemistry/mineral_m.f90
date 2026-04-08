!> Mineral phase subclass
module mineral_m
    use phase_m
    use solid_m
    implicit none
    save
    type, public, extends(phase_c) :: mineral_c
        type(solid_c) :: mineral !> mineral that defines mineral phase
    contains
    end type
    
!> PFLOTRAN:
    
      !type, public :: mineral_rxn_type
      !>  PetscInt :: id
      !>  PetscInt :: itype
      !>  character(len=MAXWORDLENGTH) :: name
      !>  PetscReal :: molar_volume
      !>  PetscReal :: molar_weight
      !>  PetscBool :: print_me
      !>  type(database_rxn_type), pointer :: dbaserxn
      !>  type(transition_state_rxn_type), pointer :: tstrxn
      !>  type(mineral_rxn_type), pointer :: next
      !end type mineral_rxn_type
  !>  

!*************************************************************************************************!>    
    interface
      
    end interface
    
    
    contains
        !subroutine set_mol_vol(this,mol_vol)
        !>    implicit none
        !>    class(mineral_c) :: this
        !>    real(kind=8), intent(in) :: mol_vol
        !>    if (mol_vol<0d0) error stop "Molar volume cannot be negative"
        !>    this%mol_vol=mol_vol
        !end subroutine
        
       
end module