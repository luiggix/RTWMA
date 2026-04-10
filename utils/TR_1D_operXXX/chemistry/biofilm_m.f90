!> Biofilm module (to be developed)
module biofilm_m
    use imm_zone_m
    implicit none
    save
    type, public, extends(imm_zone_c) :: biofilm_c
    end type
    
!> PFLOTRAN:
    
    !type, public :: immobile_species_type
    !>    PetscInt :: id
    !>    character(len=MAXWORDLENGTH) :: name
    !>    PetscReal :: molar_weight
    !>    PetscBool :: print_me
    !>    type(immobile_species_type), pointer :: next
    !end type immobile_species_type
    
    !type, public :: microbial_rxn_type
    !>    PetscInt :: id
    !>    PetscInt :: itype
    !>    character(len=MAXSTRINGLENGTH) :: reaction
    !>    PetscReal :: rate_constant
    !>    PetscReal :: activation_energy
    !>    PetscBool :: print_me
    !>    type(reaction_equation_type), pointer :: reaction_equation
    !>    type(monod_type), pointer :: monod
    !>    type(inhibition_type), pointer :: inhibition
    !>    type(microbial_biomass_type), pointer :: biomass
    !>    type(microbial_rxn_type), pointer :: next
    !>  end type microbial_rxn_type

    !type, public :: monod_type
    !>    PetscInt :: id
    !>    character(len=MAXWORDLENGTH) :: species_name
    !>    PetscReal :: half_saturation_constant
    !>    PetscReal :: threshold_concentration
    !>    type(monod_type), pointer :: next
    !>  end type monod_type
    
    !type, public :: microbial_biomass_type
    !>    PetscInt :: id
    !>    character(len=MAXWORDLENGTH) :: species_name
    !>    PetscReal :: yield
    !end type microbial_biomass_type
end module