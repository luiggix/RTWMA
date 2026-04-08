module kin_mineral_params_m
    use kin_params_m
    use aq_phase_m
    implicit none
    save
    type, public, extends(kin_params_c) :: kin_mineral_params_c !> mineral kinetic parameters subclass
        real(kind=8) :: act_energy !> activation energy (J)
        integer(kind=4) :: num_par_reacts !> number of parallel reactions
        real(kind=8), allocatable :: k(:) !> reaction constants
        integer(kind=4) :: num_cat !> number of catalysers
        integer(kind=4), allocatable :: cat_indices(:) !> indices of catalysers in aqueous phase (dim=num_cat)
        !class(species_c), allocatable :: catalysers(:) !> esto NO va aqui
        real(kind=8), allocatable :: p(:,:) !> experimental constants
        real(kind=8), allocatable :: theta(:) !> experimental constants (usually =1)
        real(kind=8), allocatable :: eta(:) !> experimental constants (usually =1)
        real(kind=8) :: supersat_threshold
    contains
        procedure, public :: allocate_constants
        procedure, public :: allocate_cat_indices
        !procedure, public :: set_cat_indices
        !procedure, public :: allocate_catalysers
    end type

!> PFLOTRAN:
  !type, public :: transition_state_rxn_type
  !>  PetscReal :: min_scale_factor
  !>  PetscReal :: affinity_factor_sigma
  !>  PetscReal :: affinity_factor_beta
  !>  PetscReal :: affinity_threshold
  !>  PetscReal :: rate_limiter
  !>  PetscReal :: surf_area_vol_frac_pwr
  !>  PetscReal :: surf_area_porosity_pwr
  !>  PetscInt :: irreversible
  !>  PetscReal :: rate
  !>  PetscReal :: activation_energy
  !>  character(len=MAXWORDLENGTH) :: armor_min_name
  !>  PetscReal :: armor_pwr
  !>  PetscReal :: armor_crit_vol_frac
  !>  PetscReal :: surf_area_epsilon
  !>  PetscReal :: vol_frac_epsilon
  !>  type(transition_state_prefactor_type), pointer :: prefactor
  !>  type(transition_state_rxn_type), pointer :: next
  !end type transition_state_rxn_type
  !
  !type, public :: transition_state_prefactor_type
  !>  type(ts_prefactor_species_type), pointer :: species
  !>  !> these supercede the those above in transition_state_rxn_type
  !>  PetscReal :: rate
  !>  PetscReal :: activation_energy
  !>  type(transition_state_prefactor_type), pointer :: next
  !end type transition_state_prefactor_type
  !
  !type, public :: ts_prefactor_species_type
  !>  character(len=MAXWORDLENGTH) :: name
  !>  PetscInt :: id
  !>  PetscReal :: alpha
  !>  PetscReal :: beta
  !>  PetscReal :: attenuation_coef
  !>  type(ts_prefactor_species_type), pointer :: next
  !end type ts_prefactor_species_type

    interface
       
    end interface
    
    contains
        subroutine allocate_constants(this)
            implicit none
            class(kin_mineral_params_c) :: this
            allocate(this%k(this%num_par_reacts),this%theta(this%num_par_reacts),this%eta(this%num_par_reacts))
            !allocate(this%cat_indices(this%num_cat))
            allocate(this%p(this%num_par_reacts,this%num_cat))
        end subroutine
        
        subroutine allocate_cat_indices(this)
            implicit none
            class(kin_mineral_params_c) :: this
            allocate(this%cat_indices(this%num_cat))
        end subroutine
        
        !subroutine set_cat_indices(this,aq_phase)
        !    implicit none
        !    class(kin_mineral_params_c) :: this
        !    class(aq_phase_c), intent(in) :: aq_phase
        !    
        !    integer(kind=4) :: i,aq_species_ind
        !    type(aq_species_c) :: DOC
        !    logical :: flag
        !    
        !    call this%allocate_cat_indices()
        !    
        !    do i=1,this%num_cat
        !        call aq_phase%is_species_in_aq_phase(this%catalysers(i),flag,aq_species_ind)
        !        if (flag==.true.) then
        !            this%cat_indices(i)=aq_species_ind
        !        end if
        !    end do
        !end subroutine
        
        !subroutine allocate_catalysers(this)
        !    implicit none
        !    class(kin_mineral_params_c) :: this
        !    allocate(this%catalysers(this%num_cat))
        !end subroutine
end module