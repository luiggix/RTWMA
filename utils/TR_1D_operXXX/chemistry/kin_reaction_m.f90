!> Kinetic reaction module
module kin_reaction_m
    use reaction_m
    use kin_params_m
    implicit none
    save
    type, public, abstract, extends(reaction_c) :: kin_reaction_c !> Kinetic reaction subclass
        integer(kind=4), allocatable :: indices_aq_phase(:) !> indices of species in aqueous phase class that control kinetic reaction rates
    contains
    end type
    
    type, public :: kin_reaction_ptr_c !< clase ad hoc para crear vector punteros clase reaccion cinetica (cosas de Fortran)
        class(kin_reaction_c), pointer :: kin_reaction
    end type
    
    abstract interface
        !subroutine read_kin_reaction(this)
        !    import kin_reaction_c
        !    implicit none
        !    class(kin_reaction_c) :: this
        !end subroutine
        !
        !subroutine compute_drk_dc(this,conc,rk,drk_dc)
        !    import kin_reaction_c
        !    implicit none
        !    class(kin_reaction_c), intent(in) :: this
        !    real(kind=8), intent(in) :: conc(:)
        !    real(kind=8), intent(in) :: rk
        !    real(kind=8), intent(out) :: drk_dc(:) !> tiene que estar alocatado
        !end subroutine
        !    
        !
        !subroutine get_conc_kin(this,species,conc,conc_kin,kin_ind)
        !    import kin_reaction_c
        !    import species_c
        !    implicit none
        !    class(kin_reaction_c), intent(in) :: this
        !    class(species_c), intent(in) :: species(:)
        !    real(kind=8), intent(in) :: conc(:) !> species concentrations
        !    real(kind=8), intent(out) :: conc_kin(:) !> concentration of species relevant to compute kinetic reaction rate
        !    integer(kind=4), intent(out), optional :: kin_ind(:)
        !end subroutine
        !
        !subroutine compute_rk(this,conc,rk)
        !    import kin_reaction_c
        !    implicit none
        !    class(kin_reaction_c), intent(in) :: this
        !    real(kind=8), intent(in) :: conc(:) !> depends on type of kinetic parameters
        !    real(kind=8), intent(out) :: rk !> reaction rate
        !end subroutine
        
    end interface
    
    interface 
        
        
        
    end interface
    
    contains
        
    
    
end module