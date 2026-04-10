!> Kinetic parameters module
module kin_params_m
    use species_m
    implicit none
    save
    type, public, abstract :: kin_params_c !> kinetic parameters superclass
        real(kind=8) :: rate_cst !> reaction rate constant
    contains
        !procedure(get_conc_kin), public, deferred :: get_conc_kin !> gets concentrations needed to compute rk
        !procedure(compute_rk), public, deferred :: compute_rk !> computes rk
        !procedure(read_kin_params), public, deferred :: read_kin_params
        !procedure(deallocate_kin_params), public, deferred :: deallocate_kin_params
    end type
    
    abstract interface
    
        !subroutine get_conc_kin(this,species,conc,i,conc_kin,kin_ind)
        !>    import kin_params_c
        !>    import species_c
        !>    implicit none
        !>    class(kin_params_c), intent(in) :: this
        !>    class(species_c), intent(in) :: species(:)
        !>    real(kind=8), intent(in) :: conc(:) !> species concentrations
        !>    integer(kind=4), intent(in) :: i !> local reaction index
        !>    real(kind=8), intent(out) :: conc_kin(:) !> concentration of species relevant to kinetic reaction rates
        !>    integer(kind=4), intent(out), optional :: kin_ind(:)
        !end subroutine
        !
        !subroutine compute_rk(this,conc,i,rk,saturation)
        !>    import kin_params_c
        !>    implicit none
        !>    class(kin_params_c), intent(in) :: this
        !>    real(kind=8), intent(in) :: conc(:) !> depends on type of kinetic parameters
        !>    integer(kind=4), intent(in) :: i !> local reaction index
        !>    real(kind=8), intent(out) :: rk !> reaction rate
        !>    real(kind=8), intent(in), optional :: saturation
        !end subroutine
        
        !subroutine read_kin_params(this,react_name,filename)
        !>    import kin_params_c
        !>    implicit none
        !>    class(kin_params_c) :: this
        !>    character(len=*), intent(in) :: react_name
        !>    !integer(kind=4), intent(in) :: n_paths
        !>    character(len=*), intent(in) :: filename
        !end subroutine
        
        !subroutine deallocate_kin_params(this)
        !>    import kin_params_c
        !>    implicit none
        !>    class(kin_params_c) :: this
        !end subroutine
    end interface
end module