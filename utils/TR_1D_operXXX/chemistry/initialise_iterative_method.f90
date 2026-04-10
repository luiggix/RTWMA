!> This subroutine initialises concentrations for any iterative method
!> It takes a linear combination of the concentrations in the two previous time steps
subroutine initialise_iterative_method(conc_old_old,conc_old,param,initial_guess)
    implicit none
    real(kind=8), intent(in) :: conc_old_old(:) !> concentrations at time step k-1
    real(kind=8), intent(in) :: conc_old(:) !> concentrations at time step k
    real(kind=8), intent(in) :: param !> parameter for linear combination
    real(kind=8), intent(out) :: initial_guess(:) !> initial guess concentrations at time step k+1
    
    if (param<0d0 .or. param>1d0) error stop "Initialisation parameter in iterative method must be in [0,1]"
    initial_guess=(1d0+param)*conc_old-param*conc_old_old
end subroutine