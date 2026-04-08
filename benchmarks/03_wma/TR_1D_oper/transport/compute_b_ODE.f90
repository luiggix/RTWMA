!> dy/dt=-Ay+b
!> b=rc^*inv(F): tridiagonal, symmetric, positive semi-definite
function compute_b_ODE(this) result(b)
    use PDE_transient_m
    implicit none
    class(PDE_1D_transient_c), intent(in) :: this
    real(kind=8), allocatable :: b(:)
    b=this%source_term_PDE/sqrt(this%F_mat%diag)
end function