subroutine compute_B_mat(this,theta,E_mat)
!> B=(Id+(1-theta)*E) (tridiagonal, negative semi-definite)
    use PDE_transient_m
    implicit none
    
    class(PDE_1D_transient_c) :: this
    real(kind=8), intent(in) :: theta
    class(tridiag_matrix_c), intent(in) :: E_mat
    
    integer(kind=4) :: n
    real(kind=8) :: B_norm_inf,B_norm_1
        
    !call this%B_mat%allocate_matrix(n)
    this%B_mat%sub=(1d0-theta)*E_mat%sub
    this%B_mat%diag=1d0+(1d0-theta)*E_mat%diag
    this%B_mat%super=(1d0-theta)*E_mat%super
end subroutine 