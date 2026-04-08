subroutine LU_lin_syst(A,b,tol,x) !> Ax=b
    use vectors_m
    use matrices_m, only : LU
    use metodos_sist_lin_m, only : forward_substitution,backward_substitution
    implicit none
    real(kind=8), intent(in) :: A(:,:) !> square matrix (A=LU)
    real(kind=8), intent(in) :: b(:) !> vector
    real(kind=8), intent(in) :: tol !> tolerance for solution
    real(kind=8), intent(out) :: x(:) !> solution of linear system
    
    
    real(kind=8), allocatable :: L(:,:), U(:,:), y(:)
    integer(kind=4) :: n
    
    n=size(b)
    allocate(L(n,n),U(n,n),y(n))
    call LU(A,L,U)
    call forward_substitution(L,b,y)
    call backward_substitution(U,y,x)
    
    if (inf_norm_vec_real(matmul(A,x)-b)>=tol) then
        print *, "Wrong solution in LU_lin_syst"
        print *, "Residual: ", inf_norm_vec_real(matmul(A,x)-b)
        error stop
    end if
end subroutine