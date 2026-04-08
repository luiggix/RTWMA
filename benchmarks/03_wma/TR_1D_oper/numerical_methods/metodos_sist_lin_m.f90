!> This module contains methods to solve linear systems of equations
module metodos_sist_lin_m
    use matrices_m
    implicit none
    save
    interface
        subroutine Gauss_Jordan(A,b,tol,x,error)
            implicit none
            real(kind=8), intent(in) :: A(:,:) 
            real(kind=8), intent(in) :: b(:)
            real(kind=8), intent(in) :: tol !> tolerance for solution
            real(kind=8), intent(out) :: x(:)
            integer(kind=4), intent(out), optional :: error
        end subroutine
        
        subroutine forward_substitution(L,b,x)
            implicit none
            real(kind=8), intent(in) :: L(:,:)
            real(kind=8), intent(in) :: b(:)
            real(kind=8), intent(out) :: x(:)
        end subroutine
        
        subroutine backward_substitution(U,b,x)
            implicit none
            real(kind=8), intent(in) :: U(:,:)
            real(kind=8), intent(in) :: b(:)
            real(kind=8), intent(out) :: x(:)
        end subroutine
        
        subroutine LU_lin_syst(A,b,tol,x)
            implicit none
            real(kind=8), intent(in) :: A(:,:)
            real(kind=8), intent(in) :: b(:)
            real(kind=8), intent(in) :: tol
            real(kind=8), intent(out) :: x(:)
        end subroutine
        
        subroutine Thomas(A,b,tol,x)
            import tridiag_matrix_c
            implicit none
            class(tridiag_matrix_c), intent(in) :: A
            real(kind=8), intent(in) :: b(:)
            real(kind=8), intent(in) :: tol
            real(kind=8), intent(out) :: x(:)
        end subroutine
        
        subroutine Thomas_Toeplitz(a,b,c,d,x)
            implicit none
            real(kind=8), intent(in) :: a,b,c
            real(kind=8), intent(in) :: d(:)
            real(kind=8), intent(out) :: x(:)
        end subroutine
        
        subroutine Jacobi(A,b,x0,x,niter)
            implicit none
            real(kind=8), intent(in) :: A(:,:)
            real(kind=8), intent(in) :: b(:)
            real(kind=8), intent(inout) :: x0(:)
            real(kind=8), intent(out) :: x(:)
            integer(kind=4), intent(out) :: niter !> number of iterations
        end subroutine
        
        subroutine Gauss_seidel(A,b,x0,x,niter)
            implicit none
            real(kind=8), intent(in) :: A(:,:)
            real(kind=8), intent(in) :: b(:)
            real(kind=8), intent(inout) :: x0(:)
            real(kind=8), intent(out) :: x(:)
            integer(kind=4), intent(out) :: niter !> number of iterations
        end subroutine
    end interface
end module 