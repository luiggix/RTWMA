!> Solves linear system Ax=b using Gauss-seidel iterative method
subroutine Gauss_seidel(A,b,x0,x,niter)
    use vectors_m
    use matrices_m
    implicit none
    real(kind=8), intent(in) :: A(:,:)
    real(kind=8), intent(in) :: b(:)
    real(kind=8), intent(inout) :: x0(:)
    real(kind=8), intent(out) :: x(:)
    integer(kind=4), intent(out) :: niter !> number of iterations
    
    real(kind=8), allocatable :: L(:,:), R(:,:), D(:), C_mat(:,:), inv(:,:), c(:)
    real(kind=8) :: sum
    integer(kind=4) :: i,j,k,n
    real(kind=8), parameter :: tol=1d-9
    n=size(A,1)
    allocate(L(n,n),R(n,n),D(n),C_mat(n,n),inv(n,n),c(n))
    forall (i=1:n)
        D(i)=A(i,i)
    end forall
    L=0d0
    R=0d0
    do j=1,n
        L((j+1):n,j)=-A((j+1):n,j)
        R(j,(j+1):n)=-A(j,(j+1):n)
    end do
    forall (i=1:n)
        L(i,i)=L(i,i)-D(i)
    end forall
    call inv_matrix(-L,tol,inv)
    C_mat=matmul(inv,R)
    c=matmul(inv,b)
    do k=1,n
        niter=niter+1 !> we update number of iterations
        do i=1,n
            sum=0
            do j=1,n
                if (j<i .and. i>1) then
                    sum=sum+A(i,j)*x(j)
                else if (j>i) then
                    sum=sum+A(i,j)*x0(j)
                else
                    continue
                end if
            end do
            x(i)=(1d0/D(i))*(b(i)-sum)
        end do
        if (inf_norm_vec_real(x-x0)<tol) exit
        x0=x
    end do
end subroutine Gauss_seidel