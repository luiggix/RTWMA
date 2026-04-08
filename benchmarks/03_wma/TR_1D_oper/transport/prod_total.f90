function prod_total(lambda,P,y0,b,time_discr_obj,k) result(y)
    use time_discr_m
    use matrices_m, only : inv_matrix
    implicit none
    real(kind=8), intent(in) :: lambda(:)
    real(kind=8), intent(in) :: P(:,:)
    real(kind=8), intent(in) :: y0(:)
    real(kind=8), intent(in) :: b(:)
    class(time_discr_c), intent(in) :: time_discr_obj
    integer(kind=4), intent(in), optional :: k
    real(kind=8), allocatable :: y(:)
    
    real(kind=8), allocatable :: Q_lambda(:,:),x(:),Pt_x(:),inv_P(:,:)
    integer(kind=4) :: i,n,time_step
    real(kind=8), parameter :: tol=1d-9
    
    n=size(lambda)
    allocate(Q_lambda(2,n),x(2*n),Pt_x(2*n),inv_P(n,n))
    Q_lambda(1,:)=1d0
    if (present(k)) then
        time_step=k
    else
        time_step=time_discr_obj%Num_time
    end if
    select type (time=>time_discr_obj)
    type is (time_discr_homog_c)
        do i=1,n
            if (abs(lambda(i))<tol) then
                Q_lambda(2,i)=0d0
            else
                Q_lambda(2,i)=(1d0-exp(-lambda(i)*time_step*time%Delta_t))/lambda(i)
            end if
        end do
        Q_lambda(1,:)=exp(-lambda*time_step*time%Delta_t)
    type is (time_discr_heterog_c)
        do i=1,n
            if (abs(lambda(i))<tol) then
                Q_lambda(2,i)=0d0
            else
                Q_lambda(2,i)=(1d0-exp(-lambda(i)*sum(time%Delta_t(1:time_step))))/lambda(i)
            end if
        end do
        Q_lambda(1,:)=exp(-lambda*sum(time%Delta_t(1:time_step)))
    end select
    x(1:n)=y0
    x(n+1:2*n)=b
    call inv_matrix(P,tol,inv_P)
    Pt_x(1:n)=matmul(inv_P,x(1:n))
    Pt_x(n+1:2*n)=matmul(inv_P,x(n+1:2*n))
    y=matmul(P,Q_lambda(1,:)*Pt_x(1:n))+matmul(P,Q_lambda(2,:)*Pt_x(n+1:2*n))
end function