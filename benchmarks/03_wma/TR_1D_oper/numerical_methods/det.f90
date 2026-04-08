!> Determinant of square matrix using LU decomposition
function det(A)
    use matrices_m, only : LU
    implicit none
    real(kind=8), intent(in) :: A(:,:) !> square matrix
    real(kind=8) :: det
    
    real(kind=8) :: det_L, det_U
    real(kind=8), allocatable :: L(:,:), U(:,:)
    integer(kind=4) :: n,i
    
    n=size(A,1)
    if (n/=size(A,2)) then
        error stop "Matrix must be square (det)"
    end if
    if (n==2) then
        det=A(1,1)*A(2,2)-A(2,1)*A(1,2)
    !else if (n==3) then
    !>    det=A(1,1)*A(2,2)*A(3,3)+A(2,1)*A(3,2)*A(1,3)+A(1,2)*A(2,3)*A(3,1)-(A(3,1)*A(2,2)*A(1,3)+A(2,1)*A(1,2)*A(3,3)+A(3,2)*A(2,3)*A(1,1))
    else
        allocate(L(n,n),U(n,n))
        call LU(A,L,U)
        det_L=1d0
        det_U=1d0
        do i=1,n
            det_L=det_L*L(i,i)
            det_U=det_U*U(i,i)
        end do
        det=det_L*det_U
    end if
end function