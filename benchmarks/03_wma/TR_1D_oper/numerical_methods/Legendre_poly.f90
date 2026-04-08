!> Computes Legendre polynomial
function Legendre_poly(n,x) result(p)
    use polynomials_m
    implicit none
    integer(kind=4), intent(in) :: n !> degree
    real(kind=8), intent(in) :: x(:) !> input
    real(kind=8), allocatable :: p(:) !> output
    
    allocate(p(size(x)))
    if (n==0) then
        p=1d0
    else if (n==1) then
        p=x*sqrt(3d0)/sqrt(2d0)
    else if (n==2) then
        p=0.5*(3*(x**2)-1)*sqrt(5d0)/sqrt(2d0)
    else if (n==3) then
        p=0.5*(5*(x**3)-3*x)*sqrt(7d0)/sqrt(2d0)
    else
        error stop "Legendre polynomial not implemented yet"
    end if
end function