module polynomials_m
    implicit none
    save
    interface
        function Legendre_poly(n,x) result(p)
            implicit none
            integer(kind=4), intent(in) :: n !> degree
            real(kind=8), intent(in) :: x(:) !> input
            real(kind=8), allocatable :: p(:) !> output
        end function
        
    end interface
    
    contains
        function real_poly_1D(coeffs,x) result(p)
            implicit none
            real(kind=8), intent(in) :: coeffs(:) !> orden decreciente
            real(kind=8), intent(in) :: x !> input
            real(kind=8) :: p !> output
            
            integer(kind=4) :: i,deg
            
            deg=size(coeffs)-1
            p=coeffs(deg+1)
            do i=1,deg
                p=p+coeffs(i)*x**(deg-i+1)
            end do
            
        end function
        
        function der_real_poly_1D(coeffs,x) result(dp_dx)
            implicit none
            real(kind=8), intent(in) :: coeffs(:) !> orden decreciente
            real(kind=8), intent(in) :: x !> input
            real(kind=8) :: dp_dx !> output
            
            integer(kind=4) :: i,deg
            
            deg=size(coeffs)-1
            dp_dx=coeffs(deg)
            do i=1,deg-1
                dp_dx=dp_dx+(deg-i+1)*coeffs(i)*x**(deg-i)
            end do
        end function
        
        function sec_der_real_poly_1D(coeffs,x) result(d2p_dx2)
            implicit none
            real(kind=8), intent(in) :: coeffs(:) !> orden decreciente
            real(kind=8), intent(in) :: x !> input
            real(kind=8) :: d2p_dx2 !> output
            
            integer(kind=4) :: i,deg
            
            deg=size(coeffs)-1
            d2p_dx2=coeffs(deg-1)
            do i=1,deg-2
                d2p_dx2=d2p_dx2+(deg-i+1)*(deg-i)*coeffs(i)*x**(deg-i-1)
            end do
        end function
                
end module