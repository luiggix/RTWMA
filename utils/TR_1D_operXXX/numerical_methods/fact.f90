!> Coimputes factorial of integer
recursive function fact(n)
    implicit none
    integer(kind=4), intent(in) :: n
    integer(kind=4) :: fact
    
    if (n<0) error stop "Factorial must be non-negative"
    if (n==0) then
        fact=1
    else 
        fact=n*fact(n-1)
    end if
end function