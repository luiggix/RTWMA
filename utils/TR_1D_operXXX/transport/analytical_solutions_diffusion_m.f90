module analytical_solutions_diffusion_m
    use diffusion_m
    use diffusion_transient_m
    implicit none
    save
    contains
        function fund_sol_diff_eqn_1D(x,t,phi,D) result(conc)
        !> Fundamental solution of diffusion equation in 1D
            implicit none
            real(kind=8), intent(in) :: x
            real(kind=8), intent(in) :: t
            real(kind=8), intent(in) :: phi
            real(kind=8), intent(in) :: D
            real(kind=8) :: conc
            
            real(kind=8), parameter :: pi=4d0*atan(1d0)
            integer(kind=4) :: n
            real(kind=8) :: mu
            real(kind=8), allocatable :: mesh_size(:)
            
            if (mod(n,2)==0) then
                mu=(n-1)/2d0
            else
                mu=floor(n/2d0)
            end if
            
            if (t==0d0) then
                if (x==0d0) then
                    conc=1d0
                else
                    conc=0d0
                end if
            else
                conc=(1d0/sqrt(4d0*pi*D*t))*exp(-25d-2*x**2)/(D*t)
            end if
        end function
        
       
        
end module