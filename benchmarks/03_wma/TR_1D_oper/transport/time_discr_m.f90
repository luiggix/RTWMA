!> Time discretisation class
module time_discr_m
    implicit none
    save
    type, public, abstract :: time_discr_c
        real(kind=8) :: Final_time
        integer(kind=4) :: Num_time                         !> Number of time steps
        integer(kind=4) :: int_method                       !> Time integration method:
                                                                !> 1: Euler explicit
                                                                !> 2: Euler fully implicit
                                                                !> 3: Crank-Nicolson
                                                                !> 4: RKF45
    contains
        procedure, public :: set_Final_time
        procedure, public :: set_Num_time
        procedure, public :: set_int_method
        procedure, public :: compute_Final_time
        procedure, public :: compute_Num_time
        procedure(read_time_discr), public, deferred :: read_time_discr
        procedure(get_Delta_t), public, deferred :: get_Delta_t
    end type
!****************************************************************************************************************************************************
    abstract interface 
        subroutine read_time_discr(this,filename)
            import time_discr_c
            implicit none
            class(time_discr_c) :: this
            character(len=*), intent(in) :: filename
        end subroutine
        
        function get_Delta_t(this,k) result(Delta_t)
            import time_discr_c
            implicit none
            class(time_discr_c) :: this
            integer(kind=4), intent(in), optional :: k
            real(kind=8) :: Delta_t
        end function
    end interface
!****************************************************************************************************************************************************
    type, public, extends(time_discr_c) :: time_discr_homog_c
        real(kind=8) :: Delta_t
    contains
        procedure, public :: read_time_discr=>read_time_discr_homog
        procedure, public :: set_Delta_t_homog
        procedure, public :: get_Delta_t=>get_Delta_t_homog
    end type
    
    type, public, extends(time_discr_c) :: time_discr_heterog_c
        real(kind=8), allocatable :: Delta_t(:)
    contains
        procedure, public :: read_time_discr=>read_time_discr_heterog
        procedure, public :: set_Delta_t_heterog
        procedure, public :: get_Delta_t=>get_Delta_t_heterog
    end type
!****************************************************************************************************************************************************
    contains
        subroutine set_Final_time(this,Final_time)
            implicit none
            class(time_discr_c) :: this
            real(kind=8), intent(in) :: Final_time
            this%Final_time=Final_time
        end subroutine
        
        subroutine set_Num_time(this,Num_time)
            implicit none
            class(time_discr_c) :: this
            integer(kind=4) :: Num_time
            this%Num_time=Num_time
        end subroutine
        
        subroutine set_int_method(this,int_method)
            implicit none
            class(time_discr_c) :: this
            integer(kind=4), intent(in) :: int_method
            if (int_method>4 .and. int_method<1) error stop "Method not implemented yet"
            this%int_method=int_method
        end subroutine
        
        subroutine compute_Final_time(this)
            implicit none
            class(time_discr_c) :: this
            select type (this)
            type is (time_discr_homog_c)
                this%Final_time=this%Num_time*this%Delta_t
            type is (time_discr_heterog_c)
                this%Final_time=sum(this%Delta_t)
            end select
        end subroutine
        
        subroutine compute_Num_time(this)
            implicit none
            class(time_discr_c) :: this
            select type (this)
            type is (time_discr_homog_c)
                this%Num_time=nint(this%Final_time/this%Delta_t)
            type is (time_discr_heterog_c)
                this%Num_time=size(this%Delta_t)
            end select
        end subroutine
        
        
        
        subroutine set_Delta_t_homog(this,Delta_t)
            implicit none
            class(time_discr_homog_c) :: this
            real(kind=8), intent(in) :: Delta_t
            this%Delta_t=Delta_t
        end subroutine
        
        subroutine set_Delta_t_heterog(this,Delta_t)
            implicit none
            class(time_discr_heterog_c) :: this
            real(kind=8), intent(in) :: Delta_t(:)
            this%Delta_t=Delta_t
        end subroutine
        
        subroutine read_time_discr_homog(this,filename)
            implicit none
            class(time_discr_homog_c) :: this
            character(len=*), intent(in) :: filename
            !> We assume filename contains integration method, time step and final time
            open(unit=1,file=filename,status='old',action='read')
            read(1,*) this%int_method
            if (this%int_method>4 .and. this%int_method<1) then
                error stop "Method not implemented yet"
            end if
            read(1,*) this%Delta_t
            if (this%Delta_t<0d0) then
                error stop
            end if
            !read(1,*) this%Final_time
            read(1,*) this%Num_time
            if (this%Num_time<0) then
                error stop
            end if
            close(1)
            !call this%compute_Num_time()
            call this%compute_Final_time()
        end subroutine
        
        subroutine read_time_discr_heterog(this,filename)
            implicit none
            class(time_discr_heterog_c) :: this
            character(len=*), intent(in) :: filename
            
            real(kind=8), allocatable :: t_vec(:)
            integer(kind=4), allocatable :: n_vec(:)
            integer(kind=4) :: i,j,k,size_t_vec
            !> We assume filename contains a vector of time values and a vector of time steps
            open(unit=2,file=filename,status='old',action='read')
            read(2,*) size_t_vec
            allocate(t_vec(size_t_vec))
            read(2,*) t_vec
            allocate(n_vec(size_t_vec-1))
            read(2,*) n_vec
            allocate(this%Delta_t(sum(n_vec)))
            i=1
            k=1
            do
                this%Delta_t(i)=(t_vec(k+1)-t_vec(k))/n_vec(k)
                if (n_vec(k)>1) then
                    do j=1,n_vec(k)-1
                        this%Delta_t(i+j)=this%Delta_t(i)
                    end do
                end if
                i=i+n_vec(k)
                k=k+1
                if (k==size_t_vec) exit
            end do
            close(2)
            this%Final_time=sum(this%Delta_t)
            this%Num_time=size(this%Delta_t)
        end subroutine 
        
        function get_Delta_t_homog(this,k) result(Delta_t)
            implicit none
            class(time_discr_homog_c) :: this
            integer(kind=4), intent(in), optional :: k
            real(kind=8) :: Delta_t
            Delta_t=this%Delta_t
        end function
        
        function get_Delta_t_heterog(this,k) result(Delta_t)
            implicit none
            class(time_discr_heterog_c) :: this
            integer(kind=4), intent(in), optional :: k
            real(kind=8) :: Delta_t
            if (present(k)) then
                Delta_t=this%Delta_t(k)
            else
                Delta_t=this%Delta_t(1)
            end if
        end function
end module