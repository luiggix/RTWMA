!> Radial spatial discretisation module
module spatial_discr_rad_m
    use spatial_discr_m
    implicit none
    save
    type, public, extends(spatial_discr_c) :: spatial_discr_rad_c
        integer(kind=4) :: dim !> dimension
        real(kind=8) :: radius
        real(kind=8), allocatable :: Delta_r(:)
    contains
        procedure, public :: set_dim
        procedure, public :: set_radius
        procedure, public :: set_Delta_r
        procedure, public :: read_mesh=>read_mesh_rad
        procedure, public :: get_mesh_size=>get_Delta_r
        procedure, public :: get_dim=>get_dim_rad
        procedure, public :: compute_radius
        procedure, public :: compute_Delta_r
        procedure, public :: compute_measure=>compute_measure_rad
        procedure, public :: refine_mesh=>refine_mesh_rad
    end type
    
    interface
        subroutine refine_mesh_rad(this,conc,conc_ext,rel_tol)
            import spatial_discr_rad_c
            implicit none
            class(spatial_discr_rad_c) :: this
            real(kind=8), intent(inout), allocatable :: conc(:,:) !> Num_columns=Num_targets
            real(kind=8), intent(inout), allocatable :: conc_ext(:,:) !> Num_columns=Num_targets
            real(kind=8), intent(in) :: rel_tol !> relative tolerance
        end subroutine
    end interface
    
    contains
        subroutine set_dim(this,dim)
            implicit none
            class(spatial_discr_rad_c) :: this
            integer(kind=4), intent(in) :: dim
            this%dim=dim
        end subroutine
        
        subroutine set_radius(this,radius)
            implicit none
            class(spatial_discr_rad_c) :: this
            real(kind=8), intent(in) :: radius
            this%radius=radius
        end subroutine
        
        subroutine set_Delta_r(this,Delta_r)
            implicit none
            class(spatial_discr_rad_c) :: this
            real(kind=8), intent(in) :: Delta_r(:)
            this%Delta_r=Delta_r
        end subroutine
        
        
        
        subroutine read_mesh_rad(this,filename)
            implicit none
            class(spatial_discr_rad_c) :: this
            character(len=*), intent(in) :: filename
            open(unit=1,file=filename,status='old',action='read')
            read(1,*) this%scheme
            read(1,*) this%targets_flag
            read(1,*) this%Num_targets
            read(1,*) this%Delta_r
            this%Num_targets_defined=.true.
        end subroutine
        
        function get_Delta_r(this,i) result(Delta_r)
            implicit none
            class(spatial_discr_rad_c) :: this
            integer(kind=4), intent(in), optional :: i
            real(kind=8) :: Delta_r
            if (present(i)) then
                Delta_r=this%Delta_r(i)
            else
                error stop "Heterogeneous mesh"
            end if
        end function
        
        function get_dim_rad(this) result(dim)
            implicit none
            class(spatial_discr_rad_c) :: this
            integer(kind=4) :: dim
            dim=this%dim
        end function
        
        subroutine compute_radius(this)
            implicit none
            class(spatial_discr_rad_c) :: this
            this%radius=sum(this%Delta_r)
        end subroutine
        
        subroutine compute_Delta_r(this)
            implicit none
            class(spatial_discr_rad_c) :: this
            if (.not. allocated(this%Delta_r)) then
                allocate(this%Delta_r(this%Num_targets-this%targets_flag))
            end if
            this%Delta_r=this%radius/(this%Num_targets-this%targets_flag)
        end subroutine
        
        subroutine compute_measure_rad(this)
            implicit none
            class(spatial_discr_rad_c) :: this
            real(kind=8), parameter :: pi=4d0*atan(1d0)
            if (this%dim==1) then
                this%measure=this%radius
            else if (this%dim==2) then
                this%measure=pi*this%radius**2
            else if (this%dim==3) then
                this%measure=(4d0/3d0)*pi*this%radius**3
            else
                error stop "Dimension not implemented yet"
            end if
        end subroutine
end module
