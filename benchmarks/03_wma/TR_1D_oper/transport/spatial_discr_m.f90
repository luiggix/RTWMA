!> Spatial discretisation module
module spatial_discr_m
    implicit none
    save
    type, public, abstract :: spatial_discr_c !> spatial discretisation superclass
        integer(kind=4) :: Num_targets      !> number of targets
        logical :: Num_targets_defined      !> TRUE if Num_targets defined, FALse otherwise
        integer(kind=4) :: targets_flag     !> 0: cells
                                            !> 1: interfaces
        real(kind=8) :: measure
        integer(kind=4) :: scheme           !> Spatial discretisation scheme:
                                                !> 1: CFD
                                                !> 2: IFD
                                                !> 3: Upwind
        integer(kind=4) :: adapt_ref        !> adaptive refinement (0: NO, 1: YES)
    contains
        procedure, public :: set_targets
        procedure, public :: set_Num_targets
        procedure, public :: set_measure
        procedure, public :: set_scheme
        procedure(read_mesh), public, deferred :: read_mesh
        procedure(get_mesh_size), public, deferred :: get_mesh_size
        procedure(get_dim), public, deferred :: get_dim
        procedure(compute_measure), public, deferred :: compute_measure
        !procedure(compute_Num_targets), public, deferred :: compute_Num_targets
        procedure(refine_mesh), public, deferred :: refine_mesh        
    end type
        
    abstract interface
        !subroutine set_mesh_1D(this,Delta_x)
        !>    import spatial_discr_c
        !>    implicit none
        !>    class(spatial_discr_c) :: this
        !>    real(kind=8), intent(in) :: Delta_x
        !end subroutine
        
        subroutine read_mesh(this,filename)
            import spatial_discr_c
            implicit none
            class(spatial_discr_c) :: this
            character(len=*), intent(in) :: filename
        end subroutine
        
        function get_mesh_size(this,i) result(Delta_x)
            import spatial_discr_c
            implicit none
            class(spatial_discr_c) :: this
            integer(kind=4), intent(in), optional :: i
            real(kind=8) :: Delta_x
        end function
        
        function get_dim(this) result(dim)
            import spatial_discr_c
            implicit none
            class(spatial_discr_c) :: this
            integer(kind=4) :: dim
        end function
        
        subroutine compute_measure(this)
            import spatial_discr_c
            implicit none
            class(spatial_discr_c) :: this
        end subroutine
        
        subroutine refine_mesh(this,conc,conc_ext,rel_tol)
            import spatial_discr_c
            implicit none
            class(spatial_discr_c) :: this
            real(kind=8), intent(inout), allocatable :: conc(:,:) !> Num_columns=Num_targets
            real(kind=8), intent(inout), allocatable :: conc_ext(:,:) !> Num_columns=Num_targets
            real(kind=8), intent(in) :: rel_tol !> relative tolerance
            !integer(kind=4), intent(out) :: n_new
        end subroutine
    end interface
    
    contains
        subroutine set_targets(this,Num_targets,flag)
            implicit none
            class(spatial_discr_c) :: this
            integer(kind=4), intent(in) :: Num_targets
            integer(kind=4), intent(in) :: flag
            this%Num_targets=Num_targets
            if (flag>1 .or. flag<0) error stop "Error in targets_flag"
            this%targets_flag=flag
        end subroutine
        
        subroutine set_Num_targets(this,Num_targets)
            implicit none
            class(spatial_discr_c) :: this
            integer(kind=4), intent(in) :: Num_targets
            if (Num_targets<1) then
                error stop "Number of targets must be positive"
            end if
            this%Num_targets=Num_targets
        end subroutine 
        
        !subroutine read_mesh_1D(this,filename)
        !>    implicit none
        !>    class(spatial_discr_c) :: this
        !>    character(len=*), intent(in) :: filename
        !>    
        !>    real(kind=8), allocatable :: Delta_x(:)
        !>    
        !>    open(unit=1,file=filename,status='old',action='read')
        !>    read(1,*) this%Num_targets
        !>    select type (this)
        !>    type is (mesh_1D_Lagr_heterog_c)
        !>        allocate(Delta_x(this%Num_targets)) !> size(Delta_x_vec)=Num_targets
        !>        read(1,*) Delta_x
        !>        allocate(this%Delta_x(this%Num_targets)) 
        !>        this%Delta_x=Delta_x
        !>    end select
        !>    close(1)
        !>    this%Num_targets_defined=.true.
        !end subroutine 
        !
        !subroutine set_mesh_1D(this,Delta_x,Num_targets)
        !>    implicit none
        !>    class(spatial_discr_c) :: this
        !>    real(kind=8), intent(in) :: Delta_x
        !>    integer(kind=4), intent(in), optional :: Num_targets
        !>    select type (this)
        !>    type is (mesh_1D_Lagr_homog_c)
        !>        this%Delta_x=Delta_x
        !>        if (this%Num_targets_defined==.false. .and. present(Num_targets)) then
        !>            this%Num_targets=Num_targets
        !>            this%Num_targets_defined=.true.
        !>        else if (this%Num_targets_defined==.true. .and. present(Num_targets)) then
        !>            error stop "Num_targets already defined"
        !>        else if (this%Num_targets==.false. .and. (.not. present(Num_targets))) then
        !>            error stop "Num_targets missing"
        !>        end if
        !>    end select
        !end subroutine
        !
        !subroutine set_mesh_1D_Lagr_homog(this,Delta_x,Num_targets)
        !>    implicit none
        !>    class(mesh_1D_Lagr_homog_c) :: this
        !>    real(kind=8), intent(in) :: Delta_x
        !>    integer(kind=4), intent(in), optional :: Num_targets
        !>    this%Delta_x=Delta_x
        !>    if (this%Num_targets_defined==.false. .and. present(Num_targets)) then
        !>        this%Num_targets=Num_targets
        !>        this%Num_targets_defined=.true.
        !>    else if (this%Num_targets_defined==.true. .and. present(Num_targets)) then
        !>        error stop "Num_targets already defined"
        !>    else if (this%Num_targets==.false. .and. (.not. present(Num_targets))) then
        !>        error stop "Num_targets missing"
        !>    end if
        !end subroutine
        !
        !subroutine set_mesh_1D_Lagr_heterog(this,Delta_x,Num_targets)
        !>    implicit none
        !>    class(mesh_1D_Lagr_heterog_c) :: this
        !>    real(kind=8), intent(in) :: Delta_x(:)
        !>    integer(kind=4), intent(in), optional :: Num_targets
        !>    this%Delta_x=Delta_x
        !>    if (this%Num_targets_defined==.true. .and. size(Delta_x)/=this%Num_targets) then
        !>        error stop "Dimension error in heterogeneous mesh"
        !>    else if (this%Num_targets_defined==.false.) then
        !>        this%Num_targets=size(Delta_x)
        !>        this%Num_targets_defined=.true.
        !>    end if
        !end subroutine 
        
        !subroutine set_var_Delta_x(this,file_Delta_x)
        !>    implicit none
        !>    class(mesh_transport_1D) :: this
        !>    character(len=*), intent(in) :: file_Delta_x
        !>    
        !>    integer(kind=4) :: i,j,Num_elements
        !>    real(kind=8), allocatable :: Delta_x_vec(:)
        !>    !integer(kind=4), allocatable :: n_vec(:)
        !>    !> We assume file_Delta_x contains element sizes
        !>    select type (this)
        !>    type is (heterog_mesh_transport_1D)
        !>        open(unit=2,file=file_Delta_x,status='old',action='read')
        !>        read(2,*) Num_elements
        !>        allocate(Delta_x_vec(Num_elements)) !> size(Delta_x_vec)=Num_elements
        !>        read(2,*) Delta_x_vec
        !>        allocate(this%Delta_x(Num_elements)) 
        !>        this%Delta_x=Delta_x_vec
        !>        !print *, this%Delta_x
        !>        close(2)
        !>    class default
        !>        error stop "Wrong subclass"
        !>    end select
        !end subroutine set_var_Delta_x
        
        subroutine set_measure(this,measure)
            implicit none
            class(spatial_discr_c) :: this
            real(kind=8), intent(in) :: measure
            this%measure=measure
            !select type (this)
            !type is (homog_mesh_transport_1D)
            !>    this%measure=this%Num_elements*this%Delta_x
            !type is (heterog_mesh_transport_1D)
            !>    this%measure=sum(this%Delta_x)
            !end select
        end subroutine set_measure
        
        subroutine set_scheme(this,scheme)
            implicit none
            class(spatial_discr_c) :: this
            integer(kind=4), intent(in) :: scheme
            if (scheme>3 .or. scheme<1) then
                error stop "Scheme not implemented yet"
            !else if (scheme==2 .and. this%targets_flag==0) then
                !error stop "Targets must be interfaces with IFDS"
            else
                this%scheme=scheme
            end if
        end subroutine 
        
        !function get_mesh_size(this) result(Delta_x)
        !>    implicit none
        !>    class(spatial_discr_c) :: this
        !>    real(kind=8), allocatable :: Delta_x(:)
        !>    select type (this)
        !>    type is (mesh_1D_Lagr_homog_c)
        !>        allocate(Delta_x(1))
        !>        Delta_x=get_mesh_size_homog(this)
        !>    type is (mesh_1D_Lagr_heterog_c)
        !>        Delta_x=get_mesh_size_heterog(this)
        !>    end select
        !end function
        !
        !function get_mesh_size_homog(this) result(Delta_x)
        !>    implicit none
        !>    class(mesh_1D_Lagr_homog_c) :: this
        !>    real(kind=8) :: Delta_x
        !>    Delta_x=this%Delta_x
        !end function
        !
        !function get_mesh_size_heterog(this) result(Delta_x)
        !>    implicit none
        !>    class(mesh_1D_Lagr_heterog_c) :: this
        !>    real(kind=8), allocatable :: Delta_x(:)
        !>    Delta_x=this%Delta_x
        !end function
            
end module
