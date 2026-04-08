module probability_m
    use spatial_discr_1D_m
    implicit none
    save
    type, public, abstract :: prob_law_c !> probability law class
        character(len=256) :: name
        real(kind=8), allocatable :: pdf(:)
    contains
        !procedure, public :: set_name
        procedure, public :: allocate_pdf
        procedure(compute_pdf), public, deferred :: compute_pdf
        !procedure, public :: read_prob_law
    end type
    
    type, public, extends(prob_law_c) :: unif_law_c
        real(kind=8) :: min
        real(kind=8) :: max
    contains
        procedure, public :: set_interval
        procedure, public :: compute_pdf=>compute_unif_pdf
    end type
    
     type, public :: random_var_c !> random variable class
        character(len=256) :: name
        class(prob_law_c), pointer :: law
     end type
     
     type, public :: PCE_c !> polynomial chaos expansion class
         integer(kind=4) :: p !> degree polynomial chaos expansion
         real(kind=8), allocatable :: coeffs(:)
         real(kind=8), allocatable :: Sobol_indices(:)
         class(random_var_c), allocatable :: inputs(:) 
         class(prob_law_c), pointer :: law
         type(random_var_c) :: output
     contains
        procedure, public :: set_p
        procedure, public :: set_inputs
        procedure, public :: read_inputs_output
        procedure, public :: set_law
        procedure, public :: allocate_coeffs
        procedure, public :: allocate_Sobol_indices
        procedure, public :: compute_PCE_coeffs
        procedure, public :: compute_Sobol_indices
     end type
    
    abstract interface
        subroutine compute_pdf(this,mesh)
            import prob_law_c
            import spatial_discr_c
            implicit none
            class(prob_law_c) :: this
            class(spatial_discr_c), intent(in), optional :: mesh
            !real(kind=8), intent(in) :: x
        end subroutine
    end interface
        
    interface
        subroutine compute_Sobol_indices(this,index)
            import PCE_c
            implicit none
            class(PCE_c) :: this
            integer(kind=4), intent(in), optional :: index
        end subroutine
        
        subroutine compute_PCE_coeffs(this,samples,results)
            import PCE_c
            implicit none
            class(PCE_c) :: this
            real(kind=8), intent(in) :: samples(:,:)
            real(kind=8), intent(in) :: results(:)
        end subroutine
        
    end interface
    
    contains
        !subroutine set_name(this,name)
        !>    implicit none
        !>    class(prob_law_c) :: this
        !>    character(len=*), intent(in) :: name
        !>    this%name=name
        !end subroutine
        
        subroutine allocate_pdf(this,mesh)
            implicit none
            class(prob_law_c) :: this
            class(spatial_discr_c), intent(in) :: mesh
            allocate(this%pdf(mesh%Num_targets))
        end subroutine
        
        subroutine set_interval(this,min,max)
            implicit none
            class(unif_law_c) :: this
            real(kind=8), intent(in) :: min
            real(kind=8), intent(in) :: max
            this%min=min
            this%max=max
        end subroutine
        
        subroutine compute_unif_pdf(this,mesh)
            implicit none
            class(unif_law_c) :: this
            class(spatial_discr_c), intent(in), optional :: mesh
            !real(kind=8), intent(in) :: x
            !select type (mesh)
            !type is (mesh_1D_Euler_homog_c)
            this%pdf=1d0/(this%max-this%min)
        end subroutine
        
        subroutine set_p(this,p)
            implicit none
            class(PCE_c) :: this
            integer(kind=4), intent(in) :: p
            this%p=p
        end subroutine
        
        subroutine set_law(this,law)
            implicit none
            class(PCE_c) :: this
            class(prob_law_c), intent(in), target :: law
            this%law=>law
        end subroutine
        
        subroutine set_inputs(this,inputs)
            implicit none
            class(PCE_c) :: this
            class(random_var_c), intent(in) :: inputs(:)
            !select type (law=>input(1)%law)
            !type is (unif_law_c)
                this%inputs=inputs
            !class default
            !>    error stop "PCE not implemented yet for this law"
            !end select
        end subroutine
        
        subroutine allocate_coeffs(this)
            implicit none
            class(PCE_c) :: this
            !integer(kind=4), intent(in) :: p !> degree polynomial chaos expansion
            if (size(this%inputs)==1 .and. this%p==1) then
                allocate(this%coeffs(2))
            end if
        end subroutine
    
        subroutine allocate_Sobol_indices(this)
            implicit none
            class(PCE_c) :: this
            !if (allocated(this%coeffs)) then
            !>    allocate(this%Sobol_indices(size(this%coeffs)))
            !else
            !>    error stop "PCE coefficients not computed"
            !end if
            if (size(this%inputs)==1) then
                allocate(this%Sobol_indices(1))
            else if (size(this%inputs)==2) then
                allocate(this%Sobol_indices(3))
            end if
        end subroutine
    
        subroutine read_inputs_output(this,filename)
            implicit none
            class(PCE_c) :: this
            character(len=*), intent(in) :: filename
            integer(kind=4) :: j,m
            open(unit=5,file=filename,status='old',action='read')
            read(5,*) m
            allocate(this%inputs(m))
            backspace(5)
            read(5,*) m, (this%inputs(j)%name, j=1,m)!, this%law%name
            read(5,*) this%output%name
            close(5)
        end subroutine
end module