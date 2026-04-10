module properties_m
    use BCs_m
    use spatial_discr_m
    implicit none
    save
    type, public, abstract :: props_c !> properties superclass
        integer(kind=4) :: source_term_order                    !> if source term is polynomial, this is the degree
        real(kind=8), allocatable :: source_term(:)             !> (r)
        integer(kind=4), allocatable :: source_term_flag(:)     !> 0 if r<0 & no evaporation
                                                                !> 1 otherwise
        logical :: homog_flag                                  !> True if all properties are homogenous, False otherwise
    contains
        procedure, public :: set_source_term
        procedure, public :: set_source_term_order
        procedure, public :: set_source_term_flag
        procedure(read_props), public, deferred :: read_props
        procedure(are_props_homog), public, deferred :: are_props_homog
    end type
    
    abstract interface
    
        subroutine read_props(this,filename,spatial_discr)
            import props_c
            import spatial_discr_c
            class(props_c) :: this
            character(len=*), intent(in) :: filename
            class(spatial_discr_c), intent(in), optional :: spatial_discr
        end subroutine
        
        subroutine are_props_homog(this)
            import props_c
            class(props_c) :: this
        end subroutine
            
    end interface
    
    contains
        subroutine set_source_term(this,source_term)
            implicit none
            class(props_c) :: this
            real(kind=8), intent(in) :: source_term(:)
            this%source_term=source_term
        end subroutine
        
        subroutine set_source_term_order(this,source_term_order)
            implicit none
            class(props_c) :: this
            integer(kind=4), intent(in) :: source_term_order
            this%source_term_order=source_term_order
        end subroutine
        
         subroutine set_source_term_flag(this,BCs)
            implicit none
            class(props_c) :: this
            class(BCs_t), intent(in) :: BCs
            
            integer(kind=4) :: i
            allocate(this%source_term_flag(size(this%source_term)))
            this%source_term_flag=1
            do i=1,size(this%source_term)
                if (this%source_term(i)<0 .and. BCs%evap==.false.) then !> disvalence
                    this%source_term_flag(i)=0
                end if
            end do
        end subroutine
        
end module