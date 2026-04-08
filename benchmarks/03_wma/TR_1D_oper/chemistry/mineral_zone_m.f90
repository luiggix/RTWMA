module mineral_zone_m
    use mineral_m
    implicit none
    save
!**************************************************************************************************
    type, public :: mineral_zone_c !> mineral zone class
        !integer(kind=4) :: num_species
        integer(kind=4) :: num_minerals
        integer(kind=4) :: num_minerals_eq !> number of minerals in equilibrium
        type(mineral_c), allocatable :: minerals(:)
        type(mineral_c), allocatable :: minerals_eq(:)
        real(kind=8), allocatable :: stoich_mat(:,:)
        real(kind=8), allocatable :: comp_mat(:,:)
    contains
        procedure, public :: set_num_minerals
        procedure, public :: allocate_minerals_eq
        procedure, public :: allocate_minerals
        procedure, public :: set_minerals_eq
        procedure, public :: update_mineral_zone
        procedure, public :: is_min_zone_equal_to
    end type
!**************************************************************************************************
    interface
        subroutine update_mineral_zone(this,old_min_ind)
            import mineral_zone_c
            class(mineral_zone_c) :: this
            integer(kind=4), intent(in) :: old_min_ind(:)
        end subroutine
    end interface
    
    
    
    contains
        subroutine set_num_minerals(this,num_minerals)
            implicit none
            class(mineral_zone_c) :: this
            integer(kind=4), intent(in), optional :: num_minerals
            if (present(num_minerals)) then
                this%num_minerals=num_minerals
            else
                this%num_minerals=size(this%minerals)
            end if
        end subroutine
        
        
        subroutine allocate_minerals_eq(this,num_minerals_eq)
            implicit none
            class(mineral_zone_c) :: this
            integer(kind=4), intent(in), optional :: num_minerals_eq
            if (present(num_minerals_eq)) then
                allocate(this%minerals_eq(num_minerals_eq))
            else
                allocate(this%minerals_eq(this%num_minerals_eq))
            end if
        end subroutine
        
        subroutine allocate_minerals(this,num_minerals)
            implicit none
            class(mineral_zone_c) :: this
            integer(kind=4), intent(in), optional :: num_minerals
            if (present(num_minerals)) then
                allocate(this%minerals(num_minerals))
            else
                allocate(this%minerals(this%num_minerals))
            end if
        end subroutine
        
      
        
        subroutine set_minerals_eq(this,minerals_eq)
            implicit none
            class(mineral_zone_c) :: this
            class(mineral_c), intent(in) :: minerals_eq(:)
            integer(kind=4) :: i,j
            
            if (size(minerals_eq)>this%num_minerals) error stop "Dimension error in minerals in equilibrium"
            this%minerals_eq=minerals_eq
        end subroutine
        
        subroutine is_min_zone_equal_to(this,min_zone,flag)
            implicit none
            class(mineral_zone_c), intent(in) :: this
            class(mineral_zone_c), intent(in) :: min_zone
            logical, intent(out) :: flag
            
            integer(kind=4) :: i,j
            
            flag=.true.
            if (this%num_minerals/=min_zone%num_minerals) then
                flag=.false.
            else if (this%num_minerals_eq/=min_zone%num_minerals_eq) then
                flag=.false.
            else
                do i=1,this%num_minerals
                    if (this%minerals(i)%name/=min_zone%minerals(i)%name) then
                        flag=.false.
                        exit
                    end if
                end do
                do i=1,this%num_minerals_eq
                    if (this%minerals_eq(i)%name/=min_zone%minerals_eq(i)%name) then
                        flag=.false.
                        exit
                    end if
                end do
            end if
        end subroutine
        
end module