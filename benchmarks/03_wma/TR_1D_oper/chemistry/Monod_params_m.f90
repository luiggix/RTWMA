!> This subclass contains the parameters of a Monod reaction rate
module Monod_params_m
    use kin_params_m
    implicit none
    save
    type, public, extends(kin_params_c) :: Monod_params_c
        integer(kind=4) :: num_terms !> number of terms
        integer(kind=4) :: n_M !> number of TEAs
        integer(kind=4) :: n_inh !> number of inhibitors
        real(kind=8), allocatable :: conc_thr_TEAs(:) !> threshold concentration electron donors & acceptors
        !type(species_c), allocatable :: TEAs(:) !> terminal electron acceptors
        !type(species_c), allocatable :: inhibitors(:)
        !type(species_c), allocatable :: catalysts(:)
        real(kind=8), allocatable :: k_M(:) !> half-saturation constants
        real(kind=8), allocatable :: k_inh(:) !> inhibition constants
        !integer(kind=4) :: n_cat !> number of catalysts
        real(kind=8), allocatable :: conc_thr_inh(:) !> threshold concentration inhibitors
        
    contains
        procedure, public :: allocate_inhibitors
        procedure, public :: allocate_TEAs
        procedure, public :: compute_num_terms
    end type
        
    interface
     
    end interface
    
    contains
      
        subroutine allocate_TEAs(this,n_M)
            implicit none
            class(Monod_params_c) :: this
            integer(kind=4), intent(in), optional :: n_M
            if (present(n_M)) then
                if (n_M<0) then
                    error stop
                else
                    this%n_M=n_M
                end if
            end if
            allocate(this%k_M(this%n_M))
        end subroutine
        
        subroutine allocate_inhibitors(this,n_inh)
            implicit none
            class(Monod_params_c) :: this
            integer(kind=4), intent(in), optional :: n_inh
            if (present(n_inh)) then
                if (n_inh<0) then
                    error stop
                else
                    this%n_inh=n_inh
                end if
            end if
            allocate(this%k_inh(this%n_inh))
        end subroutine
        
        subroutine compute_num_terms(this)
            implicit none
            class(Monod_params_c) :: this
            this%num_terms=this%n_inh+this%n_M
        end subroutine
        
end module