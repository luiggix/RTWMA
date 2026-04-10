!> Local chemistry abstract superclass
!!> This class contains the local chemical information
module local_chemistry_m
    implicit none
    save
    type, public, abstract :: local_chemistry_c
        real(kind=8) :: temp !> temperature [K]
        real(kind=8) :: density !> [kg/L]
        real(kind=8) :: pressure !> [atm]
        real(kind=8), allocatable :: concentrations(:) !> species concentrations
        !real(kind=8), allocatable :: conc_comp(:) !> component concentrations
        real(kind=8), allocatable :: activities(:) !> species activities
        real(kind=8), allocatable :: log_act_coeffs(:) !> log_10 activity coefficients
        real(kind=8), allocatable :: log_Jacobian_act_coeffs(:,:) !> log_10 Jacobian activity coefficients
        real(kind=8), allocatable :: rk(:) !> kinetic reaction rates
        real(kind=8), allocatable :: r_eq(:) !> equilibrium reaction rates
        real(kind=8) :: volume=1d0 !> volume of solution or volumetric fraction (1 by default)
        integer(kind=4), allocatable :: var_act_species_indices(:)
        integer(kind=4), allocatable :: cst_act_species_indices(:)
    contains
    !> Set
        procedure, public :: set_density
        procedure, public :: set_pressure
        procedure, public :: set_temp
        procedure, public :: set_volume
        procedure, public :: set_concentrations
    !> Allocate
        procedure, public :: allocate_var_act_species_indices
        procedure, public :: allocate_cst_act_species_indices
    end type
    
    contains
        subroutine set_temp(this,temp)
            implicit none
            class(local_chemistry_c) :: this
            real(kind=8), intent(in), optional :: temp !> Kelvin
            if (present(temp)) then
                this%temp=temp
            else
                this%temp=298.15 !> default
            end if
        end subroutine
        
        subroutine set_volume(this,vol)
            implicit none
            class(local_chemistry_c) :: this
            real(kind=8), intent(in), optional :: vol !> [L]
            if (present(vol)) then
                this%volume=vol
            else
                this%volume=1d0 !> default
            end if
        end subroutine
        
        subroutine set_pressure(this,pressure)
            implicit none
            class(local_chemistry_c) :: this
            real(kind=8), intent(in), optional :: pressure
            if (present(pressure)) then
                this%pressure=pressure
            else
                this%pressure=1d0 !> default
            end if
        end subroutine
        
        subroutine set_density(this,density)
            implicit none
            class(local_chemistry_c) :: this
            real(kind=8), intent(in), optional :: density
            if (present(density)) then
                this%density=density
            else
                this%density=0.9970749 !> [kg/L] (water)
            end if
        end subroutine
        
        subroutine set_concentrations(this,conc)
            implicit none
            class(local_chemistry_c) :: this
            real(kind=8), intent(in) :: conc(:)
            this%concentrations=conc
        end subroutine
        
        subroutine allocate_var_act_species_indices(this,num_var)
            implicit none
            class(local_chemistry_c) :: this
            integer(kind=4), intent(in) :: num_var
            allocate(this%var_act_species_indices(num_var))
        end subroutine
        
        subroutine allocate_cst_act_species_indices(this,num_cst)
            implicit none
            class(local_chemistry_c) :: this
            integer(kind=4), intent(in) :: num_cst
            allocate(this%cst_act_species_indices(num_cst))
        end subroutine
end module