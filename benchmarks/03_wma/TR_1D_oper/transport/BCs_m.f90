!> Boundary conditions type:
!>   contains information of boundary conditions
module BCs_m
    implicit none
    save
    type, public :: BCs_t
        integer(kind=4) :: BCs_label(2)         !> First element: inflow
                                                !> second element: outflow
                                                !> 1: Dirichlet
                                                !> 2: Neumann homogeneous
                                                !> 3: Robin inflow
        logical :: evap                         !> evaporation flag
        real(kind=8) :: conc_inf                !> concentration at inflow
        real(kind=8) :: conc_out                !> concentration at outflow
        real(kind=8) :: flux_inf                !> flux at inflow
        real(kind=8) :: flux_out                !> flux at outflow
    contains
        procedure, public :: set_BCs_label
        procedure, public :: set_evap
        !procedure, public :: set_Neumann_homog_BCs
        procedure, public :: read_BCs
        procedure, public :: read_Dirichlet_BCs
        procedure, public :: read_Robin_BC_inflow
        procedure, public :: read_flux_inf
        procedure, public :: set_conc_boundary
        procedure, public :: set_cst_flux_boundary
    end type
    
    contains
        subroutine set_BCs_label(this,BCs)
            implicit none
            class(BCs_t) :: this
            integer(kind=4), intent(in) :: BCs(2)
            if (BCs(1)>3 .or. BCs(2)>3) error stop "BCs not implemented yet"
            this%BCs_label=BCs
        end subroutine
        
        subroutine set_evap(this,evap)
            implicit none
            class(BCs_t) :: this
            logical, intent(in) :: evap
            this%evap=evap
        end subroutine 
        
        subroutine set_conc_boundary(this,conc_inf,conc_out)
            implicit none
            class(BCs_t) :: this
            real(kind=8), intent(in) :: conc_inf,conc_out
            this%conc_inf=conc_inf
            this%conc_out=conc_out
        end subroutine
        
        subroutine set_cst_flux_boundary(this,flux)
            implicit none
            class(BCs_t) :: this
            real(kind=8), intent(in) :: flux
            this%flux_inf=flux
            this%flux_out=this%flux_inf
        end subroutine
        
        subroutine read_BCs(this,filename)
            implicit none
            class(BCs_t) :: this
            character(len=*), intent(in) :: filename
            open(unit=1,file=filename,status='old',action='read')
            read(1,*) this%BCs_label
            read(1,*) this%evap
            close(1)
        end subroutine
        
        subroutine read_Dirichlet_BCs(this,filename)
            implicit none
            class(BCs_t) :: this
            character(len=*), intent(in) :: filename
            open(unit=1,file=filename,status='old',action='read')
            read(1,*) this%conc_inf
            read(1,*) this%conc_out
            close(1)
        end subroutine
        
        subroutine read_Robin_BC_inflow(this,filename)
            implicit none
            class(BCs_t) :: this
            character(len=*), intent(in) :: filename
            open(unit=1,file=filename,status='old',action='read')
            read(1,*) this%flux_inf
            read(1,*) this%conc_inf
            close(1)
        end subroutine
        
        subroutine read_flux_inf(this,filename)
            implicit none
            class(BCs_t) :: this
            character(len=*), intent(in) :: filename
            open(unit=1,file=filename,status='old',action='read')
            read(1,*) this%flux_inf
            close(1)
        end subroutine
end module