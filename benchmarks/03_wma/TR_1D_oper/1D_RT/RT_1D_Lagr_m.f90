!> 1D reactive transport module (Lagrangian version)
module RT_1D_m
    use PDE_m
    use chemistry_Lagr_m
    use time_discr_m
    use transport_transient_m
    use char_params_m
    implicit none
    save
    type, public :: RT_1D_c !> 1D reactive transport superclass
        type(chemistry_c) :: chemistry
    contains
        procedure, public :: set_transport
        procedure, public :: set_chemistry
        procedure, public :: solve_RT_1D
        procedure, public :: write_RT_1D
        procedure, public :: write_python
        procedure, public :: compute_Delta_t_crit_RT
        procedure, public :: check_Delta_t_RT
        procedure, public :: write_transport_data
        procedure, public :: read_discretisation
    end type
    
    type,public,extends(RT_1D_c) :: RT_1D_transient_c !> 1D transient reactive transport subclass
        type(transport_1D_transient_c) :: transport
        real(kind=8) :: Delta_t_crit !> critical time step
        integer(kind=4) :: int_method_chem_reacts !> intregration method chemical reactions
    contains
        procedure, public :: set_int_method_chem_reacts
    end type
    
    interface
        
        subroutine solve_RT_1D(this)
            import RT_1D_c
            import props_c
            implicit none
            class(RT_1D_c) :: this
            !integer(kind=4), intent(out), optional :: niter
            !class(props_c), intent(in), optional :: props
        end subroutine
        
        subroutine write_RT_1D(this,unit,file_out,path_py)!,mixing_ratios)
            import RT_1D_c
            implicit none
            class(RT_1D_c), intent(in) :: this
            integer(kind=4), intent(in) :: unit
            character(len=*), intent(in) :: file_out
            character(len=*), intent(in), optional :: path_py
            !real(kind=8), intent(in) :: mixing_ratios(:)
        end subroutine
        
        subroutine write_python(this,path)
            import RT_1D_c
            implicit none
            class(RT_1D_c), intent(in) :: this
            character(len=*), intent(in) :: path !> path output files
        end subroutine
       
        
        subroutine compute_Delta_t_crit_RT(this)
            import RT_1D_c
            import chem_system_c
            implicit none
            class(RT_1D_c) :: this
        end subroutine
        
       subroutine read_discretisation(this,path,unit,file_discr)
            import RT_1D_c
            class(RT_1D_c) :: this
            character(len=*), intent(in) :: path
            integer(kind=4), intent(in) :: unit
            character(len=*), intent(in) :: file_discr
        end subroutine
        
       subroutine read_transport_data(this,unit,file_tpt,mixing_ratios)!,f_vec)!,tpt_props,BCs,mesh,time_discr)
            import RT_1D_c
            import tpt_props_heterog_c
            import BCs_t
            import mesh_1D_Euler_homog_c
            import time_discr_homog_c
            class(RT_1D_c) :: this
            integer(kind=4), intent(in) :: unit
            character(len=*), intent(in) :: file_tpt
            !real(kind=8), intent(out), allocatable :: mixing_ratios(:,:)
            !real(kind=8), intent(out), allocatable :: f_vec(:)
            !type(tpt_props_heterog_c), intent(out) :: tpt_props
            !type(BCs_t), intent(out) :: BCs
            !!real(kind=8), intent(in) :: Delta_x
            !type(mesh_1D_Euler_homog_c), intent(out) :: mesh !> homogeneous Euler mesh 1D
            !type(time_discr_homog_c), intent(out) :: time_discr !> homogeneous time discretisation
        end subroutine
        
        subroutine write_transport_data(this,unit,file_out)!,mixing_ratios)
            import RT_1D_c
            class(RT_1D_c) :: this
            integer(kind=4), intent(in) :: unit
            character(len=*), intent(in) :: file_out
            !real(kind=8), intent(in) :: mixing_ratios(:)
        end subroutine

    end interface
    
    contains
        subroutine set_transport(this,transport_obj)
            implicit none
            class(RT_1D_c) :: this
            !class(PDE_1D_c), intent(in), target :: transport_obj
            class(transport_1D_transient_c), intent(in) :: transport_obj
            select type(this)
            type is (RT_1D_transient_c)
                this%transport=transport_obj
            end select
        end subroutine
        
        subroutine set_chemistry(this,chemistry_obj)
            implicit none
            class(RT_1D_c) :: this
            class(chemistry_c), intent(in) :: chemistry_obj
            this%chemistry=chemistry_obj
        end subroutine
        
      
        
        subroutine check_Delta_t_RT(this)
            implicit none
            class(RT_1D_c) :: this    
            
            real(kind=8), parameter :: eps=1d-9
            
            select type (this)
            class is (RT_1D_transient_c)
                if (this%transport%time_discr%int_method==1) then
                    call this%compute_Delta_t_crit_RT()
                    if (abs(this%Delta_t_crit)<eps) then
                        continue
                    else
                        select type (time=>this%transport%time_discr)
                        type is (time_discr_homog_c)
                            if (time%Delta_t>this%Delta_t_crit) then
                                print *, this%Delta_t_crit
                                error stop "Delta_t is larger than Delta_t_crit"
                            end if
                        end select
                    end if
                end if
            end select
        end subroutine
        
        subroutine set_char_params_RT(this,char_params_obj)
            implicit none
            class(RT_1D_c) :: this
            class(char_params_c), intent(in), target :: char_params_obj
            select type (this)
            class is (RT_1D_transient_c)
                !this%char_params=>char_params_obj
            end select
        end subroutine
        
        subroutine set_int_method_chem_reacts(this,int_method)
            implicit none
            class(RT_1D_transient_c) :: this
            integer(kind=4), intent(in) :: int_method
            if (int_method<1 .or. int_method>2) error stop "Integration method for chemical reactions not implemented yet"
            this%int_method_chem_reacts=int_method
        end subroutine
end module