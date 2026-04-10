module transport_transient_m
    use diffusion_transient_m
    use transport_properties_heterog_m
    use transport_stab_params_m
    implicit none
    save
    type, public, extends(diffusion_1D_transient_c) :: transport_1D_transient_c !> 1D transient transport subclass
        type(tpt_props_heterog_c) :: tpt_props_heterog      !> properties
        type(stab_params_tpt_c) :: stab_params_tpt          !> stability parameters
    contains
        procedure, public :: set_conc_r_flag=>set_conc_r_flag_tpt
        procedure, public :: compute_F_mat_PDE=>compute_F_mat_tpt
        procedure, public :: compute_trans_mat_PDE=>compute_trans_mat_tpt_transient
        procedure, public :: set_stab_params_tpt
        procedure, public :: initialise_PDE=>initialise_transport_1D_transient
        procedure, public :: initialise_transport_1D_transient_RT
        procedure, public :: write_PDE_1D=>write_transport_1D_transient
        procedure, public :: set_tpt_props_heterog_obj
        procedure, public :: mass_balance_error_ADE_trans_PMF_evap
        procedure, public :: mass_balance_error_ADE_trans_Dirichlet_evap
        procedure, public :: mass_balance_error_ADE_trans_PMF_revalence
        procedure, public :: mass_balance_error_ADE_trans_Dirichlet_revalence
        procedure, public :: mass_balance_error_ADE_trans_PMF_disvalence
        procedure, public :: mass_balance_error_ADE_trans_Dirichlet_disvalence
        procedure, public :: check_Delta_t
        procedure, public :: read_transport_data_WMA
        !procedure, public :: read_discretisation_WMA
    end type
    
    interface
        subroutine compute_F_mat_tpt(this)
            import transport_1D_transient_c
            implicit none
            class(transport_1D_transient_c) :: this
        end subroutine
        
        subroutine compute_trans_mat_tpt_transient(this)
            import transport_1D_transient_c
            implicit none
            class(transport_1D_transient_c) :: this
        end subroutine
        
        subroutine initialise_transport_1D_transient(this)
            import transport_1D_transient_c
            implicit none
            class(transport_1D_transient_c) :: this
        end subroutine
        
        subroutine initialise_transport_1D_transient_RT(this,path,file_BCs,file_spatial_discr,file_time_discr,file_tpt_props)
            import transport_1D_transient_c
            implicit none
            class(transport_1D_transient_c) :: this
            character(len=*), intent(in) :: path
            character(len=*), intent(in) :: file_BCs
            character(len=*), intent(in) :: file_spatial_discr
            character(len=*), intent(in) :: file_time_discr
            character(len=*), intent(in) :: file_tpt_props
        end subroutine
        
        subroutine write_transport_1D_transient(this,Time_out,output)
            import transport_1D_transient_c
            import props_c
            implicit none
            class(transport_1D_transient_c), intent(in) :: this !> transport object
            real(kind=8), intent(in) :: Time_out(:)
            real(kind=8), intent(in) :: output(:,:)
        end subroutine
        
        function mass_balance_error_ADE_trans_Dirichlet_revalence(this,conc_old,conc_new,Delta_t,Delta_x) result(mass_bal_err)
            import transport_1D_transient_c
            implicit none
            class(transport_1D_transient_c), intent(in) :: this
            real(kind=8), intent(in) :: conc_old(:)
            real(kind=8), intent(in) :: conc_new(:)
            real(kind=8), intent(in) :: Delta_t
            real(kind=8), intent(in) :: Delta_x
            real(kind=8) :: mass_bal_err
        end function
        
        function mass_balance_error_ADE_trans_Dirichlet_disvalence(this,conc_old,conc_new,Delta_t,Delta_x) result(mass_bal_err)
            import transport_1D_transient_c
            implicit none
            class(transport_1D_transient_c), intent(in) :: this
            real(kind=8), intent(in) :: conc_old(:)
            real(kind=8), intent(in) :: conc_new(:)
            real(kind=8), intent(in) :: Delta_t
            real(kind=8), intent(in) :: Delta_x
            real(kind=8) :: mass_bal_err
        end function
        
        function mass_balance_error_ADE_trans_Dirichlet_evap(this,conc_old,conc_new,Delta_t,Delta_x) result(mass_bal_err)
            import transport_1D_transient_c
            implicit none
            class(transport_1D_transient_c), intent(in) :: this
            real(kind=8), intent(in) :: conc_old(:)
            real(kind=8), intent(in) :: conc_new(:)
            real(kind=8), intent(in) :: Delta_t
            real(kind=8), intent(in) :: Delta_x
            real(kind=8) :: mass_bal_err
        end function
        
        function mass_balance_error_ADE_trans_PMF_revalence(this,conc_old,conc_new,Delta_t,Delta_x) result(mass_bal_err)
            import transport_1D_transient_c
            implicit none
            class(transport_1D_transient_c), intent(in) :: this
            real(kind=8), intent(in) :: conc_old(:)
            real(kind=8), intent(in) :: conc_new(:)
            real(kind=8), intent(in) :: Delta_t
            real(kind=8), intent(in) :: Delta_x
            real(kind=8) :: mass_bal_err
        end function
        
        function mass_balance_error_ADE_trans_PMF_disvalence(this,conc_old,conc_new,Delta_t,Delta_x) result(mass_bal_err)
            import transport_1D_transient_c
            implicit none
            class(transport_1D_transient_c), intent(in) :: this
            real(kind=8), intent(in) :: conc_old(:)
            real(kind=8), intent(in) :: conc_new(:)
            real(kind=8), intent(in) :: Delta_t
            real(kind=8), intent(in) :: Delta_x
            real(kind=8) :: mass_bal_err
        end function
        
        function mass_balance_error_ADE_trans_PMF_evap(this,conc_old,conc_new,Delta_t,Delta_x) result(mass_bal_err)
            import transport_1D_transient_c
            implicit none
            class(transport_1D_transient_c), intent(in) :: this
            real(kind=8), intent(in) :: conc_old(:)
            real(kind=8), intent(in) :: conc_new(:)
            real(kind=8), intent(in) :: Delta_t
            real(kind=8), intent(in) :: Delta_x
            real(kind=8) :: mass_bal_err
        end function
        
        subroutine read_transport_data_WMA(this,path,unit,file_tpt)!,mixing_ratios)!,f_vec)!,tpt_props,BCs,mesh,time_discr)
            import transport_1D_transient_c
            !import tpt_props_heterog_c
            !import BCs_t
            !import mesh_1D_Euler_homog_c
            !import time_discr_homog_c
            class(transport_1D_transient_c) :: this
            character(len=*), intent(in) :: path
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
        
        subroutine read_discretisation_WMA(this,path,unit,file_discr)
            import transport_1D_transient_c
            class(transport_1D_transient_c) :: this
            character(len=*), intent(in) :: path
            integer(kind=4), intent(in) :: unit
            character(len=*), intent(in) :: file_discr
        end subroutine
    end interface
    
    contains
      
        subroutine set_stab_params_tpt(this,stab_params_tpt)
            implicit none
            class(transport_1D_transient_c) :: this
            class(stab_params_tpt_c), intent(in) :: stab_params_tpt
            this%stab_params_tpt=stab_params_tpt
        end subroutine
        
        subroutine set_tpt_props_heterog_obj(this,tpt_props_heterog)
            implicit none
            class(transport_1D_transient_c) :: this
            class(tpt_props_heterog_c), intent(in) :: tpt_props_heterog
            this%tpt_props_heterog=tpt_props_heterog
        end subroutine
        
        subroutine set_conc_r_flag_tpt(this)
            implicit none
            class(transport_1D_transient_c) :: this
            
            integer(kind=4) :: i
            
            allocate(this%conc_r_flag(this%spatial_discr%Num_targets))
            this%conc_r_flag=0
            do i=1,this%spatial_discr%Num_targets-this%spatial_discr%targets_flag
                if (this%tpt_props_heterog%source_term(i)>0) then
                    this%conc_r_flag(i)=1
                end if
            end do
        end subroutine
        
        subroutine check_Delta_t(this)
            implicit none
            class(transport_1D_transient_c) :: this
            select type (time=>this%time_discr)
            type is (time_discr_homog_c)
                if (time%Delta_t>this%stab_params_tpt%Delta_t_crit) then
                    !print *, "Critical time step: ", this%stab_params_tpt%Delta_t_crit
                    !error stop "You must reduce time step to have stability"
                end if
            end select
        end subroutine
end module