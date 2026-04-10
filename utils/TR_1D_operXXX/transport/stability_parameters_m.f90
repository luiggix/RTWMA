module stability_parameters_m
    use time_discr_m
    use properties_m
    implicit none
    save
    type, public, abstract :: stab_params_c !> stability parameters superclass
        real(kind=8) :: Delta_t_crit !> critical time step
    contains
        procedure(compute_stab_params), public, deferred :: compute_stab_params
    end type
    
    abstract interface        
        subroutine compute_stab_params(this,props_obj,mesh_size,time_step)
            import stab_params_c
            import props_c
            import spatial_discr_c
            import time_discr_c
            implicit none
            class(stab_params_c) :: this
            class(props_c), intent(in) :: props_obj
            real(kind=8), intent(in) :: mesh_size
            real(kind=8), intent(in) :: time_step
        end subroutine
    end interface
end module