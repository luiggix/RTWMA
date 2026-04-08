module transport_stab_params_m
    use diff_stab_params_m
    use transport_properties_heterog_m
    use spatial_discr_1D_m
    
    implicit none
    save
    type, public, extends(stab_params_diff_c) :: stab_params_tpt_c !> 1D transport stability parameters subclass
        real(kind=8) :: Courant !> advection stability parameter (=q*Delta_t/(phi*Delta_x))
        real(kind=8) :: Peclet !> Pe=|q|*Delta_x/D
    contains
        procedure, public :: compute_stab_params=>compute_stab_params_tpt
    end type
    
    contains
        subroutine compute_stab_params_tpt(this,props_obj,mesh_size,time_step)
            implicit none
            class(stab_params_tpt_c) :: this
            class(props_c), intent(in) :: props_obj
            real(kind=8), intent(in) :: mesh_size
            real(kind=8), intent(in) :: time_step
            
            real(kind=8) :: D,phi,q
            real(kind=8), parameter :: epsilon=1d-12
            logical :: flag
            
            call compute_stab_params_diff(this,props_obj,mesh_size,time_step)
            
            select type (props_obj)
            type is (tpt_props_heterog_c)
                call are_tpt_props_homog(props_obj)
                if (props_obj%homog_flag==.true.) then
                    phi=props_obj%porosity(1)
                    D=props_obj%dispersion(1)
                    q=props_obj%flux(1)
                    this%Delta_t_crit=phi*mesh_size**2/(2d0*D)
                    this%Courant=q*time_step/(phi*mesh_size)
                    if (this%Courant>1d0) then
                        error stop  "Courant condition violated"
                    end if
                    this%Peclet=abs(q)*mesh_size/D
                    if (this%Peclet>2d0) then
                        error stop  "Peclet condition violated"
                    end if
                else
                    error stop "Stability parameters for heterogenous properties not implemented yet"
                end if
            end select            
        end subroutine
end module