subroutine initialise_transport_1D_transient(this)
    use BCs_subroutines_m
    use transport_stab_params_m
    implicit none

    class(transport_1D_transient_c) :: this
    !character(len=*), intent(in) :: path
    !character(len=*), intent(in) :: file_BCs
    !character(len=*), intent(in) :: file_spatial_discr
    !character(len=*), intent(in) :: file_time_discr
    !character(len=*), intent(in) :: file_tpt_props
    
    type(tpt_props_heterog_c) :: my_props_tpt
    class(spatial_discr_c), pointer :: my_mesh=>null()
    type(mesh_1D_Euler_homog_c), target :: my_homog_mesh
    type(mesh_1D_Euler_heterog_c), target :: my_heterog_mesh
    class(time_discr_c), pointer :: my_time_discr=>null()
    type(time_discr_homog_c), target :: my_homog_time_discr
    type(time_discr_heterog_c), target :: my_heterog_time_discr
    type(BCs_t) :: my_BCs
    type(stab_params_tpt_c) :: my_stab_params_tpt
    type(char_params_tpt_c) :: my_char_params_tpt
    
    real(kind=8) :: q0,Delta_t,Delta_x,theta,measure,Final_time,x_1,x_2,x_3,x
    real(kind=8), allocatable :: c0(:),c_e(:),source_term_vec(:),porosity_vec(:),dispersion_vec(:),flux_vec(:),flux_coeffs(:)
    real(kind=8), allocatable :: d(:),e(:)
    integer(kind=4) :: parameters_flag,i,Num_cells,Num_time,info,n,adapt_ref_flag,scheme,int_method,r_flag,flux_ord,half_num_tar
    real(kind=8), parameter :: pi=4d0*atan(1d0), eps=1d-12, epsilon_x=1d-2, epsilon_t=1d-6
    character(len=200) :: filename
    logical :: evap,dimless
!****************************************************************************************************************************************************
!> Dimensionless form flag
    dimless=.false.
    this%dimensionless=dimless
!> Boundary conditions
    call my_BCs%read_BCs("C:\Users\Jordi\source\repos\jordipg10\RT_Lagrange\input\BCs.dat")
    if (my_BCs%BCs_label(1)==1 .and. my_BCs%BCs_label(2)==1) then
        call my_BCs%read_Dirichlet_BCs("C:\Users\Jordi\source\repos\jordipg10\RT_Lagrange\input\Dirichlet_BCs.dat")
        call my_BCs%read_flux_inf("C:\Users\Jordi\source\repos\jordipg10\RT_Lagrange\input\flux_inflow.dat")
    else if (my_BCs%BCs_label(1)==3) then
        call my_BCs%read_Robin_BC_inflow("C:\Users\Jordi\source\repos\jordipg10\RT_Lagrange\input\Robin_BC_inflow.dat")
    end if
    call my_BCs%read_BCs("C:\Users\Jordi\source\repos\jordipg10\RT_Lagrange\input\BCs.dat")
    if (my_BCs%BCs_label(1)==1 .and. my_BCs%BCs_label(2)==1) then
        call my_BCs%read_Dirichlet_BCs("C:\Users\Jordi\source\repos\jordipg10\RT_Lagrange\input\Dirichlet_BCs.dat")
        call my_BCs%read_flux_inf("C:\Users\Jordi\source\repos\jordipg10\RT_Lagrange\input\flux_inflow.dat")
    else if (my_BCs%BCs_label(1)==3) then
        call my_BCs%read_Robin_BC_inflow("C:\Users\Jordi\source\repos\jordipg10\RT_Lagrange\input\Robin_BC_inflow.dat")
    end if
    call this%set_BCs(my_BCs)
 !> Uniform mesh
    my_mesh=>my_homog_mesh
    call my_mesh%read_mesh("C:\Users\Jordi\source\repos\jordipg10\RT_Lagrange\input\Delta_x_homog.dat")
    call this%set_spatial_discr(my_mesh)
    Num_cells=this%spatial_discr%Num_targets-this%spatial_discr%targets_flag
!> Uniform time discretisation
    my_time_discr=>my_homog_time_discr
    Delta_t=1d-1 !> [yr]
    Final_time=Delta_t!1.5d3
    int_method=2                                                !> 1: Euler explicit
                                                                !> 2: Euler fully implicit
                                                                !> 3: Crank-Nicolson
                                                                !> 4: RKF45
    call my_homog_time_discr%set_Delta_t_homog(Delta_t)
    call my_time_discr%set_int_method(int_method)
    call my_time_discr%set_Final_time(Final_time)
    call my_time_discr%compute_Num_time()
    call this%set_time_discr(my_time_discr)
!****************************************************************************************************************************************************
!> Transport properties
    call my_props_tpt%read_props("C:\Users\Jordi\source\repos\jordipg10\RT_Lagrange\input\tpt_props.dat",this%spatial_discr)
    if (my_props_tpt%source_term_order==0) then
        if (inf_norm_vec_real(my_props_tpt%source_term)<eps) then
            call this%BCs%set_cst_flux_boundary(my_props_tpt%flux(1))
        else
            call my_props_tpt%compute_flux_lin(this%BCs%flux_inf,this%spatial_discr,this%BCs%flux_out)
        end if
    else if (my_props_tpt%source_term_order>0) then
        open(unit=2,file='flux_coeffs.dat',status='old',action='read')
        open(unit=2,file='flux_coeffs.dat',status='old',action='read')
        read(2,*) flux_ord
        allocate(flux_coeffs(flux_ord+1))
        read(2,*) flux_coeffs
        close(2)
        call my_props_tpt%set_source_term_order(flux_ord-1)
        call my_props_tpt%compute_flux_nonlin(flux_coeffs,this%spatial_discr,this%BCs%flux_out)
        call my_props_tpt%compute_source_term(this%spatial_discr,flux_coeffs)
    end if
    call my_props_tpt%set_source_term_flag(this%BCs)
    call this%set_tpt_props_heterog_obj(my_props_tpt)
!****************************************************************************************************************************************************
!> Stability parameters
    call my_stab_params_tpt%compute_stab_params(this%tpt_props_heterog,my_homog_mesh%Delta_x,my_homog_time_discr%Delta_t)
    call this%set_stab_params_tpt(my_stab_params_tpt)
    !call this%check_Delta_t()
    !print *, this%stab_params_tpt%Delta_t_crit
!****************************************************************************************************************************************************
!> Critical time step test
    !select type (time=>this%time_discr)
    !type is (time_discr_homog_c)
    !>    call time%set_Delta_t_homog(this%stab_params_tpt%Delta_t_crit-epsilon_t)
    !>    call time%compute_Num_time()
    !>    call this%stab_params_tpt%compute_stab_params(this%tpt_props_heterog,my_homog_mesh%Delta_x,time%Delta_t)
    !end select
!****************************************************************************************************************************************************
!> External concentration
    allocate(c_e(this%spatial_discr%Num_targets))
    c_e=0d0
    call this%set_conc_ext(c_e)
    call this%set_conc_r_flag()
!> Initial concentration
    allocate(c0(this%spatial_discr%Num_targets))
    half_num_tar=nint(this%spatial_discr%Num_targets/2d0)
    if (this%BCs%BCs_label(1)<3 .and. this%BCs%BCs_label(2)<3) then
        c0(1:half_num_tar)=1d0
        c0(half_num_tar+1:half_num_tar)=0d0
        !c0=0d0
    else
        c0=0d0
    end if
    if (this%spatial_discr%targets_flag==1) then
        if (this%BCs%BCs_label(1)==1) then !> Dirichlet inflow
            c0(1)=this%BCs%conc_inf
        end if
        if (this%BCs%BCs_label(2)==1) then !> Dirichlet outflow
            c0(Num_cells)=this%BCs%conc_out
        end if
    end if
    call this%set_conc_init(c0)
!****************************************************************************************************************************************************
!!> We compute transport arrays
!>    call this%allocate_trans_mat()
!>    call this%compute_trans_mat_PDE()
!>    call this%compute_source_term_PDE()
!>    call this%F_mat%allocate_matrix(Num_cells)
!>    call this%compute_F_mat_PDE()
!!> We impose BCs
!>    if (this%BCs%BCs_label(1)==1 .and. this%BCs%BCs_label(2)==1) then
!>        call Dirichlet_BCs_PDE(this)
!>    else if (this%BCs%BCs_label(1)==2 .and. this%BCs%BCs_label(2)==2) then
!>        call Neumann_homog_BCs(this)
!>    else if (this%BCs%BCs_label(1)==3 .and. this%BCs%BCs_label(2)==2) then
!>        call Robin_Neumann_homog_BCs(this)
!>    else
!>        error stop "Boundary conditions not implemented yet"
!>    end if
!****************************************************************************************************************************************************
    nullify(my_mesh,my_time_discr)
end subroutine