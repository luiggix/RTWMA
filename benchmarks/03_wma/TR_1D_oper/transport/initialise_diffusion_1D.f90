subroutine initialise_diffusion_1D(this)
    use BCs_subroutines_m
    use diffusion_m
    use diff_props_heterog_m
    use spatial_discr_1D_m
    implicit none

    class(diffusion_1D_c) :: this
    
    type(diff_props_heterog_c) :: my_props_diff
    class(spatial_discr_c), pointer :: my_mesh=>null()
    type(mesh_1D_Euler_homog_c), target :: my_homog_mesh
    type(mesh_1D_Euler_heterog_c), target :: my_heterog_mesh
    type(spatial_discr_rad_c), target :: my_radial_mesh
    type(BCs_t) :: my_BCs
    
    real(kind=8) :: q0,Delta_t,Delta_x,theta
    real(kind=8), allocatable :: c0(:),c_e(:),source_term_vec(:),porosity_vec(:),dispersion_vec(:),flux_vec(:),Dirichlet_BCs(:)
    real(kind=8), allocatable :: d(:),e(:)
    integer(kind=4) :: parameters_flag,i,Num_cells,Num_time,info,n,adapt_ref_flag
    integer(kind=4) :: BCs(2)
    real(kind=8), parameter :: pi=4d0*atan(1d0), epsilon_x=1d-2, epsilon_t=1d-4
    character(len=200) :: filename
    logical :: evap,dimless
!****************************************************************************************************************************************************
!> Dimensionless form flag
    dimless=.false.
    this%dimensionless=dimless
!> Boundary conditions
    call my_BCs%read_BCs("BCs.dat")
    if (my_BCs%BCs_label(1)==1 .and. my_BCs%BCs_label(2)==1) then
        call my_BCs%read_Dirichlet_BCs("Dirichlet_BCs.dat")
        call my_BCs%read_flux_inf("flux_inflow.dat")
    else if (my_BCs%BCs_label(1)==3) then
        call my_BCs%read_Robin_BC_inflow("Robin_BC_inflow.dat")
    end if
    call this%set_BCs(my_BCs)
 !> Uniform mesh
    my_mesh=>my_radial_mesh
    call my_mesh%read_mesh("Delta_r_homog.dat")
    call this%set_spatial_discr(my_mesh)
    Num_cells=this%spatial_discr%Num_targets-this%spatial_discr%targets_flag
!> Diffusion properties
        call my_props_diff%read_props("diff_props.dat",this%spatial_discr)
        call my_props_diff%set_source_term_flag(this%BCs)
        call this%set_diff_props_heterog(my_props_diff)
!****************************************************************************************************************************************************
!> External concentration
    allocate(c_e(Num_cells))
    c_e=0d0
    call this%set_conc_ext(c_e)
!****************************************************************************************************************************************************
!> Matrices y vectors
    call this%allocate_trans_mat()
    call this%compute_trans_mat_PDE()
    call this%compute_source_term_PDE()
!> Condiciones contorno
    if (this%BCs%BCs_label(1)==1 .and. this%BCs%BCs_label(2)==1) then
        call Dirichlet_BCs_PDE(this)
    else if (this%BCs%BCs_label(1)==2 .and. this%BCs%BCs_label(2)==2) then
        call Neumann_homog_BCs(this)
    else if (this%BCs%BCs_label(1)==3 .and. this%BCs%BCs_label(2)==2) then
        call Robin_Neumann_homog_BCs(this)
    else
        error stop "Boundary conditions not implemented yet"
    end if
!****************************************************************************************************************************************************
    nullify(my_mesh)
end subroutine