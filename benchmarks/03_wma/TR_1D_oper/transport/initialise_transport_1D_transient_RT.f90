!> This subroutine initialises a 1D transient transport object for a RT simulation
!! It reads:
!!      discretisation parameters
!!      transport properties
!!      BCs
!! It computes stability parameters (in the explicit case)
subroutine initialise_transport_1D_transient_RT(this,path,file_BCs,file_spatial_discr,file_time_discr,file_tpt_props)
    use BCs_subroutines_m
    use transport_stab_params_m
    implicit none

    class(transport_1D_transient_c) :: this
    character(len=*), intent(in) :: path
    character(len=*), intent(in) :: file_BCs
    character(len=*), intent(in) :: file_spatial_discr
    character(len=*), intent(in) :: file_time_discr
    character(len=*), intent(in) :: file_tpt_props
    
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
    dimless=.false. !> esto habria que leerlo
    this%dimensionless=dimless
!> Boundary conditions
    call my_BCs%read_BCs(trim(path)//file_BCs)
    if (my_BCs%BCs_label(1)==1 .and. my_BCs%BCs_label(2)==1) then
        call my_BCs%read_Dirichlet_BCs(trim(path)//"Dirichlet_BCs.dat")
        call my_BCs%read_flux_inf(trim(path)//"flux_inflow.dat")
    else if (my_BCs%BCs_label(1)==3) then
        call my_BCs%read_Robin_BC_inflow(trim(path)//"Robin_BC_inflow.dat")
    end if
    call this%set_BCs(my_BCs)
 !> Uniform mesh
    my_mesh=>my_homog_mesh
    call my_mesh%read_mesh(trim(path)//file_spatial_discr)
    call this%set_spatial_discr(my_mesh)
!> Uniform time discretisation
    my_time_discr=>my_homog_time_discr
    call my_time_discr%read_time_discr(trim(path)//file_time_discr)
    call this%set_time_discr(my_time_discr)
!****************************************************************************************************************************************************
!> Transport properties
    call my_props_tpt%read_props(trim(path)//file_tpt_props,this%spatial_discr)
    if (my_props_tpt%source_term_order==0) then !> constant source term
        if (inf_norm_vec_real(my_props_tpt%source_term)<eps) then
            call this%BCs%set_cst_flux_boundary(my_props_tpt%flux(1))
        else
            call my_props_tpt%compute_flux_lin(this%BCs%flux_inf,this%spatial_discr,this%BCs%flux_out)
        end if
    else if (my_props_tpt%source_term_order>0) then !> flux is polynomic
    !> chapuza
        open(unit=2,file=trim(path)//"flux_coeffs.dat",status='old',action='read')
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
    if (this%time_discr%int_method==1) then !> Euler explicit
        call my_stab_params_tpt%compute_stab_params(this%tpt_props_heterog,my_homog_mesh%Delta_x,my_homog_time_discr%Delta_t)
        call this%set_stab_params_tpt(my_stab_params_tpt)
    end if
!**************************************************************************************************************************************************
    nullify(my_mesh,my_time_discr)
end subroutine