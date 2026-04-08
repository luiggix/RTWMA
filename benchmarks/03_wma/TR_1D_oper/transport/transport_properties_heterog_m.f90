module transport_properties_heterog_m
    use diff_props_heterog_m
    use spatial_discr_1D_m
    use polynomials_m
    use vectors_m
    implicit none
    save
    type, public, extends(diff_props_heterog_c) :: tpt_props_heterog_c !> heterogeneous 1D transport properties subclass
        real(kind=8), allocatable :: flux(:) !> q
    contains
        procedure, public :: set_tpt_props_heterog
        procedure, public :: read_props=>read_tpt_props_heterog
        procedure, public :: compute_flux_lin
        procedure, public :: compute_flux_nonlin
        procedure, public :: compute_source_term
        procedure, public :: are_props_homog=>are_tpt_props_homog
    end type
    
    contains
        subroutine read_source_term_tpt(this,filename,mesh)
            implicit none
            class(tpt_props_heterog_c) :: this
            character(len=*), intent(in) :: filename
            class(spatial_discr_c), intent(in) :: mesh
            
            real(kind=8) :: r
            integer(kind=4) :: r_flag
            logical :: cst_source_term
            
            open(unit=1,file=filename,status='old',action='read')
            read(1,*) cst_source_term
            if (cst_source_term==.true.) then
                backspace(1)
                read(1,*) cst_source_term, r
                allocate(this%source_term(mesh%Num_targets-mesh%targets_flag))
                this%source_term=r
                this%source_term_order=0
            else if (allocated(this%flux)) then
                continue
            else
                read(1,*) this%source_term
                if (size(this%source_term)/=mesh%Num_targets-mesh%targets_flag) error stop "Dimension error in source term"
            end if
            close(1)
        end subroutine
        
        subroutine set_tpt_props_heterog(this,porosity,dispersion,flux)
            implicit none
            class(tpt_props_heterog_c) :: this
            real(kind=8), intent(in) :: porosity(:),dispersion(:)
            real(kind=8), intent(in), optional :: flux(:)
            
            integer(kind=4) :: i
            
            this%porosity=porosity
            this%dispersion=dispersion
            if (present(flux)) then
                this%flux=flux
            end if
        end subroutine
        
        subroutine read_tpt_props_heterog(this,filename,spatial_discr)
            implicit none
            class(tpt_props_heterog_c) :: this
            character(len=*), intent(in) :: filename
            class(spatial_discr_c), intent(in), optional :: spatial_discr
            
            integer(kind=4) :: n_flux
            real(kind=8), parameter :: epsilon=1d-12
            real(kind=8) :: phi,D,q,r
            logical :: flag
            
            open(unit=1,file=filename,status='old',action='read')
            read(1,"(/,L)") flag
            if (flag==.true.) then
                backspace(1)
                read(1,*) flag, r
                allocate(this%source_term(spatial_discr%Num_targets))
                this%source_term=r
                this%source_term_order=0
            else if (allocated(this%flux)) then
                continue
            else
                read(1,*) this%source_term
                if (size(this%source_term)/=spatial_discr%Num_targets) error stop "Dimension error in source term"
            end if
            read(1,*) flag
            if (flag==.true.) then
                backspace(1)
                read(1,*) flag, phi
                allocate(this%porosity(spatial_discr%Num_targets-spatial_discr%targets_flag))
                this%porosity=phi
            end if
            read(1,*) flag
            if (flag==.true.) then
                backspace(1)
                read(1,*) flag, D
                allocate(this%dispersion(spatial_discr%Num_targets))
                this%dispersion=D
            end if
            read(1,*) flag
            if (flag==.true. .and. allocated(this%flux)==.true.) then
                error stop "Flux already allocated"
            else if (flag==.true. .and. sum_squares(this%source_term)>epsilon) then
                error stop "Flux cannot be constant"
            else if (flag==.true.) then
                backspace(1)
                read(1,*) flag, q
                if (spatial_discr%scheme==2 .and. spatial_discr%targets_flag==0) then
                    n_flux=spatial_discr%Num_targets+1
                else
                    n_flux=spatial_discr%Num_targets
                end if
                allocate(this%flux(n_flux))
                this%flux=q
            end if
            close(1)
            this%homog_flag=.true. !> chapuza
        end subroutine
        
        subroutine compute_flux_lin(this,q_inf,spatial_discr_obj,q_out)
            implicit none
            class(tpt_props_heterog_c) :: this
            real(kind=8), intent(in) :: q_inf !> flux entering domain
            class(spatial_discr_c), intent(in) :: spatial_discr_obj
            real(kind=8), intent(out) :: q_out !> flux leaving domain
            
            integer(kind=4) :: Num_cells,i
            
            Num_cells=spatial_discr_obj%Num_targets-spatial_discr_obj%targets_flag
            
            if (spatial_discr_obj%scheme==2 .and. spatial_discr_obj%targets_flag==0) then !> IFDS
                allocate(this%flux(Num_cells+1))
                this%flux(1)=q_inf
                select type (spatial_discr_obj)
                type is (mesh_1D_Euler_homog_c)
                    do i=2,Num_cells+1
                        this%flux(i)=this%flux(i-1)+this%source_term(i-1)*spatial_discr_obj%Delta_x
                    end do
                type is (mesh_1D_Euler_heterog_c)
                    do i=2,Num_cells+1
                        this%flux(i)=this%flux(i-1)+this%source_term(i-1)*spatial_discr_obj%Delta_x(i-1)
                    end do
                end select
                q_out=this%flux(Num_cells+1)
            else if (spatial_discr_obj%scheme==1 .and. spatial_discr_obj%targets_flag==0) then
                allocate(this%flux(Num_cells))
                select type (spatial_discr_obj)
                type is (mesh_1D_Euler_homog_c)
                    this%flux(1)=q_inf+this%source_term(1)*spatial_discr_obj%Delta_x/2d0
                    do i=2,Num_cells
                        this%flux(i)=this%flux(i-1)+this%source_term(i-1)*spatial_discr_obj%Delta_x
                    end do
                type is (mesh_1D_Euler_heterog_c)
                    this%flux(1)=q_inf+this%source_term(1)*spatial_discr_obj%Delta_x(1)/2d0
                    do i=2,Num_cells
                        this%flux(i)=this%flux(i-1)+this%source_term(i-1)*(spatial_discr_obj%Delta_x(i-1)+spatial_discr_obj%Delta_x(i))/2d0
                    end do
                end select
                q_out=this%flux(Num_cells)+this%source_term(Num_cells)*spatial_discr_obj%get_mesh_size(Num_cells)/2d0
            end if
        end subroutine
        
        subroutine compute_flux_nonlin(this,flux_coeffs,spatial_discr_obj,q_out) !> we assume domain starts at x=0
            implicit none
            class(tpt_props_heterog_c) :: this
            real(kind=8), intent(in) :: flux_coeffs(:) !> coefficients of flux polynomial decreasing order
            class(spatial_discr_c), intent(in) :: spatial_discr_obj
            real(kind=8), intent(out) :: q_out !> flux leaving domain
            
            integer(kind=4) :: i,deg,Num_cells
            real(kind=8) :: x,dqn_dx,d2qn_dx2,Delta_x_n
            
            deg=size(flux_coeffs)-1
            Num_cells=spatial_discr_obj%Num_targets-spatial_discr_obj%targets_flag
            
            if (spatial_discr_obj%scheme==1) then
                allocate(this%flux(Num_cells))
                select type (spatial_discr_obj)     
                type is (mesh_1D_Euler_homog_c)
                    do i=1,Num_cells
                        this%flux(i)=real_poly_1D(flux_coeffs,spatial_discr_obj%Delta_x*(2*i-1)/2d0)
                    end do
                type is (mesh_1D_Euler_heterog_c)
                    x=spatial_discr_obj%Delta_x(1)/2d0
                    this%flux(1)=real_poly_1D(flux_coeffs,x)
                    do i=2,Num_cells
                        x=x+(spatial_discr_obj%Delta_x(i-1)+spatial_discr_obj%Delta_x(i))/2d0
                        this%flux(i)=real_poly_1D(flux_coeffs,x)
                    end do
                end select
                Delta_x_n=spatial_discr_obj%get_mesh_size(Num_cells)
                dqn_dx=der_real_poly_1D(flux_coeffs,spatial_discr_obj%measure-Delta_x_n/2d0)
                d2qn_dx2=sec_der_real_poly_1D(flux_coeffs,spatial_discr_obj%measure-Delta_x_n/2d0)
                if (this%source_term_order==1) then
                    q_out=this%flux(Num_cells) + Delta_x_n*dqn_dx/2d0 + (Delta_x_n**2)*d2qn_dx2/8d0 !> Taylor
                end if
            else if (spatial_discr_obj%scheme==2) then
                allocate(this%flux(Num_cells+1))
                select type (spatial_discr_obj)     
                type is (mesh_1D_Euler_homog_c)
                    do i=1,Num_cells+1
                        this%flux(i)=real_poly_1D(flux_coeffs,spatial_discr_obj%Delta_x*(i-1))
                    end do
                type is (mesh_1D_Euler_heterog_c)
                    x=-spatial_discr_obj%Delta_x(1)
                    do i=1,Num_cells+1
                        x=x+spatial_discr_obj%Delta_x(i)
                        this%flux(i)=real_poly_1D(flux_coeffs,x)
                    end do
                end select
                q_out=this%flux(Num_cells+1)
            else
                error stop "subroutine 'compute_flux_nonlin' not implemented yet for this scheme"
            end if
        end subroutine
        
        subroutine compute_source_term(this,spatial_discr_obj,flux_coeffs) !> computes sink/source term from flux polynomial
            implicit none
            class(tpt_props_heterog_c) :: this
            class(spatial_discr_c), intent(in) :: spatial_discr_obj
            real(kind=8), intent(in) :: flux_coeffs(:) !> orden decreciente
            
            integer(kind=4) :: i
            real(kind=8) :: dq_dx
            allocate(this%source_term(spatial_discr_obj%Num_targets-spatial_discr_obj%targets_flag))
            
            select type (spatial_discr_obj)
            type is (mesh_1D_Euler_homog_c)
                if (spatial_discr_obj%scheme==1 .and. spatial_discr_obj%targets_flag==0) then
                    if (size(flux_coeffs)==2) then
                        dq_dx=flux_coeffs(1) !> linear flux
                        do i=1,size(this%flux)
                            this%source_term(i)=dq_dx
                        end do
                    else if (size(flux_coeffs)==3) then
                        do i=1,size(this%flux)
                            dq_dx=der_real_poly_1D(flux_coeffs,spatial_discr_obj%Delta_x*(2*i-1)/2d0) !> quadratic flux
                            this%source_term(i)=dq_dx
                        end do
                    else
                        error stop "Subroutine 'compute_source_term' not implemented yet for cubic fluxes"
                    end if
                else if (spatial_discr_obj%scheme==2 .and. spatial_discr_obj%targets_flag==0) then
                    do i=1,size(this%source_term)
                        this%source_term(i)=(this%flux(i+1)-this%flux(i))/spatial_discr_obj%Delta_x
                    end do
                else
                    error stop
                end if
            end select
        end subroutine
        
        subroutine are_tpt_props_homog(this)
            implicit none
            class(tpt_props_heterog_c) :: this
            
            integer(kind=4) :: i
            real(kind=8), parameter :: eps=1d-12
            
            call are_diff_props_homog(this)
            if (this%homog_flag==.true.) then
                do i=2,size(this%flux)
                    if (abs(this%flux(1)-this%flux(i))>eps) then
                        this%homog_flag=.false.
                        exit
                    end if
                end do
            end if
        end subroutine
end module