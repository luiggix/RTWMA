!> This module contains the subroutines that impose BCs in transition matrix and source term of a PDE
module BCs_subroutines_m
    use spatial_discr_rad_m
    use transport_m
    use transport_transient_m
    implicit none
    save
    contains
         
         subroutine Dirichlet_BCs_PDE(this)
         !> Imposes Dirichlet boundary conditions in transition matrix & source term
            implicit none
            class(PDE_1D_c) :: this
            
            integer(kind=4) :: n,opcion,n_flux
            real(kind=8) :: r_n,r_n_12
            
            n=this%spatial_discr%Num_targets
            
            select type (this)
            type is (transport_1D_c)
                select type (mesh=>this%spatial_discr)
                type is (mesh_1D_Euler_homog_c)
                    if (mesh%targets_flag==1) then
                        this%trans_mat%diag(1)=0d0
                        this%trans_mat%super(1)=0d0
                        this%trans_mat%diag(n)=0d0
                        this%trans_mat%sub(n-1)=0d0
                        !this%source_term_PDE(1)=this%BCs%conc_inf
                        !this%source_term_PDE(1)=this%BCs%conc_out
                    else if (mesh%targets_flag==0 .and. mesh%scheme==1) then !> CFDS
                        this%trans_mat%diag(1)=this%trans_mat%diag(1)-this%tpt_props_heterog%flux(1)/(2d0*mesh%Delta_x) - 3d0*this%tpt_props_heterog%dispersion(1)/(mesh%Delta_x**2)
                        this%trans_mat%super(1)=-this%tpt_props_heterog%flux(1)/(2d0*mesh%Delta_x) + this%tpt_props_heterog%dispersion(1)/(mesh%Delta_x**2)
                        this%trans_mat%diag(n)=this%trans_mat%diag(n)+this%tpt_props_heterog%flux(n)/(2d0*mesh%Delta_x) - 3d0*this%tpt_props_heterog%dispersion(n)/(mesh%Delta_x**2)
                        this%trans_mat%sub(n-1)=this%tpt_props_heterog%flux(n)/(2d0*mesh%Delta_x) + this%tpt_props_heterog%dispersion(n)/(mesh%Delta_x**2)
                        this%source_term_PDE(1)=this%source_term_PDE(1)+this%BCs%conc_inf*(this%tpt_props_heterog%flux(1)+2d0*this%tpt_props_heterog%dispersion(1)/mesh%Delta_x)/mesh%Delta_x
                        this%source_term_PDE(n)=this%source_term_PDE(n)-this%BCs%conc_out*(this%tpt_props_heterog%flux(n)-2d0*this%tpt_props_heterog%dispersion(n)/mesh%Delta_x)/mesh%Delta_x
                    else if (mesh%targets_flag==0 .and. mesh%scheme==2) then !> IFDS
                        this%trans_mat%diag(1)=this%trans_mat%diag(1)-this%tpt_props_heterog%flux(1)/mesh%Delta_x + this%tpt_props_heterog%flux(2)/(2d0*mesh%Delta_x) - 3d0*this%tpt_props_heterog%dispersion(1)/(mesh%Delta_x**2)
                        this%trans_mat%super(1)=-this%tpt_props_heterog%flux(2)/(2d0*mesh%Delta_x) + this%tpt_props_heterog%dispersion(1)/(mesh%Delta_x**2)
                        this%trans_mat%diag(n)=this%trans_mat%diag(n)+this%tpt_props_heterog%flux(n+1)/mesh%Delta_x - this%tpt_props_heterog%flux(n)/(2d0*mesh%Delta_x) - 3d0*this%tpt_props_heterog%dispersion(n)/(mesh%Delta_x**2)
                        this%trans_mat%sub(n-1)=this%tpt_props_heterog%flux(n)/(2d0*mesh%Delta_x) + this%tpt_props_heterog%dispersion(n)/(mesh%Delta_x**2)
                        this%source_term_PDE(1)=this%source_term_PDE(1)+this%BCs%conc_inf*(this%tpt_props_heterog%flux(1)+2d0*this%tpt_props_heterog%dispersion(1)/mesh%Delta_x)/mesh%Delta_x
                        this%source_term_PDE(n)=this%source_term_PDE(n)-this%BCs%conc_out*(this%tpt_props_heterog%flux(n+1)-2d0*this%tpt_props_heterog%dispersion(n)/mesh%Delta_x)/mesh%Delta_x
                    else
                        error stop "BCs not implemented yet"
                    end if
                end select
            type is (transport_1D_transient_c)
                select type (mesh=>this%spatial_discr) 
                type is (mesh_1D_Euler_homog_c)
                    if (mesh%targets_flag==1) then
                        this%trans_mat%diag(1)=0d0
                        this%trans_mat%super(1)=0d0
                        this%trans_mat%diag(n)=0d0
                        this%trans_mat%sub(n-1)=0d0
                        !this%source_term_PDE(1)=this%conc_init(1)
                        !this%source_term_PDE(n)=this%conc_init(n)
                    else if (mesh%targets_flag==0 .and. mesh%scheme==1) then !> CFDS
                        this%trans_mat%diag(1)=this%trans_mat%diag(1)-this%tpt_props_heterog%flux(1)/(2d0*mesh%Delta_x) - 3d0*this%tpt_props_heterog%dispersion(1)/(mesh%Delta_x**2)
                        this%trans_mat%super(1)=-this%tpt_props_heterog%flux(1)/(2d0*mesh%Delta_x) + this%tpt_props_heterog%dispersion(1)/(mesh%Delta_x**2)
                        this%trans_mat%diag(n)=this%trans_mat%diag(n)+this%tpt_props_heterog%flux(n)/(2d0*mesh%Delta_x) - 3d0*this%tpt_props_heterog%dispersion(n)/(mesh%Delta_x**2)
                        this%trans_mat%sub(n-1)=this%tpt_props_heterog%flux(n)/(2d0*mesh%Delta_x) + this%tpt_props_heterog%dispersion(n)/(mesh%Delta_x**2)
                        this%source_term_PDE(1)=this%source_term_PDE(1)+this%BCs%conc_inf*(this%tpt_props_heterog%flux(1)+2d0*this%tpt_props_heterog%dispersion(1)/mesh%Delta_x)/mesh%Delta_x
                        this%source_term_PDE(n)=this%source_term_PDE(n)-this%BCs%conc_out*(this%tpt_props_heterog%flux(n)-2d0*this%tpt_props_heterog%dispersion(n)/mesh%Delta_x)/mesh%Delta_x
                    else if (mesh%targets_flag==0 .and. mesh%scheme==2) then !> IFDS
                        this%trans_mat%diag(1)=this%trans_mat%diag(1)-this%tpt_props_heterog%flux(1)/mesh%Delta_x + this%tpt_props_heterog%flux(2)/(2d0*mesh%Delta_x) - 3d0*this%tpt_props_heterog%dispersion(1)/(mesh%Delta_x**2)
                        this%trans_mat%super(1)=-this%tpt_props_heterog%flux(2)/(2d0*mesh%Delta_x) + this%tpt_props_heterog%dispersion(1)/(mesh%Delta_x**2)
                        this%trans_mat%diag(n)=this%trans_mat%diag(n)+this%tpt_props_heterog%flux(n+1)/mesh%Delta_x - this%tpt_props_heterog%flux(n)/(2d0*mesh%Delta_x) - 3d0*this%tpt_props_heterog%dispersion(n)/(mesh%Delta_x**2)
                        this%trans_mat%sub(n-1)=this%tpt_props_heterog%flux(n)/(2d0*mesh%Delta_x) + this%tpt_props_heterog%dispersion(n)/(mesh%Delta_x**2)
                        this%source_term_PDE(1)=this%source_term_PDE(1)+this%BCs%conc_inf*(this%tpt_props_heterog%flux(1)+2d0*this%tpt_props_heterog%dispersion(1)/mesh%Delta_x)/mesh%Delta_x
                        this%source_term_PDE(n)=this%source_term_PDE(n)-this%BCs%conc_out*(this%tpt_props_heterog%flux(n+1)-2d0*this%tpt_props_heterog%dispersion(n)/mesh%Delta_x)/mesh%Delta_x
                    else
                        error stop "BCs not implemented yet"
                    end if
                end select
            type is (diffusion_1D_transient_c)
                select type (mesh=>this%spatial_discr)
                type is (spatial_discr_rad_c)
                    if (this%dimensionless==.true.) then
                        this%trans_mat%diag(n)=this%trans_mat%diag(n)-2d0/mesh%Delta_r(n) !> dimensionless
                        this%source_term_PDE(n)=2d0/mesh%Delta_r(n) !> dimensionless
                    end if
                end select
            end select
         end subroutine
         
         subroutine Dirichlet_Neumann_BCs_PDE(this)
         !> Imposes Dirichlet & Neumann boundary conditions in transition matrix & source term
            implicit none
            class(PDE_1D_c) :: this
            
            integer(kind=4) :: n,opcion,n_flux
            real(kind=8) :: r_n,r_n_12
            
            n=this%spatial_discr%Num_targets
            
            select type (this)
            !type is (transport_1D_c)
            !    select type (mesh=>this%spatial_discr)
            !    type is (mesh_1D_Euler_homog_c)
            !        if (mesh%targets_flag==1) then
            !            this%trans_mat%diag(1)=0d0
            !            this%trans_mat%super(1)=0d0
            !            this%trans_mat%diag(n)=0d0
            !            this%trans_mat%sub(n-1)=0d0
            !            !this%source_term_PDE(1)=this%BCs%conc_inf
            !            !this%source_term_PDE(1)=this%BCs%conc_out
            !        else if (mesh%targets_flag==0 .and. mesh%scheme==1) then !> CFDS
            !            this%trans_mat%diag(1)=this%trans_mat%diag(1)-this%tpt_props_heterog%flux(1)/(2d0*mesh%Delta_x) - 3d0*this%tpt_props_heterog%dispersion(1)/(mesh%Delta_x**2)
            !            this%trans_mat%super(1)=-this%tpt_props_heterog%flux(1)/(2d0*mesh%Delta_x) + this%tpt_props_heterog%dispersion(1)/(mesh%Delta_x**2)
            !            this%trans_mat%diag(n)=this%trans_mat%diag(n)+this%tpt_props_heterog%flux(n)/(2d0*mesh%Delta_x) - 3d0*this%tpt_props_heterog%dispersion(n)/(mesh%Delta_x**2)
            !            this%trans_mat%sub(n-1)=this%tpt_props_heterog%flux(n)/(2d0*mesh%Delta_x) + this%tpt_props_heterog%dispersion(n)/(mesh%Delta_x**2)
            !            this%source_term_PDE(1)=this%source_term_PDE(1)+this%BCs%conc_inf*(this%tpt_props_heterog%flux(1)+2d0*this%tpt_props_heterog%dispersion(1)/mesh%Delta_x)/mesh%Delta_x
            !            this%source_term_PDE(n)=this%source_term_PDE(n)-this%BCs%conc_out*(this%tpt_props_heterog%flux(n)-2d0*this%tpt_props_heterog%dispersion(n)/mesh%Delta_x)/mesh%Delta_x
            !        else if (mesh%targets_flag==0 .and. mesh%scheme==2) then !> IFDS
            !            this%trans_mat%diag(1)=this%trans_mat%diag(1)-this%tpt_props_heterog%flux(1)/mesh%Delta_x + this%tpt_props_heterog%flux(2)/(2d0*mesh%Delta_x) - 3d0*this%tpt_props_heterog%dispersion(1)/(mesh%Delta_x**2)
            !            this%trans_mat%super(1)=-this%tpt_props_heterog%flux(2)/(2d0*mesh%Delta_x) + this%tpt_props_heterog%dispersion(1)/(mesh%Delta_x**2)
            !            this%trans_mat%diag(n)=this%trans_mat%diag(n)+this%tpt_props_heterog%flux(n+1)/mesh%Delta_x - this%tpt_props_heterog%flux(n)/(2d0*mesh%Delta_x) - 3d0*this%tpt_props_heterog%dispersion(n)/(mesh%Delta_x**2)
            !            this%trans_mat%sub(n-1)=this%tpt_props_heterog%flux(n)/(2d0*mesh%Delta_x) + this%tpt_props_heterog%dispersion(n)/(mesh%Delta_x**2)
            !            this%source_term_PDE(1)=this%source_term_PDE(1)+this%BCs%conc_inf*(this%tpt_props_heterog%flux(1)+2d0*this%tpt_props_heterog%dispersion(1)/mesh%Delta_x)/mesh%Delta_x
            !            this%source_term_PDE(n)=this%source_term_PDE(n)-this%BCs%conc_out*(this%tpt_props_heterog%flux(n+1)-2d0*this%tpt_props_heterog%dispersion(n)/mesh%Delta_x)/mesh%Delta_x
            !        else
            !            error stop "BCs not implemented yet"
            !        end if
            !    end select
            type is (transport_1D_transient_c)
                select type (mesh=>this%spatial_discr) 
                type is (mesh_1D_Euler_homog_c)
                    if (mesh%targets_flag==1) then
                        this%trans_mat%diag(1)=0d0
                        this%trans_mat%super(1)=0d0
                        this%trans_mat%sub(n-1)=this%tpt_props_heterog%flux(n)/(2d0*mesh%Delta_x)+this%tpt_props_heterog%dispersion(n)/(mesh%Delta_x**2)
                        this%trans_mat%diag(n)=this%trans_mat%diag(n)-this%trans_mat%sub(n-1)
                    else if (mesh%targets_flag==0 .and. mesh%scheme==1) then !> CFDS
                        this%trans_mat%diag(1)=this%trans_mat%diag(1)-this%tpt_props_heterog%flux(1)/(2d0*mesh%Delta_x) - 3d0*this%tpt_props_heterog%dispersion(1)/(mesh%Delta_x**2)
                        this%trans_mat%super(1)=-this%tpt_props_heterog%flux(1)/(2d0*mesh%Delta_x) + this%tpt_props_heterog%dispersion(1)/(mesh%Delta_x**2)
                        this%trans_mat%sub(n-1)=this%tpt_props_heterog%flux(n)/(2d0*mesh%Delta_x)+this%tpt_props_heterog%dispersion(n)/(mesh%Delta_x**2)
                        this%trans_mat%diag(n)=this%trans_mat%diag(n)-this%trans_mat%sub(n-1)
                        this%source_term_PDE(1)=this%source_term_PDE(1)+this%BCs%conc_inf*(this%tpt_props_heterog%flux(1)+2d0*this%tpt_props_heterog%dispersion(1)/mesh%Delta_x)/mesh%Delta_x
                        !this%source_term_PDE(n)=this%source_term_PDE(n)-this%BCs%conc_out*(this%tpt_props_heterog%flux(n)-2d0*this%tpt_props_heterog%dispersion(n)/mesh%Delta_x)/mesh%Delta_x
                    else if (mesh%targets_flag==0 .and. mesh%scheme==2) then !> IFDS
                        this%trans_mat%diag(1)=this%trans_mat%diag(1)-this%tpt_props_heterog%flux(1)/mesh%Delta_x + this%tpt_props_heterog%flux(2)/(2d0*mesh%Delta_x) - 3d0*this%tpt_props_heterog%dispersion(1)/(mesh%Delta_x**2)
                        this%trans_mat%super(1)=-this%tpt_props_heterog%flux(2)/(2d0*mesh%Delta_x) + this%tpt_props_heterog%dispersion(1)/(mesh%Delta_x**2)
                        this%trans_mat%sub(n-1)=this%tpt_props_heterog%flux(n)/(2d0*mesh%Delta_x)+this%tpt_props_heterog%dispersion(n)/(mesh%Delta_x**2)
                        this%trans_mat%diag(n)=this%trans_mat%diag(n)-this%trans_mat%sub(n-1)
                        this%source_term_PDE(1)=this%source_term_PDE(1)+this%BCs%conc_inf*(this%tpt_props_heterog%flux(1)+2d0*this%tpt_props_heterog%dispersion(1)/mesh%Delta_x)/mesh%Delta_x
                        !this%source_term_PDE(n)=this%source_term_PDE(n)-this%BCs%conc_out*(this%tpt_props_heterog%flux(n+1)-2d0*this%tpt_props_heterog%dispersion(n)/mesh%Delta_x)/mesh%Delta_x
                    else
                        error stop "BCs not implemented yet"
                    end if
                end select
            !type is (diffusion_1D_transient_c)
            !    select type (mesh=>this%spatial_discr)
            !    type is (spatial_discr_rad_c)
            !        if (this%dimensionless==.true.) then
            !            this%trans_mat%diag(n)=this%trans_mat%diag(n)-2d0/mesh%Delta_r(n) !> dimensionless
            !            this%source_term_PDE(n)=2d0/mesh%Delta_r(n) !> dimensionless
            !        end if
            !    end select
            end select
         end subroutine
        
         subroutine Neumann_homog_BCs(this)
         !> Imposes Neumann homogeneous boundary conditions in transition matrix
            implicit none
            class(PDE_1D_c) :: this
            
            integer(kind=4) :: n
            
            n=this%spatial_discr%Num_targets

            select type (this)
            type is (transport_1D_transient_c)
                select type (mesh=>this%spatial_discr)
                type is (mesh_1D_Euler_homog_c)
                    if (mesh%scheme==1 .and. mesh%targets_flag==0) then !> CFD
                        this%trans_mat%super(1)=-this%tpt_props_heterog%flux(1)/(2d0*mesh%Delta_x)+this%tpt_props_heterog%dispersion(1)/(mesh%Delta_x**2)
                        this%trans_mat%sub(n-1)=this%tpt_props_heterog%flux(n)/(2d0*mesh%Delta_x)+this%tpt_props_heterog%dispersion(n)/(mesh%Delta_x**2)
                        this%trans_mat%diag(1)=this%trans_mat%diag(1)-this%trans_mat%super(1)
                        this%trans_mat%diag(n)=this%trans_mat%diag(n)-this%trans_mat%sub(n-1)
                    else if (mesh%scheme==2 .and. mesh%targets_flag==0) then !> IFD
                        this%trans_mat%super(1)=-this%tpt_props_heterog%flux(2)/(2d0*mesh%Delta_x)+this%tpt_props_heterog%dispersion(1)/(mesh%Delta_x**2)
                        this%trans_mat%sub(n-1)=this%tpt_props_heterog%flux(n)/(2d0*mesh%Delta_x)+this%tpt_props_heterog%dispersion(n)/(mesh%Delta_x**2)
                        this%trans_mat%diag(1)=this%trans_mat%diag(1)-this%trans_mat%super(1)
                        this%trans_mat%diag(n)=this%trans_mat%diag(n)-this%trans_mat%sub(n-1)
                    end if
                end select
            end select
         end subroutine 
         
         subroutine Robin_Neumann_homog_BCs(this)
         !> Imposes Robin BC inflow & Neumann homogeneous BC outflow in transition matrix & sink/source term
            implicit none
            class(PDE_1D_c) :: this
            
            integer(kind=4) :: n
            real(kind=8) :: q_1,q_32,q_n,q_inf,q_out,D_1,D_n,c_inf
            
            n=this%spatial_discr%Num_targets
            
            select type (this)
            type is (transport_1D_transient_c)
                q_1=this%tpt_props_heterog%flux(1)
                q_32=this%tpt_props_heterog%flux(2)
                q_n=this%tpt_props_heterog%flux(n)
                q_inf=this%BCs%flux_inf
                q_out=this%BCs%flux_out
                D_1=this%tpt_props_heterog%dispersion(1)
                D_n=this%tpt_props_heterog%dispersion(n)
                c_inf=this%BCs%conc_inf
                select type (mesh=>this%spatial_discr)
                type is (mesh_1D_Euler_homog_c)
                    if (mesh%scheme==1 .and. mesh%targets_flag==0) then !> CFDS
                        this%trans_mat%super(1)=-q_1/(2d0*mesh%Delta_x)+D_1/(mesh%Delta_x**2)
                        this%trans_mat%sub(n-1)=q_n/(2d0*mesh%Delta_x)+D_n/(mesh%Delta_x**2)
                        this%trans_mat%diag(1)=this%trans_mat%diag(1)+(2d0*D_1-q_inf*mesh%Delta_x)*(q_1*mesh%Delta_x+2*D_1)/(2*q_inf*mesh%Delta_x**3+4*D_1*mesh%Delta_x**2) - 2*D_1/(mesh%Delta_x**2)
                        this%trans_mat%diag(n)=this%trans_mat%diag(n)-this%trans_mat%sub(n-1)
                        this%source_term_PDE(1)=this%source_term_PDE(1)+q_inf*c_inf*(q_1*mesh%Delta_x+2*D_1)/(q_inf*mesh%Delta_x**2+2*D_1*mesh%Delta_x)
                    else if (mesh%scheme==2 .and. mesh%targets_flag==0) then !> IFDS
                        this%trans_mat%super(1)=-q_32/(2d0*mesh%Delta_x)+D_1/(mesh%Delta_x**2)
                        this%trans_mat%sub(n-1)=this%tpt_props_heterog%flux(n)/(2d0*mesh%Delta_x)+this%tpt_props_heterog%dispersion(n)/(mesh%Delta_x**2)
                        !this%trans_mat%diag(1)=this%trans_mat%diag(1)-(q_inf**2+D_1*(3*q_inf*mesh%Delta_x+2*D_1)/(mesh%Delta_x**2))/(q_inf*mesh%Delta_x+2*D_1) + q_32/(2*mesh%Delta_x)
                        this%trans_mat%diag(1)=this%trans_mat%diag(1) - q_inf/mesh%Delta_x + q_32/(2*mesh%Delta_x) - D_1/(mesh%Delta_x**2)
                        this%trans_mat%diag(n)=this%trans_mat%diag(n)-this%trans_mat%sub(n-1)
                        this%source_term_PDE(1)=this%source_term_PDE(1)+q_inf*c_inf/mesh%Delta_x
                    end if
                end select
            end select
         end subroutine
end module