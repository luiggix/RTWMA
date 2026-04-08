subroutine compute_trans_mat_tpt_transient(this)
!> T: transition matrix (tridiagonal, negative stoich_mat_react_zonemi-definite)
!> rows sum = 0 if r=0
!> F*dc/dt=T*c+g
    use transport_transient_m
    use spatial_discr_1D_m
    use transport_properties_heterog_m
    implicit none
    
    class(transport_1D_transient_c) :: this
    
    real(kind=8) :: sign_flux !> sign of flux
    integer(kind=4) :: i,n
    
    n=this%spatial_discr%Num_targets

    if (this%spatial_discr%adapt_ref==1) then
        deallocate(this%trans_mat%sub,this%trans_mat%diag,this%trans_mat%super)
        call this%allocate_trans_mat()
    end if
    
    select type (mesh=>this%spatial_discr)
    type is (mesh_1D_Euler_homog_c)
        if (mesh%targets_flag==0 .and. this%dimensionless==.true.) then
            if (mesh%scheme==1) then
                this%trans_mat%sub=1d0/(mesh%Delta_x**2) + 1d0/(2*mesh%Delta_x)
                this%trans_mat%super=1d0/(mesh%Delta_x**2) - 1d0/(2*mesh%Delta_x)
            else if (mesh%scheme==2) then
                this%trans_mat%super(1)=1d0/(mesh%Delta_x**2) - 1d0/(2*mesh%Delta_x)
                do i=2,n-1
                    this%trans_mat%sub(i-1)=1d0/(mesh%Delta_x**2) + 1d0/(2*mesh%Delta_x)
                    this%trans_mat%super(i)=1d0/(mesh%Delta_x**2) - 1d0/(2*mesh%Delta_x)
                end do
                this%trans_mat%sub(n-1)=1d0/(mesh%Delta_x**2) + 1d0/(2*mesh%Delta_x)
            else if (mesh%scheme==3) then !> upwind
                if (minval(this%tpt_props_heterog%flux)>=0d0 .or. maxval(this%tpt_props_heterog%flux)<0d0) then
                    sign_flux=sign(1d0,this%tpt_props_heterog%flux(1))
                    this%trans_mat%sub=1d0/(mesh%Delta_x**2)+((sign_flux+1d0)/2)*1d0/mesh%Delta_x
                    this%trans_mat%super=1d0/(mesh%Delta_x**2)+((sign_flux-1d0)/2)*1d0/mesh%Delta_x
                end if
            else
                error stop "Scheme not implemented yet"
            end if
        else if (mesh%targets_flag==0 .and. this%dimensionless==.false.) then
            if (mesh%scheme==1) then
                this%trans_mat%sub=this%tpt_props_heterog%dispersion(2:n)/(mesh%Delta_x**2) + this%tpt_props_heterog%flux(2:n)/(2*mesh%Delta_x)
                this%trans_mat%super=this%tpt_props_heterog%dispersion(1:n-1)/(mesh%Delta_x**2) - this%tpt_props_heterog%flux(1:n-1)/(2*mesh%Delta_x)
            else if (mesh%scheme==2) then
                this%trans_mat%super(1)=this%tpt_props_heterog%dispersion(1)/(mesh%Delta_x**2) - this%tpt_props_heterog%flux(2)/(2*mesh%Delta_x)
                do i=2,n-1
                    this%trans_mat%sub(i-1)=this%tpt_props_heterog%dispersion(i)/(mesh%Delta_x**2) + this%tpt_props_heterog%flux(i)/(2*mesh%Delta_x)
                    this%trans_mat%super(i)=this%tpt_props_heterog%dispersion(i)/(mesh%Delta_x**2) - this%tpt_props_heterog%flux(i+1)/(2*mesh%Delta_x)
                end do
                this%trans_mat%sub(n-1)=this%tpt_props_heterog%dispersion(n)/(mesh%Delta_x**2) + this%tpt_props_heterog%flux(n)/(2*mesh%Delta_x)
            else if (mesh%scheme==3) then !> upwind
                if (minval(this%tpt_props_heterog%flux)>=0d0 .or. maxval(this%tpt_props_heterog%flux)<0d0) then
                    sign_flux=sign(1d0,this%tpt_props_heterog%flux(1))
                    this%trans_mat%sub=this%tpt_props_heterog%dispersion(2:n)/(mesh%Delta_x**2)+((sign_flux+1d0)/2)*this%tpt_props_heterog%flux(2:n)/mesh%Delta_x
                    this%trans_mat%super=this%tpt_props_heterog%dispersion(1:n-1)/(mesh%Delta_x**2)+((sign_flux-1d0)/2)*this%tpt_props_heterog%flux(1:n-1)/mesh%Delta_x
                end if
            else
                error stop "Scheme not implemented yet"
            end if
        else if (mesh%targets_flag==1 .and. this%dimensionless==.false. .and. this%tpt_props_heterog%homog_flag==.true.) then
            this%trans_mat%sub=this%tpt_props_heterog%dispersion(1)/(mesh%Delta_x**2) + this%tpt_props_heterog%flux(1)/(2*mesh%Delta_x)
            this%trans_mat%super=this%tpt_props_heterog%dispersion(1)/(mesh%Delta_x**2) - this%tpt_props_heterog%flux(1)/(2*mesh%Delta_x)
        end if
    end select
    this%trans_mat%diag=-this%tpt_props_heterog%source_term_flag*this%tpt_props_heterog%source_term
    this%trans_mat%diag(2:n-1)=this%trans_mat%diag(2:n-1)-this%trans_mat%sub(1:n-2)-this%trans_mat%super(2:n-1)
end subroutine 