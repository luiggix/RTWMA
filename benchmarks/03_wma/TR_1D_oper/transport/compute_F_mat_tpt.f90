subroutine compute_F_mat_tpt(this) !> diagonal matrix
!> F_ii=phi_i
    use transport_transient_m
    use transport_properties_heterog_m
    implicit none
    class(transport_1D_transient_c) :: this
    
    integer(kind=4) :: n
    
    n=this%spatial_discr%Num_targets
     
    if (this%spatial_discr%adapt_ref==1) then
        deallocate(this%F_mat%diag)
        call this%F_mat%allocate_matrix(n)
    end if
    
    if (this%tpt_props_heterog%homog_flag==.true.) then
        this%F_mat%diag=this%tpt_props_heterog%porosity(1)
    end if
    
        
end subroutine