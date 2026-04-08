subroutine link_target_waters_target_solid(this,i,tw_indices)
    use chemistry_Lagr_m
    use array_ops_m
    use vectors_m
    implicit none
    class(chemistry_c) :: this
    integer(kind=4), intent(in) :: i !> target solid index
    integer(kind=4), intent(out), allocatable :: tw_indices(:) !> indices of target waters associated to i_th target solid
    
    integer(kind=4) :: j,k,n_tw_sol
    logical :: flag
    real(kind=8), parameter :: eps=1d-12

    allocate(tw_indices(0))
    
    if (this%num_target_waters_init==this%num_target_solids) then !> biyeccion
        if (i==1) then
            call append_int_1D_array(tw_indices,i)
            call append_int_1D_array(tw_indices,i+1)
        else if (i<this%num_target_solids) then
            call append_int_1D_array(tw_indices,i-1)
            call append_int_1D_array(tw_indices,i)
            call append_int_1D_array(tw_indices,i+1)
        else
            call append_int_1D_array(tw_indices,i-1)
            call append_int_1D_array(tw_indices,i)
        end if 
    else if (this%num_target_waters_init>this%num_target_solids) then
        do j=1,this%num_target_waters_init
            if (inf_norm_vec_real(this%target_waters_init(j)%solid_chemistry%concentrations-this%target_solids(i)%concentrations)<eps) then
                call append_int_1D_array(tw_indices,j)
            end if
        end do
    else
        error stop
    end if
end subroutine