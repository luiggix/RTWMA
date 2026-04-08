subroutine link_target_waters_init_reactive_zone(this,i,tw_indices)
    use chemistry_Lagr_m
    use vectors_m
    use array_ops_m
    implicit none
    
    class(chemistry_c), intent(in) :: this
    integer(kind=4), intent(in) :: i !> reactive zone index
    integer(kind=4), intent(out), allocatable :: tw_indices(:) !> indices of target waters associated to i_th reactive zone
    
    integer(kind=4) :: j,k,n,nf_sp_ind
    logical :: flag
    real(kind=8), parameter :: eps=1d-12
    
    j=1
    k=1
    flag=.true.
    allocate(tw_indices(0))
    
    do
        if (associated(this%target_waters_init(j)%solid_chemistry)) then
            call this%target_waters_init(j)%solid_chemistry%reactive_zone%is_nf_species_in_react_zone(this%reactive_zones(i)%non_flowing_species(k),flag,nf_sp_ind)
            if (flag==.false.) then
                if (j<this%num_target_waters_init) then
                    j=j+1
                    k=1
                else
                    exit
                end if
            else if (k<this%reactive_zones(i)%num_non_flowing_species) then
                k=k+1
            else if (j<this%num_target_waters_init) then
                call append_int_1D_array(tw_indices,j)
                j=j+1
            else
                call append_int_1D_array(tw_indices,j)
                exit
            end if
        else if (associated(this%target_waters_init(j)%gas_chemistry)) then
            call this%target_waters_init(j)%gas_chemistry%reactive_zone%is_nf_species_in_react_zone(this%reactive_zones(i)%non_flowing_species(k),flag,nf_sp_ind)
            if (flag==.false.) then
                if (j<this%num_target_waters_init) then
                    j=j+1
                    k=1
                else
                    exit
                end if
            else if (k<this%reactive_zones(i)%num_non_flowing_species) then
                k=k+1
            else if (j<this%num_target_waters_init) then
                call append_int_1D_array(tw_indices,j)
                j=j+1
            else
                call append_int_1D_array(tw_indices,j)
                exit
            end if
        else if (j<this%num_target_waters_init) then
            j=j+1
        else
            exit
        end if
    end do
end subroutine