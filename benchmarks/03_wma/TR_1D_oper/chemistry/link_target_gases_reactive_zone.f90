subroutine link_target_gases_reactive_zone(this,i,tar_gas_indices)
    use chemistry_Lagr_m
    use vectors_m
    use array_ops_m
    implicit none
    
    class(chemistry_c) :: this
    integer(kind=4), intent(in) :: i !> reactive zone index
    integer(kind=4), intent(out), allocatable :: tar_gas_indices(:) !> indices of target gases associated to i_th reactive zone
    
    integer(kind=4) :: j,k,n,nf_sp_ind
    logical :: flag
    real(kind=8), parameter :: eps=1d-12
    
    if (this%num_target_gases==0) then !> chapuza
        allocate(tar_gas_indices(0))
    else
        j=1
        k=1
        allocate(tar_gas_indices(0))
        do
            call this%target_gases(j)%reactive_zone%is_nf_species_in_react_zone(this%reactive_zones(i)%non_flowing_species(k),flag,nf_sp_ind)
            if (flag==.false.) then
                if (j<this%num_target_gases) then
                    j=j+1
                    k=1
                else
                    exit
                end if
            else if (k<this%reactive_zones(i)%num_non_flowing_species) then
                k=k+1
            else if (j<this%num_target_gases) then
                call append_int_1D_array(tar_gas_indices,j)
                j=j+1
            else
                call append_int_1D_array(tar_gas_indices,j)
                exit
            end if
        end do
    end if
end subroutine