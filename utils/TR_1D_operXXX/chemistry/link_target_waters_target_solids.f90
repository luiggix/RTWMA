subroutine link_target_waters_target_solids(this,tar_sol_indices,tar_wat_indices)
    use chemistry_Lagr_m
    use array_ops_m
    use vectors_m
    implicit none
    class(chemistry_c) :: this
    integer(kind=4), intent(in) :: tar_sol_indices(:) !> target solid indices
    integer(kind=4), intent(inout), allocatable :: tar_wat_indices(:) !> indices of target waters associated to target solids
    
    integer(kind=4) :: i,j,k,n_tw_sol
    logical :: flag
    real(kind=8), parameter :: eps=1d-12

    if (size(tar_sol_indices)>0) then
    
        allocate(tar_wat_indices(size(tar_sol_indices))) !> we assume bijection
    
    
        i=1
        j=1
        do
            if (inf_norm_vec_real(this%target_waters_init(j)%solid_chemistry%concentrations-this%target_solids(tar_sol_indices(i))%concentrations)<eps) then
                tar_wat_indices(i)=j
                if (j<this%num_target_waters_init) then
                    j=j+1
                end if
                if (i<size(tar_sol_indices)) then
                    i=i+1
                else
                    exit
                end if
            else if (j<this%num_target_waters_init) then
                j=j+1
            !else if (i<size(tar_sol_indices)) then
            !>    i=i+1
            !>    !i=1
            else
                error stop "Error in link_target_waters_target_solids"
            end if
        end do
    end if
end subroutine