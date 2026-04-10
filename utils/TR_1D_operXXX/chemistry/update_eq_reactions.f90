!> Updates equilibrium reactions attribute in reactive zone object
subroutine update_eq_reactions(this,old_eq_reacts_ind)
    use reactive_zone_Lagr_m
    implicit none
    class(reactive_zone_c) :: this
    integer(kind=4), intent(in) :: old_eq_reacts_ind(:) !> indices of old equilibrium reactions
    
    integer(kind=4) :: i,j,k
    type(eq_reaction_c), allocatable :: aux(:) !> auxiliary variable equilibrium reactions
    logical :: flag
    aux=this%eq_reactions
    deallocate(this%eq_reactions)
    call this%update_num_eq_reacts(size(old_eq_reacts_ind))
    call this%allocate_eq_reactions()
    j=1
    k=1
    
    do 
        i=1
        flag=.true.
        do
            if (old_eq_reacts_ind(i)==j) then
                flag=.false.
                exit
            else if (i<size(old_eq_reacts_ind)) then
                i=i+1
            else
                exit
            end if
        end do
        if (flag==.true.) then
            this%eq_reactions(k)=aux(j)
            if (k<this%num_eq_reactions) then
                k=k+1
            end if
        end if
        if (j<size(aux)) then
            j=j+1
        end if
    end do
    
end subroutine
