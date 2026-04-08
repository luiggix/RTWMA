!> Updates concentration aqueous primary species in iterative method
subroutine update_conc_aq_prim_species(this,Delta_c1)
    use aqueous_chemistry_m
    implicit none
    class(aqueous_chemistry_c) :: this
    real(kind=8), intent(inout) :: Delta_c1(:) !> must be already allocated
    
    integer(kind=4) :: i,n_p_aq
    real(kind=8), allocatable :: c1_old(:)
    
    n_p_aq=this%speciation_alg%num_aq_prim_species
    
    if (n_p_aq/=size(Delta_c1) .and. this%speciation_alg%num_aq_prim_species/=size(Delta_c1)) error stop "Dimension error in update_conc_aq_prim_species"
    
    c1_old=this%concentrations(1:n_p_aq)
    do i=1,n_p_aq
        if (this%concentrations(i)+Delta_c1(i)<=this%CV_params%control_factor*this%concentrations(i)) then
            this%concentrations(i)=this%CV_params%control_factor*this%concentrations(i)
        else if (this%concentrations(i)+Delta_c1(i)>=this%concentrations(i)/this%CV_params%control_factor) then
            this%concentrations(i)=this%concentrations(i)/this%CV_params%control_factor
        else
            this%concentrations(i)=this%concentrations(i)+Delta_c1(i)
        end if
        Delta_c1(i)=this%concentrations(i)-c1_old(i)
    end do
end subroutine