!> Computes Monod kinetic reaction rate gradient
!!> drk_dc=rk*dlogT/dc
!!>        =rk*k/(c*(k+c)) if TEA
!!>        =-rk/(k+c) if inhibitor
subroutine compute_drk_dc_Monod(this,conc,rk,drk_dc)
    use redox_kin_reaction_m
    implicit none
    class(redox_kin_c), intent(in) :: this
    real(kind=8), intent(in) :: conc(:) !> concentrations of relevant species
    real(kind=8), intent(in) :: rk
    real(kind=8), intent(out) :: drk_dc(:) !> must be allocated
    
    
    integer(kind=4) :: i,j,k,n,m,l,DOC_ind
    type(species_c), allocatable :: stoich_mat_react_zonec_species(:)
    type(species_c) :: DOC !> CH2O
    logical :: DOC_flag

    !DOC%name='ch2o'
    !call this%is_species_in_react(DOC,DOC_flag,DOC_ind)
    j=1
    k=1
    l=1
    drk_dc=0d0 !> chapuza
    if (this%params%n_inh>0) then
        do i=1,this%params%n_inh
            drk_dc(i)=-rk/(this%params%k_inh(i)+conc(i))
        end do
    end if
    do i=1,this%params%n_M
        drk_dc(this%params%n_inh+i)=rk*this%params%k_M(i)/(conc(this%params%n_inh+i)*(this%params%k_M(i)+conc(this%params%n_inh+i)))
    end do
    drk_dc(this%params%num_terms+1)=rk/conc(this%params%num_terms+1)
            !if (this%species(i)%name==this%params%TEAs(j)%name) then
                !drk_dc(i)=rk*this%params%k_M(j)/(conc(i)*(this%params%k_M(j)+conc(i)))
                !if (j<this%params%n_M) then
                !    j=j+1
                !end if
            !else if (this%species(i)%name==this%params%inhibitors(k)%name) then
                !drk_dc(i)=-rk/(this%params%k_inh(k)+conc(i))
                !if (k<this%params%n_inh) then
                !    k=k+1
                !end if
            !else if (this%species(i)%name=='ch2o(aq)') then
            !    drk_dc(i)=rk/conc(i)
            !else if (this%species(i)%name=='c5h7o2n') then
            !    drk_dc(i)=rk/conc(i)
            !else
            !    continue
            !end if
        !end do
    !else
    !    do i=1,this%num_species
    !        if (this%species(i)%name==this%params%TEAs(j)%name) then
    !            drk_dc(i)=rk*this%params%k_M(j)/(conc(i)*(this%params%k_M(j)+conc(i)))
    !            if (j<this%params%n_M) then
    !                j=j+1
    !            end if
    !        else if (this%species(i)%name=='ch2o(aq)') then
    !            drk_dc(i)=rk/conc(i)
    !        else if (this%species(i)%name=='c5h7o2n') then
    !            drk_dc(i)=rk/conc(i)
    !        else
    !            continue
    !        end if
    !    end do
    !end if
end subroutine