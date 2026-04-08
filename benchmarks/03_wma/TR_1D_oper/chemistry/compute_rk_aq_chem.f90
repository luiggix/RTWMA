!> Computes kinetic reaction rates associated to aqueous chemistry object
subroutine compute_rk_aq_chem(this)
    use aqueous_chemistry_m
    implicit none
!> Arguments
    class(aqueous_chemistry_c) :: this !> aqueous chemistry object
!> Variables
    integer(kind=4) :: i,n,niter,rk_ind,l
    integer(kind=4), allocatable :: kin_ind(:),species_indices(:)
    real(kind=8), allocatable :: conc_kin(:),conc(:),act_cat(:)
    real(kind=8) :: saturation
!> We compute linear kinetic reaction rates
    do i=1,this%chem_syst%num_lin_kin_reacts
        allocate(conc_kin(this%chem_syst%num_species),kin_ind(this%chem_syst%num_species))
        call this%chem_syst%lin_kin_reacts(i)%compute_rk_lin(conc_kin,this%rk(i))
        deallocate(conc_kin,kin_ind)
    end do
!> We compute mineral kinetic reaction rates
    do i=1,this%chem_syst%num_min_kin_reacts
        saturation=this%compute_saturation_min(this%chem_syst%min_kin_reacts(i))
        call this%chem_syst%min_kin_reacts(i)%compute_rk_mineral(this%activities(this%chem_syst%min_kin_reacts(i)%params%cat_indices),saturation,this%solid_chemistry%react_surfaces(i),this%solid_chemistry%temp,this%rk(this%chem_syst%num_lin_kin_reacts+i))
    end do
!> We compute Monod reaction rates
    do i=1,this%chem_syst%num_redox_kin_reacts
        call this%chem_syst%redox_kin_reacts(i)%compute_rk_Monod(this%concentrations(this%chem_syst%redox_kin_reacts(i)%indices_aq_phase),this%rk(this%chem_syst%num_lin_kin_reacts+this%chem_syst%num_min_kin_reacts+i))
    end do
end subroutine