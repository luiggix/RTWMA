!> Computes kinetic reaction rates associated to aqueous chemistry object
subroutine compute_rk_Jac_rk_anal(this,drk_dc)
    use aqueous_chemistry_m
    implicit none
!> Arguments
    class(aqueous_chemistry_c) :: this !> aqueous chemistry object
    real(kind=8), intent(out) :: drk_dc(:,:) !> Jacobian (must be allocated)
!> Variables
    integer(kind=4) :: i,n,niter,rk_ind,l
    integer(kind=4), allocatable :: indices(:)
    real(kind=8), allocatable :: drk_dc_loc(:),conc(:),act_cat(:)
    real(kind=8) :: saturation
!> We compute linear kinetic reaction rates
    do i=1,this%chem_syst%num_lin_kin_reacts
        !allocate(conc_kin(this%chem_syst%num_species),kin_ind(this%chem_syst%num_species))
        call this%chem_syst%lin_kin_reacts(i)%compute_rk_lin(this%concentrations,this%rk(i))
        call this%chem_syst%lin_kin_reacts(i)%compute_drk_dc_lin(this%concentrations,this%rk(i),drk_dc(i,:))
        !deallocate(conc_kin,kin_ind)
    end do
!> We compute mineral kinetic reaction rates
    do i=1,this%chem_syst%num_min_kin_reacts
        indices=this%chem_syst%min_kin_reacts(i)%indices_aq_phase
        allocate(drk_dc_loc(SIZE(indices)))
        saturation=this%compute_saturation_min(this%chem_syst%min_kin_reacts(i))
        call this%chem_syst%min_kin_reacts(i)%compute_rk_mineral(this%activities(this%chem_syst%min_kin_reacts(i)%params%cat_indices),saturation,this%solid_chemistry%react_surfaces(i),this%solid_chemistry%temp,this%rk(this%chem_syst%num_lin_kin_reacts+i))
        call this%chem_syst%min_kin_reacts(i)%compute_drk_dc_mineral(this%concentrations(indices),this%activities(this%chem_syst%min_kin_reacts(i)%params%cat_indices),saturation,this%solid_chemistry%react_surfaces(i),this%solid_chemistry%temp,drk_dc_loc)
        drk_dc(this%chem_syst%num_lin_kin_reacts+i,indices)=drk_dc_loc !> chapuza
        deallocate(indices,drk_dc_loc)
    end do
!> We compute Monod reaction rates
    do i=1,this%chem_syst%num_redox_kin_reacts
        indices=this%chem_syst%redox_kin_reacts(i)%indices_aq_phase
        allocate(drk_dc_loc(SIZE(indices)))
        !call this%chem_syst%redox_kin_reacts(i)%get_conc_kin_Monod(this%chem_syst%aq_phase%aq_species,this%concentrations,conc_kin,kin_ind)
        call this%chem_syst%redox_kin_reacts(i)%compute_rk_Monod(this%concentrations(indices),this%rk(this%chem_syst%num_lin_kin_reacts+this%chem_syst%num_min_kin_reacts+i))
        call this%chem_syst%redox_kin_reacts(i)%compute_drk_dc_Monod(this%concentrations(indices),this%rk(this%chem_syst%num_lin_kin_reacts+this%chem_syst%num_min_kin_reacts+i),drk_dc_loc)
        drk_dc(this%chem_syst%num_lin_kin_reacts+this%chem_syst%num_min_kin_reacts+i,indices)=drk_dc_loc !> chapuza
        deallocate(indices,drk_dc_loc)
    end do
end subroutine