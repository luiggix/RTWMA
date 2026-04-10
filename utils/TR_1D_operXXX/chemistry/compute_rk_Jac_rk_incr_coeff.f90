 !> Computes kinetic reaction rates associated to aqueous chemistry object
subroutine compute_rk_Jac_rk_incr_coeff(this,drk_dc)
    use aqueous_chemistry_m
    implicit none
!> Arguments
    class(aqueous_chemistry_c) :: this !> aqueous chemistry object
    real(kind=8), intent(out) :: drk_dc(:,:) !> Jacobian (must be allocated)
!> Variables
    integer(kind=4) :: i,j,niter,rk_ind,l
    integer(kind=4), allocatable :: kin_ind(:),species_indices(:)
    real(kind=8), allocatable :: rk_pert(:),c_pert(:)
    real(kind=8) :: saturation,saturation_pert

    type(matrix_int_c) :: indices_Monod,indices_min
    
    allocate(rk_pert(this%chem_syst%num_kin_reacts))
    
    drk_dc=0d0 !> chapuza
!> We compute linear kinetic reaction rates
    do i=1,this%chem_syst%num_lin_kin_reacts
        !allocate(conc_kin(this%chem_syst%num_species),kin_ind(this%chem_syst%num_species))
        call this%chem_syst%lin_kin_reacts(i)%compute_rk_lin(this%concentrations,this%rk(i))
    end do
!> We compute mineral kinetic reaction rates
    call indices_min%allocate_matrix(this%chem_syst%num_min_kin_reacts)
    do i=1,this%chem_syst%num_min_kin_reacts
        !call this%chem_syst%min_kin_reacts(i)%get_solid_chem_mineral(this%aq_phase%aq_species,this%activities,this%activities)
        indices_min%cols(i)%col_1=this%chem_syst%min_kin_reacts(i)%indices_aq_phase
        indices_min%cols(i)%dim=SIZE(indices_min%cols(i)%col_1) !> chapuza
        saturation=this%compute_saturation_min(this%chem_syst%min_kin_reacts(i))
        call this%chem_syst%min_kin_reacts(i)%compute_rk_mineral(this%activities(this%chem_syst%min_kin_reacts(i)%params%cat_indices),saturation,this%solid_chemistry%react_surfaces(i),this%solid_chemistry%temp,this%rk(this%chem_syst%num_lin_kin_reacts+i))
    end do
!> We compute Monod reaction rates
    call indices_Monod%allocate_matrix(this%chem_syst%num_redox_kin_reacts)
    do i=1,this%chem_syst%num_redox_kin_reacts
        !indices_Monod%cols(i)%col_1=this%get_indices_reaction(this%chem_syst%redox_kin_reacts(i)) !> chapuza (siempre seran los mismos indices)
        indices_Monod%cols(i)%col_1=this%chem_syst%redox_kin_reacts(i)%indices_aq_phase
        indices_Monod%cols(i)%dim=SIZE(indices_Monod%cols(i)%col_1) !> chapuza
        call this%chem_syst%redox_kin_reacts(i)%compute_rk_Monod(this%concentrations(indices_Monod%cols(i)%col_1),this%rk(this%chem_syst%num_lin_kin_reacts+this%chem_syst%num_min_kin_reacts+i))
    end do
!> We compute perturbed mineral kinetic reaction rates
    do i=1,this%chem_syst%num_min_kin_reacts
        do j=1,indices_min%cols(i)%dim
         !> chapuza
            this%concentrations(indices_min%cols(i)%col_1(j))=this%concentrations(indices_min%cols(i)%col_1(j))+this%CV_params%eps
            call this%compute_ionic_act()
            call this%aq_phase%compute_log_act_coeffs_aq_phase(this%ionic_act,this%params_aq_sol,this%log_act_coeffs)
            call this%compute_activities()

            saturation_pert=this%compute_saturation_min(this%chem_syst%min_kin_reacts(i))
            call this%chem_syst%min_kin_reacts(i)%compute_rk_mineral(this%activities(this%chem_syst%min_kin_reacts(i)%params%cat_indices),saturation_pert,this%solid_chemistry%react_surfaces(i),this%solid_chemistry%temp,rk_pert(this%chem_syst%num_lin_kin_reacts+i))
            drk_dc(i,indices_min%cols(i)%col_1(j))=(rk_pert(this%chem_syst%num_lin_kin_reacts+i)-this%rk(this%chem_syst%num_lin_kin_reacts+i))/this%CV_params%eps
        !> chapuza
            this%concentrations(indices_min%cols(i)%col_1(j))=this%concentrations(indices_min%cols(i)%col_1(j))-this%CV_params%eps
            call this%compute_ionic_act()
            call this%aq_phase%compute_log_act_coeffs_aq_phase(this%ionic_act,this%params_aq_sol,this%log_act_coeffs)
            call this%compute_activities()
        end do
    end do
!> We compute perturbed Monod reaction rates
    do i=1,this%chem_syst%num_redox_kin_reacts
        do j=1,indices_Monod%cols(i)%dim !> chapuza
            c_pert=this%concentrations(indices_Monod%cols(i)%col_1)
            c_pert(j)=c_pert(j)+this%CV_params%eps
            call this%chem_syst%redox_kin_reacts(i)%compute_rk_Monod(c_pert,rk_pert(this%chem_syst%num_lin_kin_reacts+this%chem_syst%num_min_kin_reacts+i))
            drk_dc(i,indices_Monod%cols(i)%col_1(j))=(rk_pert(this%chem_syst%num_lin_kin_reacts+this%chem_syst%num_min_kin_reacts+i)-this%rk(this%chem_syst%num_lin_kin_reacts+this%chem_syst%num_min_kin_reacts+i))/this%CV_params%eps
        end do
    end do
end subroutine