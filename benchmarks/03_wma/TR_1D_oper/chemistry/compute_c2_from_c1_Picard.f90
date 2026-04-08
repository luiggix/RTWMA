!> Computes concentration of secondary species from concentration of primary species using Picard method and mass action law
!! We assume the initial guess of secondary aqueous concentrations is already set in the aquoeus chemnistry object
subroutine compute_c2_from_c1_Picard(this,c1,c2_ig,c2,niter,CV_flag)
    use vectors_m
    use aqueous_chemistry_m
    implicit none
!> Arguments
    class(aqueous_chemistry_c) :: this
    !real(kind=8), intent(in) :: act_ph(:) !> chapuza
    real(kind=8), intent(in) :: c1(:) !> chapuza (dim=n_p)
    real(kind=8), intent(in) :: c2_ig(:) !> chapuza (dim=n_eq)
    real(kind=8), intent(out) :: c2(:) !> chapuza (dim=n_eq)
    integer(kind=4), intent(out) :: niter !> number of iterations
    logical, intent(out) :: CV_flag
!> Variables
    integer(kind=4) :: n_p,n_sec_aq,n_e,n_p_aq,n_mins_eq
    real(kind=8), allocatable :: log_gamma1_old(:),log_gamma2_old(:),log_c2_new(:),c2_old(:),c2_new(:)
!> Pre-process
    CV_flag=.false.
    n_p=this%speciation_alg%num_prim_species
    n_p_aq=this%speciation_alg%num_aq_prim_species
    n_e=this%speciation_alg%num_eq_reactions
    n_sec_aq=this%speciation_alg%num_sec_aq_species
    n_mins_eq=THIS%chem_syst%num_minerals_eq
    
    allocate(log_gamma1_old(n_p),log_gamma2_old(n_e),log_c2_new(n_e),c2_new(n_e))
        
    niter=0
    
    call this%set_conc_sec_aq_species(c2_ig(1:n_sec_aq)) !> initial guess c2aq
    if (associated(this%gas_chemistry)) then !> chapuza
        call this%gas_chemistry%set_concentrations(c2_ig(n_sec_aq+n_mins_eq+1:n_e)) !> initial guess conc_gases
    end if
    c2_old=c2_ig
    c2_new=c2_old !> chapuza
    log_gamma1_old=0d0 !> chapuza
    log_gamma2_old=0d0 !> chapuza
    !log_gamma2_old=this%get_log_gamma2() !> chapuza
!> Process
    do
        niter=niter+1 !> we update number of iterations
        !call this%compute_molalities()
        call this%compute_ionic_act() !> we compute ionic activity
        !print *, this%ionic_act
        !call this%compute_salinity()
        !call this%compute_molarities()
        call this%aq_phase%compute_log_act_coeffs_aq_phase(this%ionic_act,this%params_aq_sol,this%log_act_coeffs) !> we compute log activity coefficients aqueous species
        call this%compute_activities_aq()
        call this%compute_log_act_coeff_wat()
        
        log_gamma1_old(1:n_p_aq)=this%log_act_coeffs(1:n_p_aq) !> we set log activity coefficients aqueous primary species
        log_gamma2_old(1:n_sec_aq)=this%log_act_coeffs(n_p_aq+1:this%aq_phase%num_species) !> we set log activity coefficients aqueous secondary species
        !print *, this%gas_chemistry%reactive_zone%gas_phase%num_var_act_species
        if (associated(this%gas_chemistry) .and. this%gas_chemistry%reactive_zone%gas_phase%num_var_act_species>0) then !> chapuza
            !call this%gas_chemistry%compute_vol_gas() !> we compute total volume of gas
            call this%gas_chemistry%compute_log_act_coeffs_gases()
            log_gamma2_old(n_sec_aq+n_mins_eq+1:n_e)=this%gas_chemistry%log_act_coeffs
        end if
        
        log_c2_new=matmul(this%speciation_alg%Se_1_star,log_gamma1_old+log10(c1))+this%speciation_alg%logK_tilde-log_gamma2_old !> mass action law
        c2_new=10**log_c2_new
        
        call this%update_conc_sec_aq_species(c2_new(1:n_sec_aq)) !> we update aqueous secondary species
        if (associated(this%gas_chemistry) .and. this%gas_chemistry%reactive_zone%gas_phase%num_var_act_species>0) then !> chapuza
            call this%gas_chemistry%update_conc_gases(c2_new(n_sec_aq+n_mins_eq+1:n_e)*this%volume) !> we update moles of gases
            !call this%gas_chemistry%compute_vol_gas() !> we compute total volume of gas
        end if
        if (inf_norm_vec_real((c2_new(1:n_sec_aq)-c2_old(1:n_sec_aq))/c2_old(1:n_sec_aq))<this%CV_params%rel_tol) then
            CV_flag=.true.
            exit !> CV reached
        else if (niter==this%CV_params%niter_max) then
            print *, inf_norm_vec_real((c2_new(1:n_sec_aq)-c2_old(1:n_sec_aq))/c2_old(1:n_sec_aq))
            error stop "Too many Picard iterations in speciation"
        else
            c2_old=c2_new
        end if
    end do
    c2=c2_new
    call this%compute_ionic_act() !> we compute ionic activity
    call this%aq_phase%compute_log_act_coeffs_aq_phase(this%ionic_act,this%params_aq_sol,this%log_act_coeffs) !> we compute log activity coefficients aqueous species
    call this%compute_activities_aq()
    call this%compute_log_act_coeff_wat()
    if (associated(this%gas_chemistry)) then !> chapuza
        !nullify(this%gas_chemistry)
        call this%gas_chemistry%compute_log_act_coeffs_gases() !> we compute log_10 activity coefficients of gases
        call this%gas_chemistry%compute_partial_pressures() !> we compute activities (ie. partial pressures)
        call this%gas_chemistry%compute_pressure()
    end if
 end subroutine