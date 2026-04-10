 !> Computes log_10 activity coefficients aqueous species variable activity
subroutine compute_log_act_coeffs_aq_phase(this,ionic_act,params_aq_sol,log_act_coeffs)
    use aq_phase_m
    implicit none
    class(aq_phase_c) :: this
    real(kind=8), intent(in) :: ionic_act !> ionic activity
    class(params_aq_sol_t), intent(in) :: params_aq_sol
    real(kind=8), intent(out) :: log_act_coeffs(:) !> log_10 activity coefficients (must be already allocated)
    
    integer(kind=4) :: i
    
    procedure(Debye_Huckel_restr), pointer :: p_compute_log_act_coeff=>null()
        
    !if (ionic_act<1d-2) then
    !    p_compute_log_act_coeff=>Debye_Huckel_restr
    !else if (ionic_act<=1d-1) then
    !    p_compute_log_act_coeff=>Debye_Huckel_ampl
    !else if (ionic_act<=7d-1) then
    !    p_compute_log_act_coeff=>Davies
    !else
    !    error stop
    !end if
    !p_compute_log_act_coeff=>Davies !> lo impongo (chapuza)
    do i=1,this%num_species-this%wat_flag
        !call p_compute_log_act_coeff(this%aq_species(i),ionic_act,log_act_coeffs(i))
        log_act_coeffs(this%ind_diss_solids(i))=-this%aq_species(this%ind_diss_solids(i))%params_act_coeff%alpha*(params_aq_sol%A*this%aq_species(this%ind_diss_solids(i))%valence**2)*sqrt(ionic_act)/(1d0+this%aq_species(this%ind_diss_solids(i))%params_act_coeff%beta*sqrt(ionic_act)) + this%aq_species(this%ind_diss_solids(i))%params_act_coeff%gamma*ionic_act
    end do
    !log_act_coeffs(this%ind_wat)
end subroutine