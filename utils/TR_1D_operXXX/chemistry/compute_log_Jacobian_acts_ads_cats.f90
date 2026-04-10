!> Computes Jacobian log_10 activity coefficients surface complexes with respect to log_10 concentrations
subroutine compute_log_Jacobian_act_coeffs_ads_cats(this,log_act_coeffs,log_Jacobian_act_coeffs)
    use surf_compl_m
    implicit none
    class(cat_exch_c) :: this
    real(kind=8), intent(in) :: log_act_coeffs(:) !> log_10 activity coefficients surface complexes
    real(kind=8), intent(out) :: log_Jacobian_act_coeffs(:,:) !> (must be allocated)
    
    log_Jacobian_act_coeffs=0d0 !> chapuza
end subroutine