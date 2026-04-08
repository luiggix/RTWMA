!> Computes logarithm activity coefficients adsorbed cations
subroutine compute_log_act_coeffs_ads_cats(this,log_act_coeffs)
    use surf_compl_m
    implicit none
    class(cat_exch_c) :: this
    real(kind=8), intent(out) :: log_act_coeffs(:) !> must be allocated
    
    integer(kind=4) :: i
    
    log_act_coeffs=0d0
    
    !do i=1,size(log_act_coeffs)
        !call this%convention%compute_log_act_coeff_ads_cat(this%exch_cats(i),this%CEC,log_act_coeffs(i))
    !end do
end subroutine