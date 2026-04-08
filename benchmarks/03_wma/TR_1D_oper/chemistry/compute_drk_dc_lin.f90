!> Computes linear kinetic reaction rate gradient
subroutine compute_drk_dc_lin(this,conc,rk,drk_dc)
    use lin_kin_reaction_m
    implicit none
    class(lin_kin_reaction_c), intent(in) :: this
    real(kind=8), intent(in) :: conc(:) !> conc=c_aq_j
    real(kind=8), intent(in) :: rk
    real(kind=8), intent(out) :: drk_dc(:) !> must be allocated
    
    drk_dc=0d0
    drk_dc(1)=this%lambda
    
end subroutine