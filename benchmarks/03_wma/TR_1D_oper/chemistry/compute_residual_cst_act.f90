!> Computes residual of component concentrations
!!> The component matrix INCLUDES constant activity species
!!> residual: \f$ res = U*c - u \f$
subroutine compute_residual_cst_act(this,conc_comp,conc,residual)
    use aqueous_chemistry_m
    implicit none
    !> Variables
    class(aqueous_chemistry_c), intent(in) :: this
    real(kind=8), intent(in) :: conc_comp(:)
    real(kind=8), intent(in) :: conc(:)
    real(kind=8), intent(out) :: residual(:) !> residual of component concentrations
    
    !> Process
    residual=matmul(this%speciation_alg%comp_mat_cst_act,conc) - conc_comp
end subroutine