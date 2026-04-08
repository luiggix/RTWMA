!> Computes chemical part of aqueous component concentrations for a reactive mixing Euler explicit iteration assuming there are equilibrium and kinetic reactions
subroutine reaction_iteration_EE_eq_kin_aq_chem(this,porosity,Delta_t,conc_comp_react)
    use aqueous_chemistry_m
    implicit none
!> Arguments
    class(aqueous_chemistry_c) :: this
    real(kind=8), intent(in) :: porosity !> porosity
    real(kind=8), intent(in) :: Delta_t !> time step
    real(kind=8), intent(out) :: conc_comp_react(:) !> reaction part of component concentrations (must be already allocated)
!> Process
    call this%compute_rk_aq_chem() !> we compute kinetic reactions rates
    conc_comp_react=Delta_t*matmul(this%U_SkT_prod,this%rk)/porosity
end subroutine