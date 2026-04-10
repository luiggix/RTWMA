!> Computes concentration of secondary species with variable activity from concentration of primary species explicitly using mass action law
!!  We assume the activity coefficients are ideal
!! This subroutine is meant to be used only when all primary species are aqueous
subroutine compute_c2nc_aq_from_c1_aq_expl(this)
    use aqueous_chemistry_m
    implicit none
    class(aqueous_chemistry_c) :: this
    
    integer(kind=4) :: n_p
    real(kind=8), allocatable :: log_c2nc(:)
    
    n_p=this%speciation_alg%num_prim_species
    
    log_c2nc=matmul(this%speciation_alg%Se_nc_1_star,log10(this%concentrations(1:n_p)))+this%speciation_alg%logK_star
    call this%update_conc_sec_aq_var_act_species(10**(log_c2nc))
 end subroutine