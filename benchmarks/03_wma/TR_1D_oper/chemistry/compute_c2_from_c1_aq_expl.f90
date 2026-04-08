!> Computes concentration of secondary species from concentration of primary species explicitly using mass action law
!! We assume ideal activity coefficients
!! This subroutine is meant to be used only when all primary species are aqueous
subroutine compute_c2_from_c1_aq_ideal(this)
    use aqueous_chemistry_m
    implicit none
    class(aqueous_chemistry_c) :: this
    
    integer(kind=4) :: n_p,n_sec_aq,n_nc_aq
    real(kind=8), allocatable :: log_c2(:)
    
    n_p=this%speciation_alg%num_prim_species
    n_sec_aq=this%speciation_alg%num_sec_aq_species
            
    log_c2=matmul(this%speciation_alg%Se_1_star,log10(this%concentrations(1:n_p)))+this%speciation_alg%logK_tilde !> mass action law
    call this%set_conc_sec_aq_species(10**(log_c2(1:n_sec_aq)))
 end subroutine