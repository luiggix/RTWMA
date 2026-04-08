!> Computes concentration of secondary species from concentration of primary species explicitly using mass action law
!!> We assume the activity coefficients are ideal
!!> This subroutine is meant to be used when there are aqueous and solid primary species 
subroutine compute_c2nc_from_c1_expl(this)
    use aqueous_chemistry_m
    implicit none
    class(aqueous_chemistry_c) :: this
    
    integer(kind=4) :: n_nc2_aq,n_e,n_p_aq
    real(kind=8), allocatable :: log_c2nc(:),c1(:),log_gamma1(:),log_gamma2nc(:),log_gamma_exch(:)
    
    n_p_aq=this%speciation_alg%num_aq_prim_species
    n_e=this%speciation_alg%num_eq_reactions
    n_nc2_aq=this%speciation_alg%num_aq_sec_var_act_species
    
    c1=this%get_c1_exch() !> we get primary concentartions
        
!> Mass action law to compute log_10 secondary variable activity concentrations
    log_c2nc=matmul(this%speciation_alg%Se_nc_1_star,log10(c1))+this%speciation_alg%logK_star
!> We update secondary variable activity species concentrations
    this%concentrations(n_p_aq+1:this%speciation_alg%num_aq_var_act_species)=10**(log_c2nc(1:n_nc2_aq))
    this%solid_chemistry%concentrations(this%solid_chemistry%reactive_zone%num_minerals+2:this%solid_chemistry%reactive_zone%num_solids)=10**(log_c2nc(n_nc2_aq+1:n_e))
 end subroutine