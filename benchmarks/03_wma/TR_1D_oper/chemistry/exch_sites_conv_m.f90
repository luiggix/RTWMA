!> Cation exchange convention superclass
module exch_sites_conv_m
    use species_m
    implicit none
    save
    type, public, abstract :: exch_sites_conv_c !> exchange sites convention
    contains
        procedure(compute_log_act_coeff_ads_cat), public, deferred :: compute_log_act_coeff_ads_cat
    end type
    
    abstract interface
        subroutine compute_log_act_coeff_ads_cat(this,exchangeable_cation,CEC,log_act_coeff)
            import exch_sites_conv_c
            import species_c
            implicit none
            class(exch_sites_conv_c) :: this
            class(species_c), intent(in) :: exchangeable_cation
            real(kind=8), intent(in) :: CEC
            real(kind=8), intent(out) :: log_act_coeff
        end subroutine
        

    end interface
    
    contains
    
      
end module