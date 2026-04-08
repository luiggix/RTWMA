!> Vanselow convention module
module Vanselow_m
    use exch_sites_conv_m
    implicit none
    save
    type, public, extends(exch_sites_conv_c) :: Vanselow_c
        
    contains
        procedure, public :: compute_log_act_coeff_ads_cat=>compute_log_act_coeff_Vanselow
    end type
    
    contains
        subroutine compute_log_act_coeff_Vanselow(this,exchangeable_cation,CEC,log_act_coeff)
        !> Activity of surface complex = molar fraction
            implicit none
            class(Vanselow_c) :: this
            class(species_c), intent(in) :: exchangeable_cation
            real(kind=8), intent(in) :: CEC
            real(kind=8), intent(out) :: log_act_coeff
            log_act_coeff=log10(exchangeable_cation%valence/CEC)
        end subroutine
    
    
    
      
end module