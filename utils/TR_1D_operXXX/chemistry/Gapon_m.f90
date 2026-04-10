!> Gapon convention module
module Gapon_m
    use exch_sites_conv_m
    implicit none
    save
    type, public, extends(exch_sites_conv_c) :: Gapon_c
        
    contains
        procedure, public :: compute_log_act_coeff_ads_cat=>compute_log_act_coeff_Gapon
    end type
    
    contains
        subroutine compute_log_act_coeff_Gapon(this,exchangeable_cation,CEC,log_act_coeff)
        !> Activity of surface complex = molar fraction total number exchange sites
            implicit none
            class(Gapon_c) :: this
            class(species_c), intent(in) :: exchangeable_cation
            real(kind=8), intent(in) :: CEC
            real(kind=8), intent(out) :: log_act_coeff
            
            log_act_coeff=exchangeable_cation%valence*log10(exchangeable_cation%valence/CEC)

        end subroutine
        
    
    
      
end module