!> Gaines-Thomas convention module
module Gaines_Thomas_m
    use exch_sites_conv_m
    implicit none
    save
    type, public, extends(exch_sites_conv_c) :: Gaines_Thomas_c
        
    contains
        procedure, public :: compute_log_act_coeff_ads_cat=>compute_log_act_coeff_Gaines_Thomas
    end type
    
    interface
        subroutine compute_log_act_coeff_Gaines_Thomas(this,exchangeable_cation,CEC,log_act_coeff)
            import Gaines_Thomas_c
            import species_c
            implicit none
            class(Gaines_Thomas_c) :: this
            class(species_c), intent(in) :: exchangeable_cation
            real(kind=8), intent(in) :: CEC
            real(kind=8), intent(out) :: log_act_coeff
        end subroutine
        

    end interface
    
    contains
    
      
end module