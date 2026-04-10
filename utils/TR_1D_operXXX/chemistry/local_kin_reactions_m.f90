!> Local mineral kinetic reaction subclass
module local_min_kin_reaction_m
    use aq_phase_m
    use kin_mineral_m
    implicit none
    save
    type, public :: local_min_kin_reaction_c
        real(kind=8) :: saturation !> if $=1$ then reaction is in equilibrium
        !class(aqueous_chemistry_c), pointer :: aq_chem
        class(kin_mineral_c), pointer :: min_kin_react
        integer(kind=4), allocatable :: aq_species_ind(:) !> indices of aqueous species of reaction in aqueous phase
        integer(kind=4) :: min_ind !> index of mineral in solid chemistry
        real(kind=8) :: rk !> reaction rate
    contains
        procedure, public :: compute_saturation
        procedure, public :: compute_rk_min
        procedure, public :: compute_drk_dc_min
        procedure, public :: set_aq_species_ind
        procedure, public :: set_min_ind
    end type
    
    interface
        subroutine compute_rk_min(this)
            import local_min_kin_reaction_c
            implicit none
            class(local_min_kin_reaction_c) :: this
            !real(kind=8), intent(in) :: act_cat(:)
            !real(kind=8), intent(in) :: react_surf
            !real(kind=8), intent(in) :: temp
            !real(kind=8), intent(out) :: rk
        end subroutine
        
        subroutine compute_drk_dc_min(this,drk_dc)
            import local_min_kin_reaction_c
            implicit none
            class(local_min_kin_reaction_c), intent(in) :: this
            !real(kind=8), intent(in) :: conc(:)
            !integer(kind=4), intent(in) :: species_ind(:) !> indices of species relevant for rk
            !real(kind=8), intent(in) :: act_cat(:) !> activities catalysers
            !real(kind=8), intent(in) :: rk
            !real(kind=8), intent(in) :: react_surf
            !real(kind=8), intent(in) :: temp !> Kelvin
            real(kind=8), intent(out) :: drk_dc(:) !> must be already allocated
        end subroutine
    end interface
    
    contains
        subroutine compute_saturation(this,activities)
            implicit none
            class(local_min_kin_reaction_c) :: this
            real(kind=8), intent(in) :: activities(:) !> activities of reaction species
            integer(kind=4) :: i
            real(kind=8) :: IAP
            IAP=1d0
            do i=1,this%min_kin_react%num_species-1
                IAP=IAP*activities(i)**this%min_kin_react%stoichiometry(i)
            end do
            this%saturation=IAP/this%min_kin_react%eq_cst
        end subroutine
        
        subroutine set_aq_species_ind(this,aq_phase)
            implicit none
            class(local_min_kin_reaction_c) :: this
            class(aq_phase_c), intent(in) :: aq_phase
        end subroutine
        
        subroutine set_min_ind(this)
            implicit none
            class(local_min_kin_reaction_c) :: this
        end subroutine
end module