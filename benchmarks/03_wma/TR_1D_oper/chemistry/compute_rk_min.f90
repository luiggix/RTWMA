!! Computes mineral kinetic reaction rate
!subroutine compute_rk_min(this)
!    use local_reactions_m
!    implicit none
!    class(local_min_kin_reaction_c) :: this
!    !real(kind=8), intent(in) :: act_cat(:) ! activities of catalysers (dim=num_cat)
!    !real(kind=8), intent(in) :: react_surf
!    !real(kind=8), intent(in) :: temp ! Kelvin
!    !real(kind=8), intent(out) :: rk
!    
!    integer(kind=4) :: zeta,j,i
!    real(kind=8) :: prod
!    real(kind=8), parameter :: R=8.31446261815324 ! J/(mol*K)
!    
!    this%rk=0d0
!    do j=1,this%min_kin_react%params%num_terms
!        prod=1d0
!        do i=1,this%min_kin_react%params%num_cat
!            prod=prod*this%aq_chem%activities(this%aq_species_ind(this%min_kin_react%params%cat_indices(i)))**this%min_kin_react%params%p(j,i)
!        end do
!        this%rk=this%rk+this%min_kin_react%params%k(j)*((this%saturation**this%min_kin_react%params%theta(j))-1d0)**this%min_kin_react%params%eta(j)
!    end do
!    if (this%saturation>1d0) then
!        zeta=1
!    else if (this%saturation<1d0) then
!        zeta=-1
!    end if
!    this%rk=this%rk*zeta
!    this%rk=this%rk*this%aq_chem%solid_chemistry%react_surfaces(this%min_ind)*exp(-this%min_kin_react%params%act_energy/(R*this%aq_chem%solid_chemistry%temp))
!end subroutine