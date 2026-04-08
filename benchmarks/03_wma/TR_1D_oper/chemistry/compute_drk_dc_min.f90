!!> Computes local mineral kinetic reaction rate gradient
!subroutine compute_drk_dc_min(this,drk_dc)
!>    use local_reactions_m
!>    implicit none
!>    class(local_min_kin_reaction_c), intent(in) :: this
!>    !real(kind=8), intent(in) :: conc(:) !> c_j or [c1,c2nc]_j
!>    !integer(kind=4), intent(in) :: species_ind(:) !> indices of species relevant for rk
!>    !real(kind=8), intent(in) :: act_cat(:) !> activities catalysers
!>    !!real(kind=8), intent(in) :: rk
!>    !real(kind=8), intent(in) :: react_surf
!>    !real(kind=8), intent(in) :: temp !> Kelvin
!>    real(kind=8), intent(out) :: drk_dc(:) !> must be already allocated
!>    
!>    integer(kind=4) :: n_sp,i,l,j,k,zeta
!>    real(kind=8) :: rk_j,sum,prod
!>    real(kind=8), parameter :: R=8.31446261815324 !> J/(mol*K)
!>    
!>    drk_dc=0d0
!>    n_sp=size(this%aq_species_ind)
!>    do l=1,n_sp
!>        sum=0d0
!>        do j=1,this%min_kin_react%params%num_terms
!>            prod=1d0
!>            do i=1,this%min_kin_react%params%num_cat
!>                prod=prod*this%aq_chem%activities(this%aq_species_ind(this%min_kin_react%params%cat_indices(i)))**this%min_kin_react%params%p(j,i)
!>            end do
!>            sum=sum+prod*this%min_kin_react%params%theta(j)*(this%saturation**this%min_kin_react%params%theta(j))*this%min_kin_react%params%k(j)*this%min_kin_react%params%eta(j)*((this%saturation**this%min_kin_react%params%theta(j))-1d0)**(this%min_kin_react%params%eta(j)-1d0)
!>        end do
!>        drk_dc(this%aq_species_ind(l))=sum*this%min_kin_react%stoichiometry(l)/this%aq_chem%concentrations(this%aq_species_ind(l))
!>    end do
!>    if (this%saturation>1d0) then
!>        zeta=1
!>    else if (this%saturation<1d0) then
!>        zeta=-1
!>    end if
!>    drk_dc=drk_dc*zeta
!>    drk_dc=drk_dc*this%aq_chem%solid_chemistry%react_surfaces(this%min_ind)*exp(-this%min_kin_react%params%act_energy/(R*this%aq_chem%solid_chemistry%temp))
!end subroutine