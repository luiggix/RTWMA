!> Computes mineral kinetic reaction rate gradient
subroutine compute_drk_dc_mineral(this,conc,act_cat,saturation,react_surf,temp,drk_dc)
    use kin_mineral_m
    implicit none
    class(kin_mineral_c), intent(in) :: this
    real(kind=8), intent(in) :: conc(:) !> concentration aqueous species reaction
    !integer(kind=4), intent(in) :: aq_species_ind(:) !> indices of aqueous species of reaction
    real(kind=8), intent(in) :: act_cat(:) !> activities catalysers
    real(kind=8), intent(in) :: saturation
    real(kind=8), intent(in) :: react_surf !> reactive surface
    real(kind=8), intent(in) :: temp !> temperature [K]
    real(kind=8), intent(out) :: drk_dc(:) !> must be already allocated
    
    integer(kind=4) :: n_sp,i,l,j,k,zeta
    real(kind=8) :: rk_j,sum,prod
    real(kind=8), parameter :: R=8.31446261815324 !> J/(mol*K)
    
    drk_dc=0d0
    n_sp=size(conc)
    do l=1,n_sp
        sum=0d0
        do j=1,this%params%num_par_reacts
            prod=1d0
            do i=1,this%params%num_cat
                prod=prod*act_cat(i)**this%params%p(j,i)
            end do
            sum=sum+prod*this%params%theta(j)*(saturation**this%params%theta(j))*this%params%k(j)*this%params%eta(j)*((saturation**this%params%theta(j))-1d0)**(this%params%eta(j)-1d0)
        end do
        drk_dc(l)=sum*this%stoichiometry(l)/conc(l)
    end do
    if (saturation>1d0) then
        zeta=1
    else if (saturation<1d0) then
        zeta=-1
    end if
    drk_dc=drk_dc*zeta
    drk_dc=drk_dc*react_surf*exp(-this%params%act_energy/(R*temp))
end subroutine