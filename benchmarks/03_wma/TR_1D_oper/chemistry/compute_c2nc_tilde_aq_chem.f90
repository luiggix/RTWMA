!> This subroutine computes the concentration of secondary aqueous variable activity species after a mixing iteration
function compute_c2nc_tilde_aq_chem(this,mixing_ratios,mixing_waters) result(c2nc_tilde)
    use aqueous_chemistry_m
    implicit none
!> Arguments
    class(aqueous_chemistry_c), intent(in) :: this 
    real(kind=8), intent(in) :: mixing_ratios(:) !> the first element corresponds to argument "this", and the rest correspond to argument "mixing waters" (in the same order)
    class(aqueous_chemistry_c), intent(in) :: mixing_waters(:) !> dimension must be dim(mixing_ratios)-1
    real(kind=8), allocatable :: c2nc_tilde(:) !> concentration of secondary aqueous variable activity species
!> Variables
    integer(kind=4) :: i !> index aqueous secondary variable activity species
    integer(kind=4) :: j !> index mixing waters
!> Pre-process
    allocate(c2nc_tilde(this%speciation_alg%num_aq_sec_var_act_species))
!> Process
    do i=1,this%speciation_alg%num_aq_sec_var_act_species
        c2nc_tilde(i)=mixing_ratios(1)*this%concentrations(this%sec_var_act_species_indices(i))
        do j=1,size(mixing_waters)
            c2nc_tilde(i)=c2nc_tilde(i)+mixing_ratios(j+1)*mixing_waters(j)%concentrations(this%sec_var_act_species_indices(i))
        end do
    end do
end function