!> This function computes aqueous component concentrations after mixing (u_tilde)
function compute_u_tilde_aq_chem(this,c_tilde) result(u_tilde)
    use aqueous_chemistry_m
    implicit none
!> Arguments
    class(aqueous_chemistry_c), intent(in) :: this
    real(kind=8), intent(in) :: c_tilde(:) !> concentration of "mobile" species after mixing
    real(kind=8), allocatable :: u_tilde(:) !> component concentrations after mixing
!> Variables
    integer(kind=4) :: i !> index aqueous components
    integer(kind=4) :: j !> index mixing waters
!> Process
    u_tilde=matmul(this%speciation_alg%comp_mat,c_tilde)
end function