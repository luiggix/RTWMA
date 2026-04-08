!!> Computes aqueous component concentrations after iteration of WMA-EE method in equilibrium-kinetic chemical system
!subroutine water_mixing_iter_EI_tpt_EE_rk_eq_kin(this,mixing_ratios,mixing_waters,rk_mat,porosity,Delta_t)
!    use aqueous_chemistry_m
!    implicit none
!    class(aqueous_chemistry_c) :: this
!    !integer(kind=4), intent(in) :: ind                              !> index of object "this" in mixing ratios array
!    real(kind=8), intent(in) :: mixing_ratios(:,:)
!    class(aqueous_chemistry_c), intent(in) :: mixing_waters(:)
!    real(kind=8), intent(in), optional :: rk_mat(:,:) !> rk matrix
!    real(kind=8), intent(in), optional :: porosity
!    real(kind=8), intent(in), optional :: Delta_t !> time step
!            
!    real(kind=8), allocatable :: c1_init(:),conc_comp_old(:),um_tilde(:),conc_comp_react(:),r_vec(:)
!    integer(kind=4) :: i,n,n_nc_aq,n_p_aq,niter,n_mix_rat
!    logical :: CV_flag
!    
!    n_p_aq=this%speciation_alg%num_aq_prim_species
!    n_mix_rat=size(mixing_ratios,2) !> chapuza
!    
!    allocate(um_tilde(n_p_aq))
!    um_tilde=this%compute_u_tilde_aq_chem(mixing_ratios(:,1),mixing_waters) !> transport part
!    allocate(conc_comp_react(n_p_aq))
!    call this%reaction_iteration_EI_tpt_EE_rk_eq_kin_aq_chem(mixing_ratios(:,n_mix_rat),rk_mat,porosity,Delta_t,conc_comp_react) !> chemical part
!    this%conc_comp=um_tilde+conc_comp_react !> we sum both parts
!!> Speciation
!    call this%compute_c_nc_from_u_aq_Newton(niter,CV_flag)
!    call this%check_conc_aq_var_act_species()
!    call this%check_act_aq_species()
!end subroutine