!!> Computes variabl activity species concentrations after time iteration using WMA-Euler fully implicit in rk
!subroutine water_mixing_iter_EI_tpt_EfI_rk_eq_kin(this,mixing_ratios,mixing_waters,rk_mat,porosity,Delta_t)
!    use aqueous_chemistry_m
!    implicit none
!    class(aqueous_chemistry_c) :: this
!    !integer(kind=4), intent(in) :: ind                              !> index of object "this" in mixing ratios array
!    real(kind=8), intent(in) :: mixing_ratios(:,:)  !> first column: transport
!                                                    !> second column (if not fully implicit): reactions
!    class(aqueous_chemistry_c), intent(in) :: mixing_waters(:)
!    real(kind=8), intent(in), optional :: rk_mat(:,:) !> rk matrix
!    real(kind=8), intent(in), optional :: porosity
!    real(kind=8), intent(in), optional :: Delta_t !> time step
!            
!    real(kind=8), allocatable :: conc_old_old(:),conc_old(:),initial_guess(:)
!    integer(kind=4) :: i,n,n_nc_aq,n_p_aq,k,niter
!    real(kind=8) :: mu !> Newton initialisation parameter
!    logical :: CV_flag
!    
!    procedure(Newton_EI_tpt_EfI_rk_eq_kin_aq), pointer :: p_iterative_method=>null()
!    
!    n_p_aq=this%speciation_alg%num_aq_prim_species
!    n_nc_aq=this%speciation_alg%num_aq_var_act_species
!    
!    conc_old_old=this%concentrations(1:n_p_aq)
!    conc_old=this%concentrations
!    
!    allocate(initial_guess(n_p_aq))
!        
!    mu=0d0
!    !> aqui hay que poner un if para elegir método iterativo
!    p_iterative_method=>Newton_EI_tpt_EfI_rk_eq_kin_aq
!        do !> iterative loop
!            call initialise_iterative_method(conc_old_old,conc_old(1:n_p_aq),mu,initial_guess)
!            call this%set_conc_aq_prim_species(initial_guess)
!            call p_iterative_method(this,mixing_ratios,rk_mat,porosity,Delta_t,mixing_waters,niter,CV_flag)
!            if (CV_flag==0) then !> no CV
!                if (mu<1d0) then
!                    mu=mu+0.25
!                else
!                    print *, "Iterative method does not CV"
!                    error stop
!                end if
!            else 
!                exit
!            end if
!        end do
!    deallocate(conc_old,conc_old_old,initial_guess)
!    call this%compute_conc_comp_aq()
!end subroutine