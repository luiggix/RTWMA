!> Computes Jacobian of secondary variable activity species concentrations
!> \f$\partial c_{2,nc} / \partial c_1\f$
subroutine compute_dc2nc_dc1(this,c1,c2nc,out_prod,dc2nc_dc1)
    use aqueous_chemistry_m
    use metodos_sist_lin_m
    implicit none
!> Arguments
    class(aqueous_chemistry_c) :: this
    real(kind=8), intent(in) :: c1(:) !> chapuza
    real(kind=8), intent(in) :: c2nc(:) !> chapuza
    real(kind=8), intent(in) :: out_prod(:,:) !> outer product between d_log_gamma_nc_d_I and z_nc^2
    real(kind=8), intent(out) :: dc2nc_dc1(:,:)
!> Variables
    integer(kind=4) :: i,j
    type(diag_matrix_c) :: diag_c1,diag_c2nc
    real(kind=8), allocatable :: aux_mat(:,:),mat_lin_syst(:,:),indep_mat(:,:),log_Jacobian(:,:),log_dc2nc_dc1(:,:)
    !real(kind=8), allocatable :: d_log_gamma1_d_log_c1(:,:),d_log_gamma1_d_log_c2nc(:,:),d_log_gamma2nc_d_log_c1(:,:),d_log_gamma2nc_d_log_c2nc(:,:),d_log_gamma_surf_d_log_c_surf(:,:)
    
    !> We compute linear system with logarithms
    allocate(log_dc2nc_dc1(this%speciation_alg%num_eq_reactions,this%speciation_alg%num_prim_species))
    
    call diag_c2nc%set_diag_matrix(c2nc)
    aux_mat=5d-1*log(1d1)*(matmul(this%speciation_alg%Se_nc_1_star,out_prod(1:this%speciation_alg%num_prim_species,this%speciation_alg%num_prim_species+1:this%speciation_alg%num_var_act_species))-out_prod(this%speciation_alg%num_prim_species+1:this%speciation_alg%num_var_act_species,this%speciation_alg%num_prim_species+1:this%speciation_alg%num_var_act_species))
    !> We obtain the matrix that premultiplies the log-Jacobian
    mat_lin_syst=id_matrix(this%speciation_alg%num_eq_reactions)-diag_c2nc%prod_mat_diag_mat(aux_mat)
    !> We compute independent matrix
    call diag_c1%set_diag_matrix(c1)
    aux_mat=5d-1*log(1d1)*diag_c1%prod_mat_diag_mat(out_prod(1:this%speciation_alg%num_prim_species,1:this%speciation_alg%num_prim_species))+id_matrix(this%speciation_alg%num_prim_species)
    indep_mat=matmul(this%speciation_alg%Se_nc_1_star,aux_mat)-5d-1*log(1d1)*diag_c1%prod_mat_diag_mat(out_prod(this%speciation_alg%num_prim_species+1:this%speciation_alg%num_var_act_species,1:this%speciation_alg%num_prim_species)) 
    do j=1,this%speciation_alg%num_prim_species
    !> Linear system solver
        call LU_lin_syst(mat_lin_syst,indep_mat(:,j),this%CV_params%zero,log_dc2nc_dc1(:,j))
    end do
    !> Finally, we compute dc2nc_dc1
    do i=1,this%speciation_alg%num_eq_reactions
        do j=1,this%speciation_alg%num_prim_species
            dc2nc_dc1(i,j)=c2nc(i)*log_dc2nc_dc1(i,j)/c1(j)
        end do
    end do
    
    
!    allocate(d_log_gamma1_d_log_c1(this%speciation_alg%num_prim_species,this%speciation_alg%num_prim_species))
!    allocate(d_log_gamma1_d_log_c2nc(this%speciation_alg%num_prim_species,this%speciation_alg%num_eq_reactions))
!    allocate(d_log_gamma2nc_d_log_c1(this%speciation_alg%num_eq_reactions,this%speciation_alg%num_prim_species))
!    allocate(d_log_gamma2nc_d_log_c2nc(this%speciation_alg%num_eq_reactions,this%speciation_alg%num_eq_reactions))
!    allocate(d_log_gamma_surf_d_log_c_surf(this%solid_chemistry%reactive_zone%cat_exch_zone%num_exch_cats,this%solid_chemistry%reactive_zone%cat_exch_zone%num_exch_cats))
!!> We compute Jacobian logarithm activity coefficients
!    call this%aq_phase%compute_log_Jacobian_act_coeffs_aq_phase(out_prod,this%concentrations(1:this%speciation_alg%num_aq_var_act_species),this%log_Jacobian_act_coeffs(1:this%speciation_alg%num_aq_var_act_species,1:this%speciation_alg%num_aq_var_act_species))
!    call this%solid_chemistry%reactive_zone%cat_exch_zone%compute_log_Jacobian_act_coeffs_ads_cats(this%solid_chemistry%log_act_coeffs(this%solid_chemistry%reactive_zone%num_minerals+2:this%solid_chemistry%reactive_zone%num_solids),d_log_gamma_surf_d_log_c_surf)
!!> log Jacobiano primarias-primarias
!    d_log_gamma1_d_log_c1(1:this%speciation_alg%num_aq_prim_species,1:this%speciation_alg%num_aq_prim_species)=this%log_Jacobian_act_coeffs(1:this%speciation_alg%num_aq_prim_species,1:this%speciation_alg%num_aq_prim_species)
!    !> Chapuza
!    d_log_gamma1_d_log_c1(this%speciation_alg%num_prim_species,:)=0d0
!    d_log_gamma1_d_log_c1(:,this%speciation_alg%num_prim_species)=0d0
!!> log Jacobiano primarias-secundarias actividad variable
!    d_log_gamma1_d_log_c2nc(1:this%speciation_alg%num_aq_prim_species,1:this%speciation_alg%num_aq_sec_var_act_species)=this%log_Jacobian_act_coeffs(1:this%speciation_alg%num_aq_prim_species,this%speciation_alg%num_aq_prim_species:this%speciation_alg%num_aq_var_act_species)
!    !> Chapuza
!    d_log_gamma1_d_log_c2nc(this%speciation_alg%num_prim_species,:)=0d0
!    d_log_gamma1_d_log_c2nc(:,this%speciation_alg%num_aq_sec_var_act_species+1:this%speciation_alg%num_eq_reactions)=0d0
!!> log Jacobiano secundarias actividad variable-primarias
!    d_log_gamma2nc_d_log_c1(1:this%speciation_alg%num_aq_sec_var_act_species,1:this%speciation_alg%num_prim_species-1)=this%log_Jacobian_act_coeffs(this%speciation_alg%num_prim_species:this%speciation_alg%num_aq_var_act_species,1:this%speciation_alg%num_prim_species-1)
!    !> Chapuza
!    d_log_gamma2nc_d_log_c1(this%speciation_alg%num_aq_sec_var_act_species+1:this%speciation_alg%num_eq_reactions,:)=0d0
!    d_log_gamma2nc_d_log_c1(:,this%speciation_alg%num_prim_species)=0d0
!!> log Jacobiano secundarias actividad variable-secundarias actividad variable    
!    d_log_gamma2nc_d_log_c2nc(1:this%speciation_alg%num_aq_sec_var_act_species,1:this%speciation_alg%num_aq_sec_var_act_species)=this%log_Jacobian_act_coeffs(this%speciation_alg%num_prim_species:this%speciation_alg%num_aq_var_act_species,this%speciation_alg%num_prim_species:this%speciation_alg%num_aq_var_act_species)
!    !> Chapuza
!    d_log_gamma2nc_d_log_c2nc(this%speciation_alg%num_aq_sec_var_act_species+1:this%speciation_alg%num_eq_reactions,:)=0d0
!    d_log_gamma2nc_d_log_c2nc(:,this%speciation_alg%num_aq_sec_var_act_species+1:this%speciation_alg%num_eq_reactions)=0d0
!!> We solve linear system logarithms
!    mat_lin_syst=id_matrix(this%speciation_alg%num_eq_reactions)-matmul(this%speciation_alg%Se_nc_1_star,d_log_gamma1_d_log_c2nc)+d_log_gamma2nc_d_log_c2nc
!    indep_term=matmul(this%speciation_alg%Se_nc_1_star,d_log_gamma1_d_log_c1+id_matrix(this%speciation_alg%num_prim_species))-d_log_gamma2nc_d_log_c1
!    allocate(log_Jacobian(this%speciation_alg%num_eq_reactions,this%speciation_alg%num_prim_species))
!    do j=1,this%speciation_alg%num_prim_species
!        call LU_lin_syst(mat_lin_syst,indep_term(:,j),this%CV_params%zero,log_Jacobian(:,j))
!    end do
!> We apply chain rule to obtain dc2nc_dc1
    !c1=this%get_c1_exch()
    !c2nc=this%get_c2nc_exch()
    !do i=1,this%speciation_alg%num_eq_reactions
    !    do j=1,this%speciation_alg%num_prim_species
    !        dc2nc_dc1(i,j)=c2nc(i)*log_Jacobian(i,j)/c1(j)
    !    end do
    !end do
end subroutine