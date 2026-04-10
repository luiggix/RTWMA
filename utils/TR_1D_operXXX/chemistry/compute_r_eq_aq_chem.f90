!> Computes equilibrium reaction rates from secondary variable activity species concentrations
subroutine compute_r_eq_aq_chem(this,c2nc_tilde,Delta_t,porosity)
    use aqueous_chemistry_m
    use metodos_sist_lin_m
    implicit none
    class(aqueous_chemistry_c) :: this !> aqueous chemistry object at time step k+1
    real(kind=8), intent(in) :: c2nc_tilde(:) !> concentrations secondary variable activity species after mixing at time step k
    real(kind=8), intent(in) :: Delta_t !> (k+1)-th time step
    real(kind=8), intent(in) :: porosity
!> Variables
    real(kind=8), allocatable :: A(:,:),b(:),c2nc(:),R_eq(:)
    integer(kind=4) :: err
!> Pre-process
    allocate(R_eq(this%speciation_alg%num_eq_reactions)) !> R_eq=Delta_t*r_eq/phi
!> Process
    !> Linear least squares
        !> Autentica chapuza
            c2nc=this%get_c2nc()
            if (associated(this%solid_chemistry)) then
                A=matmul(this%solid_chemistry%reactive_zone%stoich_mat(:,this%speciation_alg%num_prim_species+1:this%speciation_alg%num_var_act_species),transpose(this%solid_chemistry%reactive_zone%stoich_mat(:,this%speciation_alg%num_prim_species+1:this%speciation_alg%num_var_act_species)))
                !print *, this%rk, this%chem_syst%Sk
                b=matmul(this%solid_chemistry%reactive_zone%stoich_mat(:,this%speciation_alg%num_prim_species+1:this%speciation_alg%num_var_act_species),c2nc-c2nc_tilde) - matmul(this%solid_chemistry%reactive_zone%stoich_mat(:,this%speciation_alg%num_prim_species+1:this%speciation_alg%num_var_act_species),matmul(transpose(this%chem_syst%Sk(:,this%speciation_alg%num_prim_species+1:this%speciation_alg%num_var_act_species)),this%rk))
                if (inf_norm_vec_real(b)<this%CV_params%zero) then
                    R_eq=0d0
                else
                    call Gauss_Jordan(A,b,this%CV_params%zero,R_eq,err) !> linear system solver
                end if
                this%solid_chemistry%r_eq(1:this%solid_chemistry%reactive_zone%num_minerals_cst_act)=R_eq(1:this%solid_chemistry%reactive_zone%num_minerals_cst_act)*porosity/Delta_t !> r_eq_j=R_eq*phi_j/Delta_t
                this%solid_chemistry%r_eq(this%solid_chemistry%reactive_zone%num_minerals_cst_act+1:this%solid_chemistry%reactive_zone%num_minerals_cst_act+this%solid_chemistry%reactive_zone%cat_exch_zone%num_exch_cats)=R_eq(this%speciation_alg%num_cst_act_species-this%aq_phase%wat_flag+this%aq_phase%num_aq_complexes+this%chem_syst%num_redox_eq_reacts+1:this%solid_chemistry%reactive_zone%num_eq_reactions-this%solid_chemistry%reactive_zone%gas_phase%num_var_act_species)*porosity/Delta_t !> r_eq_j=R_eq*phi_j/Delta_t
                if (associated(this%gas_chemistry)) then
                    this%gas_chemistry%r_eq(1:this%solid_chemistry%reactive_zone%gas_phase%num_cst_act_species)=R_eq(this%solid_chemistry%reactive_zone%num_minerals_cst_act+1:this%speciation_alg%num_cst_act_species-this%aq_phase%wat_flag)*porosity/Delta_t !> r_eq_j=R_eq*phi_j/Delta_t
                    this%gas_chemistry%r_eq(this%solid_chemistry%reactive_zone%gas_phase%num_cst_act_species+1:this%solid_chemistry%reactive_zone%gas_phase%num_species)=R_eq(this%speciation_alg%num_cst_act_species-this%aq_phase%wat_flag+this%aq_phase%num_aq_complexes+this%chem_syst%num_redox_eq_reacts+this%solid_chemistry%reactive_zone%cat_exch_zone%num_exch_cats+1:this%solid_chemistry%reactive_zone%num_eq_reactions)*porosity/Delta_t !> r_eq_j=R_eq*phi_j/Delta_t
                end if
            else if (associated(this%gas_chemistry)) then
                A=matmul(this%gas_chemistry%reactive_zone%stoich_mat(:,this%speciation_alg%num_prim_species+1:this%speciation_alg%num_var_act_species),transpose(this%gas_chemistry%reactive_zone%stoich_mat(:,this%speciation_alg%num_prim_species+1:this%speciation_alg%num_var_act_species)))
                b=matmul(this%gas_chemistry%reactive_zone%stoich_mat(:,this%speciation_alg%num_prim_species+1:this%speciation_alg%num_var_act_species),c2nc-c2nc_tilde) - matmul(this%gas_chemistry%reactive_zone%stoich_mat(:,this%speciation_alg%num_prim_species+1:this%speciation_alg%num_var_act_species),matmul(transpose(this%chem_syst%Sk(:,this%speciation_alg%num_prim_species+1:this%speciation_alg%num_var_act_species)),this%rk))
                if (inf_norm_vec_real(b)<this%CV_params%zero) then
                    R_eq=0d0
                else
                    call Gauss_Jordan(A,b,this%CV_params%zero,R_eq,err) !> linear system solver
                end if
                this%gas_chemistry%r_eq(1:this%gas_chemistry%reactive_zone%gas_phase%num_cst_act_species)=R_eq(this%gas_chemistry%reactive_zone%num_minerals_cst_act+1:this%speciation_alg%num_cst_act_species-this%aq_phase%wat_flag)*porosity/Delta_t !> r_eq_j=R_eq*phi_j/Delta_t
                this%gas_chemistry%r_eq(this%gas_chemistry%reactive_zone%gas_phase%num_cst_act_species+1:this%gas_chemistry%reactive_zone%gas_phase%num_species)=R_eq(this%speciation_alg%num_cst_act_species-this%aq_phase%wat_flag+this%aq_phase%num_aq_complexes+this%chem_syst%num_redox_eq_reacts+this%gas_chemistry%reactive_zone%num_minerals_var_act+this%gas_chemistry%reactive_zone%cat_exch_zone%num_exch_cats+1:this%gas_chemistry%reactive_zone%num_eq_reactions)*porosity/Delta_t !> r_eq_j=R_eq*phi_j/Delta_t
            else
                A=matmul(this%chem_syst%stoich_mat(:,this%speciation_alg%num_prim_species+1:this%speciation_alg%num_var_act_species),transpose(this%chem_syst%stoich_mat(:,this%speciation_alg%num_prim_species+1:this%speciation_alg%num_var_act_species)))
                b=matmul(this%chem_syst%stoich_mat(:,this%speciation_alg%num_prim_species+1:this%speciation_alg%num_var_act_species),c2nc-c2nc_tilde)
            end if
            this%r_eq=R_eq(this%speciation_alg%num_cst_act_species-this%aq_phase%wat_flag+1:this%speciation_alg%num_cst_act_species-this%aq_phase%wat_flag+this%aq_phase%num_aq_complexes+this%chem_syst%num_redox_eq_reacts)*porosity/Delta_t !> r_eq_j=R_eq*phi_j/Delta_t
end subroutine