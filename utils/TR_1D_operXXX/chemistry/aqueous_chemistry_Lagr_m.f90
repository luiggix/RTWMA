!> Aqueous chemistry subclass:
!!>   contains local information of aqueous solution
!!>   points to solid chemistry, gas chemistry, chemical system and aqueous phase classes
module aqueous_chemistry_m
    use solid_chemistry_m
    use gas_chemistry_m
    use matrices_m
    use aq_phase_m
    use CV_params_m
    use params_aq_sol_m
    implicit none
    save
    type, public, extends(local_chemistry_c) :: aqueous_chemistry_c
        type(params_aq_sol_t) :: params_aq_sol !> parameters of aqueous solution
        real(kind=8) :: ionic_act !> ionic activity I
        !real(kind=8), allocatable :: d_log_gamma_d_I(:) !> Jacobian of log_10(activity coefficients) with respect to ionic activity
        real(kind=8) :: pH
        real(kind=8) :: pe
        real(kind=8) :: salinity
        real(kind=8) :: alkalinity !> m_OH- + m_HCO3- + 2*m_CO3-2
        class(solid_chemistry_c), pointer :: solid_chemistry !> (if there are reactive zones)
        class(gas_chemistry_c), pointer :: gas_chemistry !> (if there are reactive zones)
        class(chem_system_c), pointer :: chem_syst !> (same chemical system as chemistry class)
        class(aq_phase_c), pointer :: aq_phase !> aqueous phase associated
        type(speciation_algebra_c) :: speciation_alg !> speciation algebra object
        real(kind=8), allocatable :: U_SkT_prod(:,:) !> =U*S_k,nc^T
        integer(kind=4), allocatable :: prim_species_indices(:) !> in aqueous phase
        integer(kind=4), allocatable :: sec_var_act_species_indices(:) !> in aqueous phase
        type(CV_params_t) :: CV_params !> convergence parameters for speciation and reactive mixing computations (should be a pointer)
    contains
    !> Set
        procedure, public :: set_conc_aq_species
        procedure, public :: set_conc_aq_prim_species
        procedure, public :: set_conc_prim_species
        procedure, public :: set_conc_var_act_species
        procedure, public :: set_conc_sec_var_act_species
        procedure, public :: set_conc_sec_aq_species
        procedure, public :: set_conc_cst_act_species
        procedure, public :: set_aq_phase
        procedure, public :: set_pH
        procedure, public :: set_pe
        procedure, public :: set_ionic_act
        procedure, public :: set_conc_single_species
        !procedure, public :: set_conc_comp_aq
        procedure, public :: set_log_act_coeffs
        procedure, public :: set_solid_chemistry
        procedure, public :: set_gas_chemistry
        procedure, public :: set_chem_syst_aq_chem
        procedure, public :: set_speciation_alg
        procedure, public :: set_speciation_alg_dimensions
        procedure, public :: set_prim_species_indices
        procedure, public :: set_sec_var_act_species_indices
    !> Allocate
        procedure, public :: allocate_reaction_rates_aq_chem
        procedure, public :: allocate_conc_aq_species
        procedure, public :: allocate_activities_aq_species
        !procedure, public :: allocate_conc_comp_aq
        procedure, public :: allocate_log_act_coeffs_aq_chem
    !> Compute
        procedure, public :: compute_dI_dc1
        procedure, public :: compute_activities_aq_var_act_species
        procedure, public :: compute_activities_aq
        procedure, public :: compute_activities
        procedure, public :: compute_activity
        procedure, public :: compute_act_from_MAL
        procedure, public :: compute_act_water
        !procedure, public :: compute_act_Henry
        procedure, public :: compute_molarities
        procedure, public :: compute_molalities
        procedure, public :: compute_salinity
        procedure, public :: compute_alkalinity
        procedure, public :: compute_pH
        procedure, public :: compute_conc_comp
        !procedure, public :: compute_conc_comp_aq
        procedure, public :: compute_conc_comp_cst_act
        !procedure, public :: compute_conc_comp_cst_act_exch
        !procedure, public :: compute_conc_comp_glob_aq
        !procedure, public :: compute_r_eq_tilde_aq_chem
        procedure, public :: compute_r_eq_aq_chem
        procedure, public :: compute_rk_aq_chem
        procedure, public :: compute_Jacobian_rk_anal
        procedure, public :: compute_rk_Jac_rk_anal
        procedure, public :: compute_rk_Jac_rk_incr_coeff
        procedure, public :: compute_log_K_aq_chem
        procedure, public :: compute_ionic_act
        !procedure, public :: compute_d_log_gamma_nc_d_log_c2nc_aq
        !procedure, public :: compute_d_log_gamma_nc_d_log_c1_aq
        procedure, public :: compute_d_log_gamma_d_I_aq_chem
        procedure, public :: compute_res_Jac_res_anal
        procedure, public :: compute_res_Jac_res_anal_exch
        procedure, public :: compute_res_Jac_res_incr_coef
        procedure, public :: compute_res_init
        procedure, public :: compute_log_act_coeff_wat
        procedure, public :: compute_saturation_min
    !> Get
        procedure, public :: get_indices_reaction
        procedure, public :: get_c1_exch
        procedure, public :: get_c1_aq
        procedure, public :: get_c2_exch
        procedure, public :: get_c2nc
        procedure, public :: get_conc_nc
        !procedure, public :: get_conc_comp_exch
        procedure, public :: get_conc
        procedure, public :: get_log_gamma2nc
        procedure, public :: get_log_gamma2
        procedure, public :: get_react_zone
    !> Speciation
        procedure, public :: compute_c2nc_aq_from_c1_aq_expl
        procedure, public :: compute_c2_from_c1_aq_ideal
        procedure, public :: compute_c2nc_from_c1_expl
        procedure, public :: compute_c2nc_from_c1_aq_Picard
        procedure, public :: compute_c2nc_from_c1_Picard
        procedure, public :: compute_c2_from_c1_aq_Picard
        procedure, public :: compute_c2_from_c1_Picard
        procedure, public :: compute_c_nc_from_u_Newton
        procedure, public :: compute_c_nc_from_u_aq_Newton
        !procedure, public :: compute_c_aq_from_u_aq_Newton
        !procedure, public :: compute_c_from_u_Newton_CHEPROO
        procedure, public :: compute_residual
        !procedure, public :: compute_residual_bis
        procedure, public :: compute_residual_cst_act
        procedure, public :: compute_dc2nc_dc1_gamma_cst
        procedure, public :: compute_dc2nc_dc1_aq_gamma_cst
        procedure, public :: compute_dc2_dc1_aq_gamma_cst
        procedure, public :: compute_dc2nc_dc1
        procedure, public :: compute_dc2nc_dc1_aq
        procedure, public :: compute_dc2_dc1
        procedure, public :: compute_speciation_alg_arrays
    !> Reactive mixing
        procedure, public :: transport_iter_comp_EE_aq_chem
        procedure, public :: transport_iter_species_EE_aq_chem
        procedure, public :: transport_iter_comp_exch_EE_aq_chem
        procedure, public :: water_mixing_iter_EE_eq_kin
        procedure, public :: water_mixing_iter_EE_kin
        procedure, public :: water_mixing_iter_EfI_eq_kin_anal
        procedure, public :: water_mixing_iter_EfI_kin_anal
        procedure, public :: compute_c_tilde_aq_chem
        procedure, public :: compute_u_tilde_aq_chem
        procedure, public :: reaction_iteration_EE_eq_kin_aq_chem
        procedure, public :: reaction_iteration_EE_kin_aq_chem
        procedure, public :: compute_dfk_dc_aq_EfI
        procedure, public :: compute_dfk_dc1_aq_EfI
        procedure, public :: Newton_EfI_rk_eq_kin_aq_anal
        procedure, public :: Newton_EfI_rk_kin_aq_anal
        procedure, public :: compute_U_SkT_prod
    !> Update
        procedure, public :: update_conc_aq_prim_species
        procedure, public :: update_conc_prim_species
        procedure, public :: update_conc_aq_species
        procedure, public :: update_conc_sec_aq_species
        procedure, public :: update_conc_sec_var_act_species
        procedure, public :: update_conc_sec_aq_var_act_species
        procedure, public :: update_c_nc
    !> Check
        procedure, public :: check_conc_aq_var_act_species
        procedure, public :: check_conc_var_act_species
        procedure, public :: check_act_aq_species
        procedure, public :: check_dc2nc_dc1_aq
        procedure, public :: check_dc2nc_dc1
        procedure, public :: check_dc2_dc1
    !> Initialise
        procedure, public :: initialise_conc_incr_coeff
        procedure, public :: initialise_conc_anal
        procedure, public :: initialise_conc_anal_exch
    !> Rearrange
        procedure, public :: rearrange_state_vars
    end type
    
    interface 
        subroutine compute_c2nc_aq_from_c1_aq_expl(this)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
        end subroutine 
    end interface
    
    interface
        subroutine compute_c2nc_from_c1_expl(this)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
        end subroutine
    end interface
    
    interface
        
        subroutine compute_c2_from_c1_aq_ideal(this)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
        end subroutine 
    end interface
    
    interface
        
        subroutine compute_c2nc_from_c1_Picard(this,c2nc_ig,c2nc,niter,CV_flag)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: c2nc_ig(:) !> initial guess secondary variable activity concentrations
            real(kind=8), intent(out) :: c2nc(:) !> secondary variable activity concentrations (must be already allocated)
            integer(kind=4), intent(out) :: niter !> number of iterations
            logical, intent(out) :: CV_flag !> TRUE if converges, FALSE otherwise
        end subroutine
    end interface
    
    interface
        
        subroutine compute_c2nc_from_c1_aq_Picard(this,c2nc_ig,c2nc,niter,CV_flag)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: c2nc_ig(:) !> initial guess secondary variable activity concentrations
            real(kind=8), intent(out) :: c2nc(:) !> secondary variable activity concentrations (must be already allocated)
            integer(kind=4), intent(out) :: niter !> number of iterations
            logical, intent(out) :: CV_flag !> TRUE if converges, FALSE otherwise
        end subroutine 
        
        subroutine compute_c2_from_c1_aq_Picard(this,c2_init,c2,niter,CV_flag)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
            !real(kind=8), intent(in) :: act_ph(:) !> chapuza innecesaria
            !real(kind=8), intent(in) :: c1(:) !> chapuza (dim=n_p)
            real(kind=8), intent(in) :: c2_init(:) !> chapuza (dim=n_eq)
            real(kind=8), intent(out) :: c2(:) !> chapuza (dim=n_eq)
            integer(kind=4), intent(out) :: niter !> number of iterations
            logical, intent(out) :: CV_flag !> TRUE if converges, FALSE otherwise
        end subroutine
        
        subroutine compute_c2_from_c1_Picard(this,c1,c2_init,c2,niter,CV_flag)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
            !real(kind=8), intent(in) :: act_ph(:) !> chapuza innecesaria
            real(kind=8), intent(in) :: c1(:) !> chapuza (dim=n_p)
            real(kind=8), intent(in) :: c2_init(:) !> chapuza (dim=n_eq)
            real(kind=8), intent(out) :: c2(:) !> chapuza (dim=n_eq)
            integer(kind=4), intent(out) :: niter !> number of iterations
            logical, intent(out) :: CV_flag !> TRUE if converges, FALSE otherwise
        end subroutine
        
        subroutine compute_c_nc_from_u_Newton(this,c2nc_ig,conc_comp,conc_nc,niter,CV_flag)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: c2nc_ig(:) !> initial guess secondary variable activity concentrations
            real(kind=8), intent(in) :: conc_comp(:) !> component concentrations
            real(kind=8), intent(out) :: conc_nc(:) !> variable activity concentrations (already allocated)
            integer(kind=4), intent(out) :: niter !> number of iterations
            logical, intent(out) :: CV_flag !> TRUE if converges, FALSE otherwise
        end subroutine
        
        subroutine compute_c_nc_from_u_aq_Newton(this,c2nc_ig,conc_comp,conc_nc,niter,CV_flag)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: c2nc_ig(:) !> initial guess secondary variable activity concentrations
            real(kind=8), intent(in) :: conc_comp(:) !> component concentrations
            real(kind=8), intent(out) :: conc_nc(:) !> variable activity concentrations (already allocated)
            integer(kind=4), intent(out) :: niter !> number of iterations
            logical, intent(out) :: CV_flag !> TRUE if converges, FALSE otherwise
        end subroutine
        
        subroutine compute_c_aq_from_u_aq_Newton(this,ind_comps,niter,CV_flag)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
            integer(kind=4), intent(in) :: ind_comps(:) !> nº aqueous species with icon=1
            integer(kind=4), intent(out) :: niter !> number of iterations
            logical, intent(out) :: CV_flag !> TRUE if converges, FALSE otherwise
        end subroutine
        
        subroutine compute_c_from_u_Newton_CHEPROO(this,icon,n_icon,constrains,ctot,niter,CV_flag)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
            integer(kind=4), intent(in) :: icon(:)
            integer(kind=4), intent(in) :: n_icon(:)
            character(len=*), intent(in) :: constrains(:)
            real(kind=8), intent(in) :: ctot(:)
            integer(kind=4), intent(out) :: niter !> number of iterations
            logical, intent(out) :: CV_flag !> TRUE if converges, FALSE otherwise
        end subroutine
        
        subroutine initialise_conc_incr_coeff(this,icon,n_icon,indices_constrains,ctot,niter,CV_flag)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
            integer(kind=4), intent(in) :: icon(:)
            integer(kind=4), intent(in) :: n_icon(:)
            integer(kind=4), intent(in) :: indices_constrains(:)
            real(kind=8), intent(in) :: ctot(:)
            integer(kind=4), intent(out) :: niter !> number of iterations
            logical, intent(out) :: CV_flag !> TRUE if converges, FALSE otherwise
        end subroutine 
        
        subroutine initialise_conc_anal(this,icon,n_icon,indices_constrains,ctot,niter,CV_flag)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
            integer(kind=4), intent(in) :: icon(:)
            integer(kind=4), intent(in) :: n_icon(:)
            integer(kind=4), intent(in) :: indices_constrains(:)
            real(kind=8), intent(in) :: ctot(:)
            integer(kind=4), intent(out) :: niter !> number of iterations
            logical, intent(out) :: CV_flag !> TRUE if converges, FALSE otherwise
        end subroutine
        
        subroutine initialise_conc_anal_exch(this,icon,n_icon,indices_constrains,ctot,surf_chem,niter,CV_flag)
            import aqueous_chemistry_c
            import solid_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
            integer(kind=4), intent(in) :: icon(:)
            integer(kind=4), intent(in) :: n_icon(:)
            integer(kind=4), intent(in) :: indices_constrains(:)
            real(kind=8), intent(in) :: ctot(:)
            !real(kind=8), intent(in) :: c1_surf !> chapuza
            !real(kind=8), intent(in) :: CEC
            !real(kind=8), intent(out) :: conc_exch(:) !> chapuza
            class(solid_chemistry_c), intent(inout) :: surf_chem !> surface chemistry
            integer(kind=4), intent(out) :: niter !> number of iterations
            logical, intent(out) :: CV_flag !> TRUE if converges, FALSE otherwise
        end subroutine 
        
        subroutine compute_residual(this,conc_comp,c_nc,residual)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            real(kind=8), intent(in) :: conc_comp(:)
            real(kind=8), intent(in) :: c_nc(:)
            real(kind=8), intent(out) :: residual(:)
        end subroutine
        
        !subroutine compute_residual_bis(this,c_nc,conc_comp,residual)
        !    import aqueous_chemistry_c
        !    implicit none
        !    class(aqueous_chemistry_c), intent(in) :: this
        !    real(kind=8), intent(in) :: c_nc(:) !> variable activity concentrations (chapuza)
        !    real(kind=8), intent(in) :: conc_comp(:) !> component concentrations (chapuza)
        !    real(kind=8), intent(out) :: residual(:)
        !end subroutine
        
        subroutine compute_residual_cst_act(this,conc_comp,conc,residual)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            real(kind=8), intent(in) :: conc_comp(:) !> component concentrations
            real(kind=8), intent(in) :: conc(:) !> species concentrations
            real(kind=8), intent(out) :: residual(:)
        end subroutine
        
        subroutine compute_dc2nc_dc1_gamma_cst(this,dc2nc_dc1)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            real(kind=8), intent(out) :: dc2nc_dc1(:,:)
        end subroutine
        
        subroutine compute_dc2nc_dc1_aq_gamma_cst(this,dc2nc_dc1)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            !real(kind=8), intent(in) :: c2(:) !> chapuza (dim=n_eq)
            real(kind=8), intent(out) :: dc2nc_dc1(:,:)
        end subroutine
        
        subroutine compute_dc2_dc1_aq_gamma_cst(this,c2,dc2_dc1)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            real(kind=8), intent(in) :: c2(:) !> chapuza (dim=n_eq)
            real(kind=8), intent(out) :: dc2_dc1(:,:)
        end subroutine
        
        subroutine compute_dc2nc_dc1(this,c1,c2nc,out_prod,dc2nc_dc1)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: c1(:) !> chapuza
            real(kind=8), intent(in) :: c2nc(:) !> chapuza
            real(kind=8), intent(in) :: out_prod(:,:) !> outer product between d_log_gamma_nc_aq_d_I and z_nc_aq^2
            real(kind=8), intent(out) :: dc2nc_dc1(:,:)
        end subroutine
        
        subroutine compute_dc2nc_dc1_aq(this,c2nc,out_prod,dc2nc_dc1)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: c2nc(:)
            real(kind=8), intent(in) :: out_prod(:,:) !> outer product between d_log_gamma_nc_d_I and z_nc^2
            real(kind=8), intent(out) :: dc2nc_dc1(:,:)
        end subroutine
        
        !subroutine compute_dc2_dc1(this,dc2_dc1)
        !    import aqueous_chemistry_c
        !    implicit none
        !    class(aqueous_chemistry_c), intent(in) :: this
        !    real(kind=8), intent(out) :: dc2_dc1(:,:)
        !end subroutine
        
        subroutine compute_dc2_dc1(this,out_prod,c1,c2,dc2_dc1)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: out_prod(:,:) !> outer product between d_log_gamma_d_I and z^2
            real(kind=8), intent(in) :: c1(:) !> chapuza (dim=n_p)
            real(kind=8), intent(in) :: c2(:) !> chapuza (dim=n_eq)
            real(kind=8), intent(out) :: dc2_dc1(:,:)
        end subroutine
        
        subroutine compute_dI_dc1(this,dc2aq_dc1,dI_dc1)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: dc2aq_dc1(:,:)
            real(kind=8), intent(out) :: dI_dc1(:) !> must be allocated
        end subroutine
        
        subroutine compute_c2nc_from_c1_expl_homog(this)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
        end subroutine 
        
        subroutine compute_c2nc_from_c1_Picard_homog(this,tolerance,niter_max,niter)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: tolerance
            integer(kind=4), intent(in) :: niter_max
            integer(kind=4), intent(out) :: niter !> number of iterations
        end subroutine 
        
        subroutine compute_c_nc_from_u_Newton_homog(this,tolerance,rel_tolerance,control_factor,niter_max,niter,CV_flag)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: tolerance
            real(kind=8), intent(in) :: rel_tolerance
            real(kind=8), intent(in) :: control_factor
            integer(kind=4), intent(in) :: niter_max
            integer(kind=4), intent(out) :: niter !> number of iterations
            logical, intent(out) :: CV_flag !> TRUE if converges, FALSE otherwise
        end subroutine 
        
        subroutine compute_residual_homog(this,residual)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            real(kind=8), intent(out) :: residual(:)
        end subroutine
        
        subroutine compute_dc2nc_dc1_gamma_cst_homog(this,tol,dc2nc_dc1)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            real(kind=8), intent(in) :: tol
            real(kind=8), intent(out) :: dc2nc_dc1(:,:)
        end subroutine
        
        subroutine compute_dc2nc_dc1_homog(this,tol,dc2nc_dc1)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            real(kind=8), intent(in) :: tol
            real(kind=8), intent(out) :: dc2nc_dc1(:,:)
        end subroutine 
        
        subroutine update_conc_aq_species(this,Delta_c)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(inout) :: Delta_c(:)
            !real(kind=8), intent(in) :: control_factor !> must $\in (0,1)$
        end subroutine
        
    !> Updates concentration aqueous primary species in iterative method
        subroutine update_conc_aq_prim_species(this,Delta_c1)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(inout) :: Delta_c1(:)
        end subroutine
        
    !> Updates concentration aqueous and solid primary species in iterative method
        subroutine update_conc_prim_species(this,c1,Delta_c1)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(inout) :: c1(:)
            real(kind=8), intent(inout) :: Delta_c1(:)
        end subroutine
        
        subroutine initialise_iterative_method(conc_old_old,conc_old,param,initial_guess)
            implicit none
            real(kind=8), intent(in) :: conc_old_old(:)
            real(kind=8), intent(in) :: conc_old(:)
            real(kind=8), intent(in) :: param
            real(kind=8), intent(out) :: initial_guess(:)
        end subroutine
        
        subroutine compute_log_act_coeffs(this)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
        end subroutine
        
        subroutine compute_log_Jacobian_act_coeffs(this,d_log_gamma_d_log_c)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(out) :: d_log_gamma_d_log_c(:,:)
        end subroutine
        
        subroutine transport_iter_comp_EE_aq_chem(this,c1_old,c2nc_ig,c_tilde,conc_nc,conc_comp,porosity,Delta_t)
            import aqueous_chemistry_c
            import diag_matrix_c 
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: c1_old(:)
            real(kind=8), intent(in) :: c2nc_ig(:)
            real(kind=8), intent(in) :: c_tilde(:)
            real(kind=8), intent(out) :: conc_nc(:)
            real(kind=8), intent(out) :: conc_comp(:) !> concentration components
            real(kind=8), intent(in), optional :: porosity
            real(kind=8), intent(in), optional :: Delta_t !> time step
        end subroutine
        
        subroutine transport_iter_comp_exch_EE_aq_chem(this,c1_old,c2nc_ig,c_tilde,conc_nc,conc_comp,porosity,Delta_t)
            import aqueous_chemistry_c
            import diag_matrix_c 
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: c1_old(:)
            real(kind=8), intent(in) :: c2nc_ig(:)
            real(kind=8), intent(in) :: c_tilde(:)
            real(kind=8), intent(out) :: conc_nc(:)
            real(kind=8), intent(out) :: conc_comp(:) !> concentration components
            real(kind=8), intent(in), optional :: porosity
            real(kind=8), intent(in), optional :: Delta_t !> time step
        end subroutine
        
        subroutine transport_iter_species_EE_aq_chem(this,c1_old,c2nc_ig,c_tilde,conc_nc,conc_comp,porosity,Delta_t)
            import aqueous_chemistry_c
            import diag_matrix_c 
            implicit none
            class(aqueous_chemistry_c) :: this
            !class(aqueous_chemistry_c), intent(in) :: this_old !> aqueous chemistry object at previous time step
            real(kind=8), intent(in) :: c1_old(:)
            real(kind=8), intent(in) :: c2nc_ig(:)
            real(kind=8), intent(in) :: c_tilde(:)
            real(kind=8), intent(out) :: conc_nc(:)
            real(kind=8), intent(out) :: conc_comp(:) !> concentration components
            real(kind=8), intent(in), optional :: porosity !> storage matrix
            real(kind=8), intent(in), optional :: Delta_t !> time step
        end subroutine
            
        
        
        subroutine transport_iter_EI_aq_chem(this,c_tilde,rk_mat,porosity,Delta_t)
            import aqueous_chemistry_c
            import diag_matrix_c 
            implicit none
            class(aqueous_chemistry_c) :: this
            !integer(kind=4), intent(in) :: ind                              !> index of object "this" in mixing ratios array
            real(kind=8), intent(in) :: c_tilde(:,:)
            !class(aqueous_chemistry_c), intent(in) :: mixing_waters(:)
            real(kind=8), intent(in), optional :: rk_mat(:,:) !> rk matrix
            real(kind=8), intent(in), optional :: porosity !> storage matrix
            real(kind=8), intent(in), optional :: Delta_t !> time step
            !real(kind=8), intent(in), optional :: tolerance !> for Newton CV
            !real(kind=8), intent(in), optional :: rel_tolerance !> for Picard CV
            !real(kind=8), intent(in), optional :: control_factor !> controls Delta_c1_j in Newton
            !integer(kind=4), intent(in), optional :: niter_max !> number maximum of iterations
        end subroutine
        
        subroutine transport_iter_EfI_aq_chem(this,porosity,Delta_t,tolerance,rel_tolerance,control_factor,niter_max)
            import aqueous_chemistry_c
            import diag_matrix_c 
            implicit none
            class(aqueous_chemistry_c) :: this
            !class(aqueous_chemistry_c), intent(in) :: mixing_waters(:) !> all of them have same aqueous species and components
            !real(kind=8), intent(in) :: mixing_ratios(:) !> A^(-T)_j
            real(kind=8), intent(in), optional :: porosity !> storage matrix
            real(kind=8), intent(in), optional :: Delta_t !> time step
            real(kind=8), intent(in), optional :: tolerance !> for Newton CV
            real(kind=8), intent(in), optional :: rel_tolerance !> for Newton/Picard CV
            real(kind=8), intent(in), optional :: control_factor !> controls Delta_c1_j in Newton
            integer(kind=4), intent(in), optional :: niter_max !> number maximum of iterations
        end subroutine
        
        subroutine water_mixing_iter_EE_eq_kin(this,c1_old,c2nc_ig,c_tilde,conc_nc,conc_comp,porosity,Delta_t)
            import aqueous_chemistry_c
            import diag_matrix_c 
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: c1_old(:)
            real(kind=8), intent(in) :: c2nc_ig(:)
            real(kind=8), intent(in) :: c_tilde(:)
            real(kind=8), intent(out) :: conc_nc(:)
            real(kind=8), intent(out) :: conc_comp(:)
            real(kind=8), intent(in), optional :: porosity !> storage matrix
            real(kind=8), intent(in), optional :: Delta_t !> time step
        end subroutine
        
        !subroutine water_mixing_iter_EI_tpt_EE_rk_eq_kin(this,ind,mixing_ratios,mixing_waters,rk_mat,porosity,Delta_t)
        !    import aqueous_chemistry_c
        !    import diag_matrix_c 
        !    implicit none
        !    class(aqueous_chemistry_c) :: this
        !    integer(kind=4), intent(in) :: ind                              !> index of object "this" in mixing ratios array
        !    real(kind=8), intent(in) :: mixing_ratios(:,:)
        !    class(aqueous_chemistry_c), intent(in) :: mixing_waters(:)
        !    real(kind=8), intent(in), optional :: rk_mat(:,:) !> rk matrix
        !    real(kind=8), intent(in), optional :: porosity !> storage matrix
        !    real(kind=8), intent(in), optional :: Delta_t !> time step
        !    !real(kind=8), intent(in), optional :: tolerance !> for Newton CV
        !    !real(kind=8), intent(in), optional :: rel_tolerance !> for Newton/Picard CV
        !    !real(kind=8), intent(in), optional :: control_factor !> controls Delta_c1_j in Newton/Picard
        !    !integer(kind=4), intent(in), optional :: niter_max !> number maximum of iterations
        !end subroutine
        
        !subroutine water_mixing_iter_EI_tpt_EfI_rk_eq_kin(this,ind,mixing_ratios,mixing_waters,rk_mat,porosity,Delta_t)
        !    import aqueous_chemistry_c
        !    import diag_matrix_c 
        !    implicit none
        !    class(aqueous_chemistry_c) :: this
        !    integer(kind=4), intent(in) :: ind                              !> index of object "this" in mixing ratios array
        !    real(kind=8), intent(in) :: mixing_ratios(:,:)
        !    class(aqueous_chemistry_c), intent(in) :: mixing_waters(:)
        !    real(kind=8), intent(in), optional :: rk_mat(:,:) !> rk matrix
        !    real(kind=8), intent(in), optional :: porosity !> storage matrix
        !    real(kind=8), intent(in), optional :: Delta_t !> time step
        !    !real(kind=8), intent(in), optional :: tolerance !> for Newton CV
        !    !real(kind=8), intent(in), optional :: rel_tolerance !> for Newton/Picard CV
        !    !real(kind=8), intent(in), optional :: control_factor !> controls Delta_c1_j in Newton/Picard
        !    !integer(kind=4), intent(in), optional :: niter_max !> number maximum of iterations
        !end subroutine
        
        !subroutine water_mixing_iter_EE_eq_heterog_kin(this,mixing_ratios,mixing_waters,porosity,Delta_t,tolerance,rel_tolerance,control_factor,niter_max)
        !>    import aqueous_chemistry_c
        !>    import diag_matrix_c 
        !>    implicit none
        !>    class(aqueous_chemistry_c) :: this
        !>    real(kind=8), intent(in) :: mixing_ratios(:)
        !>    class(aqueous_chemistry_c), intent(in) :: mixing_waters(:)
        !>    real(kind=8), intent(in), optional :: porosity !> storage matrix
        !>    real(kind=8), intent(in), optional :: Delta_t !> time step
        !>    real(kind=8), intent(in), optional :: tolerance !> for Newton CV
        !>    real(kind=8), intent(in), optional :: rel_tolerance !> for Newton/Picard CV
        !>    real(kind=8), intent(in), optional :: control_factor !> controls Delta_c1_j in Newton/Picard
        !>    integer(kind=4), intent(in), optional :: niter_max !> number maximum of iterations
        !end subroutine
        !
        !subroutine water_mixing_iter_EE_eq_homog_kin(this,mixing_ratios,mixing_waters,porosity,Delta_t,tolerance,rel_tolerance,control_factor,niter_max)
        !>    import aqueous_chemistry_c
        !>    import diag_matrix_c 
        !>    implicit none
        !>    class(aqueous_chemistry_c) :: this
        !>    real(kind=8), intent(in) :: mixing_ratios(:)
        !>    class(aqueous_chemistry_c), intent(in) :: mixing_waters(:)
        !>    real(kind=8), intent(in), optional :: porosity !> storage matrix
        !>    real(kind=8), intent(in), optional :: Delta_t !> time step
        !>    real(kind=8), intent(in), optional :: tolerance !> for Newton CV
        !>    real(kind=8), intent(in), optional :: rel_tolerance !> for Newton/Picard CV
        !>    real(kind=8), intent(in), optional :: control_factor !> controls Delta_c1_j in Newton/Picard
        !>    integer(kind=4), intent(in), optional :: niter_max !> number maximum of iterations
        !end subroutine
        
        subroutine water_mixing_iter_EE_kin(this,c1_old,c2nc_ig,c_tilde,conc_nc,conc_comp,porosity,Delta_t)
            import aqueous_chemistry_c
            import diag_matrix_c 
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: c1_old(:)
            real(kind=8), intent(in) :: c2nc_ig(:)
            real(kind=8), intent(in) :: c_tilde(:)
            real(kind=8), intent(out) :: conc_nc(:)
            real(kind=8), intent(out) :: conc_comp(:)
            real(kind=8), intent(in), optional :: porosity !> storage matrix
            real(kind=8), intent(in), optional :: Delta_t !> time step
        end subroutine
        
        subroutine water_mixing_iter_EfI_eq_kin_anal(this,c1_old,c2nc_ig,c_tilde,conc_nc,conc_comp,porosity,Delta_t)
            import aqueous_chemistry_c
            import diag_matrix_c 
            import time_discr_c
            implicit none
            class(aqueous_chemistry_c) :: this
            !class(aqueous_chemistry_c), intent(in) :: this_old !> aqueous chemistry object at previous time step
            !integer(kind=4), intent(in) :: ind                              !> index of object "this" in mixing ratios array
            real(kind=8), intent(in) :: c1_old(:)
            real(kind=8), intent(in) :: c2nc_ig(:)
            real(kind=8), intent(in) :: c_tilde(:)
            real(kind=8), intent(out) :: conc_nc(:)
            real(kind=8), intent(out) :: conc_comp(:)
            real(kind=8), intent(in), optional :: porosity
            real(kind=8), intent(in), optional :: Delta_t !> time step
            !real(kind=8), intent(in), optional :: tolerance !> for Newton CV
            !real(kind=8), intent(in), optional :: rel_tolerance !> for Newton/Picard CV
            !real(kind=8), intent(in), optional :: control_factor !> controls Delta_c1_j in Newton
            !integer(kind=4), intent(in), optional :: niter_max !> number maximum of iterations
        end subroutine
        
        subroutine water_mixing_iter_EfI_kin_anal(this,c1_old,c2nc_ig,c_tilde,conc_nc,conc_comp,porosity,Delta_t)
            import aqueous_chemistry_c
            import diag_matrix_c 
            import time_discr_c
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: c1_old(:)
            real(kind=8), intent(in) :: c2nc_ig(:)
            real(kind=8), intent(in) :: c_tilde(:)
            real(kind=8), intent(out) :: conc_nc(:)
            real(kind=8), intent(out) :: conc_comp(:)
            real(kind=8), intent(in) :: porosity
            real(kind=8), intent(in) :: Delta_t !> time step
        end subroutine
        
        function compute_c2nc_tilde_aq_chem(this,mixing_ratios,mixing_waters) result(c2nc_tilde)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            !integer(kind=4), intent(in) :: ind                              !> index of object "this" in mixing ratios array
            real(kind=8), intent(in) :: mixing_ratios(:)
            class(aqueous_chemistry_c), intent(in) :: mixing_waters(:)
            real(kind=8), allocatable :: c2nc_tilde(:)
        end function
        
        function compute_c_tilde_aq_chem(this,mixing_ratios,conc_old) result(c_tilde)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            !integer(kind=4), intent(in) :: ind
            real(kind=8), intent(in) :: mixing_ratios(:)
            real(kind=8), allocatable :: conc_old(:,:) !> concentration of variable activity species before mixing
            !class(aqueous_chemistry_c), intent(in) :: mixing_waters(:)
            real(kind=8), allocatable :: c_tilde(:)
        end function
        
        function compute_u_tilde_aq_chem(this,c_tilde) result(u_tilde)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            real(kind=8), intent(in) :: c_tilde(:) !> concentration of "mobile" species after mixing
            real(kind=8), allocatable :: u_tilde(:)
        end function
        
        !subroutine reaction_iteration_EE_WMA_eq_homog_kin_aq_chem(this,porosity,Delta_t,conc_comp_react)
        !>    import aqueous_chemistry_c
        !>    import diag_matrix_c
        !>    implicit none
        !>    class(aqueous_chemistry_c) :: this
        !>    real(kind=8), intent(in) :: porosity
        !>    real(kind=8), intent(in) :: Delta_t
        !>    real(kind=8), intent(out) :: conc_comp_react(:) !> must be allocated
        !end subroutine
        
        subroutine reaction_iteration_EE_eq_kin_aq_chem(this,porosity,Delta_t,conc_comp_react)
            import aqueous_chemistry_c
            import diag_matrix_c
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: porosity
            real(kind=8), intent(in) :: Delta_t
            real(kind=8), intent(out) :: conc_comp_react(:) !> must be allocated
        end subroutine
        
        subroutine reaction_iteration_EI_tpt_EE_rk_eq_kin_aq_chem(this,mixing_ratios,rk_mat,porosity,Delta_t,conc_comp_react)
            import aqueous_chemistry_c
            import diag_matrix_c
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: mixing_ratios(:) !> implicit transport
            real(kind=8), intent(in) :: rk_mat(:,:) !> rk matrix
            real(kind=8), intent(in) :: porosity
            real(kind=8), intent(in) :: Delta_t
            real(kind=8), intent(out) :: conc_comp_react(:) !> must be allocated
        end subroutine
        
        subroutine reaction_iteration_EE_kin_aq_chem(this,porosity,Delta_t,conc_react)
            import aqueous_chemistry_c
            import diag_matrix_c
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: porosity
            real(kind=8), intent(in) :: Delta_t
            real(kind=8), intent(out) :: conc_react(:) !> must be allocated
        end subroutine
        
        subroutine compute_dfk_dc1_aq_EfI(this,c2nc,drk_dc,porosity,Delta_t,dfk_dc1)
            import aqueous_chemistry_c
            import diag_matrix_c
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            real(kind=8), intent(in) :: c2nc(:)
            real(kind=8), intent(in) :: drk_dc(:,:)
            real(kind=8), intent(in) :: porosity
            real(kind=8), intent(in) :: Delta_t
            real(kind=8), intent(out) :: dfk_dc1(:,:)
        end subroutine
        
        subroutine compute_dfk_dc_aq_EfI(this,drk_dc,porosity,Delta_t,dfk_dc)
            import aqueous_chemistry_c
            import diag_matrix_c
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            real(kind=8), intent(in) :: drk_dc(:,:)
            real(kind=8), intent(in) :: porosity
            real(kind=8), intent(in) :: Delta_t
            real(kind=8), intent(out) :: dfk_dc(:,:)
        end subroutine
        
        subroutine Newton_EfI_rk_eq_kin_aq_anal(this,c2nc_ig,c_tilde,porosity,Delta_t,conc_nc,niter,CV_flag)
            import aqueous_chemistry_c
            import diag_matrix_c
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: c2nc_ig(:) !> initial guess secondary variable activity concentrations
            real(kind=8), intent(in) :: c_tilde(:)
            real(kind=8), intent(in) :: porosity
            real(kind=8), intent(in) :: Delta_t !> time step
            real(kind=8), intent(out) :: conc_nc(:) !> variable activity concentrations (already allocated)
            integer(kind=4), intent(out) :: niter !> number of iterations
            logical, intent(out) :: CV_flag !> FALSE: no CV, TRUE: CV
        end subroutine
        
        subroutine Newton_EI_tpt_EfI_rk_eq_kin_aq(this,c_tilde,rk_mat,porosity,Delta_t,niter,CV_flag)
            import aqueous_chemistry_c
            import diag_matrix_c
            implicit none
            class(aqueous_chemistry_c) :: this
            !integer(kind=4), intent(in) :: ind                              !> index of object "this" in mixing ratios array
            real(kind=8), intent(in) :: c_tilde(:,:)
            real(kind=8), intent(in) :: rk_mat(:,:) !> rk matrix
            real(kind=8), intent(in) :: porosity
            real(kind=8), intent(in) :: Delta_t !> time step
            !class(aqueous_chemistry_c), intent(in) :: mixing_waters(:)
            integer(kind=4), intent(out) :: niter !> number of iterations
            logical, intent(out) :: CV_flag !> FALSE: no CV, TRUE: CV
        end subroutine
        
         subroutine Newton_EfI_rk_kin_aq_anal(this,c_tilde,porosity,Delta_t,niter,CV_flag)
            import aqueous_chemistry_c
            import diag_matrix_c
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: c_tilde(:)
            real(kind=8), intent(in) :: porosity
            real(kind=8), intent(in) :: Delta_t !> time step
            !class(aqueous_chemistry_c), intent(in) :: mixing_waters(:)
            integer(kind=4), intent(out) :: niter !> number of iterations
            logical, intent(out) :: CV_flag !> FALSE: no CV, TRUE: CV
        end subroutine
        
        subroutine Picard_EfI_eq_kin_aq_chem(this,mixing_ratios,porosity,Delta_t,mixing_waters,niter,CV_flag)
            import aqueous_chemistry_c
            import tridiag_matrix_c
            import diag_matrix_c
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: mixing_ratios(:)
            real(kind=8), intent(in) :: porosity
            real(kind=8), intent(in) :: Delta_t !> time step
            class(aqueous_chemistry_c), intent(in) :: mixing_waters(:)
            integer(kind=4), intent(out) :: niter !> number of iterations
            logical, intent(out) :: CV_flag !> FALSE: no CV, TRUE: CV
        end subroutine
        

        
        subroutine compute_r_eq_tilde_aq_chem(this,this_old,mixing_ratios,mixing_waters,Delta_t,porosity,r_eq_tilde)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
            !integer(kind=4), intent(in) :: ind                              !> index of object "this" in mixing ratios array
            class(aqueous_chemistry_c), intent(in) :: this_old              !> nombre muy cutre
            real(kind=8), intent(in) :: mixing_ratios(:)
            class(aqueous_chemistry_c), intent(in) :: mixing_waters(:)
            real(kind=8), intent(in) :: Delta_t !> time step
            real(kind=8), intent(in) :: porosity
            real(kind=8), intent(out) :: r_eq_tilde(:)
        end subroutine
        
        subroutine compute_r_eq_aq_chem(this,c2nc_tilde,Delta_t,porosity)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: c2nc_tilde(:) !> concentrations secondary variable activity species after mixing at time step k
            real(kind=8), intent(in) :: Delta_t !> time step
            real(kind=8), intent(in) :: porosity
        end subroutine
        
        subroutine compute_rk_aq_chem_eq_heterog(this)!,rk)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
        end subroutine
        
        subroutine compute_rk_aq_chem(this)!,rk)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c) :: this
        end subroutine
        
        subroutine compute_Jacobian_rk_anal(this,drk_dc)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            real(kind=8), intent(out) :: drk_dc(:,:)
        end subroutine
        
        subroutine compute_rk_Jac_rk_anal(this,drk_dc)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            real(kind=8), intent(out) :: drk_dc(:,:)
        end subroutine
        
        subroutine compute_rk_Jac_rk_incr_coeff(this,drk_dc)
            import aqueous_chemistry_c
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            real(kind=8), intent(out) :: drk_dc(:,:)
        end subroutine
        
        function get_indices_reaction(this,reaction) result(indices)
            import aqueous_chemistry_c
            import reaction_c
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            class(reaction_c), intent(in) :: reaction
            integer(kind=4), allocatable :: indices(:)
        end function
        
        subroutine compute_d_log_gamma_d_I_aq_chem(this,d_log_gamma_d_I)
            import aqueous_chemistry_c
            import params_aq_sol_t
            implicit none
            class(aqueous_chemistry_c) :: this
            !real(kind=8), intent(in) :: ionic_act
            !class(params_aq_sol_t), intent(in) :: params_aq_sol
            real(kind=8), intent(out) :: d_log_gamma_d_I(:) !> must be allocated
        end subroutine
        
        subroutine compute_res_Jac_res_anal_exch(this,conc,indices_icon,n_icon,indices_constrains,ctot,dc2_dc1,log_Jacobian_act_coeffs,CEC,res,Jac_res)
            import aqueous_chemistry_c
            import matrix_int_c
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: conc(:)
            class(matrix_int_c), intent(in) :: indices_icon
            integer(kind=4), intent(in) :: n_icon(:) !> number of each icon
            integer(kind=4), intent(in) :: indices_constrains(:)
            real(kind=8), intent(in) :: ctot(:)
            real(kind=8), intent(in) :: dc2_dc1(:,:) !> Jacobian secondary-primary
            real(kind=8), intent(in) :: log_Jacobian_act_coeffs(:,:)
            real(kind=8), intent(in) :: CEC
            real(kind=8), intent(out) :: res(:) !> residual in Newton-Raphson
            real(kind=8), intent(out) :: Jac_res(:,:) !> Jacobian of residual in Newton-Raphson
        end subroutine
        
        subroutine compute_res_Jac_res_anal(this,indices_icon,n_icon,indices_constrains,ctot,dc2_dc1,log_Jacobian_act_coeffs,res,Jac_res)
            import aqueous_chemistry_c
            import matrix_int_c
            implicit none
            class(aqueous_chemistry_c) :: this
            class(matrix_int_c), intent(in) :: indices_icon
            integer(kind=4), intent(in) :: n_icon(:) !> number of each icon
            integer(kind=4), intent(in) :: indices_constrains(:)
            real(kind=8), intent(in) :: ctot(:)
            real(kind=8), intent(in) :: dc2_dc1(:,:) !> Jacobian secondary-primary
            real(kind=8), intent(in) :: log_Jacobian_act_coeffs(:,:)
            real(kind=8), intent(out) :: res(:) !> residual in Newton-Raphson
            real(kind=8), intent(out) :: Jac_res(:,:) !> Jacobian of residual in Newton-Raphson
        end subroutine
        
        subroutine compute_res_Jac_res_incr_coef(this,c2_init,indices_icon,n_icon,indices_constrains,ctot,res,Jac_res)
            import aqueous_chemistry_c
            import matrix_int_c
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: c2_init(:)
            class(matrix_int_c), intent(in) :: indices_icon
            integer(kind=4), intent(in) :: n_icon(:) !> number of each icon
            integer(kind=4), intent(in) :: indices_constrains(:)
            real(kind=8), intent(in) :: ctot(:)
            real(kind=8), intent(out) :: res(:) !> residual in Newton-Raphson
            real(kind=8), intent(out) :: Jac_res(:,:) !> Jacobian of residual in Newton-Raphson
        end subroutine
        
        subroutine compute_res_init(this,indices_icon,n_icon,indices_constrains,ctot,res)
            import aqueous_chemistry_c
            import matrix_int_c
            implicit none
            !> Pre-process
            class(aqueous_chemistry_c) :: this
            class(matrix_int_c), intent(in) :: indices_icon
            integer(kind=4), intent(in) :: n_icon(:) !> number of each icon
            integer(kind=4), intent(in) :: indices_constrains(:)
            real(kind=8), intent(in) :: ctot(:)
            real(kind=8), intent(out) :: res(:) !> residual in Newton-Raphson
        end subroutine
    end interface
    
    contains
        !subroutine set_model(this,model)
        !    implicit none
        !    class(aqueous_chemistry_c) :: this
        !    integer(kind=4), intent(in) :: model
        !    if (model<1 .or. model>5) error stop "model not implemnted yet"
        !    this%act_coeffs_model=model
        !end subroutine
        
        subroutine allocate_conc_aq_species(this)
            implicit none
            class(aqueous_chemistry_c) :: this
            allocate(this%concentrations(this%aq_phase%num_species))
        end subroutine
        
        subroutine allocate_activities_aq_species(this)
            implicit none
            class(aqueous_chemistry_c) :: this
            allocate(this%activities(this%aq_phase%num_species))
        end subroutine
        
        subroutine compute_activities_aq_var_act_species(this)
            implicit none
            class(aqueous_chemistry_c) :: this
            this%activities(this%prim_species_indices)=this%concentrations(this%prim_species_indices)*(10**(this%log_act_coeffs(this%prim_species_indices)))
            this%activities(this%sec_var_act_species_indices)=this%concentrations(this%sec_var_act_species_indices)*(10**(this%log_act_coeffs(this%sec_var_act_species_indices)))
        end subroutine
        
        subroutine compute_activities_aq(this)
            implicit none
            class(aqueous_chemistry_c) :: this
            call this%compute_activities_aq_var_act_species()
            call this%compute_act_water()
        end subroutine
        
        subroutine compute_activities(this)
            implicit none
            class(aqueous_chemistry_c) :: this
            call this%compute_activities_aq_var_act_species()
            call this%compute_act_water()
            call this%solid_chemistry%compute_activities_solids()
        end subroutine
        
        subroutine compute_activity(this,ind)
            implicit none
            class(aqueous_chemistry_c) :: this
            integer(kind=4), intent(in) :: ind
            this%activities(ind)=this%concentrations(ind)*(10**this%log_act_coeffs(ind))
        end subroutine
        
        subroutine compute_act_water(this)
            implicit none
            class(aqueous_chemistry_c) :: this
            if (this%aq_phase%ind_wat>0) then
                this%activities(this%aq_phase%ind_wat)=1d0-0.018*sum(this%concentrations(this%aq_phase%ind_diss_solids)) !> 1st order approximation (chapuza)
            end if
        end subroutine
        
        !subroutine compute_act_Henry(this,ind_aq_sp,part_press,eq_cst) !> Henry's law
        !    implicit none
        !    class(aqueous_chemistry_c) :: this
        !    integer(kind=4), intent(in) :: ind_aq_sp !> index of aqueous species in aqueous phase
        !    real(kind=8), intent(in) :: part_press !> partial pressure gas equilibrium
        !    real(kind=8), intent(in) :: eq_cst !> equilibrium constant gas reaction
        !    this%activities(ind_aq_sp)=part_press/eq_cst
        !end subroutine
        
        !subroutine allocate_conc_comp_aq(this,n_p_aq)
        !    implicit none
        !    class(aqueous_chemistry_c) :: this
        !    integer(kind=4), intent(in), optional :: n_p_aq
        !    if (present(n_p_aq)) then
        !        this%speciation_alg%num_prim_species=n_p_aq
        !    end if
        !    if (allocated(this%conc_comp)) then
        !        deallocate(this%conc_comp)
        !    end if
        !    allocate(this%conc_comp(this%speciation_alg%num_aq_prim_species))
        !end subroutine
        
        !subroutine allocate_conc_comp(this,np)
        !    implicit none
        !    class(aqueous_chemistry_c) :: this
        !    integer(kind=4), intent(in), optional :: n_p_aq
        !    if (present(n_p_aq)) then
        !        this%speciation_alg%num_aq_prim_species=n_p_aq
        !    end if
        !    allocate(this%conc_comp(this%speciation_alg%num_aq_prim_species))
        !end subroutine
        
        subroutine allocate_log_act_coeffs_aq_chem(this)
            implicit none
            class(aqueous_chemistry_c) :: this
            allocate(this%log_act_coeffs(this%aq_phase%num_species))
            allocate(this%log_Jacobian_act_coeffs(this%aq_phase%num_species,this%aq_phase%num_species))
        end subroutine
        
        subroutine set_conc_aq_species(this,concentrations)
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: concentrations(:)
            if (size(concentrations)/=this%aq_phase%num_species) then
                error stop "Dimension error in concentration of aqueous species"
            else
                this%concentrations=concentrations
            end if
        end subroutine
        
        subroutine set_log_act_coeffs(this,log_act_coeffs) !> we set or initialise the logarithm of activity coefficientes of aqueous species
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in), optional :: log_act_coeffs(:)
            integer(kind=4) :: i
            if (present(log_act_coeffs)) then
                this%log_act_coeffs=log_act_coeffs
            else if (.not. allocated(this%log_act_coeffs)) then
                error stop "Activity coefficients not allocated"
            else
                do i=1,this%aq_phase%num_species
                    this%log_act_coeffs(i)=0d0 !> chapuza
                end do
            end if
        end subroutine
        
        subroutine set_conc_aq_prim_species(this,conc_aq_prim_species)
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: conc_aq_prim_species(:)
            if (size(conc_aq_prim_species)/=this%speciation_alg%num_aq_prim_species) then
                error stop "Dimension error in concentration of primary species"
            else
                this%concentrations(1:this%speciation_alg%num_aq_prim_species)=conc_aq_prim_species
            end if
        end subroutine
        
        subroutine set_conc_prim_species(this,conc_prim_species)
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: conc_prim_species(:)
            if (size(conc_prim_species)/=this%speciation_alg%num_prim_species) then
                error stop "Dimension error in concentration of primary species"
            else if (ASSOCIATED(this%solid_chemistry)) then
                this%solid_chemistry%concentrations(this%solid_chemistry%reactive_zone%num_minerals+1)=conc_prim_species(this%speciation_alg%num_prim_species)
            end if
            this%concentrations(1:this%speciation_alg%num_aq_prim_species)=conc_prim_species(1:this%speciation_alg%num_aq_prim_species)
        end subroutine
        
        subroutine set_conc_var_act_species(this,conc_var_act_species) !> we assume there are aqueous and solid variable activity species (no gases)
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: conc_var_act_species(:)
            if (size(conc_var_act_species)/=this%speciation_alg%num_var_act_species) then
                error stop "Dimension error in concentration of variable activity species"
            else
                this%concentrations(1:this%speciation_alg%num_aq_var_act_species)=conc_var_act_species(1:this%speciation_alg%num_aq_var_act_species)
                this%solid_chemistry%concentrations(this%solid_chemistry%reactive_zone%num_minerals+2:this%solid_chemistry%reactive_zone%num_solids)=conc_var_act_species(this%speciation_alg%num_aq_var_act_species+1:this%speciation_alg%num_var_act_species) 
            end if
        end subroutine
        
        subroutine set_conc_sec_var_act_species(this,c2nc) !> we assume there are aqueous and solid variable activity species (no gases)
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: c2nc(:) 
            if (size(c2nc)/=this%speciation_alg%num_eq_reactions) then
                error stop "Dimension error in concentration of secondary variable activity species"
            else if (size(c2nc)==this%speciation_alg%num_aq_sec_var_act_species) then
                this%concentrations(this%speciation_alg%num_aq_prim_species+1:this%speciation_alg%num_aq_var_act_species)=c2nc
            else
                this%concentrations(this%speciation_alg%num_aq_prim_species+1:this%speciation_alg%num_aq_var_act_species)=c2nc(1:this%speciation_alg%num_aq_sec_var_act_species)
                this%solid_chemistry%concentrations(this%solid_chemistry%reactive_zone%num_minerals+2:this%solid_chemistry%reactive_zone%num_solids)=c2nc(this%speciation_alg%num_aq_sec_var_act_species+1:this%speciation_alg%num_eq_reactions)
                !> faltan gases
            end if
        end subroutine
        
        subroutine set_conc_sec_aq_species(this,c2aq) !> sets concentration of secondary aqueous species
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in), optional :: c2aq(:) !> concentration of secondary aqueous species
            if (present(c2aq)) then
                if (size(c2aq)/=this%aq_phase%num_species-this%speciation_alg%num_aq_prim_species) then
                    error stop "Dimension error in concentration of secondary aqueous species"
                else
                    this%concentrations(this%speciation_alg%num_aq_prim_species+1:this%aq_phase%num_species)=c2aq
                end if
            else
                this%concentrations(this%speciation_alg%num_aq_prim_species+1:this%aq_phase%num_species)=1d-16 !> chapuza
            end if
        end subroutine
        
        subroutine update_conc_sec_aq_species(this,c2aq) !> updates concentration of secondary aqueous species
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: c2aq(:) !> concentration of secondary aqueous species
            this%concentrations(this%speciation_alg%num_aq_prim_species+1:this%aq_phase%num_species)=c2aq
        end subroutine
        
        subroutine update_conc_sec_var_act_species(this,c2nc) !> updates concentration of secondary variable activity species assuming there are aqueous and solid species
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: c2nc(:) !> concentration of secondary variable activity species
            this%concentrations(this%speciation_alg%num_aq_prim_species+1:this%speciation_alg%num_aq_var_act_species)=c2nc(1:this%speciation_alg%num_sec_aq_species)
            this%solid_chemistry%concentrations(this%solid_chemistry%reactive_zone%num_minerals+2:this%solid_chemistry%reactive_zone%num_solids)=c2nc(this%speciation_alg%num_aq_sec_var_act_species+1:this%speciation_alg%num_eq_reactions)
        end subroutine
        
        subroutine update_conc_sec_aq_var_act_species(this,c2nc_aq) !> updates concentration of secondary aqueous variable activity species
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: c2nc_aq(:) !> concentration of secondary aqueous species
            this%concentrations(this%speciation_alg%num_aq_prim_species+1:this%speciation_alg%num_aq_var_act_species)=c2nc_aq
        end subroutine
        
       subroutine set_conc_cst_act_species(this,c_c)
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: c_c(:)

            if (size(c_c)/=this%aq_phase%num_species-this%speciation_alg%num_aq_var_act_species) then
                error stop "Dimension error in concentration of aqueous constant activity species"
            else
                this%concentrations(this%speciation_alg%num_aq_var_act_species+1:this%aq_phase%num_species)=c_c
            end if

       end subroutine
       
       function compute_conc_comp(this,c_nc) result(conc_comp)
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: c_nc(:) !> variable activity species
            real(kind=8), allocatable :: conc_comp(:) !> component concentrations 
            conc_comp=matmul(this%speciation_alg%comp_mat,c_nc)
            !this%conc_comp=matmul(this%speciation_alg%comp_mat(1:this%speciation_alg%num_aq_prim_species,:),c_nc)
            !this%solid_chemistry%conc_comp=matmul(this%speciation_alg%comp_mat(this%speciation_alg%num_aq_prim_species+1:this%speciation_alg%num_prim_species,:),c_nc)
        end function
        
        !subroutine compute_conc_comp_aq(this)
        !    implicit none
        !    class(aqueous_chemistry_c) :: this
        !    this%conc_comp=matmul(this%speciation_alg%comp_mat(1:this%speciation_alg%num_aq_prim_species,1:this%speciation_alg%num_aq_var_act_species),this%concentrations(1:this%speciation_alg%num_aq_var_act_species))
        !end subroutine
        
        function compute_conc_comp_cst_act(this,conc) result(conc_comp)
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: conc(:) !> concentration of all species
            real(kind=8), allocatable :: conc_comp(:)
            conc_comp=matmul(this%speciation_alg%comp_mat_cst_act,conc)
        end function
        !
        !subroutine compute_conc_comp_cst_act_exch(this,c1,c2)
        !    implicit none
        !    class(aqueous_chemistry_c) :: this
        !    real(kind=8), intent(in) :: c1(:) !> autentica chapuza
        !    real(kind=8), intent(in) :: c2(:) !> autentica chapuza
        !    this%conc_comp=matmul(this%speciation_alg%comp_mat_cst_act(1:this%speciation_alg%num_aq_prim_species,:),[c1,c2]) !> chapuza
        !    this%solid_chemistry%conc_comp=matmul(this%speciation_alg%comp_mat(this%speciation_alg%num_aq_prim_species+1:this%speciation_alg%num_prim_species,:),[c1,c2])
        !end subroutine
        !
        !subroutine compute_conc_comp_glob_aq(this)
        !    implicit none
        !    class(aqueous_chemistry_c) :: this
        !    this%conc_comp=matmul(this%speciation_alg%comp_mat(1:this%speciation_alg%num_aq_prim_species,1:this%aq_phase%num_species),this%concentrations)
        !end subroutine
        !
        !subroutine set_conc_comp_aq(this,conc_comp)
        !    implicit none
        !    class(aqueous_chemistry_c) :: this
        !    real(kind=8), intent(in) :: conc_comp(:)
        !    if (size(conc_comp)/=this%speciation_alg%num_aq_prim_species) then
        !        error stop "Dimension error in aqueous component concentrations"
        !    else
        !        this%conc_comp=conc_comp
        !    end if
        !end subroutine
        
      
        
        subroutine set_ionic_act(this,ionic_act)
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: ionic_act
            this%ionic_act=ionic_act
        end subroutine
        
        subroutine compute_ionic_act(this) !> computes ionic activity of solution (concentrations are molalities)
            implicit none
            class(aqueous_chemistry_c) :: this
            integer(kind=4) :: i
            this%ionic_act=0d0
            do i=1,this%aq_phase%num_species
                this%ionic_act=this%ionic_act+this%concentrations(i)*this%aq_phase%aq_species(i)%valence**2
            end do
            this%ionic_act=0.5*this%ionic_act
        end subroutine
        
        subroutine compute_log_K_aq_chem(this)
            implicit none
            class(aqueous_chemistry_c) :: this
            integer(kind=4) :: i
            do i=1,this%chem_syst%num_eq_reacts
                call this%chem_syst%eq_reacts(i)%compute_logK_dep_T(this%temp)
            end do
            if (associated(this%solid_chemistry)) then
                do i=1,this%speciation_alg%num_eq_reactions
                    call this%solid_chemistry%reactive_zone%eq_reactions(i)%compute_logK_dep_T(this%temp)
                end do
            end if
        end subroutine
       
        subroutine set_solid_chemistry(this,solid_chemistry)
            implicit none
            class(aqueous_chemistry_c) :: this
            class(solid_chemistry_c), intent(in), target :: solid_chemistry
            if (associated(solid_chemistry%reactive_zone)) then
                this%solid_chemistry=>solid_chemistry
            else
                error stop "Solid chemistry object is not associated to a reactive zone"
            end if
        end subroutine
        
        subroutine set_gas_chemistry(this,gas_chemistry)
            implicit none
            class(aqueous_chemistry_c) :: this
            class(gas_chemistry_c), intent(in), target :: gas_chemistry
            if (associated(gas_chemistry%reactive_zone)) then
                this%gas_chemistry=>gas_chemistry
            else
                error stop "gas chemistry object is not associated to a reactive zone"
            end if
        end subroutine
        
        subroutine set_aq_phase(this,aq_phase)
            implicit none
            class(aqueous_chemistry_c) :: this
            class(aq_phase_c), intent(in), target :: aq_phase
            this%aq_phase=>aq_phase
        end subroutine
        
        subroutine set_pH(this,pH)
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in), optional :: pH
            if (present(pH)) then
                this%pH=pH
            else
                this%pH=7d0 !> default
            end if
        end subroutine
        
        subroutine compute_pH(this)
            implicit none
            class(aqueous_chemistry_c) :: this
            if (this%aq_phase%ind_proton/=0) then !> chapuza
                this%pH=-log10(this%activities(this%aq_phase%ind_proton))
            end if
        end subroutine
        
        subroutine set_pe(this,pe)
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in), optional :: pe
            if (present(pe)) then
                this%pe=pe
            else
                this%pe=4d0 !> default
            end if
        end subroutine
        
        subroutine set_conc_single_species(this,conc_sp,sp_ind)
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: conc_sp !> concentration of species
            integer(kind=4), intent(in) :: sp_ind !> index of species
            if (sp_ind<1 .or. sp_ind>this%aq_phase%num_species) then
                error stop "Error in aqueous species index"
            else if (.not. allocated(this%concentrations)) then
                error stop "Aqueous species concentrations must be allocated"
            else if (conc_sp<0d0) then
                error stop "Concentrations cannot be negative"
            end if
            this%concentrations(sp_ind)=conc_sp
        end subroutine
        
        subroutine allocate_reaction_rates_aq_chem(this)
            implicit none
            class(aqueous_chemistry_c) :: this
            if (associated(this%chem_syst)) then
                allocate(this%rk(this%chem_syst%num_kin_reacts))
            end if
            allocate(this%r_eq(this%aq_phase%num_aq_complexes+this%chem_syst%num_redox_eq_reacts))
            if (associated(this%solid_chemistry)) then
                allocate(this%solid_chemistry%r_eq(this%solid_chemistry%reactive_zone%num_minerals+this%solid_chemistry%reactive_zone%cat_exch_zone%num_exch_cats))
            end if
            if (associated(this%gas_chemistry)) then
                allocate(this%gas_chemistry%r_eq(this%gas_chemistry%reactive_zone%gas_phase%num_species))
            end if
        end subroutine
        
        subroutine set_chem_syst_aq_chem(this,chem_syst)
            implicit none
            class(aqueous_chemistry_c) :: this
            class(chem_system_c), intent(in), target :: chem_syst
            this%chem_syst=>chem_syst
        end subroutine
        
        subroutine set_speciation_alg_dimensions(this,flag_comp)
            implicit none
            class(aqueous_chemistry_c) :: this
            logical, intent(in) :: flag_comp !> TRUE if component matrix has no constant activity species (De Simoni et al, 2005), FALSE otherwise
            
            integer(kind=4) :: i,n_sp,n_c,n_eq
            logical :: flag_cat_exch
            
            if (.not. associated(this%chem_syst)) then
                error stop
            else if (associated(this%solid_chemistry)) then !> aqueous chemistry is associated to a solid zone
                n_sp=this%aq_phase%num_species+this%solid_chemistry%reactive_zone%num_non_flowing_species+this%chem_syst%num_min_kin_reacts
                n_c=this%aq_phase%wat_flag
                do i=1,this%solid_chemistry%reactive_zone%num_minerals
                    if (this%solid_chemistry%reactive_zone%minerals(I)%mineral%cst_act_flag==.true.) then
                        n_c=n_c+1
                    end if
                end do
                do i=1,this%chem_syst%num_min_kin_reacts
                    if (this%chem_syst%min_kin_reacts(I)%mineral%mineral%cst_act_flag==.true.) then
                        n_c=n_c+1
                    end if
                end do
                do i=1,this%solid_chemistry%reactive_zone%gas_phase%num_species
                    if (this%solid_chemistry%reactive_zone%gas_phase%gases(i)%cst_act_flag==.true.) then
                        n_c=n_c+1
                    end if
                end do
                n_eq=this%solid_chemistry%reactive_zone%num_eq_reactions
                if (this%solid_chemistry%reactive_zone%cat_exch_zone%num_surf_compl>0) then
                    flag_cat_exch=.true.
                else
                    flag_cat_exch=.false.
                end if
            else if (associated(this%gas_chemistry)) then !> aqueous chemistry is associated to a gas zone
                n_sp=this%aq_phase%num_species+this%gas_chemistry%reactive_zone%num_non_flowing_species+this%chem_syst%num_min_kin_reacts
                n_c=this%aq_phase%wat_flag
                n_eq=this%gas_chemistry%reactive_zone%num_eq_reactions
                do i=1,this%chem_syst%num_min_kin_reacts
                    if (this%chem_syst%min_kin_reacts(I)%mineral%mineral%cst_act_flag==.true.) then
                        n_c=n_c+1
                    end if
                end do
                do i=1,this%gas_chemistry%reactive_zone%gas_phase%num_species
                    if (this%gas_chemistry%reactive_zone%gas_phase%gases(i)%cst_act_flag==.true.) then
                        n_c=n_c+1
                    end if
                end do
                if (this%gas_chemistry%reactive_zone%cat_exch_zone%num_surf_compl>0) then
                    flag_cat_exch=.true.
                else
                    flag_cat_exch=.false.
                end if
            else !> all equilibrium reactions are homogeneous
                n_sp=this%chem_syst%num_species
                n_c=this%chem_syst%num_cst_act_species
                n_eq=this%chem_syst%num_eq_reacts
                if (this%chem_syst%cat_exch%num_surf_compl>0) then
                    flag_cat_exch=.true.
                else
                    flag_cat_exch=.false.
                end if
            end if
            call this%speciation_alg%set_flag_comp(flag_comp)
            call this%speciation_alg%set_flag_cat_exch(flag_cat_exch)
            call this%speciation_alg%set_dimensions(n_sp,n_eq,n_c,this%aq_phase%num_species,this%aq_phase%num_species-this%aq_phase%wat_flag,this%chem_syst%num_min_kin_reacts)
        end subroutine
       
        subroutine compute_speciation_alg_arrays(this)
            implicit none
            class(aqueous_chemistry_c) :: this
            !logical, intent(in) :: flag_comp !> TRUE if component matrix has no constant activity species (De Simoni et al, 2005), FALSE otherwise
            !real(kind=8), intent(in) :: tol !> for computing inverse matrix
            
            real(kind=8), allocatable :: Se(:,:),K(:)
            
            if (associated(this%solid_chemistry)) then
                Se=this%solid_chemistry%reactive_zone%stoich_mat
                K=this%solid_chemistry%reactive_zone%get_eq_csts_react_zone()
            else if (associated(this%chem_syst)) then
                Se=this%chem_syst%Se
                K=this%chem_syst%get_eq_csts()
            else
                error stop
            end if
            call this%speciation_alg%compute_arrays(Se,K,this%CV_params%zero)
        end subroutine
        
        subroutine compute_U_SkT_prod(this)
            implicit none
            class(aqueous_chemistry_c) :: this
            integer(kind=4) :: n_k
            !allocate(this%U_SkT_prod(this%speciation_alg%num_prim_species,THIS%chem_syst%num_kin_reacts))
            if (this%speciation_alg%num_eq_reactions==0) then
                this%U_SkT_prod=transpose(this%chem_syst%Sk) !> chapuza
            else
                if (associated(this%chem_syst)) then
                    this%U_SkT_prod=matmul(this%speciation_alg%comp_mat,transpose(this%chem_syst%Sk(:,1:this%speciation_alg%num_var_act_species)))
                else
                    error stop
                end if
            end if
        end subroutine
        
        subroutine set_prim_species_indices(this) !> indices in aqueous phase
            implicit none
            class(aqueous_chemistry_c) :: this
            
            integer(kind=4) :: i
            
            if (allocated(this%prim_species_indices)) then
                deallocate(this%prim_species_indices)
            end if
            allocate(this%prim_species_indices(this%speciation_alg%num_aq_prim_species))      
            do i=1,this%speciation_alg%num_aq_prim_species
                this%prim_species_indices(i)=i !> chapuza
            end do

        end subroutine
        
        subroutine set_sec_var_act_species_indices(this) !> indices in aqueous phase of aquoeus secondary variable activity species
            implicit none
            class(aqueous_chemistry_c) :: this
            
            integer(kind=4) :: i,j,k
            logical :: flag
            
            if (allocated(this%sec_var_act_species_indices)) then
                deallocate(this%sec_var_act_species_indices)
            end if
            allocate(this%sec_var_act_species_indices(this%speciation_alg%num_aq_sec_var_act_species))
            
            j=0
            
            do i=1,this%aq_phase%num_species-this%speciation_alg%num_aq_prim_species
                if (this%speciation_alg%num_aq_prim_species+i/=this%aq_phase%ind_wat) then
                    j=j+1
                    this%sec_var_act_species_indices(j)=this%speciation_alg%num_aq_prim_species+i
                end if
            end do
        end subroutine
        
        subroutine compute_act_from_MAL(this,ind_act_aq_phase,eq_react,act_non_aq_sp) !> computes activity aqueous species from mass action law
            implicit none
            class(aqueous_chemistry_c) :: this
            integer(kind=4), intent(in) :: ind_act_aq_phase !> index species unknown activity in aqueous phase
            class(eq_reaction_c), intent(in) :: eq_react !> equilibrium reaction
            real(kind=8), intent(in), optional :: act_non_aq_sp !> activity of non-aqueous species (partial pressure if gas)
            
            real(kind=8) :: log_a,dot_prod
            integer(kind=4) :: ind_act_react,i
            integer(kind=8), allocatable :: react_indices(:)
            
            react_indices=this%get_indices_reaction(eq_react)
            
            dot_prod=0d0
            do i=1,size(react_indices)
                if (ind_act_aq_phase==react_indices(i)) then
                    ind_act_react=i
                else
                    dot_prod=dot_prod+eq_react%stoichiometry(i)*log10(this%activities(react_indices(i)))
                end if
            end do
            if (present(act_non_aq_sp)) then
                if (size(react_indices)==eq_react%num_species-1) then
                    dot_prod=dot_prod+log10(act_non_aq_sp)
                else
                    error stop
                end if
            end if
            log_a=(log10(eq_react%eq_cst)-dot_prod)/eq_react%stoichiometry(ind_act_react)
            this%activities(ind_act_aq_phase)=10**log_a
        end subroutine
        
        subroutine compute_molarities(this) !> from molalities
            implicit none
            class(aqueous_chemistry_c) :: this
            
            integer(kind=4) :: i
            
            this%concentrations(1:this%aq_phase%num_species)=this%concentrations(1:this%aq_phase%num_species)*this%density*(1d0-this%salinity)
            
            if (this%aq_phase%wat_flag==1) then
                this%concentrations(this%aq_phase%ind_wat)=this%concentrations(this%aq_phase%ind_wat)/(1d0-this%salinity)
            end if

        end subroutine
        
        subroutine compute_salinity(this) !> concentrations are molalities
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8) :: TDS,fluid_mass
            integer(kind=4) :: i
            TDS=0d0 !> [kg_soluto/kg_w]
            do i=1,this%aq_phase%num_species-this%aq_phase%wat_flag
                TDS=TDS+this%concentrations(this%aq_phase%ind_diss_solids(i))*this%aq_phase%aq_species(this%aq_phase%ind_diss_solids(i))%molecular_weight
            end do
            fluid_mass=1d0+TDS
            this%salinity=TDS/fluid_mass
        end subroutine
        
        subroutine compute_alkalinity(this) !> concentrations are molalities
            implicit none
            class(aqueous_chemistry_c) :: this
            !this%alkalinity=this%concentrations(this%aq_phase%ind_oh)+this%concentrations(this%aq_phase%ind_bicarb)+2d0*this%concentrations(this%aq_phase%ind_carb)
        end subroutine
        
        subroutine compute_molalities(this) !> from molarities
            implicit none
            class(aqueous_chemistry_c) :: this
            
            real(kind=8) :: log_a,dot_prod
            integer(kind=4) :: ind_act_react,i
            integer(kind=8), allocatable :: react_indices(:)
            
            this%concentrations(1:this%aq_phase%num_species)=this%concentrations(1:this%aq_phase%num_species)/(this%density*(1d0-this%salinity))
            this%concentrations(this%aq_phase%ind_wat)=this%concentrations(this%aq_phase%ind_wat)*(1d0-this%salinity)
        end subroutine
        
        function get_c1_exch(this) result(c1) !> gets primary concentrations
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            real(kind=8), allocatable :: c1(:)
            
            integer(kind=4) :: i
            
            allocate(c1(this%speciation_alg%num_prim_species))
            c1(1:this%speciation_alg%num_aq_prim_species)=this%concentrations(1:this%speciation_alg%num_aq_prim_species)
            c1(this%speciation_alg%num_prim_species)=this%solid_chemistry%concentrations(this%solid_chemistry%reactive_zone%num_minerals+1)
        end function
        
        function get_c1_aq(this) result(c1) !> gets aqueous primary concentrations
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            real(kind=8), allocatable :: c1(:)
            
            integer(kind=4) :: i
            
            allocate(c1(this%speciation_alg%num_aq_prim_species))
            c1(1:this%speciation_alg%num_aq_prim_species)=this%concentrations(1:this%speciation_alg%num_aq_prim_species)
        end function
        
        function get_c2_exch(this) result(c2) !> gets secondary concentrations assuming there are surface complexation reactions
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            real(kind=8), allocatable :: c2(:)
            integer(kind=4) :: i
            allocate(c2(this%speciation_alg%num_eq_reactions))
        !> Aqueous secondary species
            do i=1,this%speciation_alg%num_sec_aq_species
                c2(i)=this%concentrations(this%speciation_alg%num_aq_prim_species+i)
            end do
        !> Minerals
            do i=1,this%solid_chemistry%reactive_zone%num_minerals
                c2(this%speciation_alg%num_sec_aq_species+i)=this%solid_chemistry%concentrations(i)
            end do
        !> Gases
            !do i=1,this%gas_chemistry%reactive_zone%gas_phase
            !>    c2(this%speciation_alg%num_sec_aq_species+this%solid_chemistry%reactive_zone%num_minerals+i)=this%gas_chemistry%concentrations(i)
            !end do
        !> Surface complexes
            do i=1,this%solid_chemistry%reactive_zone%cat_exch_zone%num_exch_cats
                c2(this%speciation_alg%num_sec_aq_species+this%solid_chemistry%reactive_zone%num_minerals+i)=this%solid_chemistry%concentrations(this%solid_chemistry%reactive_zone%num_minerals+1+i)
            end do
        end function
        
        function get_c2nc(this) result(c2nc)
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            real(kind=8), allocatable :: c2nc(:)
            
            integer(kind=4) :: i
            
            allocate(c2nc(this%speciation_alg%num_eq_reactions))
            do i=1,this%speciation_alg%num_aq_sec_var_act_species
                c2nc(i)=this%concentrations(this%speciation_alg%num_aq_prim_species+i)
            end do
            if (associated(this%solid_chemistry) .and. this%speciation_alg%flag_cat_exch==.true.) then
                c2nc(this%speciation_alg%num_aq_sec_var_act_species+1:this%speciation_alg%num_aq_sec_var_act_species+this%solid_chemistry%reactive_zone%cat_exch_zone%num_exch_cats)=this%solid_chemistry%concentrations(this%solid_chemistry%reactive_zone%num_minerals+2:this%solid_chemistry%reactive_zone%num_solids)
            end if
            if (associated(this%gas_chemistry)) then
                if (associated(this%solid_chemistry)) then
                    c2nc(this%speciation_alg%num_aq_sec_var_act_species+this%solid_chemistry%reactive_zone%cat_exch_zone%num_exch_cats+1:this%speciation_alg%num_eq_reactions)=this%gas_chemistry%concentrations
                else
                    c2nc(this%speciation_alg%num_aq_sec_var_act_species+1:this%speciation_alg%num_eq_reactions)=this%gas_chemistry%concentrations
                end if
            end if
        end function
        
        function get_conc_nc(this) result(conc_nc)
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            real(kind=8), allocatable :: conc_nc(:)
            
            integer(kind=4) :: i
            
            allocate(conc_nc(this%speciation_alg%num_var_act_species))
            do i=1,this%speciation_alg%num_aq_prim_species
                conc_nc(i)=this%concentrations(i)
            end do
            do i=1,this%speciation_alg%num_aq_sec_var_act_species
                conc_nc(this%speciation_alg%num_prim_species+i)=this%concentrations(this%speciation_alg%num_aq_prim_species+i)
            end do
            if (associated(this%solid_chemistry)) then
                if (this%speciation_alg%flag_cat_exch==.true.) then
                    conc_nc(this%speciation_alg%num_prim_species)=this%solid_chemistry%concentrations(this%solid_chemistry%reactive_zone%num_minerals+1)
                    conc_nc(this%speciation_alg%num_prim_species+this%speciation_alg%num_aq_sec_var_act_Species+1:this%speciation_alg%num_prim_species+this%speciation_alg%num_aq_sec_var_act_Species+this%solid_chemistry%reactive_zone%cat_exch_zone%num_exch_cats)=this%solid_chemistry%concentrations(this%solid_chemistry%reactive_zone%num_minerals+2:this%solid_chemistry%reactive_zone%num_solids)
                end if
            end if
            if (associated(this%gas_chemistry)) then
                if (associated(this%solid_chemistry)) then
                    conc_nc(this%speciation_alg%num_aq_var_act_species+this%solid_chemistry%reactive_zone%cat_exch_zone%num_surf_compl+1:this%speciation_alg%num_var_act_species)=this%gas_chemistry%concentrations
                else
                    conc_nc(this%speciation_alg%num_aq_var_act_species+1:this%speciation_alg%num_var_act_species)=this%gas_chemistry%concentrations
                end if
            end if
        end function
        
        function get_log_gamma2nc(this) result(log_gamma2nc) !> gets component concentrations assuming there are surface complexation reactions
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            real(kind=8), allocatable :: log_gamma2nc(:)
            integer(kind=4) :: i
            allocate(log_gamma2nc(this%speciation_alg%num_eq_reactions))
            do i=1,this%speciation_alg%num_aq_sec_var_act_species
                log_gamma2nc(i)=this%log_act_coeffs(this%speciation_alg%num_aq_prim_species+i)
            end do
            if (associated(this%solid_chemistry)) then
                do i=1,this%solid_chemistry%reactive_zone%cat_Exch_zone%num_exch_cats
                    log_gamma2nc(this%speciation_alg%num_aq_sec_var_act_species+i)=this%solid_chemistry%log_act_coeffs(this%solid_chemistry%reactive_zone%num_minerals+1+i)
                end do
            end if
            if (associated(this%gas_chemistry)) then
                !if (associated(this%solid_chemistry)) then
                !    do i=1,this%gas_chemistry%reactive_zone%gas_phase%num_species
                !        log_gamma2nc(this%speciation_alg%num_aq_sec_var_act_species+this%solid_chemistry%reactive_zone%cat_Exch_zone%num_exch_cats+i)=this%gas_chemistry%log_act_coeffs(i)
                !    end do
                !else
                !    do i=1,this%gas_chemistry%reactive_zone%gas_phase%num_species
                !        log_gamma2nc(this%speciation_alg%num_aq_sec_var_act_species+i)=this%gas_chemistry%log_act_coeffs(i)
                !    end do
                !end if
            end if
        end function
        
        function get_log_gamma2(this) result(log_gamma2) !> gets log_10 secondary activity coefficients
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            real(kind=8), allocatable :: log_gamma2(:)
            integer(kind=4) :: i
            allocate(log_gamma2(this%speciation_alg%num_eq_reactions))
            do i=1,this%speciation_alg%num_sec_aq_species
                log_gamma2(i)=this%log_act_coeffs(this%speciation_alg%num_aq_prim_species+i)
            end do
            if (associated(this%solid_chemistry)) then
                do i=1,this%solid_chemistry%reactive_zone%cat_Exch_zone%num_exch_cats
                    log_gamma2(this%speciation_alg%num_sec_aq_species+i)=this%solid_chemistry%log_act_coeffs(this%solid_chemistry%reactive_zone%num_minerals+1+i)
                end do
            end if
            if (associated(this%gas_chemistry)) then
                if (associated(this%solid_chemistry)) then
                    do i=1,this%gas_chemistry%reactive_zone%gas_phase%num_species
                        log_gamma2(this%speciation_alg%num_sec_aq_species+this%solid_chemistry%reactive_zone%cat_Exch_zone%num_exch_cats+i)=this%gas_chemistry%log_act_coeffs(i)
                    end do
                else
                    do i=1,this%gas_chemistry%reactive_zone%gas_phase%num_species
                        log_gamma2(this%speciation_alg%num_sec_aq_species+i)=this%gas_chemistry%log_act_coeffs(i)
                    end do
                end if
            end if
        end function
        
        function get_conc(this) result(conc)
        !> gets all concentrations of species
        !! concentrations are ordered in aqueous, solid, gas & primary, secondary variable activity, secondary constant activity
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            real(kind=8), allocatable :: conc(:)
            
            integer(kind=4) :: i
            allocate(conc(this%speciation_alg%num_species))
        !> Aqueous primary species
            do i=1,this%speciation_alg%num_aq_prim_species
                conc(i)=this%concentrations(i)
            end do
        !> Aqueous secondary variable activity
            do i=1,this%speciation_alg%num_aq_Sec_var_act_species
                conc(this%speciation_alg%num_prim_species+i)=this%concentrations(this%speciation_alg%num_aq_prim_species+i)
            end do
        !> Solids
            if (associated(this%solid_chemistry)) then
            !> Solid variable activity species
                if (this%speciation_alg%flag_cat_exch==.true.) then
                    conc(this%speciation_alg%num_prim_species)=this%solid_chemistry%concentrations(this%solid_chemistry%reactive_zone%num_minerals+1)
                !> Solid secondary variable activity
                    do i=1,this%solid_chemistry%reactive_zone%cat_exch_zone%num_exch_cats
                        conc(this%speciation_alg%num_prim_species+this%speciation_alg%num_aq_Sec_var_act_species+i)=this%solid_chemistry%concentrations(this%solid_chemistry%reactive_zone%num_minerals+1+i)
                    end do
                end if
            !> Minerals
                do i=1,this%solid_chemistry%reactive_zone%num_minerals
                    conc(this%speciation_alg%num_var_act_species+this%aq_phase%wat_flag+i)=this%solid_chemistry%concentrations(i)
                end do
            end if
        !> Gases
            if (associated(this%gas_chemistry)) then
                if (associated(this%solid_chemistry)) then
                    do i=1,this%gas_chemistry%reactive_zone%gas_phase%num_species
                        conc(this%speciation_alg%num_aq_var_act_species+this%solid_chemistry%reactive_zone%cat_exch_zone%num_surf_compl+i)=this%gas_chemistry%concentrations(i)
                    end do
                else
                    do i=1,this%gas_chemistry%reactive_zone%gas_phase%num_species
                        conc(this%speciation_alg%num_aq_var_act_species+i)=this%gas_chemistry%concentrations(i)
                    end do
                end if
            end if
        end function
        
        subroutine set_speciation_alg(this,speciation_alg)
            implicit none
            class(aqueous_chemistry_c) :: this
            type(speciation_algebra_c), intent(in) :: speciation_alg
            this%speciation_alg=speciation_alg
        end subroutine
        
        subroutine check_conc_aq_var_act_species(this,conc_comp) !> checks concentration aqueous variable activity species from components
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            real(kind=8), intent(in) :: conc_comp(:)
            
            real(kind=8), allocatable :: res(:) !> residual
            
            res=conc_comp-matmul(this%speciation_alg%comp_mat,this%concentrations(1:this%speciation_alg%num_aq_var_act_species))
            if (inf_norm_vec_real(res)>this%CV_params%abs_tol) then
                error stop "Error in aqueous variable activity concentrations"
            end if
        end subroutine
        
        subroutine check_conc_var_act_species(this,conc_nc,conc_comp) !> checks concentration variable activity species from components
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            real(kind=8), intent(in) :: conc_nc(:) !> concentration variable activity species
            real(kind=8), intent(in) :: conc_comp(:) !> concentration components
            
            real(kind=8), allocatable :: res(:) !> residual

            
            res=conc_comp-matmul(this%speciation_alg%comp_mat,conc_nc)
            if (inf_norm_vec_real(res)>this%CV_params%abs_tol) then
                error stop "Error in variable activity concentrations"
            end if
        end subroutine
        
        subroutine check_act_aq_species(this) !> checks activity aqueous species
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            
            real(kind=8), allocatable :: log_res(:) !> log_10(residual)
            
            log_res=log10(this%solid_chemistry%reactive_zone%get_eq_csts_react_zone())-matmul(this%solid_chemistry%reactive_zone%stoich_mat(:,1:this%aq_phase%num_species),log10(this%activities))
            if (inf_norm_vec_real(log_res)>this%CV_params%log_abs_tol) then
                print *, inf_norm_vec_real(log_res)
                error stop "Error in aqueous activities"
            end if
        end subroutine
        
        !function get_c_nc_exch(this) result(c_nc)
        !    implicit none
        !    class(aqueous_chemistry_c), intent(in) :: this
        !    real(kind=8), allocatable :: c_nc(:)
        !    integer(kind=4) :: i
        !    allocate(c_nc(this%speciation_alg%num_var_act_species))
        !    do i=1,this%speciation_alg%num_prim_species-1
        !        c_nc(i)=this%concentrations(i)
        !    end do
        !    c_nc(this%speciation_alg%num_prim_species)=this%solid_chemistry%concentrations(this%solid_chemistry%reactive_zone%num_minerals+1)
        !    c_nc(this%speciation_alg%num_prim_species+1:this%speciation_alg%num_aq_var_act_species+1)=this%concentrations(this%speciation_alg%num_prim_species:this%speciation_alg%num_aq_var_act_species)
        !    c_nc(this%speciation_alg%num_aq_var_act_species+1:this%speciation_alg%num_var_act_species)=this%solid_chemistry%concentrations(this%solid_chemistry%reactive_zone%num_minerals+2:this%solid_chemistry%reactive_zone%num_minerals+this%solid_chemistry%reactive_zone%cat_exch_zone%num_surf_compl)
        !end function
        
        !subroutine compute_d_log_gamma_nc_d_log_c2nc_aq(this,d_log_gamma_d_I)
        !    implicit none
        !    class(aqueous_chemistry_c) :: this
        !    real(kind=8), intent(in) :: d_log_gamma_d_I(:)  
        !    !real(kind=8), intent(in) :: c2_aq(:)  
        !    !real(kind=8), intent(out) :: d_log_gamma_d_c2_aq(:,:) 
        !    
        !    integer(kind=4) :: j
        !                            
        !    do j=1,this%speciation_alg%num_aq_sec_var_act_species
        !        this%log_Jacobian_act_coeffs(1:this%speciation_alg%num_aq_var_act_species,this%speciation_alg%num_aq_prim_species+j)=5d-1*this%aq_phase%z2(this%speciation_alg%num_aq_prim_species+j)*this%concentrations(this%speciation_alg%num_aq_prim_species+j)*log(1d1)*d_log_gamma_d_I(1:this%speciation_alg%num_aq_var_act_species)
        !    end do
        !end subroutine
        
        subroutine rearrange_state_vars(this,old_aq_phase)
            implicit none
            class(aqueous_chemistry_c) :: this
            class(aq_phase_c), intent(in) :: old_aq_phase
            
            integer(kind=4) :: i,j
            real(kind=8), allocatable :: aux(:,:)
            
            if (this%aq_phase%num_species/=old_aq_phase%num_species) error stop
            !> falta comparar especies acuosas
            allocate(aux(this%aq_phase%num_species,3)) !> chapuza
            aux(:,1)=this%concentrations
            aux(:,2)=this%activities
            aux(:,3)=this%log_act_coeffs
            i=1
            j=1
            do 
                if (this%aq_phase%aq_species(i)%name==old_aq_phase%aq_species(j)%name) then
                    this%concentrations(i)=aux(j,1)
                    this%activities(i)=aux(j,2)
                    this%log_act_coeffs(i)=aux(j,3)
                    if (i<this%aq_phase%num_species) then
                        i=i+1
                        j=1
                    else 
                        exit
                    end if
                else if (j<old_aq_phase%num_species) then
                    j=j+1
                else if (i<this%aq_phase%num_species) then
                    i=i+1
                    j=1
                else 
                    exit
                end if
            end do
        end subroutine
        
        subroutine check_dc2nc_dc1_aq(this,c2nc,dc2nc_dc1,log_Jacobian_act_coeffs)
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            real(kind=8), intent(in) :: c2nc(:)
            real(kind=8), intent(in) :: dc2nc_dc1(:,:)
            real(kind=8), intent(in) :: log_Jacobian_act_coeffs(:,:)
            
            real(kind=8), allocatable :: res(:,:),lhs(:,:),rhs(:,:),lhs_1(:,:),lhs_2(:,:)
            type(diag_matrix_c) :: c1_diag,c2nc_inv_diag
            
            call c1_diag%set_diag_matrix(this%concentrations(1:this%speciation_alg%num_prim_species))
            call c2nc_inv_diag%set_diag_matrix(1d0/c2nc)
            lhs_1=c2nc_inv_diag%prod_mat_diag_mat(id_matrix(this%speciation_alg%num_eq_reactions)-matmul(this%speciation_alg%Se_nc_1_star,log_Jacobian_act_coeffs(1:this%speciation_alg%num_prim_species,this%speciation_alg%num_prim_species+1:this%speciation_alg%num_var_act_species))+log_Jacobian_act_coeffs(this%speciation_alg%num_prim_species+1:this%speciation_alg%num_var_act_species,this%speciation_alg%num_prim_species+1:this%speciation_alg%num_var_act_species))
            lhs_2=c1_diag%prod_mat_diag_mat(dc2nc_dc1)
            lhs=matmul(lhs_1,lhs_2)
            rhs=matmul(this%speciation_alg%Se_nc_1_star,log_Jacobian_act_coeffs(1:this%speciation_alg%num_prim_species,1:this%speciation_alg%num_prim_species)+id_matrix(this%speciation_alg%num_prim_species))-log_Jacobian_act_coeffs(this%speciation_alg%num_prim_species+1:this%speciation_alg%num_var_act_species,1:this%speciation_alg%num_prim_species)
            res=lhs-rhs
            if (norm_mat_inf(res)>this%CV_params%log_abs_tol) then
                print *, "Error in dc2nc_dc1", norm_mat_inf(res)
                error stop
            end if
        end subroutine
        
        subroutine check_dc2nc_dc1(this,c1,c2nc,dc2nc_dc1,log_Jacobian_act_coeffs)
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            real(kind=8), intent(in) :: c1(:) !> chapuza
            real(kind=8), intent(in) :: c2nc(:) !> chapuza
            real(kind=8), intent(in) :: dc2nc_dc1(:,:) !> Jacobiano
            real(kind=8), intent(in) :: log_Jacobian_act_coeffs(:,:)
            
            real(kind=8), allocatable :: res(:,:),lhs(:,:),rhs(:,:),lhs_1(:,:),lhs_2(:,:)
            type(diag_matrix_c) :: c1_diag,c2_inv_diag
            
            call c1_diag%set_diag_matrix(c1)
            call c2_inv_diag%set_diag_matrix(1d0/c2nc)
            lhs_1=c2_inv_diag%prod_mat_diag_mat(id_matrix(this%speciation_alg%num_eq_reactions)-matmul(this%speciation_alg%Se_nc_1_star,log_Jacobian_act_coeffs(1:this%speciation_alg%num_prim_species,this%speciation_alg%num_prim_species+1:this%speciation_alg%num_var_act_species))+log_Jacobian_act_coeffs(this%speciation_alg%num_prim_species+1:this%speciation_alg%num_var_act_species,this%speciation_alg%num_prim_species+1:this%speciation_alg%num_var_act_species))
            lhs_2=c1_diag%prod_mat_diag_mat(dc2nc_dc1)
            lhs=matmul(lhs_1,lhs_2)
            rhs=matmul(this%speciation_alg%Se_nc_1_star,log_Jacobian_act_coeffs(1:this%speciation_alg%num_prim_species,1:this%speciation_alg%num_prim_species)+id_matrix(this%speciation_alg%num_prim_species))-log_Jacobian_act_coeffs(this%speciation_alg%num_prim_species+1:this%speciation_alg%num_var_act_species,1:this%speciation_alg%num_prim_species)
            res=lhs-rhs
            if (norm_mat_inf(res)>this%CV_params%log_abs_tol) then
                print *, "Error in dc2nc_dc1", norm_mat_inf(res)
                error stop
            end if
        end subroutine
        
        subroutine check_dc2_dc1(this,c1,c2,dc2_dc1,log_Jacobian_act_coeffs)
            implicit none
            class(aqueous_chemistry_c), intent(in) :: this
            real(kind=8), intent(in) :: c1(:) !> chapuza
            real(kind=8), intent(in) :: c2(:) !> chapuza
            real(kind=8), intent(in) :: dc2_dc1(:,:) !> Jacobiano
            real(kind=8), intent(in) :: log_Jacobian_act_coeffs(:,:)
            
            real(kind=8), allocatable :: res(:,:),lhs(:,:),rhs(:,:),lhs_1(:,:),lhs_2(:,:)
            type(diag_matrix_c) :: c1_diag,c2_inv_diag
            
            call c1_diag%set_diag_matrix(c1(1:this%speciation_alg%num_prim_species))
            call c2_inv_diag%set_diag_matrix(1d0/c2)
            lhs_1=c2_inv_diag%prod_mat_diag_mat(id_matrix(this%speciation_alg%num_eq_reactions)-matmul(this%speciation_alg%Se_1_star,log_Jacobian_act_coeffs(1:this%speciation_alg%num_prim_species,this%speciation_alg%num_prim_species+1:this%speciation_alg%num_species))+log_Jacobian_act_coeffs(this%speciation_alg%num_prim_species+1:this%speciation_alg%num_species,this%speciation_alg%num_prim_species+1:this%speciation_alg%num_species))
            lhs_2=c1_diag%prod_mat_diag_mat(dc2_dc1)
            lhs=matmul(lhs_1,lhs_2)
            rhs=matmul(this%speciation_alg%Se_1_star,log_Jacobian_act_coeffs(1:this%speciation_alg%num_prim_species,1:this%speciation_alg%num_prim_species)+id_matrix(this%speciation_alg%num_prim_species))-log_Jacobian_act_coeffs(this%speciation_alg%num_prim_species+1:this%speciation_alg%num_species,1:this%speciation_alg%num_prim_species)
            res=lhs-rhs
            if (norm_mat_inf(res)>this%CV_params%log_abs_tol) then
                print *, "Error in dc2_dc1", norm_mat_inf(res)
                error stop
            end if
        end subroutine
        
        !subroutine compute_d_log_gamma_nc_d_log_c1_aq(this,dc2nc_dc1_aq,d_log_gamma_d_I)
        !    implicit none
        !    class(aqueous_chemistry_c) :: this
        !    real(kind=8), intent(in) :: dc2nc_dc1_aq(:,:)
        !    real(kind=8), intent(in) :: d_log_gamma_d_I(:) 
        !    
        !    integer(kind=4) :: j
        !                            
        !    do j=1,this%speciation_alg%num_prim_species
        !        this%log_Jacobian_act_coeffs(1:this%speciation_alg%num_aq_var_act_species,j)=5d-1*(this%aq_phase%z2(j)+dot_product(this%aq_phase%z2(1:this%speciation_alg%num_aq_sec_var_act_species),dc2nc_dc1_aq(:,j)))*this%concentrations(j)*log(1d1)*d_log_gamma_d_I(1:this%speciation_alg%num_aq_var_act_species)
        !    end do
        !end subroutine

        subroutine compute_log_act_coeff_wat(this)
            implicit none
            class(aqueous_chemistry_c) :: this
            if (this%aq_phase%ind_wat>0) then
                this%log_act_coeffs(this%aq_phase%ind_wat)=log10(this%activities(this%aq_phase%ind_wat))-log10(this%concentrations(this%aq_phase%ind_wat))
            end if
        end subroutine
        
        function compute_saturation_min(this,kin_react) result(saturation)
            implicit none
            class(aqueous_chemistry_c) :: this
            class(kin_reaction_c), intent(in) :: kin_react
            real(kind=8) :: saturation
            
            real(kind=8) :: IAP
            integer(kind=4) :: i
            IAP=1d0
            do i=1,kin_react%num_species-1
                IAP=IAP*this%activities(kin_react%indices_aq_phase(i))**kin_react%stoichiometry(i)
            end do
            saturation=IAP/kin_react%eq_cst
        end function
        
        subroutine update_c_nc(this,c_nc)
            implicit none
            class(aqueous_chemistry_c) :: this
            real(kind=8), intent(in) :: c_nc(:)
            this%concentrations(1:this%speciation_alg%num_aq_prim_species)=c_nc(1:this%speciation_alg%num_aq_prim_species)
            this%solid_chemistry%concentrations(this%solid_chemistry%reactive_zone%num_minerals+1)=c_nc(this%speciation_alg%num_prim_species)
            this%concentrations(this%speciation_alg%num_prim_species+1:this%speciation_alg%num_aq_var_act_species+1)=c_nc(this%speciation_alg%num_prim_species+1:this%speciation_alg%num_aq_var_act_species+1)
            this%solid_chemistry%concentrations(this%solid_chemistry%reactive_zone%num_minerals+2:this%solid_chemistry%reactive_zone%num_solids)=c_nc(this%speciation_alg%num_aq_var_act_species+2:this%speciation_alg%num_var_act_species)
        end subroutine
        
        function get_react_zone(this) result(react_zone)
            implicit none
            class(aqueous_chemistry_c) :: this
            type(reactive_zone_c) :: react_zone
            if (associated(this%solid_chemistry)) then
                react_zone=this%solid_chemistry%reactive_zone
            else if (associated(this%gas_chemistry)) then
                react_zone=this%gas_chemistry%reactive_zone
            else
                continue
            end if
        end function
end module 
        
