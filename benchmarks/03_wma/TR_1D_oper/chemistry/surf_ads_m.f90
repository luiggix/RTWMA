!> Surface complexation module
!!> Contains properties of surface sites and complexes
module surf_compl_m
    use solid_m
    use aq_phase_m
    use exch_sites_conv_m
    implicit none
    save
    type, public, extends(solid_c) :: surf_site_c !> surface site subclass
        type(solid_c) :: surf_site
    contains
   
    end type
    
    type, public, extends(phase_c) :: surface_c !> surface subclass
        type(solid_c) :: free_site
        integer(kind=4) :: num_surf_compl=0 !> number of surface complexes
        type(solid_c), allocatable :: surf_compl(:) !> surface complexes
    contains
        procedure, public :: set_num_surf_compl
        procedure, public :: set_surf_compl
        procedure, public :: allocate_surf_compl
        procedure, public :: is_surf_compl_in
    end type
    
    type, public, extends(surface_c) :: cat_exch_c !> cation exchange subclass
        integer(kind=4) :: num_exch_cats=0 !> number of exchangeable cations
        type(aq_species_c), allocatable :: exch_cats(:) !> exchangeable cations (NOT NECESSARY)
        class(exch_sites_conv_c), pointer :: convention !> to compute activity
    contains
    !> Set
        procedure, public :: set_convention
        procedure, public :: set_num_exch_cats
        procedure, public :: set_exch_cats
    !> Allocate
        procedure, public :: allocate_exch_cats
    !> Compute
        procedure, public :: compute_log_act_coeffs_ads_cats
        procedure, public :: compute_log_Jacobian_act_coeffs_ads_cats
    end type
!>************************************************************************************************!
!> CHEPROO:
!type, public::t_parentsurface
!> 
!character(len=100)                       :: name          !> Name of interphase
!> 
!type(t_pspecies), pointer, dimension(:)  :: pspecies      !> Pointer to chemical species
!> 
!type(t_phase), pointer                   :: paqph
!> 
!real*8, pointer, dimension(:,:)          :: propsite      !> Value of the properties of the sites
!> 
!character(len=100), pointer, dimension(:):: namepropsite  !> Name of sites properties
!> 
!integer                                  :: numsite       !> Number of site of the interphase
!
!integer                                  :: numsp         !> Total number of species of the interp
!
!integer                                  :: numpropsite   !> Number of properties of sites
!> 
!integer, pointer, dimension(:)           :: numspsite     !> Number of species for site [nsite]
!
!integer, pointer, dimension(:)           :: idxoh
!
!logical                                  :: locksp        !> .true. if species vector was allocate
!
!logical                                  :: lockpaqph     !> .true. if aqueous phase class was
!
!logical                                  :: lockpropsite
!
!logical                                  :: lockidxoh
!> 
!end type t_parentsurface
    
!type, public::t_surface_cexch
!> 
!private                             ::
!> 
!type (t_parentsurface), pointer     :: pp        !> Poiter to parent surface
!> 
!real*8, dimension(:,:), pointer     :: cec       !> Cation exchange capacity 
!> 
!logical                             :: lockcec
!> 
!end type t_surface_cexch
!>************************************************************************************************!
!> PFLOTRAN:
  !type, public :: surface_complex_type
  !>  PetscInt :: id
  !>  character(len=MAXWORDLENGTH) :: name
  !>  character(len=MAXWORDLENGTH) :: free_site_name
  !>  PetscReal :: free_site_stoich
  !>  PetscReal :: Z
  !>  PetscReal :: forward_rate
  !>  PetscReal :: backward_rate
  !>  PetscBool :: print_me
  !>  !> pointer that can be used to index the master list
  !>  type(surface_complex_type), pointer :: ptr
  !>  type(database_rxn_type), pointer :: dbaserxn
  !>  type(surface_complex_type), pointer :: next
  !end type surface_complex_type
    
!> type, public :: surface_complexation_rxn_type
!>    PetscInt :: id
!>    PetscInt :: itype
!>    PetscInt :: free_site_id
!>    character(len=MAXWORDLENGTH) :: free_site_name
!>    PetscBool :: free_site_print_me
!>    PetscBool :: site_density_print_me
!>    PetscInt :: surface_itype
!>    PetscInt :: mineral_id
!>    character(len=MAXWORDLENGTH) :: surface_name
!>    PetscReal :: site_density !> site density in mol/m^3 bulk
!>    PetscReal, pointer :: rates(:)
!>    PetscReal, pointer :: site_fractions(:)
!>    PetscReal :: kinmr_scale_factor
!>    type(surface_complex_type), pointer :: complex_list
!>    type (surface_complexation_rxn_type), pointer :: next
!end type surface_complexation_rxn_type
    
!type, public :: ion_exchange_rxn_type
!>  PetscInt :: id
!>  character(len=MAXWORDLENGTH) :: mineral_name
!>  type(ion_exchange_cation_type), pointer :: cation_list
!>  PetscReal :: CEC
!>  type(ion_exchange_rxn_type), pointer :: next
!end type ion_exchange_rxn_type
!
!type, public :: ion_exchange_cation_type
!>  character(len=MAXWORDLENGTH) :: name
!>  PetscReal :: k
!>  type(ion_exchange_cation_type), pointer :: next
!end type ion_exchange_cation_type
    
    interface
        subroutine compute_log_act_coeffs_ads_cats(this,log_act_coeffs)
            import cat_exch_c
            implicit none
            class(cat_exch_c) :: this
            real(kind=8), intent(out) :: log_act_coeffs(:) !> must be allocated
        end subroutine
        
        subroutine compute_log_Jacobian_act_coeffs_ads_cats(this,log_act_coeffs,log_Jacobian_act_coeffs)
            import cat_exch_c
            implicit none
            class(cat_exch_c) :: this
            real(kind=8), intent(in) :: log_act_coeffs(:)
            real(kind=8), intent(out) :: log_Jacobian_act_coeffs(:,:) !> must be allocated
        end subroutine
    end interface
    
    contains
    
      
        subroutine set_num_exch_cats(this,num_exch_cats)
            implicit none
            class(cat_exch_c) :: this
            integer(kind=4), intent(in), optional :: num_exch_cats
            if (present(num_exch_cats)) then
                this%num_exch_cats=num_exch_cats
            else
                this%num_exch_cats=size(this%exch_cats)
            end if
        end subroutine
        
        subroutine set_exch_cats(this,exch_cats)
            implicit none
            class(cat_exch_c) :: this
            class(aq_species_c), intent(in) :: exch_cats(:)
            if (allocated(this%exch_cats) .and. size(exch_cats)>this%num_exch_cats) then
                error stop "Dimension error in exchangeable cations"
            else
                this%exch_cats=exch_cats
                this%num_exch_cats=size(exch_cats)
            end if
        end subroutine
        
        subroutine allocate_exch_cats(this,num_exch_cats)
            implicit none
            class(cat_exch_c) :: this
            integer(kind=4), intent(in), optional :: num_exch_cats
            if (present(num_exch_cats)) then
                this%num_exch_cats=num_exch_cats
            end if
            allocate(this%exch_cats(this%num_exch_cats))
        end subroutine
        
        subroutine set_num_surf_compl(this,num_surf_compl)
            implicit none
            class(surface_c) :: this
            integer(kind=4), intent(in) :: num_surf_compl
            this%num_surf_compl=num_surf_compl
        end subroutine
        
        subroutine set_surf_compl(this,surf_compl)
            implicit none
            class(surface_c) :: this
            class(solid_c), intent(in) :: surf_compl(:)
            
            if (allocated(this%surf_compl) .and. size(surf_compl)>this%num_surf_compl) then
                error stop "Number of surface complexes is wrong"
            else
                this%surf_compl=surf_compl
            end if
        end subroutine
        
        subroutine allocate_surf_compl(this,num_surf_compl)
            implicit none
            class(surface_c) :: this
            integer(kind=4), intent(in), optional :: num_surf_compl
            
            if (present(num_surf_compl) .and. num_surf_compl>=0) then
                this%num_surf_compl=num_surf_compl
            else if (present(num_surf_compl)) then
                error stop "Number of surface complexes must be non-negative"
            else
                continue
            end if
            allocate(this%surf_compl(this%num_surf_compl))
        end subroutine
        
        !subroutine set_CEC(this,CEC)
        !    implicit none
        !    class(cat_exch_c) :: this
        !    real(kind=8), intent(in) :: CEC
        !    if (CEC<0d0) then
        !        error stop "CEC cannot be negative"
        !    else
        !        this%CEC=CEC
        !    end if
        !end subroutine
        !
        !subroutine compute_CEC(this,equivalents)
        !    implicit none
        !    class(cat_exch_c) :: this
        !    real(kind=8), intent(in) :: equivalents(:)
        !    this%CEC=sum(equivalents)
        !end subroutine
        
        subroutine set_convention(this,convention)
            implicit none
            class(cat_exch_c) :: this
            class(exch_sites_conv_c), intent(in), target :: convention
            this%convention=>convention
        end subroutine
        
        subroutine is_surf_compl_in(this,surf_compl,flag,surf_compl_ind)
            implicit none
            class(surface_c), intent(in) :: this
            class(solid_c), intent(in) :: surf_compl
            logical, intent(out) :: flag
            integer(kind=4), intent(out), optional :: surf_compl_ind
            
            integer(kind=4) :: i
            
            flag=.false.
            if (present(surf_compl_ind)) then
                surf_compl_ind=0
            end if
            do i=1,this%num_surf_compl
                if (surf_compl%name==this%surf_compl(i)%name) then
                    flag=.true.
                    if (present(surf_compl_ind)) then
                        surf_compl_ind=i
                    end if
                    exit
                end if
            end do
        end subroutine
end module