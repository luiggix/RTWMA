!> Solid chemistry subclass:This class contains the concentrations, activities, volumetric fractions and specific surfaces of solid species
module solid_chemistry_m
    use local_chemistry_m
    use reactive_zone_Lagr_m
    implicit none
    save
!**************************************************************************************************
    type, public, extends(local_chemistry_c) :: solid_chemistry_c
        real(kind=8), allocatable :: equivalents(:) !> of adsorbed cations
        real(kind=8), allocatable :: vol_fracts(:) !> volumetric fractions minerals (dimensionless)
        real(kind=8), allocatable :: react_surfaces(:) !> specific surface minerals per unit volume
        class(reactive_zone_c), pointer :: reactive_zone !> solids belong to a certain reactive zone
        real(kind=8) :: CEC !> Cation exchange capacity [eq/L] (we assume we have one or zero adsorption surfaces)
    contains
    !> Set
        procedure, public :: set_conc_solids
        procedure, public :: set_indices_solids
        procedure, public :: set_vol_fracts
        procedure, public :: set_react_surfaces
        procedure, public :: set_reactive_zone
        procedure, public :: set_CEC
    !> Allocate
        procedure, public :: allocate_vol_fracts
        procedure, public :: allocate_react_surfaces
        procedure, public :: allocate_conc_solids
        !procedure, public :: allocate_conc_comp_solids
        procedure, public :: allocate_activities
        procedure, public :: allocate_log_act_coeffs_solid_chem
        procedure, public :: allocate_equivalents
    !> Compute
        procedure, public :: compute_activities_solids
        procedure, public :: compute_equivalents
        procedure, public :: compute_conc_minerals_iter
        procedure, public :: compute_mass_bal_mins
        
    end type
    
    interface
        !subroutine read_solid_chem_init(this,filename,reactive_zones,line,num_tar)
        !>    import reactive_zone_c
        !>    import solid_chemistry_c
        !>    implicit none
        !>    class(solid_chemistry_c) :: this
        !>    character(len=*), intent(in) :: filename
        !>    class(reactive_zone_c), intent(in) :: reactive_zones(:)
        !>    integer(kind=4), intent(inout) :: line
        !>    integer(kind=4), intent(out) :: num_tar
        !end subroutine
        
        subroutine compute_conc_minerals_iter(this,Delta_t)
            import solid_chemistry_c
            implicit none
            class(solid_chemistry_c) :: this
            !real(kind=8), intent(in) :: r_eq(:) !> reaction rates
            real(kind=8), intent(in) :: Delta_t !> time step
        end subroutine
        
        subroutine compute_mass_bal_mins(this,Delta_t)
            import solid_chemistry_c
            implicit none
            class(solid_chemistry_c) :: this
            !real(kind=8), intent(in) :: r_eq(:) !> equilibrium reaction rates
            real(kind=8), intent(in) :: Delta_t !> time step
        end subroutine
        
        subroutine compute_conc_solids(this,r_vec,porosity,time)
            import solid_chemistry_c
            implicit none
            class(solid_chemistry_c) :: this
            real(kind=8), intent(in) :: r_vec(:) !> reaction rates
            real(kind=8), intent(in) :: porosity
            real(kind=8), intent(in) :: time
        end subroutine
    end interface
    
    
    
    contains
        
       
        
        subroutine set_conc_solids(this,conc_solids)
            implicit none
            class(solid_chemistry_c) :: this
            real(kind=8), intent(in) :: conc_solids(:)
            if (this%reactive_zone%num_non_flowing_species<size(conc_solids)) error stop "Dimension error in set_conc_solids"
            this%concentrations=conc_solids
        end subroutine
        
        subroutine set_vol_fracts(this,vol_fracts)
            implicit none
            class(solid_chemistry_c) :: this
            real(kind=8), intent(in) :: vol_fracts(:)
            if (this%reactive_zone%num_minerals/=size(vol_fracts)) error stop "Dimension error in set_vol_fracts"
            this%vol_fracts=vol_fracts
        end subroutine
        
        subroutine allocate_vol_fracts(this)
            implicit none
            class(solid_chemistry_c) :: this
            allocate(this%vol_fracts(this%reactive_zone%num_minerals))
            this%vol_fracts=0d0
        end subroutine
        
        subroutine set_react_surfaces(this,react_surfaces)
            implicit none
            class(solid_chemistry_c) :: this
            real(kind=8), intent(in) :: react_surfaces(:)
            if (this%reactive_zone%num_minerals/=size(react_surfaces)) error stop "Dimension error in set_react_surfaces"
            this%react_surfaces=react_surfaces
        end subroutine
        
        subroutine allocate_react_surfaces(this)
            implicit none
            class(solid_chemistry_c) :: this
            allocate(this%react_surfaces(this%reactive_zone%num_minerals))
            this%react_surfaces=0d0
        end subroutine
        
        subroutine allocate_conc_solids(this)
            implicit none
            class(solid_chemistry_c) :: this
            allocate(this%concentrations(this%reactive_zone%num_solids))
            this%concentrations=0d0
        end subroutine
        
        subroutine allocate_activities(this)
            implicit none
            class(solid_chemistry_c) :: this
            allocate(this%activities(this%reactive_zone%num_solids))
            this%activities=0d0
        end subroutine
        
        subroutine allocate_log_act_coeffs_solid_chem(this)
            implicit none
            class(solid_chemistry_c) :: this
            allocate(this%log_act_coeffs(this%reactive_zone%num_non_flowing_species))
            this%log_act_coeffs=0d0 !> chapuza
        end subroutine
        
        subroutine allocate_equivalents(this)
            implicit none
            class(solid_chemistry_c) :: this
            allocate(this%equivalents(this%reactive_zone%cat_exch_zone%num_exch_cats))
        end subroutine
        
        subroutine set_reactive_zone(this,reactive_zone)
            implicit none
            class(solid_chemistry_c) :: this
            class(reactive_zone_c), intent(in), target :: reactive_zone
            if (associated(reactive_zone%chem_syst)) then
                this%reactive_zone=>reactive_zone
            else
                error stop "Reactive zone object is not associated to a chemical system"
            end if
        end subroutine
        
        !subroutine allocate_conc_comp_solids(this,n_p_sol)
        !    implicit none
        !    class(solid_chemistry_c) :: this
        !    integer(kind=4), intent(in) :: n_p_sol
        !    if (allocated(this%conc_comp)) then
        !        deallocate(this%conc_comp)
        !    end if
        !    allocate(this%conc_comp(n_p_sol))
        !end subroutine
        
        subroutine compute_activities_solids(this)
            implicit none
            class(solid_chemistry_c) :: this
            integer(kind=4) :: i
            real(kind=8), parameter :: eps=1d-15
            !call this%reactive_zone%cat_exch_zone%compute_log_act_coeffs_ads_cats(this%log_act_coeffs(this%reactive_zone%num_minerals+2:this%reactive_zone%num_solids))
            !this%activities=(10**this%log_act_coeffs)*this%concentrations
            this%activities=this%concentrations
            !do i=1,this%reactive_zone%cat_exch_zone%num_exch_cats-1
            !    this%activities(this%reactive_zone%num_minerals+1+i)=(10**this%log_act_coeffs(this%reactive_zone%num_minerals+1+i))*this%concentrations(this%reactive_zone%num_minerals+1+i)
            !end do
            !this%activities(this%reactive_zone%num_solids)=1d0-sum(this%activities(this%reactive_zone%num_minerals+2:this%reactive_zone%num_solids-1))
            !this%log_act_coeffs(this%reactive_zone%num_solids)=log10(this%activities(this%reactive_zone%num_solids))-log10(this%concentrations(this%reactive_zone%num_solids))
            if (abs(sum(this%activities(this%reactive_zone%num_minerals+2:this%reactive_zone%num_solids))-1d0)>=eps) then
                !print *, sum(this%activities(this%reactive_zone%num_minerals+2:this%reactive_zone%num_solids))
                !error stop "Error in equivalent fractions"
            end if
        end subroutine
        
        subroutine compute_equivalents(this)
            implicit none
            class(solid_chemistry_c) :: this
            integer(kind=4) :: i
            do i=1,this%reactive_zone%cat_exch_zone%num_exch_cats
                this%equivalents(i)=this%reactive_zone%cat_exch_zone%exch_cats(i)%valence*this%concentrations(this%reactive_zone%num_minerals+1+i)
            end do
        end subroutine
        
        subroutine set_CEC(this,CEC)
            implicit none
            class(solid_chemistry_c) :: this
            real(kind=8), intent(in) :: CEC
            if (CEC<0d0) then
                error stop "CEC cannot be negative"
            else
                this%CEC=CEC
            end if
        end subroutine
        
        subroutine set_indices_solids(this)
            implicit none
            class(solid_chemistry_c) :: this
            integer(kind=4) :: i,j,k 
            j=0
            k=0
            do i=1,this%reactive_zone%num_minerals
                if (this%reactive_zone%minerals(i)%mineral%cst_act_flag==.false.) then
                    j=j+1
                    this%var_act_species_indices(j)=i
                else
                    k=k+1
                    this%cst_act_species_indices(k)=i
                end if
            end do
        end subroutine
end module