!< Mineral kinetic reaction subclass
module kin_mineral_m
    use kin_mineral_params_m
    use kin_reaction_m
    use mineral_m
    use aq_phase_m
    implicit none
    save
    type, public, extends(kin_reaction_c) :: kin_mineral_c
        type(mineral_c) :: mineral !> mineral that dissolves or precipitates
        type(kin_mineral_params_c) :: params !> parameters to compute reaction rate
    contains
    !> Set
        procedure, public :: set_mineral_params
        procedure, public :: set_mineral
    !> Compute
        procedure, public :: compute_rk_mineral
        procedure, public :: compute_drk_dc_mineral
    end type
        
    interface
        !subroutine compute_rk_mineral(this,conc,rk)
        !    import kin_mineral_c
        !    implicit none
        !    class(kin_mineral_c), intent(in) :: this
        !    real(kind=8), intent(in) :: conc(:)
        !    real(kind=8), intent(out) :: rk
        !end subroutine
        
        subroutine compute_rk_mineral(this,act_cat,saturation,react_surf,temp,rk)
            import kin_mineral_c
            implicit none
            class(kin_mineral_c), intent(in) :: this
            real(kind=8), intent(in) :: act_cat(:)
            real(kind=8), intent(in) :: saturation !> (chapuza)
            real(kind=8), intent(in) :: react_surf
            real(kind=8), intent(in) :: temp
            real(kind=8), intent(out) :: rk
        end subroutine
        
        !subroutine compute_rk_j_mineral(this,j,rk_j)
        !>    import kin_mineral_c
        !>    implicit none
        !>    class(kin_mineral_c), intent(in) :: this
        !>    integer(kind=4), intent(in) :: j
        !>    real(kind=8), intent(out) :: rk_j
        !end subroutine
        
        !subroutine compute_drk_dc_mineral(this,conc,rk,drk_dc)
        !    import kin_mineral_c
        !    implicit none
        !    class(kin_mineral_c), intent(in) :: this
        !    real(kind=8), intent(in) :: conc(:)
        !    real(kind=8), intent(in) :: rk
        !    real(kind=8), intent(out) :: drk_dc(:)
        !end subroutine
        
        subroutine compute_drk_dc_mineral(this,conc,act_cat,saturation,react_surf,temp,drk_dc)
            import kin_mineral_c
            implicit none
            class(kin_mineral_c), intent(in) :: this
            real(kind=8), intent(in) :: conc(:)
            real(kind=8), intent(in) :: act_cat(:) !> activities catalysers
            real(kind=8), intent(in) :: saturation
            real(kind=8), intent(in) :: react_surf
            real(kind=8), intent(in) :: temp !> Kelvin
            real(kind=8), intent(out) :: drk_dc(:) !> must be already allocated
        end subroutine
        
        subroutine read_kin_mineral(this)
            import kin_mineral_c
            implicit none
            class(kin_mineral_c) :: this
            !character(len=*), intent(in) :: react_name
            !integer(kind=4), intent(in) :: n_paths
            !character(len=*), intent(in) :: filename
        end subroutine
        
        subroutine get_conc_kin_mineral(this,species,conc,conc_kin,kin_ind)
            import kin_mineral_c
            import species_c
            implicit none
            class(kin_mineral_c), intent(in) :: this
            class(species_c), intent(in) :: species(:)
            real(kind=8), intent(in) :: conc(:) !> species concentrations
            real(kind=8), intent(out) :: conc_kin(:) !> concentration of species relevant to kinetic reaction rates
            integer(kind=4), intent(out), optional :: kin_ind(:)
        end subroutine
        
        !subroutine get_aq_species_indices(this,aq_species,conc,aq_species_ind)
        !>    import kin_mineral_c
        !>    import species_c
        !>    implicit none
        !>    class(kin_mineral_c), intent(in) :: this
        !>    class(aq_species_c), intent(in) :: aq_species(:)
        !>    !real(kind=8), intent(in) :: conc(:) !> species concentrations
        !>    real(kind=8), intent(out) :: aq_species_ind(:) !> concentration of species relevant to kinetic reaction rates
        !end subroutine
        
        subroutine get_solid_chem_mineral(this,aq_species,activities,act_cat,aq_species_ind)!,react_surf,temp)
            import kin_mineral_c
            import aq_species_c
            implicit none
            class(kin_mineral_c), intent(in) :: this
            class(aq_species_c), intent(in) :: aq_species(:)
            real(kind=8), intent(in) :: activities(:) !> aqueous species activities
            real(kind=8), intent(out) :: act_cat(:)
            integer(kind=4), intent(out), optional :: aq_species_ind(:)
            !real(kind=8), intent(out) :: react_surf
            !real(kind=8), intent(out) :: temp
        end subroutine

    end interface
    
    contains
        subroutine set_mineral_params(this,mineral_params)
            implicit none
            class(kin_mineral_c) :: this
            class(kin_mineral_params_c), intent(in) :: mineral_params
            this%params=mineral_params
        end subroutine
        
        subroutine set_mineral(this,mineral)
            implicit none
            class(kin_mineral_c) :: this
            class(mineral_c), intent(in) :: mineral
            this%mineral=mineral
        end subroutine
        
       subroutine append_kin_min_reaction(this,kin_reactions)
        implicit none
        class(kin_mineral_c), intent(in) :: this
        type(kin_mineral_c), intent(inout), allocatable :: kin_reactions(:)
        
        type(kin_mineral_c), allocatable :: aux(:)
        
        aux=kin_reactions
        deallocate(kin_reactions)
        allocate(kin_reactions(size(aux)+1))
        kin_reactions(1:size(aux))=aux
        kin_reactions(size(kin_reactions))=this
       end subroutine
       
       subroutine set_indices_aq_phase_min(this,aq_phase)
            implicit none
            class(kin_mineral_c) :: this
            class(aq_phase_c), intent(in) :: aq_phase
            
            integer(kind=4) :: i,aq_species_ind
            type(aq_species_c) :: DOC
            logical :: flag
            
            allocate(THIS%indices_aq_phase(this%num_species-1))
            do i=1,this%num_species-1
                call aq_phase%is_species_in_aq_phase(this%species(i),flag,aq_species_ind)
                if (flag==.true.) then
                    this%indices_aq_phase(i)=aq_species_ind
                else
                    error stop "Mineral reactant is not in aqueous phase"
                end if
            end do
        end subroutine
end module