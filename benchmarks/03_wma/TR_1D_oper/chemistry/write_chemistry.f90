!> Writes data and results of chemistry object
subroutine write_chemistry(this,unit,file_out)
    use chemistry_Lagr_m
    
    implicit none
    class(chemistry_c), intent(in) :: this !> chemistry object
    integer(kind=4), intent(in) :: unit !> file unit
    character(len=*), intent(in) :: file_out !> file output
    
    integer(kind=4) :: i,j,n,l,k
    integer(kind=4), allocatable :: tar_sol_indices(:),tar_wat_indices(:),tar_wat_indices_init(:)
    real(kind=8), allocatable :: conc_comp(:,:)
    

    write(unit,"(/,2x,'Aqueous species:'/)")
    do i=1,this%chem_syst%aq_phase%num_species
        write(unit,"(10x,A15)") this%chem_syst%aq_phase%aq_species(i)%name
    end do
    write(unit,"(2x,'Minerals:'/)")
    do i=1,this%chem_syst%num_minerals
        write(unit,"(10x,A15)") this%chem_syst%minerals(i)%name
    end do
    write(unit,"(2x,'Gases:'/)")
    do i=1,this%chem_syst%gas_phase%num_species
        write(unit,"(10x,A15)") this%chem_syst%gas_phase%gases(i)%name
    end do
    write(unit,"(2x,'Surface complexes:'/)")
    do i=1,this%chem_syst%cat_exch%num_surf_compl
        write(unit,"(10x,A15)") this%chem_syst%cat_exch%surf_compl(i)%name
    end do
    write(unit,"(2x,'Equilibrium reactions:'/)")
    do i=1,this%chem_syst%num_eq_reacts
        write(unit,"(10x,A30/)") this%chem_syst%eq_reacts(i)%name
    end do
    write(unit,"(2x,'Kinetic reactions:'/)")
    do i=1,this%chem_syst%num_lin_kin_reacts
        write(unit,"(10x,A30/)") this%chem_syst%lin_kin_reacts(i)%name
    end do
    do i=1,this%chem_syst%num_min_kin_reacts
        write(unit,"(10x,A30/)") this%chem_syst%min_kin_reacts(i)%name
    end do
    do i=1,this%chem_syst%num_redox_kin_reacts
        write(unit,"(10x,A30/)") this%chem_syst%redox_kin_reacts(i)%name
    end do
    write(unit,"(2x,'Global stoichiometric matrix:',/)")
    do i=1,this%chem_syst%num_reacts
        write(unit,"(10x,*(F15.5))") (this%chem_syst%stoich_mat(i,j), j=1,this%chem_syst%num_species)
    end do
    write(unit,"(2x,'Equilibrium constants:',/)")
    do i=1,this%chem_syst%num_eq_reacts
        write(unit,"(10x,*(ES15.5))") this%chem_syst%eq_reacts(i)%eq_cst
    end do
    write(unit,"(/,2x,'Global component matrix:'/)")
    do i=1,this%chem_syst%speciation_alg%num_prim_species
        write(unit,"(10x,*(F15.5))") (this%chem_syst%speciation_alg%comp_mat(i,j), j=1,this%chem_syst%speciation_alg%num_var_act_species)
    end do
    !write(unit,"(/,2x,'Concentration of external waters: (rows -> aqueous species, columns -> targets)'/)")
    !do i=1,this%chem_syst%aq_phase%num_species
    !    write(unit,"(10x,*(ES15.5))") (this%ext_waters(j)%concentrations(i), j=1,this%num_ext_waters)
    !end do
    write(unit,"(/,2x,'Initial concentration of aqueous species:'/)")
    do i=1,this%chem_syst%aq_phase%num_species
        write(unit,"(10x,*(ES15.5))") (this%target_waters_init(j)%concentrations(i), j=1,this%num_target_waters_init)
    end do
    if (this%chem_syst%aq_phase%ind_wat>0) then
        write(unit,"(/,2x,'Initial activity water:'/)")
        write(unit,"(10x,*(ES15.5))") (this%target_waters_init(j)%activities(this%chem_syst%aq_phase%ind_wat), j=1,this%num_target_waters_init)
        write(unit,"(/,2x,'Initial salinity:'/)")
        write(unit,"(10x,*(ES15.5))") (this%target_waters_init(j)%salinity, j=1,this%num_target_waters_init)
    end if
    if (this%chem_syst%aq_phase%ind_proton>0) then
        write(unit,"(/,2x,'Initial pH:'/)")
        write(unit,"(10x,*(ES15.5))") (this%target_waters_init(j)%pH, j=1,this%num_target_waters_init)
    end if
    !write(unit,"(/,2x,'Initial alkalinity:'/)")
    !write(unit,"(10x,*(ES15.5))") (this%target_waters_init(j)%alkalinity, j=1,this%num_target_waters_init)
    write(unit,"(/,2x,'Final concentration of aqueous species:'/)")
    do i=1,this%chem_syst%aq_phase%num_species
        write(unit,"(10x,*(ES15.5))") (this%target_waters(j)%concentrations(i), j=this%num_ext_waters+1,this%num_target_waters)
    end do
    write(unit,"(/,2x,'Final concentration of aqueous species (transposed):'/)")
    do i=this%num_ext_waters+1,this%num_target_waters
        write(unit,"(10x,*(ES15.5))") (this%target_waters(i)%concentrations(j), j=1,this%chem_syst%aq_phase%num_species)
    end do
    write(unit,"(/,2x,'Final activity coefficients aqueous species:'/)")
    do i=1,this%chem_syst%aq_phase%num_species
        write(unit,"(10x,*(ES15.5))") (10**this%target_waters(j)%log_act_coeffs(i), j=this%num_ext_waters+1,this%num_target_waters)
    end do
    if (this%chem_syst%aq_phase%ind_wat>0) then
        write(unit,"(/,2x,'Activity water:'/)")
        write(unit,"(10x,*(ES15.5))") (this%target_waters(j)%activities(this%chem_syst%aq_phase%ind_wat), j=1,this%num_target_waters)
        write(unit,"(/,2x,'Final salinity:'/)")
        write(unit,"(10x,*(ES15.5))") (this%target_waters(j)%salinity, j=this%num_ext_waters+1,this%num_target_waters)
    end if
    if (this%chem_syst%aq_phase%ind_proton>0) then
        write(unit,"(/,2x,'Final pH:'/)")
        write(unit,"(10x,*(ES15.5))") (this%target_waters(j)%pH, j=this%num_ext_waters+1,this%num_target_waters)
    end if
    !write(unit,"(/,2x,'Final alkalinity:'/)")
    !write(unit,"(10x,*(ES15.5))") (this%target_waters_init(j)%alkalinity, j=1,this%num_target_waters_init)
    write(unit,"(/,2x,'Kinetic reaction rates:'/)")
    do i=1,this%chem_syst%num_kin_reacts
        write(unit,"(10x,*(ES15.5))") (this%target_waters(j)%rk(i), j=this%num_ext_waters+1,this%num_target_waters)
    end do
    if (this%num_reactive_zones>0) then
        do l=1,this%num_reactive_zones
            write(unit,"(/,2x,'Reactive zone',I5,':'/)") l
            call this%link_target_waters_init_reactive_zone(l,tar_wat_indices_init)
            call this%link_target_waters_reactive_zone(l,tar_wat_indices)
            write(unit,"(10x,'Non-flowing species:'/)")
            do i=1,this%reactive_zones(l)%num_non_flowing_species
                write(unit,"(20x,A15/)") this%reactive_zones(l)%non_flowing_species(i)%name
            end do
            write(unit,"(10x,'Number of equilibrium reactions:',I10/)") this%reactive_zones(l)%num_eq_reactions
            write(unit,"(10x,'Stoichiometric matrix:',/)")
            do i=1,this%reactive_zones(l)%num_eq_reactions
                write(unit,"(10x,*(F15.5))") (this%reactive_zones(l)%stoich_mat(i,j), j=1,this%target_waters_init(tar_wat_indices(1))%speciation_alg%num_species)
            end do
            write(unit,"(/,10x,'Equilibrium constants:',/)")
            do i=1,this%reactive_zones(l)%num_eq_reactions
                write(unit,"(10x,*(ES15.5))") this%reactive_zones(l)%eq_reactions(i)%eq_cst
            end do
            write(unit,"(/,10x,'Primary species:',/)")
            do i=1,this%target_waters_init(tar_wat_indices_init(1))%speciation_alg%num_aq_prim_species
                write(unit,"(20x,A15)") this%target_waters_init(tar_wat_indices_init(1))%aq_phase%aq_species(this%target_waters_init(tar_wat_indices(1))%prim_species_indices(i))%name
            end do
            if (associated(this%target_waters_init(tar_wat_indices(1))%solid_chemistry)) then
                if (this%target_waters_init(tar_wat_indices_init(1))%solid_chemistry%reactive_zone%cat_exch_zone%num_surf_compl>0) then
                    write(unit,"(20x,A15)") this%target_waters_init(tar_wat_indices_init(1))%solid_chemistry%reactive_zone%cat_exch_zone%surf_compl(1)%name
                !else
                !    write(unit,"(20x,A15/)") this%target_waters_init(tar_wat_indices(1))%aq_phase%aq_species(this%target_waters_init(tar_wat_indices(1))%prim_species_indices(this%target_waters_init(tar_wat_indices(1))%speciation_alg%num_prim_species))%name
                end if
            end if
            write(unit,"(10x,'Component matrix:'/)")
            do i=1,this%target_waters_init(tar_wat_indices(1))%speciation_alg%num_prim_species
                write(unit,"(10x,*(F15.5))") (this%target_waters_init(tar_wat_indices(1))%speciation_alg%comp_mat(i,j), j=1,this%target_waters_init(tar_wat_indices(1))%speciation_alg%num_var_act_species)
            end do           
            write(unit,"(/,10x,'Initial concentration of components:'/)")
            !allocate(conc_comp(this%target_waters_init(tar_wat_indices(1))%speciation_alg%num_aq_prim_species,size(tar_wat_indices)))
            !do j=1,size(tar_wat_indices)
                do i=1,this%target_waters_init(tar_wat_indices_init(1))%speciation_alg%num_prim_species
                    !conc_comp(i,j)=dot_product(this%target_waters_init(tar_wat_indices(j))%speciation_alg%comp_mat(i,:),this%target_waters_init(tar_wat_indices(j))%concentrations(1:this%target_waters_init(tar_wat_indices(j))%speciation_alg%num_var_act_species))
                    write(unit,"(10x,*(ES15.5))") (dot_product(this%target_waters_init(tar_wat_indices_init(j))%speciation_alg%comp_mat(i,:),this%target_waters_init(tar_wat_indices_init(j))%get_conc_nc()), j=1,size(tar_wat_indices_init))
                end do
            !end do
            !write(unit,"(/,10x,'Initial concentration of solid components:'/)")
            !do i=1,this%target_waters_init(tar_wat_indices(1))%speciation_alg%num_prim_species-this%target_waters_init(tar_wat_indices(1))%speciation_alg%num_aq_prim_species
            !    write(unit,"(10x,*(ES15.5))") (this%target_waters_init(tar_wat_indices(j))%solid_chemistry%conc_comp(i), j=1,size(tar_wat_indices))
            !end do
            write(unit,"(/,10x,'Concentration of components:'/)")
            !do j=1,size(tar_wat_indices)
                do i=1,this%target_waters(tar_wat_indices(1))%speciation_alg%num_prim_species
                    !conc_comp(i,j)=dot_product(this%target_waters_init(tar_wat_indices(j))%speciation_alg%comp_mat(i,:),this%target_waters_init(tar_wat_indices(j))%concentrations(1:this%target_waters_init(tar_wat_indices(j))%speciation_alg%num_var_act_species))
                    write(unit,"(10x,*(ES15.5))") (dot_product(this%target_waters(tar_wat_indices(j))%speciation_alg%comp_mat(i,:),this%target_waters(tar_wat_indices(j))%get_conc_nc()), j=1,size(tar_wat_indices))
                end do
            write(unit,"(/,10x,'Concentration of components (transposed):'/)")
            do i=1,size(tar_wat_indices)
                write(unit,"(10x,*(ES15.5))") (dot_product(this%target_waters(tar_wat_indices(i))%speciation_alg%comp_mat(j,:),this%target_waters(tar_wat_indices(i))%get_conc_nc()), j=1,this%target_waters(tar_wat_indices(1))%speciation_alg%num_prim_species)
            end do
            !end do
            !write(unit,"(/,10x,'Concentration of solid components:'/)")
            !do i=1,this%target_waters_init(tar_wat_indices(1))%speciation_alg%num_prim_species-this%target_waters_init(tar_wat_indices(1))%speciation_alg%num_aq_prim_species
            !    write(unit,"(10x,*(ES15.5))") (this%target_waters_init(tar_wat_indices(j))%solid_chemistry%conc_comp(i), j=1,size(tar_wat_indices))
            !end do
            write(unit,"(/,10x,'Initial volumetric fractions of minerals:'/)")
            do i=1,this%reactive_zones(l)%num_minerals
                write(unit,"(10x,*(ES15.5))") (this%target_solids(tar_wat_indices_init(j))%vol_fracts(i), j=1,size(tar_wat_indices_init))
            end do
            write(unit,"(/,10x,'Volumetric fractions of minerals:'/)")
            do i=1,this%reactive_zones(l)%num_minerals
                write(unit,"(10x,*(ES15.5))") (this%target_waters(tar_wat_indices(j))%solid_chemistry%vol_fracts(i), j=1,size(tar_wat_indices))
            end do
            write(unit,"(/,10x,'Initial concentration of minerals:'/)")
            do i=1,this%reactive_zones(l)%num_minerals
                write(unit,"(10x,*(ES15.5))") (this%target_solids(tar_wat_indices_init(j))%concentrations(i), j=1,size(tar_wat_indices_init))
            end do
            write(unit,"(/,10x,'Concentration of minerals:'/)")
            do i=1,this%reactive_zones(l)%num_minerals
                write(unit,"(10x,*(ES15.5))") (this%target_waters(tar_wat_indices(j))%solid_chemistry%concentrations(i), j=1,size(tar_wat_indices))
            end do
            !write(unit,"(/,10x,'Initial concentration of surface complexes:'/)")
            !do i=1,this%reactive_zones(l)%cat_exch_zone%num_surf_compl
            !    write(unit,"(10x,*(ES15.5))") (this%target_waters_init(tar_wat_indices(j))%solid_chemistry%concentrations(this%reactive_zones(l)%num_minerals+i), j=1,size(tar_wat_indices))
            !end do
            write(unit,"(/,10x,'Concentration of surface complexes:'/)")
            do i=1,this%reactive_zones(l)%cat_exch_zone%num_surf_compl
                write(unit,"(10x,*(ES15.5))") (this%target_waters(tar_wat_indices(j))%solid_chemistry%concentrations(this%reactive_zones(l)%num_minerals+i), j=1,size(tar_wat_indices))
            end do
            write(unit,"(/,10x,'Concentration of gases:'/)")
            do i=1,this%reactive_zones(l)%gas_phase%num_species
                write(unit,"(10x,*(ES15.5))") (this%target_waters(tar_wat_indices(j))%gas_chemistry%concentrations(i), j=1,size(tar_wat_indices))
            end do
            write(unit,"(/,10x,'Initial partial pressures of gases:'/)")
            do i=1,this%reactive_zones(l)%gas_phase%num_species
                write(unit,"(10x,*(ES15.5))") (this%target_gases(tar_wat_indices_init(j))%activities(i), j=1,size(tar_wat_indices_init))
            end do
            write(unit,"(/,10x,'Partial pressures of gases:'/)")
            do i=1,this%reactive_zones(l)%gas_phase%num_species
                write(unit,"(10x,*(ES15.5))") (this%target_waters(tar_wat_indices(j))%gas_chemistry%activities(i), j=1,size(tar_wat_indices))
            end do
            write(unit,"(/,10x,'Initial volume of gas:'/)")
            if (this%reactive_zones(l)%gas_phase%num_species>0) then
                write(unit,"(10x,*(ES15.5))") (this%target_gases(tar_wat_indices_init(j))%volume, j=1,size(tar_wat_indices_init))
            end if
            write(unit,"(/,10x,'Volume of gas:'/)")
            if (this%reactive_zones(l)%gas_phase%num_species>0) then
                write(unit,"(10x,*(ES15.5))") (this%target_waters(tar_wat_indices(j))%gas_chemistry%volume, j=1,size(tar_wat_indices))
            end if
            write(unit,"(/,10x,'Aqueous equilibrium reaction rates:'/)")
            do i=1,this%reactive_zones(l)%chem_syst%aq_phase%num_aq_complexes+this%reactive_zones(l)%chem_syst%num_redox_eq_reacts
                write(unit,"(10x,*(ES15.5))") (this%target_waters(tar_wat_indices(j))%r_eq(i), j=1,size(tar_wat_indices))
            end do
            write(unit,"(/,10x,'Mineral equilibrium reaction rates:'/)")
            do i=1,this%reactive_zones(l)%num_minerals
                write(unit,"(10x,*(ES15.5))") (this%target_waters(tar_wat_indices(j))%solid_chemistry%r_eq(i), j=1,size(tar_wat_indices))
            end do
            write(unit,"(/,10x,'Gas equilibrium reaction rates:'/)")
            do i=1,this%reactive_zones(l)%gas_phase%num_species
                write(unit,"(10x,*(ES15.5))") (this%target_waters(tar_wat_indices(j))%gas_chemistry%r_eq(i), j=1,size(tar_wat_indices))
            end do
            deallocate(tar_wat_indices)
        end do
    else if (this%chem_syst%num_eq_reacts>0) then
        allocate(conc_comp(this%target_waters_init(1)%speciation_alg%num_aq_prim_species,this%num_target_waters_init))
        write(unit,"(/,2x,'Initial concentration of components:'/)")
        do j=1,this%num_target_waters_init
            do i=1,this%target_waters_init(j)%speciation_alg%num_aq_prim_species
                conc_comp(i,j)=dot_product(this%target_waters_init(j)%speciation_alg%comp_mat(i,:),this%target_waters_init(j)%concentrations(1:this%target_waters_init(j)%speciation_alg%num_var_act_species))
                write(unit,"(10x,*(ES15.5))") conc_comp(i,j) !(this%target_waters_init(tar_wat_indices(j))%concentrations(i), j=1,size(tar_wat_indices))
            end do
        end do
        write(unit,"(/,2x,'Concentration of components:'/)")
        do j=1,this%num_target_waters_init
            do i=1,this%target_waters_init(j)%speciation_alg%num_aq_prim_species
                conc_comp(i,j)=dot_product(this%target_waters_init(j)%speciation_alg%comp_mat(i,:),this%target_waters_init(j)%concentrations(1:this%target_waters_init(j)%speciation_alg%num_var_act_species))
                write(unit,"(10x,*(ES15.5))") conc_comp(i,j) !(this%target_waters_init(tar_wat_indices(j))%concentrations(i), j=1,size(tar_wat_indices))
            end do
        end do
        deallocate(conc_comp)
    end if
end subroutine