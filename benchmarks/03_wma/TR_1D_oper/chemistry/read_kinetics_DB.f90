!> Reads 'kinetics_modif.dat' database
subroutine read_kinetics_DB(this,path,unit)
    use chem_system_m
    implicit none
    class(chem_system_c) :: this
    character(len=*), intent(in) :: path
    integer(kind=4), intent(in) :: unit
    
    integer(kind=4) :: i,j,k,l,N_k,N_c,mineral_ind,cat_ind
    integer(kind=4), allocatable :: global_stoich_counter(:),global_stoich_indices(:),aux_ind(:)
    logical :: flag
    real(kind=8) :: unk,logK,valence,act_E,supersat_thr,cst
    real(kind=8), allocatable :: global_stoich_coeffs(:),aux_coeffs(:)
    character(len=256) :: name,comments,label,filename
    character(len=256), allocatable :: str1(:),str2(:),species_str(:),global_stoich_names(:)
    
    type(mineral_c) :: mineral
    type(kin_mineral_params_c) :: min_params
    
    type(aq_species_c), allocatable :: catalysers(:)
    
    filename=trim(path)//'\kinetics_modif.dat'

    open(unit,file=filename,status='old',action='read')
    i=1 !> counter kinetic minerals chemical system
    j=1 !> counter number parallel reactions
    k=1 !> counter number catalysts
    read(unit,*) label
    do
        read(unit,*) mineral%name, N_k, N_c, act_E, supersat_thr, comments
        if (name=='null') then
            exit
        else
            call this%is_mineral_in_chem_syst(mineral,flag,mineral_ind)
            if (flag==.true. .and. mineral_ind<=this%num_min_kin_reacts) then
                min_params%num_par_reacts=N_k
                min_params%num_cat=N_c
                min_params%act_energy=act_E*1484 !> reads kcal/mol so we have to convert to Joules
                min_params%supersat_threshold=supersat_thr
                call min_params%allocate_constants()
                call min_params%allocate_cat_indices()
                allocate(catalysers(min_params%num_cat))
                do j=1,min_params%num_par_reacts
                    !read(unit,*) min_params%k(j), min_params%theta(j), min_params%eta(j), min_params%num_cat, ((min_params%catalysers(l)%name, min_params%p(j,l)), l=1,min_params%num_cat)
                    read(unit,*) min_params%k(j), min_params%theta(j), min_params%eta(j), ((catalysers(k)%name, min_params%p(j,k)), k=1,min_params%num_cat)
                    do k=1,min_params%num_cat
                        call this%aq_phase%is_species_in_aq_phase(catalysers(k),flag,cat_ind)
                        if (flag==.false.) then
                            error stop
                        else
                            min_params%cat_indices(k)=cat_ind
                        end if
                    end do
                end do
                call this%min_kin_reacts(mineral_ind)%set_mineral_params(min_params)
                !call kin_min_react%set_mineral(this%minerals(i))
                !this%min_kin_reacts(i)=kin_min_react !> para cada mineral asignamos su reaccion cinética (must be allocated)
                if (i<this%num_min_kin_reacts) then
                    i=i+1
                else
                    exit
                end if
            else
                read(unit,*) cst
            end if
        end if
        !else if (str==this%surf_compl%surf_compl(k)%name) then
        !>    n_r=n_r+1
        !>    backspace(15)
        !>    read(15,*) str, num_reactants
        !>    print *, str, num_reactants
        !>    backspace(15)
        !>    call react%kin_reaction%allocate_reaction(num_reactants+1)
        !>    read(15,*) str, num_reactants, ((react%kin_reaction%stoichiometry(j), react%kin_reaction%species(j)%name), j=1,num_reactants), logK, valence
        !>    call react%kin_reaction%set_eq_cst(10**logK)
        !>    call this%surf_compl%surf_compl(k)%set_valence(int(valence))
        !>    if (k<this%surf_compl%num_surf_compl) then
        !>        k=k+1
        !>    else
        !>        continue
        !>    end if
        !else if (str==this%cat_exch%exch_cats(l)%name) then
        !>    n_r=n_r+1
        !>    backspace(15)
        !>    read(15,*) str, num_reactants
        !>    print *, str, num_reactants
        !>    backspace(15)
        !>    call react%kin_reaction%allocate_reaction(num_reactants+1)
        !>    read(15,*) str, num_reactants, ((react%kin_reaction%stoichiometry(j), react%kin_reaction%species(j)%name), j=1,num_reactants), logK
        !>    call react%kin_reaction%set_eq_cst(10**logK)
        !>    call this%cat_exch%exch_cats(l)%set_valence(int(valence))
        !>    if (l<this%cat_exch%num_exch_cats) then
        !>        l=l+1
        !>    else
        !>        continue
        !>    end if
        !else
        !>    error stop "This species is not present in the master25 database"
        !end if
        !reacts_old=reacts_new
        !deallocate(reacts_new)
        !allocate(reacts_new(n_r))
        !do j=1,n_r-1
        !>    reacts_new(j)=reacts_old(j)
        !end do
        !reacts_new(n_r)=react
        !deallocate(reacts_old,react%kin_reaction%species,react%kin_reaction%stoichiometry)        
    end do
    close(unit)
!> Aqui meto dos reacciones Monod con calzador
    !allocate(Monod_array(2))
    !call Monod_array(1)%set_react_name('aerobic oxidation')
    !call Monod_array(2)%set_react_name('denitrification')
    !do i=1,size(Monod_array)
    !>    call Monod_array(i)%read_kin_reaction()
    !>    react%kin_reaction=>Monod_array(i)
    !>    !call Monod_array(i)%append_Monod_reaction(this%redox_kin_reacts)
    !>    this%kin_reacts=[this%kin_reacts,react] !> chapuza
    !end do
    !call this%set_redox_kin_reacts(Monod_array)
    !call this%set_lin_kin_reacts([lin_react])
!> Aqui meto una reaccion lineal con calzador
    !call lin_react%allocate_reaction(2)
    !lin_react%species(1)%name='A'
    !lin_react%species(2)%name='B'
    !lin_react%stoichiometry=[-1,1]
    !lin_react%react_type=5
    !call lin_react%set_lambda(-1d0)
    !call lin_react%set_react_name("lineal_A_B")
    !react%kin_reaction=>lin_react
    !call this%set_kin_reacts([react])
    !call this%set_lin_kin_reacts([lin_react])
end subroutine