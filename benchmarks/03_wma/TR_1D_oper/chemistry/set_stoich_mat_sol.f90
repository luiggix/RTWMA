!> This subroutine sets the solid stoichiometric matrix of the reactive zone from its reactions
subroutine set_stoich_mat_sol(this)
    use reactive_zone_Lagr_m
    implicit none
    class(reactive_zone_c) :: this
    
    integer(kind=4) :: i,j,k,l
    logical :: flag

    if (this%num_eq_reactions>0) then
        i=1 !> counter total surface complexes
        j=1 !> counter equilibrium reactions reactive zone
        k=1 !> counter species of each equilibrium reaction in reactive zone
        allocate(this%stoich_mat_sol(this%num_eq_reactions,this%num_solids))
        this%stoich_mat_sol=0d0 !> initialisation
    !> We set surface complexes
        if (this%cat_exch_zone%num_surf_compl>0) then
            do
                if (this%cat_exch_zone%surf_compl(i)%name==this%eq_reactions(j)%species(k)%name) then
                    this%stoich_mat_sol(j,i)=this%eq_reactions(j)%stoichiometry(k)
                    if (j<this%num_eq_reactions) then
                        j=j+1 
                        k=1
                    else if (i<this%cat_exch_zone%num_surf_compl) then
                        i=i+1
                        j=1
                        k=1
                    else
                        exit
                    end if
                else if (k<this%eq_reactions(j)%num_species) then
                    k=k+1
                else if (j<this%num_eq_reactions) then
                    j=j+1
                    k=1
                else if (i<this%cat_exch_zone%num_surf_compl) then
                    i=i+1
                    j=1
                    k=1
                else
                    exit
                end if
            end do
        end if
    !> We set minerals
        if (this%num_minerals>0) then
            i=1 !> counter total minerals
            j=1 !> counter equilibrium reactions reactive zone
            k=1 !> counter species of each equilibrium reaction in reactive zone
            do
                if (this%minerals(i)%mineral%name==this%eq_reactions(j)%species(k)%name) then
                    this%stoich_mat_sol(j,this%cat_exch_zone%num_surf_compl+i)=this%eq_reactions(j)%stoichiometry(k)
                    if (j<this%num_eq_reactions) then
                        j=j+1 
                        k=1
                    else if (i<this%num_minerals) then
                        i=i+1
                        j=1
                        k=1
                    else
                        exit
                    end if
                else if (k<this%eq_reactions(j)%num_species) then
                    k=k+1
                else if (j<this%num_eq_reactions) then
                    j=j+1
                    k=1
                else if (i<this%num_minerals) then
                    i=i+1
                    j=1
                    k=1
                else
                    exit
                end if
            end do
        end if
    end if
end subroutine