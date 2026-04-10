!> This subroutine sets the stoichiometric matrix of the reactive zone from its reactions
subroutine set_stoich_mat_react_zone(this)
    use reactive_zone_Lagr_m
    implicit none
    class(reactive_zone_c) :: this
    
    integer(kind=4) :: i,j,k,l,n_k,num_sp
    logical :: flag
    
    num_sp=this%compute_num_species_react_zone() !> number of species involved with reactive zone
        
    i=1 !> counter total species chemical system
    j=1 !> counter equilibrium reactions reactive zone
    k=1 !> counter species of each equilibrium reaction in reactive zone
    l=1 !> counter total species reactive zone
    if (this%num_eq_reactions>0) then
        if (allocated(this%stoich_mat)) then
            deallocate(this%stoich_mat)
        end if
        allocate(this%stoich_mat(this%num_eq_reactions,num_sp))
        this%stoich_mat=0d0 !> initialisation
        do
            if (this%chem_syst%species(i)%name==this%eq_reactions(j)%species(k)%name) then
                flag=.true. !> species is involved in reactive zone
                this%stoich_mat(j,l)=this%eq_reactions(j)%stoichiometry(k)
                if (j<this%num_eq_reactions) then
                    j=j+1 
                    k=1
                else if (l<num_sp) then
                    i=i+1
                    l=l+1
                    j=1
                    k=1
                    flag=.false.
                else
                    exit
                end if
            else if (k<this%eq_reactions(j)%num_species) then
                k=k+1
            else if (j<this%num_eq_reactions) then
                j=j+1
                k=1
            else if (i<this%chem_syst%num_species) then
                if (flag==.true. .or. i<=this%chem_syst%num_species-this%num_minerals) then
                    if (l<num_sp) then
                        l=l+1
                    else
                        exit
                    end if
                end if
                i=i+1
                j=1
                k=1
                flag=.false.
            else
                exit
            end if
        end do
    end if
end subroutine