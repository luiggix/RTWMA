!> This subroutine sets the stoichiometric matrices of the chemical system from its equilibrium and kinetic reactions
subroutine set_stoich_mat(this)
    use chem_system_m
    implicit none
    class(chem_system_c) :: this
    
    integer(kind=4) :: i,j,k,n_k

    if (allocated(this%stoich_mat)) then
        deallocate(this%stoich_mat)
    end if
    allocate(this%stoich_mat(this%num_reacts,this%num_species))
    
    this%stoich_mat=0d0 !> initialisation
    
    if (allocated(this%Se)) then
        deallocate(this%Se)
    end if
    allocate(this%Se(this%num_eq_reacts,this%num_species))
    this%Se=0d0 !> initialisation
    
    if (allocated(this%Sk)) then
        deallocate(this%Sk)
    end if
    allocate(this%Sk(this%num_kin_reacts,this%num_species))
    this%Sk=0d0 !> initialisation
    
    if (this%num_eq_reacts>0) then
        i=1 !> counter total species
        j=1 !> counter equilibrium reactions
        k=1 !> counter species of each equilibrium reaction
        do
            if (this%species(i)%name==this%eq_reacts(j)%species(k)%name) then
                this%Se(j,i)=this%eq_reacts(j)%stoichiometry(k)
                print *, this%species(i)%name
                if (j<this%num_eq_reacts) then
                    j=j+1
                    k=1
                else if (i<this%num_species) then
                    i=i+1
                    j=1
                    k=1
                else
                    exit
                end if
            else if (k<this%eq_reacts(j)%num_species) then
                k=k+1
            else if (j<this%num_eq_reacts) then
                j=j+1
                k=1
            else if (i<this%num_species) then
                i=i+1
                j=1
                k=1
            else
                exit
            end if
        end do
    end if
    if (this%num_kin_reacts>0) then
        n_k=0 !> counter number kinetic reactions
        if (this%num_lin_kin_reacts>0) then
            i=1 !> counter total species
            j=1 !> counter linear kinetic reactions
            k=1 !> counter species of each linear kinetic reaction
            do
                if (this%species(i)%name==this%lin_kin_reacts(j)%species(k)%name) then
                    this%Sk(j,i)=this%lin_kin_reacts(j)%stoichiometry(k)
                    if (j<this%num_lin_kin_reacts) then
                        j=j+1
                        k=1
                    else if (i<this%num_species) then
                        i=i+1
                        j=1
                        k=1
                    else
                        exit
                    end if
                else if (k<this%lin_kin_reacts(j)%num_species) then
                    k=k+1
                else if (j<this%num_lin_kin_reacts) then
                    j=j+1
                    k=1
                else if (i<this%num_species) then
                    i=i+1
                    j=1
                    k=1
                else
                    exit
                end if
            end do
            n_k=n_k+this%num_lin_kin_reacts
        end if
        if (this%num_redox_kin_reacts>0) then
            i=1 !> counter total species
            j=1 !> counter Monod kinetic reactions
            k=1 !> counter species of each Monod kinetic reaction
            do
                if (this%species(i)%name==this%redox_kin_reacts(j)%species(k)%name) then
                    this%Sk(n_k+j,i)=this%redox_kin_reacts(j)%stoichiometry(k)
                    if (j<this%num_redox_kin_reacts) then
                        j=j+1
                        k=1
                    else if (i<this%num_species) then
                        i=i+1
                        j=1
                        k=1
                    else
                        exit
                    end if
                else if (k<this%redox_kin_reacts(j)%num_species) then
                    k=k+1
                else if (j<this%num_redox_kin_reacts) then
                    j=j+1
                    k=1
                else if (i<this%num_species) then
                    i=i+1
                    j=1
                    k=1
                else
                    exit
                end if
            end do
            n_k=n_k+this%num_redox_kin_reacts
        end if
        if (this%num_min_kin_reacts>0) then
            i=1 !> counter total species
            j=1 !> counter mineral kinetic reactions
            k=1 !> counter species of each mineral kinetic reaction
            do
                if (this%species(i)%name==this%min_kin_reacts(j)%species(k)%name) then
                    this%Sk(n_k+j,i)=this%min_kin_reacts(j)%stoichiometry(k)
                    if (j<this%num_min_kin_reacts) then
                        j=j+1
                        k=1
                    else if (i<this%num_species) then
                        i=i+1
                        j=1
                        k=1
                    else
                        exit
                    end if
                else if (k<this%min_kin_reacts(j)%num_species) then
                    k=k+1
                else if (j<this%num_min_kin_reacts) then
                    j=j+1
                    k=1
                else if (i<this%num_species) then
                    i=i+1
                    j=1
                    k=1
                else
                    exit
                end if
            end do
            n_k=n_k+this%num_min_kin_reacts
        end if
        if (n_k/=this%num_kin_reacts) error stop "Error in n_k (set_stoich_mat)"
    end if
    this%stoich_mat(1:this%num_eq_reacts,:)=this%Se
    this%stoich_mat(this%num_eq_reacts+1:this%num_reacts,:)=this%Sk
    !open(97,file='C:\Users\Jordi\OneDrive\Documentos\trabajo\python\Se.dat')
    !do i=1,this%num_eq_reacts
    !    write(97,"(*(ES15.5))") (this%Se(i,j), j=1,this%num_species)
    !end do
    !close(97)
end subroutine