!> Writes data and results of chemistry object
subroutine write_python(this,path)
    use RT_1D_m
    
    implicit none
    class(RT_1D_c), intent(in) :: this !> chemistry object
    character(len=*), intent(in) :: path !> path output
    
    integer(kind=4) :: i,j,n,l,k
    integer(kind=4), allocatable :: tar_sol_indices(:),tar_wat_indices(:)
        
    select type(this)
    type is (RT_1D_transient_c)
        open(9988,file=trim(path)//'u_init.dat')
        do i=1,this%chemistry%chem_syst%speciation_alg%num_aq_prim_species
            write(9988,"(*(ES15.5))") (dot_product(this%chemistry%target_waters_init(j)%speciation_alg%comp_mat(i,:),this%chemistry%target_waters_init(j)%get_conc_nc()), j=1,this%chemistry%num_target_waters_init)
        end do
        close(9988)
        open(999,file=trim(path)//'c1_aq_init.dat')
        do i=1,this%chemistry%chem_syst%speciation_alg%num_aq_prim_species
            write(999,"(*(ES15.5))") (this%chemistry%target_waters_init(j)%concentrations(i), j=1,this%chemistry%num_target_waters_init)
        end do
        close(999)
        open(9999,file=trim(path)//'gamma_aq_init.dat')
        do i=1,this%chemistry%chem_syst%aq_phase%num_species
            write(9999,"(*(ES15.5))") (10**(this%chemistry%target_waters_init(j)%log_act_coeffs(i)), j=1,this%chemistry%num_target_waters_init)
        end do
        close(9999)
        open(998,file=trim(path)//'lambdas_filas.dat')
        if (this%transport%time_discr%int_method==1) then
            do i=1,this%transport%mixing_ratios%num_cols
                write(998,"(*(F15.5))") (this%transport%mixing_ratios%cols(i)%col_1(j), j=1,this%transport%mixing_ratios%cols(i)%dim)
            end do
        else if (this%transport%time_discr%int_method==2) then
            do i=1,this%transport%mixing_ratios%num_cols
                write(998,"(*(F15.5))") (this%transport%mixing_ratios_mat(j,i), j=1,this%transport%mixing_ratios%num_cols)
            end do
        end if
        close(998)
        open(997,file=trim(path)//'lambdas_cols.dat')
        if (this%transport%time_discr%int_method==1) then
            do i=1,this%transport%mixing_ratios%num_cols
                write(997,"(*(F15.5))") (this%transport%mixing_ratios%cols(i)%col_1(j), j=1,this%transport%mixing_ratios%cols(i)%dim)
            end do
        else if (this%transport%time_discr%int_method==2) then
            do i=1,this%transport%mixing_ratios%num_cols
                write(997,"(*(F15.5))") (this%transport%mixing_ratios_mat(i,j), j=1,this%transport%mixing_ratios%num_cols)
            end do
        end if
        close(997)
    end select
end subroutine