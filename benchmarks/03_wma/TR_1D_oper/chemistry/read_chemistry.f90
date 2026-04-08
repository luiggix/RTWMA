!> Lectura quimica
subroutine read_chemistry(this,path_inp,path_DB,unit_chem_syst_file,chem_syst_file,unit_loc_chem_file,loc_chem_file,unit_target_waters_init_file,target_waters_init_file)
    use chemistry_Lagr_m
    implicit none
    class(chemistry_c) :: this
    character(len=*), intent(in) :: path_inp
    character(len=*), intent(in) :: path_DB
    integer(kind=4), intent(in) :: unit_chem_syst_file
    character(len=*), intent(in) :: chem_syst_file
    integer(kind=4), intent(in) :: unit_loc_chem_file
    character(len=*), intent(in) :: loc_chem_file
    integer(kind=4), intent(in) :: unit_target_waters_init_file
    character(len=*), intent(in) :: target_waters_init_file
    
    integer(kind=4) :: i
    
    if (this%option==1) then !> CHEPROO
        call this%read_chemistry_CHEPROO(path_inp,path_DB,unit_chem_syst_file,chem_syst_file,unit_loc_chem_file,loc_chem_file,unit_target_waters_init_file,target_waters_init_file)
    else if (this%option==2) then !> PHREEQC
        !call this%read_chemistry_PHREEQC()
        error stop "PHREEQC data input not fully implemented yet"
    else if (this%option==3) then !> PFLOTRAN
        !call this%read_chemistry_PFLOTRAN()
        error stop "PFLOTRAN data input not fully implemented yet"
    else
        error stop "This data input option is not implemented yet"
    end if
!> Autentica chapuza
    if (this%num_reactive_zones==1) then
        do i=1,this%num_target_solids
            call this%target_solids_init(i)%set_reactive_zone(this%reactive_zones(1))
            call this%target_solids(i)%set_reactive_zone(this%reactive_zones(1))
        end do
        do i=1,this%num_target_gases
            call this%target_gases(i)%set_reactive_zone(this%reactive_zones(1))
        end do
    end if
end subroutine