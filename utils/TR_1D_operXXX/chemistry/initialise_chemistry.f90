subroutine initialise_chemistry(this,path_inp,path_DB,unit_chem_syst_file,chem_syst_file,unit_loc_chem_file,loc_chem_file,unit_target_waters_init_file,target_waters_init_file)
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
!**************************************************************************************************
!> Read chemistry
    call this%read_chemistry(path_inp,path_DB,unit_chem_syst_file,chem_syst_file,unit_loc_chem_file,loc_chem_file,unit_target_waters_init_file,target_waters_init_file)
end subroutine

