!> Writes data and results of 1D reactive transport model
subroutine write_RT_1D(this,unit,file_out,path_py)
    use RT_1D_m
    implicit none
    class(RT_1D_c), intent(in) :: this
    integer(kind=4), intent(in) :: unit
    character(len=*), intent(in) :: file_out
    character(len=*), intent(in), optional :: path_py
    
    open(unit,file=file_out,status='unknown',form='formatted')
    call this%write_transport_data(unit,file_out)
    call this%chemistry%write_chemistry(unit,file_out)
    close(unit)
    if (present(path_py)) then
        call this%write_python(path_py)
    end if
end subroutine