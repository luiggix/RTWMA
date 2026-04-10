module strings_m
    interface
        subroutine compare_str_arrays(str_arr_1,str_arr_2,flag,indices)
            implicit none
            character(len=*), intent(in) :: str_arr_1(:)
            character(len=*), intent(in) :: str_arr_2(:)
            logical, intent(out) :: flag
            integer(kind=4), intent(out), allocatable, optional :: indices(:)
        end subroutine
    end interface
end module