subroutine compute_eigenvalues(this)
    use matrices_m
    implicit none
    class(matrix_c) :: this
    
    real(kind=8), allocatable :: d(:),e(:)
    integer(kind=4) :: info
!> Lapack
    select type (this)
    class is (tridiag_sym_matrix_c)
        d=this%diag
        e=this%sub
        !call dsterf(size(d),d,e,info)
        this%eigenvalues=d
    end select
end subroutine