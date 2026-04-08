function prod_diag_tridiag_mat(A,B) result(C) !> A*B=C
    use matrices_m
    class(diag_matrix_c), intent(in) :: A !> diagonal matrix
    class(tridiag_matrix_c), intent(in) :: B !> tridiagonal matrix
    type(tridiag_matrix_c) :: C
    
    n=size(A%diag)
    
    call C%allocate_matrix(n)
    
    C%diag(1)=A%diag(1)*B%diag(1)
    C%super(1)=A%diag(1)*B%super(1)
    
    C%sub(n-1)=A%diag(n)*B%sub(n-1)
    C%diag(n)=A%diag(n)*B%diag(n)
    if (n>2) then
        do i=2,n-1
            C%diag(i)=A%diag(i)*B%diag(i)
            C%super(i)=A%diag(i)*B%super(i)
            C%sub(i-1)=A%diag(i)*B%sub(i-1)
        end do
    end if
    
end function