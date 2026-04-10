function prod_tridiag_diag_mat(A,B) result(C) !> A*B=C
    use matrices_m
    class(tridiag_matrix_c), intent(in) :: A !> tridiagonal matrix
    class(diag_matrix_c), intent(in) :: B !> diagonal matrix
    type(tridiag_matrix_c) :: C
    
    n=size(B%diag)
    
    call C%allocate_matrix(n)
        
    C%diag(1)=A%diag(1)*B%diag(1)
    
    if (n>1) then
        C%super(1)=A%super(1)*B%diag(2)
        C%sub(n-1)=A%sub(n-1)*B%diag(n-1)
        C%diag(n)=A%diag(n)*B%diag(n)
    end if
    if (n>2) then
        do i=2,n-1
            C%diag(i)=A%diag(i)*B%diag(i)
            C%super(i)=A%super(i)*B%diag(i+1)
            C%sub(i-1)=A%sub(i-1)*B%diag(i-1)
        end do
    end if
end function