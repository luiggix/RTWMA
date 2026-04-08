module matrices_m
    use vectors_m
    implicit none
    save
    
    type, public, abstract :: matrix_c !> matrix superclass
        integer(kind=4) :: num_cols
    contains
        procedure, public :: allocate_matrix
        procedure, public :: allocate_columns
        procedure, public :: prod_mat_vec
        !procedure, public :: prod_mat_mat
        procedure, public :: get_diag
        procedure, public :: get_sub
        procedure, public :: get_super
        procedure, public :: compute_norm_inf
        procedure, public :: compute_norm_1
    end type
    
    type, public, extends(matrix_c) :: non_sq_matrix_c
    end type
    
    type, public, extends(non_sq_matrix_c) :: matrix_real_c
        type(vector_real_c), allocatable :: cols(:)
    contains
        procedure, public :: is_matrix_full
    end type
    
    type, public, extends(non_sq_matrix_c) :: matrix_int_c
        type(vector_int_c), allocatable :: cols(:)
    contains
        procedure, public :: get_vector_int
    end type
    
    type, public, extends(matrix_c) :: sq_matrix_c
        !integer(kind=4) :: dim
        real(kind=8), allocatable :: eigenvalues(:)
        real(kind=8), allocatable :: eigenvectors(:,:)
    contains
        procedure, public :: compute_eigenvalues
        procedure, public :: compute_eigenvectors
    end type
!****************************************************************************************************************************************************
    type, public, extends(sq_matrix_c) :: tridiag_sym_Toeplitz_matrix_c
        real(kind=8) :: sub
        real(kind=8) :: diag
    end type
    
    type, public, extends(tridiag_sym_Toeplitz_matrix_c) :: tridiag_Toeplitz_matrix_c
        real(kind=8) :: super
    contains
        procedure, public :: set_tridiag_Toeplitz_matrix
    end type
!****************************************************************************************************************************************************
    type, public, extends(sq_matrix_c) :: diag_matrix_c
        real(kind=8), allocatable :: diag(:)
    contains
        procedure, public :: set_diag_matrix
        procedure, public :: prod_mat_diag_mat
    end type
    
    type, public, extends(diag_matrix_c) :: tridiag_sym_matrix_c
        real(kind=8), allocatable :: sub(:)
    contains
        procedure, public :: check_eigenvectors_tridiag_sym_matrix
    end type

    type, public, extends(tridiag_sym_matrix_c) :: tridiag_matrix_c
        real(kind=8), allocatable :: super(:)
    contains
        procedure, public :: set_tridiag_matrix
        procedure, public :: compute_transpose_tridiag_matrix
        procedure, public :: compute_inverse_tridiag_matrix
        procedure, public :: prod_tridiag_mat_mat
    end type

    !type, public, extends(tridiag_matrix_c) :: tridiag_matrix_vec_c
    !>    real(kind=8), allocatable :: vector(:)
    !end type
!****************************************************************************************************************************************************
    interface
        subroutine compute_eigenvalues(this)
            import sq_matrix_c
            implicit none
            class(sq_matrix_c) :: this
        end subroutine
        
        subroutine compute_eigenvectors(this)
            import sq_matrix_c
            implicit none
            class(sq_matrix_c) :: this
        end subroutine
        

        
        
        
        function id_matrix(n)
        !> Identity nxn matrix
            implicit none
            integer(kind=4), intent(in) :: n
            integer(kind=4) :: i
            real(kind=8) :: id_matrix(n,n)
        end function
        
        function norm_mat_inf(A) result(norm)
            implicit none
            real(kind=8), intent(in) :: A(:,:)
            real(kind=8) :: norm
        end function
        
        function norm_mat_1(A) result(norm)
            implicit none
            real(kind=8), intent(in) :: A(:,:)
            real(kind=8) :: norm
        end function
        
        function det(A)
        !> Determinant of square matrix using LU decomposition
            implicit none
            real(kind=8), intent(in) :: A(:,:)
            real(kind=8) :: det
        end function det
        
        subroutine inv_matrix(A,tol,inv)
        !> Inverse of square matrix using Gauss-Jordan
            implicit none
            real(kind=8), intent(in) :: A(:,:)
            real(kind=8), intent(in) :: tol
            real(kind=8), intent(out) :: inv(:,:)
        end subroutine inv_matrix
        
        function prod_mat_vec(this,b) result(x) !> Ab=x
            import matrix_c
            implicit none
            class(matrix_c), intent(in) :: this
            real(kind=8), intent(in) :: b(:)
            real(kind=8), allocatable :: x(:)
        end function
        
        function prod_tridiag_mat_mat(this,B_mat) result(C_mat) !> AB=C
            import tridiag_matrix_c
            implicit none
            class(tridiag_matrix_c), intent(in) :: this
            real(kind=8), intent(in) :: B_mat(:,:)
            real(kind=8), allocatable :: C_mat(:,:)
        end function 
        
        subroutine LU(A,L,U)
            implicit none
            real(kind=8), intent(in) :: A(:,:)
            real(kind=8), intent(out) :: L(:,:), U(:,:)
        end subroutine LU
        
        subroutine Cholesky(A,L)
            implicit none
            real(kind=8), intent(in) :: A(:,:)
            real(kind=8), intent(out) :: L(:,:)
        end subroutine Cholesky
        
        subroutine QR_Householder(A,Q,R)
            implicit none
            real(kind=8), intent(in) :: A(:,:) !> square matrix
            real(kind=8), intent(out) :: Q(:,:), R(:,:)
        end subroutine QR_Householder
        
        function Householder(x)
            implicit none
            real(kind=8), intent(in) :: x(:)
            real(kind=8), allocatable :: Householder(:,:)
        end function Householder
        
        subroutine Gram_Schmidt_mat(v,u)
            implicit none
            real(kind=8), intent(in) :: v(:,:)
            real(kind=8), intent(out) :: u(:,:)
        end subroutine
        
        subroutine non_zeros(A,n,ind)
            implicit none
            real(kind=8), intent(in) :: A(:,:)
            integer(kind=4), intent(out) :: n
            integer(kind=4), intent(out) :: ind(:,:)
        end subroutine
        
        function prod_AT_A(A) result(AT_A)
            implicit none
            real(kind=8), intent(in) :: A(:,:)
            real(kind=8), allocatable :: AT_A(:,:)
        end function
        
        subroutine check_eigenvectors_tridiag_sym_matrix(this,tolerance)
            import tridiag_sym_matrix_c
            implicit none
            class(tridiag_sym_matrix_c), intent(in) :: this
            real(kind=8), intent(in) :: tolerance
        end subroutine
        
        function prod_diag_tridiag_mat(A,B) result(C)
            import diag_matrix_c
            import tridiag_matrix_c
            class(diag_matrix_c), intent(in) :: A
            class(tridiag_matrix_c), intent(in) :: B
            type(tridiag_matrix_c) :: C
        end function
        
        function prod_tridiag_diag_mat(A,B) result(C)
            import diag_matrix_c
            import tridiag_matrix_c
            class(tridiag_matrix_c), intent(in) :: A
            class(diag_matrix_c), intent(in) :: B
            type(tridiag_matrix_c) :: C
        end function
        
        subroutine compute_inverse_tridiag_matrix(this,tol,inv_mat)
            import tridiag_matrix_c
            implicit none
            class(tridiag_matrix_c), intent(in) :: this
            real(kind=8), intent(in) :: tol
            real(kind=8), intent(out) :: inv_mat(:,:)
        end subroutine
    end interface
!****************************************************************************************************************************************************
    contains
        subroutine allocate_matrix(this,n)
            implicit none
            class(matrix_c) :: this
            integer(kind=4), intent(in), optional :: n
            if (present(n)) then
                this%num_cols=n
            end if
            select type (this)
            class is (diag_matrix_c)
                allocate(this%diag(this%num_cols))
                select type (this)
                class is (tridiag_sym_matrix_c)
                    allocate(this%sub(this%num_cols-1))
                    select type (this)
                    type is (tridiag_matrix_c)
                        allocate(this%super(this%num_cols-1))
                        !select type (this)
                        !type is (tridiag_matrix_vec_c)
                        !>    allocate(this%vector(n))
                        !end select
                    end select
                end select
            type is (matrix_real_c)
                allocate(this%cols(this%num_cols))
                !do i=1,this%num_cols
                !    call this%cols(i)%allocate_vector()
            type is (matrix_int_c)
                if (allocated(this%cols)) then
                    deallocate(this%cols)
                end if
                allocate(this%cols(this%num_cols))
            end select
        end subroutine
        
        subroutine allocate_columns(this)
            implicit none
            class(matrix_c) :: this
            integer(kind=4) :: i
            select type (this)
            type is (matrix_real_c)
                do i=1,this%num_cols
                    call this%cols(i)%allocate_vector()
                end do
            type is (matrix_int_c)
                do i=1,this%num_cols
                    call this%cols(i)%allocate_vector()
                end do
            end select
        end subroutine
        
        subroutine set_diag_matrix(this,diag)
            implicit none
            class(diag_matrix_c) :: this
            real(kind=8), intent(in) :: diag(:)
            this%diag=diag
            this%num_cols=size(diag)
        end subroutine
        
        subroutine set_tridiag_Toeplitz_matrix(this,sub,diag,super)
            implicit none
            class(tridiag_Toeplitz_matrix_c) :: this
            real(kind=8), intent(in) :: sub,diag,super
            this%sub=sub
            this%diag=diag
            this%super=super
        end subroutine
        
        subroutine set_tridiag_matrix(this,sub,diag,super)
            implicit none
            class(tridiag_matrix_c) :: this
            real(kind=8), intent(in) :: sub(:),diag(:),super(:)
            if (size(sub)/=size(super) .or. size(diag)/=(size(sub)+1)) error stop "Dimension error in set_tridiag_matrix" 
            this%sub=sub
            this%diag=diag
            this%super=super
        end subroutine
        
        function get_diag(this) result(diag)
            implicit none
            class(matrix_c), intent(in) :: this
            real(kind=8), allocatable :: diag(:)
            select type (this)
            class is (diag_matrix_c)
                diag=this%diag
            end select
        end function
        
        function get_sub(this) result(sub)
            implicit none
            class(matrix_c), intent(in) :: this
            real(kind=8), allocatable :: sub(:)
            select type (this)
            class is (tridiag_sym_matrix_c)
                sub=this%sub
            end select
        end function
        
        function get_super(this) result(super)
            implicit none
            class(matrix_c), intent(in) :: this
            real(kind=8), allocatable :: super(:)
            select type (this)
            class is (tridiag_matrix_c)
                super=this%super
            end select
        end function
        
        function compute_norm_inf(this) result(norm)
            implicit none
            class(matrix_c), intent(in) :: this
            real(kind=8) :: norm
            
            integer(kind=4) :: i,n
            real(kind=8) :: norm_i
            
            select type (this)
            type is (diag_matrix_c)
                n=size(this%diag)
                norm=abs(this%diag(1))
                do i=2,n
                    norm_i=abs(this%diag(1))
                    if (norm_i>norm) then
                        norm=norm_i
                    end if
                end do
            type is (tridiag_matrix_c)
                n=size(this%diag)
                norm=abs(this%diag(1))+abs(this%super(1))
                do i=2,n-1
                    norm_i=abs(this%sub(i-1))+abs(this%diag(i))+abs(this%super(i))
                    if (norm_i>norm) then
                        norm=norm_i
                    end if
                end do
                norm_i=abs(this%sub(n-1))+abs(this%diag(n))
                if (norm_i>norm) then
                    norm=norm_i
                end if
            end select
        end function
        
        function compute_norm_1(this) result(norm)
            implicit none
            class(matrix_c), intent(in) :: this
            real(kind=8) :: norm
            
            type(tridiag_matrix_c) :: transpose
            
            select type (this)
            type is (tridiag_matrix_c)
                call this%compute_transpose_tridiag_matrix(transpose)
                norm=transpose%compute_norm_inf()
            end select
        end function
        
        subroutine compute_transpose_tridiag_matrix(this,transpose)
            implicit none
            class(tridiag_matrix_c), intent(in) :: this
            class(tridiag_matrix_c), intent(out) :: transpose
            transpose%sub=this%super
            transpose%diag=this%diag
            transpose%super=this%sub
        end subroutine
        
        subroutine get_submatrix(mat,rows,cols,submat)
            implicit none
            real(kind=8), intent(in) :: mat(:,:)
            integer(kind=4), intent(in) :: rows(:),cols(:)
            real(kind=8), intent(out) :: submat(:,:) !> must be allocated
            
            integer(kind=4) :: i,j,n_r,n_c,i_r,i_c
            n_r=size(rows)
            n_c=size(cols)
            do i=1,n_r
                do j=1,n_c
                    submat(i,j)=mat(rows(i),cols(j))
                end do
            end do
        end subroutine
        
        function prod_mat_diag_mat(this,mat) result(prod)
            implicit none
            class(diag_matrix_c), intent(in) :: this
            real(kind=8), intent(in) :: mat(:,:)
            real(kind=8), allocatable :: prod(:,:)
            integer(kind=4) :: j
            if (size(mat,2)/=this%num_cols) then
                error stop
            end if
            allocate(prod(size(mat,1),this%num_cols))
            do j=1,this%num_cols
                prod(:,j)=mat(:,j)*this%diag(j)
            end do
        end function
        
        function get_vector_int(this) result(vector_int)
            implicit none
            class(matrix_int_c), intent(in) :: this
            integer(kind=4), allocatable :: vector_int(:)
            
            integer(kind=4) :: i,j,dim_vec
            
            dim_vec=0
            do i=1,this%num_cols
                dim_vec=dim_vec+this%cols(i)%dim
            end do
            allocate(vector_int(dim_vec))
            
            j=1
            do i=1,this%num_cols
                vector_int(j:j+this%cols(i)%dim-1)=this%cols(i)%col_1
                j=j+this%cols(i)%dim
            end do
        end function
        
        subroutine is_matrix_full(this,flag)
            implicit none
            class(matrix_real_c), intent(in) :: this
            logical, intent(out) :: flag
            
            integer(kind=4) :: i
            
            flag=.true.
            do i=1,this%num_cols
                if (this%cols(i)%dim<this%num_cols) then
                    flag=.false.
                    exit
                end if
            end do
        end subroutine
        
end module 