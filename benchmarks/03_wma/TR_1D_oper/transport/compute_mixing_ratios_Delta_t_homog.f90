!> Computes mixing ratios matrix with uniform time stepping
subroutine compute_mixing_ratios_Delta_t_homog(this,A_mat_lumped)
    use BCs_subroutines_m
    implicit none
    
    class(PDE_1D_transient_c) :: this
    type(diag_matrix_c), intent(out), optional :: A_mat_lumped
    
    integer(kind=4) :: i,j,n
    real(kind=8) :: lambda,theta
    real(kind=8), parameter :: tol_inv=1d-12
    real(kind=8), allocatable :: r(:),A_mat_inv(:,:),mix_ratios(:,:)
    
    type(tridiag_matrix_c) :: E_mat,B_mat_T
    
    call this%mixing_ratios%allocate_matrix(this%spatial_discr%Num_targets)
    call this%mixing_waters_indices%allocate_matrix(this%mixing_ratios%num_cols)
!> We compute PDE arrays
    call this%compute_trans_mat_PDE()
    call this%compute_source_term_PDE()
    call this%compute_F_mat_PDE()
!> We impose BCs
    if (this%BCs%BCs_label(1)==1 .and. this%BCs%BCs_label(2)==1) then
        call Dirichlet_BCs_PDE(this)
        this%mixing_ratios%cols(1)%dim=1
        this%mixing_ratios%cols(this%mixing_ratios%num_cols)%dim=1
    else if (this%BCs%BCs_label(1)==1 .and. this%BCs%BCs_label(2)==2) then
        call Dirichlet_Neumann_BCs_PDE(this)
        this%mixing_ratios%cols(1)%dim=1
        this%mixing_ratios%cols(this%mixing_ratios%num_cols)%dim=2
    else if (this%BCs%BCs_label(1)==2 .and. this%BCs%BCs_label(2)==2) then
        call Neumann_homog_BCs(this)
        this%mixing_ratios%cols(1)%dim=2
        this%mixing_ratios%cols(this%mixing_ratios%num_cols)%dim=2
    else if (this%BCs%BCs_label(1)==3 .and. this%BCs%BCs_label(2)==2) then
        call Robin_Neumann_homog_BCs(this)
        this%mixing_ratios%cols(1)%dim=2
        this%mixing_ratios%cols(this%mixing_ratios%num_cols)%dim=2
    else
        error stop "Boundary conditions not implemented yet"
    end if

    
    if (this%time_discr%int_method==1) then
        theta=0d0
        do i=2,this%mixing_ratios%num_cols-1
            this%mixing_ratios%cols(i)%dim=3
        end do
        call this%mixing_ratios%allocate_columns()
        call this%allocate_mixing_waters_indices()
    else
        do i=1,this%mixing_ratios%num_cols
            call this%mixing_ratios%cols(i)%set_dim(this%mixing_ratios%num_cols)
            call this%mixing_ratios%cols(i)%allocate_vector()
            call this%mixing_waters_indices%cols(i)%allocate_vector(this%mixing_ratios%cols(i)%dim-1)
        end do
        if (this%time_discr%int_method==2) then
            theta=1d0
        else if (this%time_discr%int_method==3) then
            theta=5d-1
        else
            error stop
        end if
    end if
!> We compute arrays for linear system
    call this%compute_E_mat(E_mat)
    call this%compute_B_mat(theta,E_mat)
    call this%compute_A_mat(theta,E_mat)
    call this%compute_f_vec()
    
    if (present(A_mat_lumped)) then
        call this%compute_lumped_A_mat(A_mat_lumped)
    else if (theta>0d0) then
        allocate(A_mat_inv(this%mixing_ratios%num_cols,this%mixing_ratios%num_cols))
        allocate(this%mixing_ratios_mat(this%mixing_ratios%num_cols,this%mixing_ratios%num_cols))
        call this%A_mat%compute_inverse_tridiag_matrix(tol_inv,A_mat_inv)
        if (theta==1d0) then
            this%mixing_ratios_mat=transpose(A_mat_inv)
        else
            this%mixing_ratios_mat=prod_tridiag_mat_mat(B_mat_T,transpose(A_mat_inv))
        end if
        do i=1,this%mixing_ratios%num_cols
            this%mixing_ratios%cols(i)%col_1(1)=this%mixing_ratios_mat(i,i)
            do j=1,i-1
                this%mixing_ratios%cols(i)%col_1(1+j)=this%mixing_ratios_mat(j,i)
                this%mixing_waters_indices%cols(i)%col_1(j)=j
            end do
            do j=i+1,this%mixing_ratios%cols(i)%dim
                this%mixing_ratios%cols(i)%col_1(j)=this%mixing_ratios_mat(j,i)
                this%mixing_waters_indices%cols(i)%col_1(j-1)=j
            end do
        end do
    else
        if (this%BCs%BCs_label(1)==1) then
            this%mixing_ratios%cols(1)%col_1(1)=this%B_mat%diag(1)
        else
            this%mixing_ratios%cols(1)%col_1=[this%B_mat%diag(1),this%B_mat%super(1)]
            this%mixing_waters_indices%cols(1)%col_1=2
        end if
        do i=2,this%mixing_ratios%num_cols-1
            this%mixing_ratios%cols(i)%col_1=[this%B_mat%diag(i),this%B_mat%sub(i-1),this%B_mat%super(i)]
            this%mixing_waters_indices%cols(i)%col_1=[i-1,i+1]
        end do
        if (this%BCs%BCs_label(2)==1) then
            this%mixing_ratios%cols(this%mixing_ratios%num_cols)%col_1(this%mixing_ratios%num_cols)=this%B_mat%diag(this%mixing_ratios%num_cols)
        else
            this%mixing_ratios%cols(this%mixing_ratios%num_cols)%col_1=[this%B_mat%diag(this%mixing_ratios%num_cols),this%B_mat%sub(this%mixing_ratios%num_cols-1)]
            this%mixing_waters_indices%cols(this%mixing_ratios%num_cols)%col_1=this%mixing_ratios%num_cols-1
        end if
    end if
end subroutine 