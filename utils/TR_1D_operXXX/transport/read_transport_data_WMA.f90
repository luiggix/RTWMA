subroutine read_transport_data_WMA(this,path,unit,file_tpt)!,tpt_props,BCs,mesh,time_discr)
    use transport_transient_m
    class(transport_1D_transient_c) :: this
    character(len=*), intent(in) :: path
    integer(kind=4), intent(in) :: unit
    character(len=*), intent(in) :: file_tpt
    
    character(len=256) :: str,label
    character(len=:), allocatable :: str_trim
    type(mesh_1D_Euler_homog_c), target :: mesh
    type(time_discr_homog_c) :: time_discr
    type(BCs_t) :: BCs
    type(tpt_props_heterog_c) :: tpt_props
    integer(kind=4) :: i,j,BCs_label(2),comm_ind,pos
    integer(kind=4), allocatable :: num_mix_waters(:),mix_wat_indices(:)
    real(kind=8) :: Delta_x,Delta_t,r,phi,D
    type(matrix_real_c) :: mixing_ratios
    logical :: flag_props
    type(diag_matrix_c) :: F_mat
    
    !type(transport_1D_transient_c) :: this
    
    
    
    open(unit,file=trim(path)//file_tpt,status='old',action='read')
    do 
        read(unit,*) label
        if (label=='end') then
            rewind(unit)
            exit
        else if (label=='MIXING RATIOS') then
            i=0 !> counter targets
            do
                read(unit,*) dim
                if (dim==0) then
                    exit
                else
                    i=i+1
                end if
            end do
            this%spatial_discr=>mesh !> chapuza
            call this%spatial_discr%set_Num_targets(i)
            !call this%allocate_mixing_ratios()
            call mixing_ratios%allocate_matrix(i)
        else
            continue
        end if
    end do
    do 
        read(unit,*) label
        if (label=='end') then
            rewind(unit)
            exit
        else if (label=='TRANSPORT PROPERTIES') then
            tpt_props%homog_flag=.true.
            read(unit,*) flag_props, phi
            if (flag_props==.true.) then
                allocate(tpt_props%porosity(this%spatial_discr%Num_targets),this%F_mat%diag(this%spatial_discr%Num_targets))
                tpt_props%porosity=phi
                this%F_mat%diag=phi
            else
                tpt_props%homog_flag=.false.
            end if
            call this%set_tpt_props_heterog_obj(tpt_props)
        else if (label=='MIXING RATIOS') then
            i=0 !> counter targets
            do
                i=i+1
                if (i<=this%spatial_discr%Num_targets) then
                    read(unit,*) mixing_ratios%cols(i)%dim
                    call mixing_ratios%cols(i)%allocate_vector()
                else
                    exit
                end if
            end do
        else
            continue
        end if
    end do
    this%mixing_ratios=mixing_ratios
    call this%allocate_mixing_waters_indices()
    do 
        read(unit,*) label
        if (label=='end') then
            exit
        else if (label=='MIXING RATIOS') then
            i=0 !> counter targets
            do
                i=i+1
                if (i<=this%spatial_discr%Num_targets) then
                    read(unit,*) tar_dim, (mixing_ratios%cols(i)%col_1(j), j=1,tar_dim)
                else
                    exit
                end if
            end do
        else if (label=='MIXING WATERS') then
            i=0 !> counter nº targets init (we assume they are in increasing order)
            do while (i<this%mixing_waters_indices%num_cols)
                i=i+1
                if (this%mixing_waters_indices%cols(i)%dim<this%mixing_waters_indices%num_cols-1) then
                    read(unit,*) mix_wat_ind, (this%mixing_waters_indices%cols(mix_wat_ind)%col_1(j), j=1,this%mixing_waters_indices%cols(mix_wat_ind)%dim)
                else
                    allocate(mix_wat_indices(this%mixing_ratios%cols(i)%dim))
                    read(unit,*) mix_wat_ind, mix_wat_indices
                    do j=1,this%mixing_ratios%cols(i)%dim
                        if (mix_wat_indices(j)==mix_wat_ind) then
                            pos=j
                            this%mixing_ratios%cols(I)%col_1(1)=mixing_ratios%cols(I)%col_1(J)
                        end if
                    end do
                    do j=1,this%mixing_ratios%cols(i)%dim
                        if (j<pos) then
                            this%mixing_ratios%cols(I)%col_1(1+j)=mixing_ratios%cols(I)%col_1(J)
                            this%mixing_waters_indices%cols(I)%col_1(j)=mix_wat_indices(J)
                        else if (j>pos) then
                            this%mixing_ratios%cols(I)%col_1(j)=mixing_ratios%cols(I)%col_1(J)
                            this%mixing_waters_indices%cols(I)%col_1(j-1)=mix_wat_indices(J)
                        end if
                    end do
                    deallocate(mix_wat_indices)
                end if
            end do
        else
            continue
        end if
    end do
    close(unit)
end subroutine