subroutine read_discretisation(this,path,unit,file_discr)
    use RT_1D_m
    class(RT_1D_c) :: this
    character(len=*), intent(in) :: path
    integer(kind=4), intent(in) :: unit
    character(len=*), intent(in) :: file_discr
    
    character(len=256) :: str,label
    character(len=:), allocatable :: str_trim
    type(mesh_1D_Euler_homog_c) :: mesh
    type(time_discr_homog_c) :: time_discr
    type(BCs_t) :: BCs
    type(tpt_props_heterog_c) :: tpt_props
    integer(kind=4) :: i,j,BCs_label(2),comm_ind,targets_flag
    integer(kind=4), allocatable :: num_mix_waters(:),mix_wat_indices(:,:)
    real(kind=8) :: Delta_x,Delta_t,r,phi,D
    logical :: flag_props
    type(diag_matrix_c) :: F_mat
    
    
    
    
    select type (this)
    type is (RT_1D_transient_c)
        open(unit,file=trim(path)//file_discr,status='old',action='read')
        do 
            read(unit,*) label
            if (label=='end') then
                exit
            !else if (label=='SPACE') then
            !    read(unit,*) Num_targets
            !    read(unit,*) targets_flag
            !    read(unit,*) Delta_x
            !    call mesh%set_targets(Num_targets,targets_flag)
            !    call mesh%set_Delta_x_homog(Delta_x)
            !    call mesh%compute_measure()
            !    call this%transport%set_spatial_discr(mesh)
            else if (label=='TIME') then
                read(unit,*) Num_time
                read(unit,*) Delta_t
                !read(unit,*) int_method_tpt
                read(unit,*) int_method_chem_reacts
                call time_discr%set_Num_time(Num_time)
                call time_discr%set_Delta_t_homog(Delta_t)
                call time_discr%compute_Final_time()
                !call time_discr%set_int_method(int_method_tpt)
                call this%set_int_method_chem_reacts(int_method_chem_reacts)
                call this%transport%set_time_discr(time_discr)
            !else if (label=='BOUNDARY CONDITIONS') then
            !>    read(unit,*) BCs_label(1), BCs_label(2)
            !>    read(unit,*) BCs%evap
            !>    call BCs%set_BCs_label(BCs_label)
            !>    call this%transport%set_BCs(BCs)
            !else if (label=='TRANSPORT PROPERTIES') then
            !>    tpt_props%homog_flag=.true.
            !>    !read(unit,*) flag_props, r
            !>    !if (flag_props==.true.) then
            !>    !>    allocate(tpt_props%source_term(Num_cells))
            !>    !>    tpt_props%source_term=r
            !>    !>    call tpt_props%set_source_term_flag(BCs)
            !>    !else
            !>    !>    tpt_props%homog_flag=.false.
            !>    !>    error stop
            !>    !end if
            !>    read(unit,*) flag_props, phi
            !>    if (flag_props==.true.) then
            !>        allocate(tpt_props%porosity(Num_cells),this%transport%F_mat%diag(Num_cells))
            !>        tpt_props%porosity=phi
            !>        this%transport%F_mat%diag=phi
            !>    else
            !>        tpt_props%homog_flag=.false.
            !>        error stop
            !>    end if
            !>    read(unit,*) flag_props, D
            !>    if (flag_props==.true.) then
            !>        allocate(tpt_props%dispersion(Num_cells))
            !>        tpt_props%dispersion=D
            !>    else
            !>        tpt_props%homog_flag=.false.
            !>        error stop
            !>    end if
            !>    !read(unit,*) flag_props, q
            !>    !if (flag_props==.true.) then
            !>    !>    allocate(tpt_props%porosity(Num_cells))
            !>    !>    tpt_props%porosity=phi
            !>    !else
            !>    !>    error stop
            !>    !end if
            !>    call this%transport%set_tpt_props_heterog_obj(tpt_props)
            !else if (label=='MIXING RATIOS') then
            !>    i=1 !> counter targets
            !>    allocate(mixing_ratios(Num_cells,4))!,f_vec(Num_cells))
            !>    mixing_ratios(1,1)=0d0
            !>    read(unit,*) (mixing_ratios(i,j), j=2,4)!, f_vec(i)
            !>    do
            !>        i=i+1
            !>        if (i<Num_cells) then
            !>            read(unit,*) (mixing_ratios(i,j), j=1,4)!, f_vec(i)
            !>        else
            !>            exit
            !>        end if
            !>    end do
            !>    read(unit,*) (mixing_ratios(Num_cells,j), j=1,2), mixing_ratios(Num_cells,4)!, f_vec(Num_cells)
            !>    mixing_ratios(Num_cells,3)=0d0
            !else if (label=='MIXING WATERS') then
            !>    !i=1 !> counter targets
            !>    !allocate(num_mix_waters(Num_cells))
            !>    !do
            !>    !>    read(unit,"(A10)") str
            !>    !>    if (str=='*') then
            !>    !>        if (i==Num_cells+1) then
            !>    !>            rewind(unit)
            !>    !>            exit
            !>    !>        else
            !>    !>            error stop
            !>    !>        end if
            !>    !>    else
            !>    !>        str_trim=trim(str)
            !>    !>        comm_ind=index(str_trim,'!')
            !>    !>        if (comm_ind>0) then
            !>    !>            str_trim=str_trim(1:comm_ind-1)
            !>    !>        else
            !>    !>            continue
            !>    !>        end if
            !>    !>        num_mix_waters(i)=nint((len(str_trim)+1)/2d0)
            !>    !>    end if
            !>    !>    i=i+1
            !>    !end do
            else
                continue
            end if
        end do
        !do
            !read(unit,*) label
            !if (label=='end') then
            !>    exit
            !else if (label=='MIXING RATIOS') then
            !>    i=1 !> counter targets
            !>    allocate(mix_wat_indices(Num_cells,maxval(num_mix_waters)))
            !>    mix_wat_indices=0
            !>    do
            !>        read(unit,"(A10)") str
            !>        if (str=='*') then
            !>            if (i==Num_cells+1) then
            !>                exit
            !>            else
            !>                error stop
            !>            end if
            !>        else
            !>            read(unit,*) (mix_wat_indices(i,j), j=1,num_mix_waters(i))
            !>        end if
            !>        i=i+1
            !>    end do
            !else
            !>    continue
            !end if
        !end do
        !deallocate(mix_wat_indices)
        close(unit)
    end select
    !call this%set_transport(this%transport)
end subroutine