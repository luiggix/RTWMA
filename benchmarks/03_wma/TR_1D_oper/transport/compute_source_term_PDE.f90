subroutine compute_source_term_PDE(this)
!> $g=r*c_r$
    use transport_m
    use transport_transient_m
    implicit none
    class(PDE_1D_c) :: this
    
    select type (this)
    type is (transport_1D_c)
        this%source_term_PDE=this%conc_r_flag*this%tpt_props_heterog%source_term*this%conc_ext
    type is (transport_1D_transient_c)
        this%source_term_PDE=this%conc_r_flag*this%tpt_props_heterog%source_term*this%conc_ext
    type is (diffusion_1D_c)
        this%source_term_PDE=this%diff_props_heterog%source_term*this%conc_ext
    type is (diffusion_1D_transient_c)
        this%source_term_PDE=this%diff_props_heterog%source_term*this%conc_ext
    end select

end subroutine