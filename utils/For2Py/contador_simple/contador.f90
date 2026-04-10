module contador_mod
  use iso_c_binding
  implicit none

  type :: Contador
     integer :: valor = 0
  contains
     procedure :: incrementar
  end type Contador

contains

  ! Método OO de Fortran
  subroutine incrementar(this)
    class(Contador), intent(inout) :: this
    this%valor = this%valor + 1
  end subroutine incrementar

  ! Método compatible con C que actúa como wrapper
  subroutine crear_contador(ptr) bind(C, name="crear_contador")
    use iso_c_binding
    type(c_ptr), intent(out) :: ptr
    type(Contador), pointer :: obj
    allocate(obj)
    ptr = c_loc(obj)
  end subroutine crear_contador

  subroutine destruir_contador(ptr) bind(C, name="destruir_contador")
    use iso_c_binding
    type(c_ptr), value :: ptr
    type(Contador), pointer :: obj
    call c_f_pointer(ptr, obj)
    deallocate(obj)
  end subroutine destruir_contador

  subroutine incrementar_c(ptr) bind(C, name="incrementar_c")
    use iso_c_binding
    type(c_ptr), value :: ptr
    type(Contador), pointer :: obj
    call c_f_pointer(ptr, obj)
    call obj%incrementar()
  end subroutine incrementar_c

  subroutine obtener_valor(ptr, valor) bind(C, name="obtener_valor")
    use iso_c_binding
    type(c_ptr), value :: ptr
    integer(c_int), intent(out) :: valor
    type(Contador), pointer :: obj
    call c_f_pointer(ptr, obj)
    valor = obj%valor
  end subroutine obtener_valor

end module contador_mod

