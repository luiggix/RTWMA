module contador_mod
  use iso_c_binding
  implicit none

  type :: Contador_Arreglo
     real(c_double), allocatable :: valores(:)
  contains
     procedure :: inicializar
     procedure :: incrementar
     procedure :: obtener
  end type Contador_Arreglo

contains

  subroutine inicializar(this, arreglo, n)
    class(Contador_Arreglo), intent(inout) :: this
    real(c_double), intent(in) :: arreglo(n)
    integer(c_int), intent(in) :: n
    allocate(this%valores(n))

    write(*, *) "N :", n

    this%valores = arreglo
  end subroutine inicializar

  subroutine incrementar(this, incremento)
    class(Contador_Arreglo), intent(inout) :: this
    real(c_double), intent(in) :: incremento
    this%valores = this%valores + incremento
  end subroutine incrementar

  subroutine obtener(this, arreglo_out)
    class(Contador_Arreglo), intent(in) :: this
    real(c_double), intent(out) :: arreglo_out(:)
    arreglo_out = this%valores
  end subroutine obtener

  ! Wrappers bind(C)

  subroutine crear_contador(ptr) bind(C, name="crear_contador")
    use iso_c_binding
    type(c_ptr), intent(out) :: ptr
    type(Contador_Arreglo), pointer :: obj

    print *, " AQUI VOY"

    allocate(obj)

    print *, " Y AHORA AQUI VOY"

    ptr = c_loc(obj)
  end subroutine crear_contador

  subroutine inicializar_contador(ptr, arreglo, n) bind(C, name="inicializar_contador")
    use iso_c_binding
    type(c_ptr), value :: ptr
    real(c_double), intent(inout) :: arreglo(n)
    integer(c_int), intent(in) :: n
    type(Contador_Arreglo), pointer :: obj

    print *, "N :"
    
    call c_f_pointer(ptr, obj)
    call obj%inicializar(arreglo, n)
  end subroutine inicializar_contador

  subroutine incrementar_contador(ptr, incremento) bind(C, name="incrementar_contador")
    use iso_c_binding
    type(c_ptr), value :: ptr
    real(c_double), intent(in) :: incremento
    type(Contador_Arreglo), pointer :: obj
    call c_f_pointer(ptr, obj)
    call obj%incrementar(incremento)
  end subroutine incrementar_contador
  
  subroutine obtener_valores(ptr, arreglo_out, n) bind(C, name="obtener_valores")
    use iso_c_binding
    type(c_ptr), value :: ptr
    integer(c_int), intent(in) :: n
    real(c_double), intent(out) :: arreglo_out(n)
    type(Contador_Arreglo), pointer :: obj
    call c_f_pointer(ptr, obj)
    call obj%obtener(arreglo_out)
  end subroutine obtener_valores

  subroutine destruir_contador(ptr) bind(C, name="destruir_contador")
    use iso_c_binding
    type(c_ptr), value :: ptr
    type(Contador_Arreglo), pointer :: obj
    call c_f_pointer(ptr, obj)
    deallocate(obj)
  end subroutine destruir_contador

end module contador_mod
