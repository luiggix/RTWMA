program prueba_contador_arreglo
  use contador_mod
  implicit none

  type(Contador_Arreglo) :: contador
  real(c_double), allocatable :: datos(:)
  real(c_double), allocatable :: resultado(:)
  integer :: i, n

  ! Inicializar datos
  n = 5
  allocate(datos(n))
  datos = [(i*1.0d0, i=1,n)]

  ! Inicializar objeto
  call contador%inicializar(datos, n)

  ! Incrementar todos los valores en 2.5
  call contador%incrementar(2.5d0)

  ! Obtener resultados
  allocate(resultado(n))
  call contador%obtener(resultado)

  ! Imprimir resultado
  print *, "Resultado después de incrementar en 2.5:"
  do i = 1, n
     print *, "resultado(", i, ") = ", resultado(i)
  end do

end program prueba_contador_arreglo
