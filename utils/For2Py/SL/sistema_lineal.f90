module sistema_mod
  use iso_c_binding
  implicit none

  type :: Sistema_Lineal
     integer :: n
     real(c_double), allocatable :: A(:,:), b(:), x(:)
  contains
     procedure :: inicializar
     procedure :: resolver_jacobi
  end type Sistema_Lineal

contains

  ! Método OO interno (no interoperable)
  subroutine inicializar(this, n_in, A_in, b_in)
    class(Sistema_Lineal), intent(inout) :: this
    integer(c_int), intent(in) :: n_in
    real(c_double), intent(in) :: A_in(n_in, n_in), b_in(n_in)

    write(*, *) "N :", n_in
    write(*, *) "A :", A_in
    write(*, *) "b :", b_in
 
    this%n = n_in
    allocate(this%A(n_in, n_in), this%b(n_in), this%x(n_in))
    this%A = A_in
    this%b = b_in
    this%x = 0.0
  end subroutine inicializar

  subroutine resolver_jacobi(this, x_out, max_iter, tol)
    class(Sistema_Lineal), intent(inout) :: this
    real(c_double), intent(out) :: x_out(this%n)
    integer(c_int), intent(in) :: max_iter
    real(c_double), intent(in) :: tol

    integer :: i, j, k
    real(c_double) :: sigma, error
    real(c_double), allocatable :: x_new(:)

    allocate(x_new(this%n))
    x_new = this%x

    do k = 1, max_iter
       do i = 1, this%n
          sigma = 0.0
          do j = 1, this%n
             if (j /= i) sigma = sigma + this%A(i, j) * this%x(j)
          end do
          x_new(i) = (this%b(i) - sigma) / this%A(i, i)
       end do
       error = maxval(abs(x_new - this%x))
       this%x = x_new
       if (error < tol) exit
    end do

    x_out = this%x
    deallocate(x_new)
  end subroutine resolver_jacobi

  ! Wrappers C interoperables

  subroutine crear_sistema(ptr) bind(C, name="crear_sistema")
    type(c_ptr), intent(out) :: ptr
    type(Sistema_Lineal), pointer :: obj
    allocate(obj)
    ptr = c_loc(obj)
  end subroutine crear_sistema

!  subroutine inicializar_sistema(ptr, n, A_in, b_in) bind(C, name="inicializar_sistema")
  subroutine inicializar_sistema(ptr, n) bind(C, name="inicializar_sistema")
    use iso_c_binding

    type(c_ptr), value :: ptr
    integer(c_int), intent(out) :: n
!    real(c_double), intent(in) :: A_in(n,n), b_in(n)
    type(Sistema_Lineal), pointer :: obj

    write(*, *) "N :", n
   
    
    call c_f_pointer(ptr, obj)
!    call obj%inicializar(n, A_in, b_in)
  end subroutine inicializar_sistema

  subroutine resolver_jacobi_c(ptr, x_out, max_iter, tol) bind(C, name="resolver_jacobi")
    type(c_ptr), value :: ptr
    real(c_double), intent(out) :: x_out(*)
    integer(c_int), intent(in) :: max_iter
    real(c_double), intent(in) :: tol
    type(Sistema_Lineal), pointer :: obj
    call c_f_pointer(ptr, obj)
    call obj%resolver_jacobi(x_out, max_iter, tol)
  end subroutine resolver_jacobi_c

  subroutine destruir_sistema(ptr) bind(C, name="destruir_sistema")
    type(c_ptr), value :: ptr
    type(Sistema_Lineal), pointer :: obj
    call c_f_pointer(ptr, obj)
    deallocate(obj)
  end subroutine destruir_sistema

end module sistema_mod
