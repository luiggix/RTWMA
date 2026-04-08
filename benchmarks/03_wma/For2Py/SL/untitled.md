Perfecto. A continuación te presento un ejemplo **completo, corregido y funcional**, que incluye:

1. Una **clase Fortran `Sistema_Lineal`** que resuelve sistemas usando el método de **Jacobi**.
2. **Wrappers `bind(C)`** para interoperar con Python mediante punteros opacos (`C_PTR`).
3. Funciones para:

   * `crear_sistema`
   * `inicializar_sistema`
   * `resolver_jacobi`
   * `destruir_sistema`
4. Código listo para compilar como `.dll` (Windows).
5. Ejemplos en Python con `ctypes` y `cffi`.

---

## 📦 1. Código Fortran: `sistema_lineal.f90`

```fortran
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

  subroutine inicializar_sistema(ptr, n, A_in, b_in) bind(C, name="inicializar_sistema")
    type(c_ptr), value :: ptr
    integer(c_int), intent(in) :: n
    real(c_double), intent(in) :: A_in(n,n), b_in(n)
    type(Sistema_Lineal), pointer :: obj
    call c_f_pointer(ptr, obj)
    call obj%inicializar(n, A_in, b_in)
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
```

---

## 🛠️ 2. Compilación en Windows (MSYS2 o MinGW)

```bash
gfortran -shared -fPIC -static-libgfortran -static-libgcc sistema_lineal.f90 -o sistema_lineal.dll
```

---

## 🐍 3. Ejemplo en Python con `ctypes`

```python
import ctypes
import numpy as np

# Cargar la biblioteca
lib = ctypes.CDLL("./sistema_lineal.dll")

# Tipos de argumentos
lib.crear_sistema.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
lib.inicializar_sistema.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                     np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
                                     np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')]
lib.resolver_jacobi.argtypes = [ctypes.c_void_p,
                                np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
                                ctypes.c_int, ctypes.c_double]
lib.destruir_sistema.argtypes = [ctypes.c_void_p]

# Crear sistema
sistema = ctypes.c_void_p()
lib.crear_sistema(ctypes.byref(sistema))

# Datos del sistema
A = np.array([[10.0, -1.0, 2.0],
              [-1.0, 11.0, -1.0],
              [2.0, -1.0, 10.0]], dtype=np.float64)
b = np.array([6.0, 25.0, -11.0], dtype=np.float64)
n = A.shape[0]

# Inicializar
lib.inicializar_sistema(sistema, n, A, b)

# Resolver
x_out = np.zeros(n, dtype=np.float64)
lib.resolver_jacobi(sistema, x_out, 100, 1e-6)

print("Solución (Jacobi, ctypes):", x_out)

# Liberar
lib.destruir_sistema(sistema)
```

---

## 🐍 4. Ejemplo en Python con `cffi`

```python
from cffi import FFI
import numpy as np

ffi = FFI()

ffi.cdef("""
    void crear_sistema(void **ptr);
    void inicializar_sistema(void *ptr, int n, double *A, double *b);
    void resolver_jacobi(void *ptr, double *x_out, int max_iter, double tol);
    void destruir_sistema(void *ptr);
""")

lib = ffi.dlopen("./sistema_lineal.dll")

n = 3
A = np.array([[10.0, -1.0, 2.0],
              [-1.0, 11.0, -1.0],
              [2.0, -1.0, 10.0]], dtype=np.float64)
b = np.array([6.0, 25.0, -11.0], dtype=np.float64)
x = np.zeros(n, dtype=np.float64)

# Crear y preparar punteros
ptr = ffi.new("void **")
lib.crear_sistema(ptr)
sistema = ptr[0]

lib.inicializar_sistema(sistema, n,
                        ffi.cast("double *", A.ctypes.data),
                        ffi.cast("double *", b.ctypes.data))

lib.resolver_jacobi(sistema, ffi.cast("double *", x.ctypes.data), 100, 1e-6)

print("Solución (Jacobi, cffi):", x)

lib.destruir_sistema(sistema)
```

---

¿Quieres que te entregue este conjunto como un archivo `.zip` con los `.f90`, scripts Python y un `.bat` para compilar en Windows?
