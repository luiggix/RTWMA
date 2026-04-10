import ctypes
import numpy as np

# Cargar DLL
lib = ctypes.CDLL("./contador_arreglo.dll")

# Firmas
lib.crear_contador.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
lib.inicializar_contador.argtypes = [
    ctypes.c_void_p,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags=["C_CONTIGUOUS"]),
    ctypes.c_int
]
lib.incrementar_contador.argtypes = [ctypes.c_void_p, ctypes.c_double]

lib.obtener_valores.argtypes = [
    ctypes.c_void_p,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags=["C_CONTIGUOUS"]),
    ctypes.c_int
]
lib.destruir_contador.argtypes = [ctypes.c_void_p]

# Crear objeto
contador = ctypes.c_void_p()
print(contador)
lib.crear_contador(ctypes.byref(contador))
print(contador)
# Inicializar con arreglo
datos = np.array([1.0, 2.0, 3.0], dtype=np.float64)

lib.inicializar_contador(contador, datos, len(datos))

# Incrementar
lib.incrementar_contador(contador, 5.0)

# Obtener valores
resultado = np.zeros(len(datos), dtype=np.float64)
lib.obtener_valores(contador, resultado, len(resultado))

print("Resultado tras incrementar:", resultado)  # Debería ser [6.0, 7.0, 8.0]

# Liberar
lib.destruir_contador(contador)
