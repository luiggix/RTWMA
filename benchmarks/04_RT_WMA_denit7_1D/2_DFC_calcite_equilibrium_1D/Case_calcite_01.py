# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 14:39:48 2026

@author: leo_teja
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
# import interpolationSchemes as inter
from PIL import Image              
import math
import read_data as rd

from time import time 
import sys


def monod_rates_model1_molL(
    CH2O, O2, NO3,
    mu1=0.5, K_DOC1=1.0e-3, K_O2=1.0e-5,
    mu2=2.0, K_DOC2=1.4e-3, K_NO3=5.9e-3, I_O2=2.52e-5,
    eps=1e-30
):
    CH2O = max(float(CH2O), 0.0)
    O2   = max(float(O2), 0.0)
    NO3  = max(float(NO3), 0.0)

    r1 = (
        mu1
        * CH2O / (K_DOC1 + CH2O + eps)
        * O2   / (K_O2   + O2   + eps)
    )

    r2 = (
        mu2
        * CH2O / (K_DOC2 + CH2O + eps)
        * NO3  / (K_NO3  + NO3  + eps)
        * I_O2 / (I_O2   + O2   + eps)
    )

    return r1, r2




def kinetics_update_U_model1_implicit(
    u_transport,
    phi,
    dt_days,
    mu1=0.5, K_DOC1=1.0e-3, K_O2=1.0e-5,
    mu2=2.0, K_DOC2=1.4e-3, K_NO3=5.9e-3, I_O2=2.52e-5
):
    """
    Actualización cinética implícita/acoplada:
        u_new = u_transport + dt * S_U * R(u_new)
    """

    u_transport = np.asarray(u_transport, dtype=float)

    def residual(u_new):
        u1, u2, u3, u4, u5, u6, u7 = u_new

        CH2O = max(u3, 0.0)
        NO3  = max(u6, 0.0)
        O2   = max(u7, 0.0)

        rk1, rk2 = monod_rates_model1_molL(
            CH2O=CH2O,
            O2=O2,
            NO3=NO3,
            mu1=mu1, K_DOC1=K_DOC1, K_O2=K_O2,
            mu2=mu2, K_DOC2=K_DOC2, K_NO3=K_NO3, I_O2=I_O2
        )

        # Primera opción implícita a probar:
        # R1 = phi * CH2O * rk1
        # R2 = phi * CH2O * rk2

        # R1 = phi*rk1
        # R2 = phi*rk2

        R1 = rk1
        R2 = rk2


        F = np.zeros(7)

        F[0] = u1 - (u_transport[0] + dt_days * ( 2.0 * R1 + 1.2 * R2 ))
        F[1] = u2 -  u_transport[1]
        F[2] = u3 - (u_transport[2] + dt_days * ( -R1 - R2 ))
        F[3] = u4 - (u_transport[3] + dt_days * ( 0.4 * R2 ))
        F[4] = u5 - (u_transport[4] + dt_days * ( -R1 - R2 ))
        F[5] = u6 - (u_transport[5] + dt_days * ( -0.8 * R2 ))
        F[6] = u7 - (u_transport[6] + dt_days * ( -R1 ))

        return F

    sol = root(residual, u_transport, method="hybr", tol=1e-08) #MINimization PACKage heredada de fortran, método hibrido de Powell 
    #Investigar más acerca del método de Powell NR-Hybrid

    if not sol.success:
        raise RuntimeError(f"No convergió la cinética implícita: {sol.message}")

    u_new = sol.x

    # Protección contra negativos numéricos pequeños
    u_new[2] = max(u_new[2], 0.0)
    u_new[3] = max(u_new[3], 0.0)
    u_new[5] = max(u_new[5], 0.0)
    u_new[6] = max(u_new[6], 0.0)

    # Recalcular tasas finales con u_new
    CH2O = u_new[2]
    NO3  = u_new[5]
    O2   = u_new[6]

    rk1, rk2 = monod_rates_model1_molL(
        CH2O=CH2O,
        O2=O2,
        NO3=NO3,
        mu1=mu1, K_DOC1=K_DOC1, K_O2=K_O2,
        mu2=mu2, K_DOC2=K_DOC2, K_NO3=K_NO3, I_O2=I_O2
    )

    # R1 = phi*rk1
    # R2 = phi*rk2

    R1 = rk1
    R2 = rk2


    return u_new, R1, R2


def flash_equilibrium_model1_from_U(
    u,
    K1=1.416772e-2,
    K2=4.690293e-11,
    tol=1e-10,
    max_iter=100,
    nscan=1000,
    verbose=False
):
    """
    Reconstrucción química local para Model 1 / Model 2
    usando directamente las variables U y las leyes de acción de masas.

    Unidades: mol/L

    Ecuaciones:
        u1 = H + HCO3
        u5 = Ca - HCO3 - CO3
        K1 = H / (Ca * HCO3)
        K2 = H * CO3 / HCO3

    Retorna
    -------
    state : dict con H, pH, Cl, CH2O, N2, Ca, NO3, O2, HCO3, CO3
    """

    u1, u2, u3, u4, u5, u6, u7 = map(float, u)

    if u1 <= 0.0:
        raise ValueError(f"u1 debe ser > 0. Valor recibido: {u1:.6e}")

    # Definimos B = HCO3
    # Evitamos extremos peligrosos:
    #   B no demasiado cerca de 0
    #   B no demasiado cerca de u1
    B_min = max(1e-14, 1e-10 * u1)
    B_max = min(u1 - 1e-14, (1.0 - 1e-10) * u1)

    if B_min >= B_max:
        raise RuntimeError(
            f"Rango inválido para HCO3. B_min={B_min:.6e}, B_max={B_max:.6e}, u1={u1:.6e}"
        )

    def f(B):
        """
        Función residual:
            f(B) = Ca(B) - B - CO3(B) - u5
        """
        H = u1 - B
        if B <= 0.0 or H <= 0.0:
            return np.nan

        try:
            CO3 = K2 * B / H
            Ca = H / (K1 * B)
            val = Ca - B - CO3 - u5
        except Exception:
            return np.nan

        if not np.isfinite(val):
            return np.nan

        return val

    # Barrido para encontrar un intervalo válido con cambio de signo
    B_vals = np.linspace(B_min, B_max, nscan)
    f_vals = np.array([f(B) for B in B_vals])

    # quedarnos solo con valores finitos
    valid = np.isfinite(f_vals)

    if not np.any(valid):
        raise RuntimeError(
            "No hay valores finitos de la función de flash en el rango explorado. "
            f"u1={u1:.6e}, u5={u5:.6e}"
        )

    # Buscar cambio de signo entre puntos consecutivos válidos
    bracket_found = False
    B_lo = B_hi = None
    f_lo = f_hi = None

    for i in range(len(B_vals) - 1):
        if not (valid[i] and valid[i + 1]):
            continue

        fi = f_vals[i]
        fj = f_vals[i + 1]

        if fi == 0.0:
            B_lo = B_hi = B_vals[i]
            f_lo = f_hi = fi
            bracket_found = True
            break

        if fi * fj < 0.0:
            B_lo = B_vals[i]
            B_hi = B_vals[i + 1]
            f_lo = fi
            f_hi = fj
            bracket_found = True
            break

    if not bracket_found:
        # diagnóstico útil
        f_valid = f_vals[valid]
        fmin = np.min(f_valid)
        fmax = np.max(f_valid)

        if verbose:
            print("DEBUG FLASH")
            print(f"u1 = {u1:.6e}")
            print(f"u5 = {u5:.6e}")
            print(f"fmin = {fmin:.6e}, fmax = {fmax:.6e}")

        raise RuntimeError(
            "No se encontró cambio de signo en la función de flash. "
            f"u1={u1:.6e}, u5={u5:.6e}, "
            f"fmin={fmin:.6e}, fmax={fmax:.6e}. "
            "Esto sugiere que el estado U no es compatible con este subproblema químico "
            "o que necesitamos revisar la definición de u5 / K1 / K2."
        )

    # Si cayó exactamente en raíz
    if B_lo == B_hi:
        B_mid = B_lo
    else:
        # Bisección
        for _ in range(max_iter):
            B_mid = 0.5 * (B_lo + B_hi)
            f_mid = f(B_mid)

            if not np.isfinite(f_mid):
                raise RuntimeError("La función de flash dejó de ser finita durante la bisección.")

            if abs(f_mid) < tol:
                break

            if f_lo * f_mid <= 0.0:
                B_hi = B_mid
                f_hi = f_mid
            else:
                B_lo = B_mid
                f_lo = f_mid

    # Reconstrucción final
    H = u1 - B_mid
    HCO3 = B_mid
    CO3 = K2 * HCO3 / H
    Ca = H / (K1 * HCO3)

    if H <= 0.0 or not np.isfinite(H):
        raise RuntimeError("El flash produjo una concentración de H no física.")

    pH = -math.log10(H)

    state = {
        "H": H,
        "pH": pH,
        "Cl": u2,
        "CH2O": u3,
        "N2": u4,
        "Ca": Ca,
        "NO3": u6,
        "O2": u7,
        "HCO3": HCO3,
        "CO3": CO3
    }

    return state


def check_mass_balance_stoichiometry(tol=1e-12, verbose=True):
    """
    Verifica conservación elemental y de carga para las reacciones cinéticas:
        A @ S_k.T = 0
    """

    species = [
        "H+", "Cl-", "CH2O", "N2", "Ca2+", "NO3-", "O2",
        "calcite", "HCO3-", "CO3--", "H2O"
    ]

    elements = ["C", "H", "O", "N", "Ca", "Cl", "charge"]

    S_k = np.array([
        [1.0, 0.0, -1.0, 0.0, 0.0,  0.0, -1.0, 0.0, 1.0, 0.0, 0.0],
        [0.2, 0.0, -1.0, 0.4, 0.0, -0.8,  0.0, 0.0, 1.0, 0.0, 0.4]
    ], dtype=float)

    A_elem = np.array([
        # H+ Cl- CH2O N2 Ca2+ NO3- O2 calcite HCO3- CO3-- H2O

        # C
        [0,  0,   1,   0,   0,    0,   0,    1,      1,     1,    0],

        # H
        [1,  0,   2,   0,   0,    0,   0,    0,      1,     0,    2],

        # O
        [0,  0,   1,   0,   0,    3,   2,    3,      3,     3,    1],

        # N
        [0,  0,   0,   2,   0,    1,   0,    0,      0,     0,    0],

        # Ca
        [0,  0,   0,   0,   1,    0,   0,    1,      0,     0,    0],

        # Cl
        [0,  1,   0,   0,   0,    0,   0,    0,      0,     0,    0],

        # carga
        [1, -1,   0,   0,   2,   -1,   0,    0,     -1,    -2,    0],
    ], dtype=float)

    balance = A_elem @ S_k.T

    if verbose:
        print("\n--- Verificación estequiométrica: A_elem @ S_k.T ---")
        print("Filas:", elements)
        print("Columnas: r1, r2")
        print(balance)

    if not np.allclose(balance, 0.0, atol=tol):
        raise ValueError(
            "La matriz estequiométrica S_k NO conserva masa/carga. "
            "Revisa los coeficientes de las reacciones."
        )

    if verbose:
        print("✅ La estequiometría conserva masa elemental y carga.\n")

    return balance


def check_discrete_mass_balance_U(A, B, u_old, u_new, phi, dx, dt,
                                  var_name="u", tol=1e-10, verbose=True):
    """
    Checa el balance de masa discreto consistente con el sistema lineal:

        A u_new = B

    separando acumulación, operador espacial y términos de frontera/fuente.
    """

    u_old = np.asarray(u_old, dtype=float)
    u_new = np.asarray(u_new, dtype=float)
    B = np.asarray(B, dtype=float)

    # Matriz sin el término de acumulación phi/dt
    A_spatial = A.copy()
    for i in range(A_spatial.shape[0]):
        A_spatial[i, i] -= phi / dt

    # Parte de B que no corresponde a acumulación vieja
    B_source = B - phi * u_old / dt

    # Balance global discreto
    accumulation = phi * np.sum(u_new - u_old) * dx / dt
    spatial_term = np.sum(A_spatial @ u_new) * dx
    source_term = np.sum(B_source) * dx

    residual = accumulation + spatial_term - source_term

    scale = max(
        abs(accumulation),
        abs(spatial_term),
        abs(source_term),
        1e-30
    )

    error_rel = abs(residual) / scale

    if verbose:
        print(f"\n--- Balance discreto de masa para {var_name} ---")
        print(f"Acumulación       = {accumulation:.6e}")
        print(f"Término espacial  = {spatial_term:.6e}")
        print(f"Término fuente/BC = {source_term:.6e}")
        print(f"Residual          = {residual:.6e}")
        print(f"Error relativo    = {error_rel:.6e}")

    if abs(residual) > tol and verbose:
        print(f"⚠️ Advertencia: posible desbalance discreto en {var_name}")

    return residual, error_rel



def check_conservative_mixing_matrix(A, phi, dt, TW0, q_inf, dx, Dh,
                                     tol=1e-10, verbose=True):
    """
    Verifica si la matriz de mezcla conserva una concentración constante.

    Para el transporte:
        A u_new = D u_old + b_bc

    donde:
        D = phi/dt * I

    Entonces:
        u_new = Lambda u_old + gamma u_inf

    La mezcla es conservativa si:
        sum_j Lambda_ij + gamma_i = 1
    """

    nx = A.shape[0]

    Dmat = np.eye(nx) * phi / dt

    # Lambda = A^{-1} D
    Lambda = np.linalg.solve(A, Dmat)

    # Vector de contribución de frontera para u_inf = 1
    b_unit_bc = np.zeros(nx)
    b_unit_bc[0] = -TW0 * ((q_inf * dx) / Dh)

    # gamma = A^{-1} b_unit_bc
    gamma = np.linalg.solve(A, b_unit_bc)

    row_sum = Lambda.sum(axis=1) + gamma

    residual = row_sum - 1.0
    max_error = np.max(np.abs(residual))

    min_lambda = np.min(Lambda)
    min_gamma = np.min(gamma)

    if verbose:
        print("\n--- Verificación de mezcla conservativa ---")
        print("Suma de pesos por celda:")
        print(row_sum)
        print("Residual row_sum - 1:")
        print(residual)
        print(f"Error máximo = {max_error:.6e}")
        print(f"Min Lambda   = {min_lambda:.6e}")
        print(f"Min gamma    = {min_gamma:.6e}")

    if max_error < tol:
        print("✅ La mezcla conserva una concentración constante.")
    else:
        print("⚠️ La mezcla NO conserva exactamente una concentración constante.")

    if min_lambda < -tol or min_gamma < -tol:
        print("⚠️ Hay pesos negativos de mezcla. Puede haber oscilaciones numéricas.")

    return Lambda, gamma, residual


def check_U_transport_as_mixing(A, B, u_old, u_new, phi, dt,
                                var_name="u", tol=1e-10, verbose=True):
    """
    Verifica que la solución transportada u_new sea consistente con:
        u_new = A^{-1} B
    y con:
        u_new = Lambda u_old + contribución de frontera
    """

    nx = A.shape[0]
    Dmat = np.eye(nx) * phi / dt

    Lambda = np.linalg.solve(A, Dmat)

    B_acc = Dmat @ u_old
    B_bc = B - B_acc

    u_mix = Lambda @ u_old + np.linalg.solve(A, B_bc)

    err = u_new - u_mix
    err_norm = np.linalg.norm(err, ord=np.inf)

    if verbose:
        print(f"\n--- Verificación de mezcla para {var_name} ---")
        print(f"Error infinito = {err_norm:.6e}")

    if err_norm < tol:
        print(f"✅ {var_name}: transporte consistente como mezcla.")
    else:
        print(f"⚠️ {var_name}: inconsistencia en la mezcla.")

    return err, err_norm

#Parametros 

nx=10
total_time = 1.0    #Total time of simulation days
nper = 1            # Number of periods
dt = 0.01   #delta of time    
nstp=int(total_time/dt)


######## Discretización espacial y temporal
Lx=1.0  #m 
delta_x=Lx/nx    #length of cells 
xi=np.linspace(0.5*delta_x,Lx-0.5*delta_x, nx)   #centers of cells 

##### 

Por=0.5
Dh=0.2
Dr=0.0 
recharge=0.0

# ------------------------------------------------------------------------------------------#
#---------- Calculando los valores de q para cada nodo y fronteras de la celdas ----------------#
#------------------------------------------------------------------------------------------#

q_i=np.ones(nx)*0.2   #Specific discharge
v_i=q_i/Por           #velocity 


#------------------------------------------------------------------------------------------#
#---------- Definiendo vectores y matrices para el enfoque de mezclas ---------------------#
#------------------------------------------------------------------------------------------#

A = np.zeros((nx, nx))          #Matriz de coeficientes A (contiene las proporciones de mezcla o lamnda)
B1 = np.zeros(nx)                #Rhs del sistema de ecuaciones de las U
B2 = np.zeros(nx)                #Rhs del sistema de ecuaciones de las U
B3 = np.zeros(nx)                #Rhs del sistema de ecuaciones de las U
B4 = np.zeros(nx)                #Rhs del sistema de ecuaciones de las U
B5 = np.zeros(nx)                #Rhs del sistema de ecuaciones de las U
B6 = np.zeros(nx)                #Rhs del sistema de ecuaciones de las U
B7 = np.zeros(nx)                #Rhs del sistema de ecuaciones de las U

TW = np.zeros(nx)
TE = np.zeros(nx)
TC = np.zeros(nx)

AL=np.zeros((nx, nx))
LAMBDAS=np.zeros((nx, nx))


#------------------------------------------------------------------------------------------#
#---------------------------- Definiendo condiciones iniciales ----------------------------#
#------------------------------------------------------------------------------------------#
    

"""
u1=H+  + HCO3-     #reacción 1  
U2=Cl
U3=CH2O
U4=N2
U5= Ca2+ - HCO3 -   CO3-   #reacción 2 
U6= NO3-
U7= O2 
"""

u=np.zeros(7)
u_old=np.zeros(7)

r1=np.zeros(nx)
r2=np.zeros(nx)

u1_inf= 1.000000e-08+5.376110E-04
u2_inf= 1.0e-3
u3_inf= 2.0e-3
u4_inf= 1.0e-16
u5_inf= 1.312900e-03 - 5.376110e-04 - 2.521550e-06
u6_inf= 1.0e-3
u7_inf= 2.529470E-04

u1_ini= 1.00000e-07+1.00000e-03
u2_ini= 1.0e-16
u3_ini= 1.0e-05
u4_ini= 1.0e-16
u5_ini= 7.05830e-03 - 1e-03 - 4.69029e-07
u6_ini= 1.0e-5
u7_ini= 1.0e-6

u1_n = np.ones(nx)*u1_ini                         #Vector de valor de c a un paso de tiempo n (necesario para la solución del sistema de ecuaciones)
u1_v = np.zeros(nx)                         #Vector de valor de c a un paso de tiempo n+1
u1_k = np.zeros(nx)                         #Vector de valor de c a un paso de tiempo n+1

u2_n = np.ones(nx)*u2_ini                         #Vector de valor de c a un paso de tiempo n (necesario para la solución del sistema de ecuaciones)
u2_v = np.zeros(nx)                         #Vector de valor de c a un paso de tiempo n+1
u2_k = np.zeros(nx)                         #Vector de valor de c a un paso de tiempo n+1

u3_n = np.ones(nx)*u3_ini                         #Vector de valor de c a un paso de tiempo n (necesario para la solución del sistema de ecuaciones)
u3_v = np.zeros(nx)                         #Vector de valor de c a un paso de tiempo n+1
u3_k = np.zeros(nx)                         #Vector de valor de c a un paso de tiempo n+1

u4_n = np.ones(nx)*u4_ini                         #Vector de valor de c a un paso de tiempo n (necesario para la solución del sistema de ecuaciones)
u4_v = np.zeros(nx)                         #Vector de valor de c a un paso de tiempo n+1
u4_k = np.zeros(nx)                         #Vector de valor de c a un paso de tiempo n+1

u5_n = np.ones(nx)*u5_ini                         #Vector de valor de c a un paso de tiempo n (necesario para la solución del sistema de ecuaciones)
u5_v = np.zeros(nx)                         #Vector de valor de c a un paso de tiempo n+1
u5_k = np.zeros(nx)                         #Vector de valor de c a un paso de tiempo n+1

u6_n = np.ones(nx)*u6_ini                         #Vector de valor de c a un paso de tiempo n (necesario para la solución del sistema de ecuaciones)
u6_v = np.zeros(nx)                         #Vector de valor de c a un paso de tiempo n+1
u6_k = np.zeros(nx)                         #Vector de valor de c a un paso de tiempo n+1

u7_n = np.ones(nx)*u7_ini                         #Vector de valor de c a un paso de tiempo n (necesario para la solución del sistema de ecuaciones)
u7_v = np.zeros(nx)                         #Vector de valor de c a un paso de tiempo n+1
u7_k = np.zeros(nx)                         #Vector de valor de c a un paso de tiempo n+1


solcnsv_u1 = np.zeros((nstp+1, nx))            #Matriz de guardado de las c para cada paso de tiempo y en toda la malla 1D (incluyendo las condiciones iniciales al tiempo cero)
solcnsv_u1[0][:] = u1_n 
solcnsk_u1 = np.zeros((nstp+1, nx))            #Matriz de guardado de las c para cada paso de tiempo y en toda la malla 1D (incluyendo las condiciones iniciales al tiempo cero)
solcnsk_u1[0][:] = u1_n 


solcnsv_u2 = np.zeros((nstp+1, nx))            #Matriz de guardado de las c para cada paso de tiempo y en toda la malla 1D (incluyendo las condiciones iniciales al tiempo cero)
solcnsv_u2[0][:] = u1_n 
solcnsk_u2 = np.zeros((nstp+1, nx))            #Matriz de guardado de las c para cada paso de tiempo y en toda la malla 1D (incluyendo las condiciones iniciales al tiempo cero)
solcnsk_u2[0][:] = u1_n 


solcnsv_u3 = np.zeros((nstp+1, nx))            #Matriz de guardado de las c para cada paso de tiempo y en toda la malla 1D (incluyendo las condiciones iniciales al tiempo cero)
solcnsv_u3[0][:] = u1_n 
solcnsk_u3 = np.zeros((nstp+1, nx))            #Matriz de guardado de las c para cada paso de tiempo y en toda la malla 1D (incluyendo las condiciones iniciales al tiempo cero)
solcnsk_u3[0][:] = u1_n 

solcnsv_u4 = np.zeros((nstp+1, nx))            #Matriz de guardado de las c para cada paso de tiempo y en toda la malla 1D (incluyendo las condiciones iniciales al tiempo cero)
solcnsv_u4[0][:] = u1_n 
solcnsk_u4 = np.zeros((nstp+1, nx))            #Matriz de guardado de las c para cada paso de tiempo y en toda la malla 1D (incluyendo las condiciones iniciales al tiempo cero)
solcnsk_u4[0][:] = u1_n 

solcnsv_u5 = np.zeros((nstp+1, nx))            #Matriz de guardado de las c para cada paso de tiempo y en toda la malla 1D (incluyendo las condiciones iniciales al tiempo cero)
solcnsv_u5[0][:] = u1_n 
solcnsk_u5 = np.zeros((nstp+1, nx))            #Matriz de guardado de las c para cada paso de tiempo y en toda la malla 1D (incluyendo las condiciones iniciales al tiempo cero)
solcnsk_u5[0][:] = u1_n 

solcnsv_u6 = np.zeros((nstp+1, nx))            #Matriz de guardado de las c para cada paso de tiempo y en toda la malla 1D (incluyendo las condiciones iniciales al tiempo cero)
solcnsv_u6[0][:] = u1_n 
solcnsk_u6 = np.zeros((nstp+1, nx))            #Matriz de guardado de las c para cada paso de tiempo y en toda la malla 1D (incluyendo las condiciones iniciales al tiempo cero)
solcnsk_u6[0][:] = u1_n 

solcnsv_u7 = np.zeros((nstp+1, nx))            #Matriz de guardado de las c para cada paso de tiempo y en toda la malla 1D (incluyendo las condiciones iniciales al tiempo cero)
solcnsv_u7[0][:] = u1_n 
solcnsk_u7 = np.zeros((nstp+1, nx))            #Matriz de guardado de las c para cada paso de tiempo y en toda la malla 1D (incluyendo las condiciones iniciales al tiempo cero)
solcnsk_u7[0][:] = u1_n 


pH_hist   = np.zeros((nstp, nx))
H_hist   = np.zeros((nstp, nx))
Ca_hist   = np.zeros((nstp, nx))
HCO3_hist = np.zeros((nstp, nx))
CO3_hist  = np.zeros((nstp, nx))
CH2O_hist = np.zeros((nstp, nx))
NO3_hist  = np.zeros((nstp, nx))
O2_hist   = np.zeros((nstp, nx))
N2_hist   = np.zeros((nstp, nx))

r1_hist = np.zeros((nstp, nx))
r2_hist = np.zeros((nstp, nx))

errK1_hist = np.zeros((nstp, nx))
errK2_hist = np.zeros((nstp, nx))


# if verb != 0:
#     print("|------------------------------------------------------------------------------|")
#     print("|----------WMA SOLUCIÓN DIFERENCIAS FINITAS TRADICIONALES IMPLÍCITO------------|")
#     print("|------------------------------------------------------------------------------|")


#------------------------------------------------------------------------------------------#
#------------------------ Inicio del ciclo solución en el tiempo --------------------------#
#------------------------------------------------------------------------------------------#
framesU1=[]
framesC1=[]  #Lista vacia para salvar las imagenes (se hará tan grande como tantas imagenes guardemos en ella)
framesC2=[]
framesR1=[]

tiempoComputo_inicial = time()       

q_inf=q_i[0]  ##Este valor se tiene que cambiar si hay una q_inf

check_mass_balance_stoichiometry()


for ti in range(0, nstp):
        
    #--------------------------------------------------------------------------------------#
    #--------- Calculando los coeficientes "A" Y las matrices diagonales            -------#
    #--------------------------------------------------------------------------------------# 
    
    
    for i in range(0, nx):

        B1[i]=Por*u1_n[i]/dt
        B2[i]=Por*u2_n[i]/dt
        B3[i]=Por*u3_n[i]/dt
        B4[i]=Por*u4_n[i]/dt
        B5[i]=Por*u5_n[i]/dt
        B6[i]=Por*u6_n[i]/dt
        B7[i]=Por*u7_n[i]/dt

        qe, qw = 0.2, 0.2   #Calclulo de los flujos en las caras (puede tomarse un esquema upwind o TVD, etc)
        
        if ((i>0)and(i<nx-1)):

            # qe, qw = inter.upwind_1D(i, h_i, q_i)   #Calclulo de los flujos en las caras (puede tomarse un esquema upwind o TVD, etc)
            
            TE[i] =  qe/(2*delta_x)-Dh/(delta_x**2)        #Transmisibilidad en xi+1/2
            TW[i] = -qw/(2*delta_x)-Dh/(delta_x**2)        #Transmisibilidad en xi-1/2    
            TC[i] = -(qe-qw)/(2*delta_x)+(2*Dh)/(delta_x**2)+Por/dt+ Dr+recharge

            A[i][i+1]= TE[i]  
            A[i][i]=   TC[i]
            A[i][i-1]= TW[i]


            BETA_E = -qe/(2*delta_x)+Dh/(delta_x**2)        #Transmisibilidad en xi+1/2
            BETA_W =  qw/(2*delta_x)+Dh/(delta_x**2)        #Transmisibilidad en xi-1/2    
            BETA_C = (qe-qw)/(2*delta_x)-(2*Dh)/(delta_x**2)-Dr-recharge

            
            AL[i][i+1]= BETA_E
            AL[i][i]=   BETA_C
            AL[i][i-1]= BETA_W
            

        if i==0:
            # if(h_i[i]<=h_i[i+1]):  #En la Frontera no sería conveniente utilizar otro esquemás mas que el upwind
            #     qe=q_i[i+1]
            # else:
            #     qe=q_i[i]
            
            qw=q_inf    #solo porque es igual se tendría que cambiar cuando CF tenga algun valor

            TE[i] =  qe/(2*delta_x)-Dh/(delta_x**2)        #Transmisibilidad en xi+1/2
            TW[i] = -qw/(2*delta_x)-Dh/(delta_x**2)        #Transmisibilidad en xi-1/2    
            TC[i] = -(qe-qw)/(2*delta_x)+(2*Dh)/(delta_x**2)+Por/dt+Dr+recharge
            
            A[i][i+1]= TE[i]   
            A[i][i]=   TC[i]+TW[i]*(1.-(q_i[i]*delta_x)/Dh)    
            
            B1[i]=B1[i]-TW[i]*((q_inf*delta_x)/Dh)*(u1_inf) #U1
            B2[i]=B2[i]-TW[i]*((q_inf*delta_x)/Dh)*(u2_inf) #U2
            B3[i]=B3[i]-TW[i]*((q_inf*delta_x)/Dh)*(u3_inf) #U3
            B4[i]=B4[i]-TW[i]*((q_inf*delta_x)/Dh)*(u4_inf) #U4
            B5[i]=B5[i]-TW[i]*((q_inf*delta_x)/Dh)*(u5_inf) #U5
            B6[i]=B6[i]-TW[i]*((q_inf*delta_x)/Dh)*(u6_inf) #U6
            B7[i]=B7[i]-TW[i]*((q_inf*delta_x)/Dh)*(u7_inf) #U7


            BETA_E = -qe/(2*delta_x)+Dh/(delta_x**2)        #Transmisibilidad en xi+1/2
            BETA_W =  qw/(2*delta_x)+Dh/(delta_x**2)        #Transmisibilidad en xi-1/2    
            BETA_C =  (qe-qw)/(2*delta_x)-(2*Dh)/(delta_x**2)-Dr-recharge
            
            AL[i][i+1]= BETA_E   
            AL[i][i]=   BETA_C+BETA_W*(1.-(q_i[i]*delta_x)/Dh)

        if i==nx-1:
            # if(h_i[i-1]<=h_i[i]):     #En la Frontera no sería conveniente utilizar otro cosa que el upwind
            #     qw=q_i[i]
            # else:
            #     qw=q_i[i-1]
            
            # qe=q_i[i]

            TE[i] =  qe/(2*delta_x)-Dh/(delta_x**2)        #Transmisibilidad en xi+1/2
            TW[i] = -qw/(2*delta_x)-Dh/(delta_x**2)        #Transmisibilidad en xi-1/2    
            TC[i] = -(qe-qw)/(2*delta_x)+(2*Dh)/(delta_x**2)+Por/dt+Dr+recharge

            A[i][i]=   TC[i]+TE[i]      #Condicion de frontera Neumann
            A[i][i-1]= TW[i]


            BETA_E = -qe/(2*delta_x)+Dh/(delta_x**2)        #Transmisibilidad en xi+1/2
            BETA_W =  qw/(2*delta_x)+Dh/(delta_x**2)        #Transmisibilidad en xi-1/2    
            BETA_C = (qe-qw)/(2*delta_x)-(2*Dh)/(delta_x**2)-Dr-recharge
    
            AL[i][i]=   BETA_C+BETA_E      #Condicion de frontera Neumann
            AL[i][i-1]= BETA_W

 


    if ti == 0:
        Lambda, gamma, mix_residual = check_conservative_mixing_matrix(
            A=A,
            phi=Por,
            dt=dt,
            TW0=TW[0],
            q_inf=q_inf,
            dx=delta_x,
            Dh=Dh,
            verbose=True
        )

       
    u1_v = np.linalg.solve(A, B1)
    u2_v = np.linalg.solve(A, B2)
    u3_v = np.linalg.solve(A, B3)
    u4_v = np.linalg.solve(A, B4)
    u5_v = np.linalg.solve(A, B5)
    u6_v = np.linalg.solve(A, B6)
    u7_v = np.linalg.solve(A, B7)


    check_discrete_mass_balance_U(A, B1, u1_n, u1_v, Por, delta_x, dt, "u1")
    check_discrete_mass_balance_U(A, B2, u2_n, u2_v, Por, delta_x, dt, "u2")
    check_discrete_mass_balance_U(A, B3, u3_n, u3_v, Por, delta_x, dt, "u3")
    check_discrete_mass_balance_U(A, B4, u4_n, u4_v, Por, delta_x, dt, "u4")
    check_discrete_mass_balance_U(A, B5, u5_n, u5_v, Por, delta_x, dt, "u5")
    check_discrete_mass_balance_U(A, B6, u6_n, u6_v, Por, delta_x, dt, "u6")
    check_discrete_mass_balance_U(A, B7, u7_n, u7_v, Por, delta_x, dt, "u7")


    # check_U_transport_as_mixing(A, B1, u1_n, u1_v, Por, dt, "u1")
    # check_U_transport_as_mixing(A, B2, u2_n, u2_v, Por, dt, "u2")
    # check_U_transport_as_mixing(A, B3, u3_n, u3_v, Por, dt, "u3")
    # check_U_transport_as_mixing(A, B4, u4_n, u4_v, Por, dt, "u4")
    # check_U_transport_as_mixing(A, B5, u5_n, u5_v, Por, dt, "u5")
    # check_U_transport_as_mixing(A, B6, u6_n, u6_v, Por, dt, "u6")
    # check_U_transport_as_mixing(A, B7, u7_n, u7_v, Por, dt, "u7")
        
    # sys.exit()

    solcnsv_u1[ti+1][:] = u1_v[:]
    solcnsv_u2[ti+1][:] = u2_v[:]
    solcnsv_u3[ti+1][:] = u3_v[:]
    solcnsv_u4[ti+1][:] = u4_v[:]
    solcnsv_u5[ti+1][:] = u5_v[:]
    solcnsv_u6[ti+1][:] = u6_v[:]
    solcnsv_u7[ti+1][:] = u7_v[:]
    
  
    #Reacciones en equilibrio 
    
    # for i in range(0, nx):
    #     sol, Ca, HCO3, CO3, pH, Ca_in = flash_equilibrium_calcite_from_u1_u5_molL(pp,
    #         u1_target=u1_v[i],
    #         u5_target=u5_v[i],
    #         Cl_molL=u2_v[i],
    #         temp=25.0
    #     )
        
    # sys.exit()   
    
    #Reacciones cinéticas 
    
    for i in range(0, nx):
        
        u[0]=u1_v[i]; 
        u[1]=u2_v[i]; 
        u[2]=u3_v[i]; 
        u[3]=u4_v[i]; 
        u[4]=u5_v[i]; 
        u[5]=u6_v[i]; 
        u[6]=u7_v[i]

        
        u, r1[i], r2[i] = kinetics_update_U_model1_implicit(u, Por, dt)

        u1_k[i]=u[0]; 
        u2_k[i]=u[1]; 
        u3_k[i]=u[2]; 
        u4_k[i]=u[3]; 
        u5_k[i]=u[4]; 
        u6_k[i]=u[5]; 
        u7_k[i]=u[6];

        state = flash_equilibrium_model1_from_U(u)

        H_hist[ti][i]   = state["H"]  
        pH_hist[ti][i]   = state["pH"]  
        Ca_hist[ti][i]   = state["Ca"]
        HCO3_hist[ti][i] = state["HCO3"]
        CO3_hist[ti][i]  = state["CO3"]
        CH2O_hist[ti][i] = state["CH2O"]
        NO3_hist[ti][i]  = state["NO3"]
        O2_hist[ti][i]   = state["O2"]
        N2_hist[ti][i]   = state["N2"]
        
        # u1_comp= H_hist[ti][i]+HCO3_hist[ti][i]      
        # u5_comp= Ca_hist[ti][i]-HCO3_hist[ti][i]-CO3_hist[ti][i]
        # print(u1_comp, u1_k[i], abs(u1_k[i]-u1_comp))  #Invariante u1
        # print(u5_comp, u5_k[i], abs(u5_k[i]-u5_comp), "\t")  #Invariante u1


        # K1_calc = state["H"] / (state["Ca"] * state["HCO3"])
        # K2_calc = state["H"] * state["CO3"] / state["HCO3"]

        # K1 = 1.416772e-2
        # K2 = 4.690293e-11

        # errork1= (K1-K1_calc)/K1
        # errork2= (K2-K2_calc)/K2
        
        # print(K1, K1_calc, errork1,"\t", K2, K2_calc, errork2)
        
        # print("pH   =", state["pH"])
        # print("Ca   =", state["Ca"])
        # print("HCO3 =", state["HCO3"])
        # print("CO3  =", state["CO3"])


    solcnsk_u1[ti+1][:] = u1_k[:]; u1_n = np.copy(u1_k)   #Guardando las soluciones en la matriz de u para cada paso de tiempo
    solcnsk_u2[ti+1][:] = u2_k[:]; u2_n = np.copy(u2_k)   #Guardando las soluciones en la matriz de u para cada paso de tiempo
    solcnsk_u3[ti+1][:] = u3_k[:]; u3_n = np.copy(u3_k)   #Guardando las soluciones en la matriz de u para cada paso de tiempo
    solcnsk_u4[ti+1][:] = u4_k[:]; u4_n = np.copy(u4_k)   #Guardando las soluciones en la matriz de u para cada paso de tiempo
    solcnsk_u5[ti+1][:] = u5_k[:]; u5_n = np.copy(u5_k)   #Guardando las soluciones en la matriz de u para cada paso de tiempo
    solcnsk_u6[ti+1][:] = u6_k[:]; u6_n = np.copy(u6_k)   #Guardando las soluciones en la matriz de u para cada paso de tiempo
    solcnsk_u7[ti+1][:] = u7_k[:]; u7_n = np.copy(u7_k)   #Guardando las soluciones en la matriz de u para cada paso de tiempo

    r1_hist[ti][:]=r1 
    r2_hist[ti][:]=r2 

            
    print("tiempo de simulacion: " +str(round(dt*(ti+1),2))+" dias" )
    
D=np.identity(nx)*Por/dt
A2=np.linalg.inv(D-AL)
LAMBDAS=A2.dot(D)


#------------------------------------------------------------------------------------------#
#--------------------------- Graficando todos los resultados ------------------------------#
#------------------------------------------------------------------------------------------#

plt.figure('Graf_u1',figsize=(12,8))
plt.style.use('fast')
plt.title('Perfil de $u_1$ vs distancia')
# plt.plot(xi, solcnsImp_u1[0][:], '-r', linewidth=3.0)
for i in range(1,nstp+1):
    
    plt.plot(xi, solcnsk_u1[i][:], '-')

    plt.xlabel("Distancia x [$m$]")
    plt.ylabel("u_1 [$mol/l$]")

    # stringTimes=round(dt*(i+1),2)
    # stringU1Save = 'U1 %s dias.png' %str(stringTimes)
    # plt.savefig(stringU1Save, dpi=200)
    # imgU1 = Image.open(stringU1Save)   #Se abre el archivo de imagen de la presion recien guardado
    # framesU1.append(imgU1)   #El archivo abierto de imagen de anexa a la lista de frames

                                                                   
plt.minorticks_on()
plt.grid(True,which='major', color='w', linestyle='-')
plt.grid(True, which='minor', color='w', linestyle='--')


plt.figure('Graf_u2',figsize=(12,8))
plt.style.use('fast')
plt.title('Perfil de $u_2$ vs distancia')
# plt.plot(xi, solcnsImp_u1[0][:], '-r', linewidth=3.0)
for i in range(1,nstp+1):
    
    plt.plot(xi, solcnsk_u2[i][:])
    plt.xlabel("Distancia x [$m$]")
    plt.ylabel("u_2 [$mol/l$]")



plt.figure('Graf_u3',figsize=(12,8))
plt.style.use('fast')
plt.title('Perfil de $u_3$ vs distancia')
# plt.plot(xi, solcnsImp_u1[0][:], '-r', linewidth=3.0)
for i in range(1,nstp+1):
    
    plt.plot(xi, solcnsk_u3[i][:])
    plt.xlabel("Distancia x [$m$]")
    plt.ylabel("u_3 [$mol/l$]")


plt.figure('Graf_u4',figsize=(12,8))
plt.style.use('fast')
plt.title('Perfil de $u_4$ vs distancia')
# plt.plot(xi, solcnsImp_u1[0][:], '-r', linewidth=3.0)
for i in range(1,nstp+1):
    
    plt.plot(xi, solcnsk_u4[i][:])
    plt.xlabel("Distancia x [$m$]")
    plt.ylabel("u_4 [$mol/l$]")


plt.figure('Graf_u5',figsize=(12,8))
plt.style.use('fast')
plt.title('Perfil de $u_5$ vs distancia')
# plt.plot(xi, solcnsImp_u1[0][:], '-r', linewidth=3.0)
for i in range(1,nstp+1):
    
    plt.plot(xi, solcnsk_u5[i][:])
    plt.xlabel("Distancia x [$m$]")
    plt.ylabel("u [$mol/l$]")


plt.figure('Graf_u6',figsize=(12,8))
plt.style.use('fast')
plt.title('Perfil de $u_6$ vs distancia')
# plt.plot(xi, solcnsImp_u1[0][:], '-r', linewidth=3.0)
for i in range(1,nstp+1):
    
    plt.plot(xi, solcnsk_u6[i][:])
    plt.xlabel("Distancia x [$m$]")
    plt.ylabel("u_6 [$mol/l$]")


plt.figure('Graf_u7',figsize=(12,8))
plt.style.use('fast')
plt.title('Perfil de $u_7$ vs distancia')
# plt.plot(xi, solcnsImp_u1[0][:], '-r', linewidth=3.0)
for i in range(1,nstp+1):
    
    plt.plot(xi, solcnsk_u7[i][:])
    plt.xlabel("Distancia x [$m$]")
    plt.ylabel("u_7 [$mol/l$]")


plt.figure('Graf_r1',figsize=(12,8))
plt.style.use('fast')
plt.title('Perfil de $r_k1$ vs distancia')
# plt.plot(xi, solcnsImp_u1[0][:], '-r', linewidth=3.0)
for i in range(0,nstp):
    
    plt.plot(xi, r1_hist[i][:])
    plt.xlabel("Distancia x [$m$]")
    plt.ylabel("r_1 [mol/l/day]")



plt.figure('Graf_r2',figsize=(12,8))
plt.style.use('fast')
plt.title('Perfil de $rk_2$ vs distancia')
# plt.plot(xi, solcnsImp_u1[0][:], '-r', linewidth=3.0)
for i in range(0,nstp):
    
    plt.plot(xi, r2_hist[i][:])
    plt.xlabel("Distancia x [$m$]")
    plt.ylabel("r_2 [mol/l/day]")


plt.figure('Graf_H',figsize=(12,8))
plt.style.use('fast')
plt.title('Perfil de $H+$ vs distancia')
# plt.plot(xi, solcnsImp_u1[0][:], '-r', linewidth=3.0)
for i in range(0,nstp):
    
    plt.plot(xi, H_hist[i][:])
    plt.xlabel("Distancia x [$m$]")
    plt.ylabel("H+ [mol/l]")



#--------------------------------------------------------------------------------------------------#
#------------------------ Graficando resultados y comparandolo vs WMAI ----------------------#
#---------------------------------------------------------------------------------......---------#
excel_file = "denit_case1_v8.xlsx"

x, times, df_long = rd.extract_domain_data(excel_file)

print("x =", x)
print("times =", times)
print(df_long.head())

available = rd.list_available_names(df_long)

print("Especies:")
print(available["species"])

print("\nComponentes:")
print(available["component"])

times_sp, x, cl_data = rd.get_variable_matrix(df_long, "cl-", kind="species")

plt.figure('Graf_ComCl',figsize=(12,8))
plt.title('Perfil de $Cl-$ vs distancia')    
plt.plot(xi, cl_data[3][:], label="WMA t=0.1")
plt.plot(xi, cl_data[4][:], label="WMA t=1.0")
plt.plot(xi,  solcnsk_u2[10][:], "k--o", label="this_code t=0.1")
plt.plot(xi,  solcnsk_u2[100][:], "m--s", label="this_code t=1.0")
plt.xlabel("Distancia x [$m$]")
plt.ylabel("Cl [mol/l]")
plt.legend(loc=0)
plt.minorticks_on()
plt.grid(True,which='major', color='k', linestyle='--')
# plt.grid(True, which='minor', color='g', linestyle='--')

times_sp, x, h_data = rd.get_variable_matrix(df_long, "h+", kind="species")
plt.figure('Graf_ComH1',figsize=(12,8))
plt.style.use('fast')
plt.title('Perfil de $H+$ vs distancia')    
plt.plot(xi, h_data[3][:], label="WMA t=0.1")
plt.plot(xi, h_data[4][:], label="WMA t=1.0")
plt.plot(xi, H_hist[9][:], "-o", label="this_code t=0.1")
plt.plot(xi, H_hist[99][:], "-s", label="this_code t=1.0")
plt.xlabel("Distancia x [$m$]")
plt.ylabel("H+ [mol/l]")
plt.legend(loc=0)
plt.minorticks_on()
plt.grid(True,which='major', color='k', linestyle='--')
# plt.grid(True, which='minor', color='g', linestyle='--')

times_sp, x, ch2o_data = rd.get_variable_matrix(df_long, "ch2o(aq)", kind="species")
plt.figure('Graf_ComCH2O',figsize=(12,8))
plt.style.use('fast')
plt.title('Perfil de $u_3(CH2O)$ vs distancia')
plt.plot(xi, ch2o_data[3][:], label="WMA t=0.1")
plt.plot(xi, ch2o_data[4][:], label="WMA t=1.0")
plt.plot(xi,  CH2O_hist[9][:], "-o", label="this_code t=0.1")
plt.plot(xi,  CH2O_hist[99][:], "-s", label="this_code t=1.0")
plt.xlabel("Distancia x [$m$]")
plt.ylabel("CH2O [$mol/l$]")
plt.legend(loc=0)
plt.minorticks_on()
plt.grid(True,which='major', color='k', linestyle='--')
# plt.grid(True, which='minor', color='g', linestyle='--')


times_sp, x, n2_data = rd.get_variable_matrix(df_long, "n2(aq)(p)", kind="species")
plt.figure('Graf_ComN2',figsize=(12,8))
plt.style.use('fast')
plt.title('Perfil de $N_2$ vs distancia')
plt.plot(xi, n2_data[3][:], label="WMA t=0.1")
plt.plot(xi, n2_data[4][:], label="WMA t=1.0")
plt.plot(xi,  N2_hist[9][:], "-o", label="this_code t=0.1")
plt.plot(xi,  N2_hist[99][:], "-s", label="this_code t=1.0")
plt.xlabel("Distancia x [$m$]")
plt.ylabel("N2 [$mol/l$]")
plt.legend(loc=0)
plt.minorticks_on()
plt.grid(True,which='major', color='k', linestyle='--')
# plt.grid(True, which='minor', color='g', linestyle='--')



times_sp, x, ca_data = rd.get_variable_matrix(df_long, "ca+2", kind="species")
plt.figure('Graf_ComCa2',figsize=(12,8))
plt.style.use('fast')
plt.title('Perfil de $Ca_2$ vs distancia')
plt.plot(xi, ca_data[3][:], label="WMA t=0.1")
plt.plot(xi, ca_data[4][:], label="WMA t=1.0")
plt.plot(xi,  Ca_hist[9][:], "-o", label="this_code t=0.1")
plt.plot(xi,  Ca_hist[99][:], "-s", label="this_code t=1.0")
plt.xlabel("Distancia x [$m$]")
plt.ylabel("Ca_2 [$mol/l$]")
plt.legend(loc=0)
plt.minorticks_on()
plt.grid(True,which='major', color='k', linestyle='--')
# plt.grid(True, which='minor', color='g', linestyle='--')


times_sp, x, no3_data = rd.get_variable_matrix(df_long, "no3-", kind="species")
plt.figure('Graf_ComNO3',figsize=(12,8))
plt.style.use('fast')
plt.title('Perfil de $NO_3$ vs distancia')
plt.plot(xi, no3_data[3][:], label="WMA t=0.1")
plt.plot(xi, no3_data[4][:], label="WMA t=1.0")
plt.plot(xi,  NO3_hist[9][:], "-o", label="this_code t=0.1")
plt.plot(xi,  NO3_hist[99][:], "-s", label="this_code t=1.0")
plt.xlabel("Distancia x [$m$]")
plt.ylabel("$NO_3$ [$mol/l$]")
plt.legend(loc=0)
plt.minorticks_on()
plt.grid(True,which='major', color='k', linestyle='--')
# plt.grid(True, which='minor', color='g', linestyle='--')


times_sp, x, o2_data = rd.get_variable_matrix(df_long, "o2(aq)", kind="species")
plt.figure('Graf_comO2',figsize=(12,8))
plt.style.use('fast')
plt.title('Perfil de $O_2$ vs distancia')
plt.plot(xi, o2_data[3][:], label="WMA t=0.1")
plt.plot(xi, o2_data[4][:], label="WMA t=1.0")
plt.plot(xi,  O2_hist[9][:], "-o", label="this_code t=0.1")
plt.plot(xi,  O2_hist[99][:], "-s", label="this_code t=1.0")
plt.xlabel("Distancia x [$m$]")
plt.ylabel("$O_2$ [$mol/l$]")
plt.legend(loc=0)
plt.minorticks_on()
plt.grid(True,which='major', color='k', linestyle='--')
# plt.grid(True, which='minor', color='g', linestyle='--')


times_sp, x, hco3_data = rd.get_variable_matrix(df_long, "hco3-", kind="species")
plt.figure('Graf_HCO3',figsize=(12,8))
plt.style.use('fast')
plt.title('Perfil de $HCO_3$ vs distancia')
plt.plot(xi, hco3_data[3][:], label="WMA t=0.1")
plt.plot(xi, hco3_data[4][:], label="WMA t=1.0")
plt.plot(xi,  HCO3_hist[9][:], "-o", label="this_code t=0.1")
plt.plot(xi,  HCO3_hist[99][:], "-s", label="this_code t=1.0")
plt.xlabel("Distancia x [$m$]")
plt.ylabel("$HCO_3$ [$mol/l$]")
plt.legend(loc=0)
plt.minorticks_on()
plt.grid(True,which='major', color='k', linestyle='--')
# plt.grid(True, which='minor', color='g', linestyle='--')



r1_data=np.array([7.85E-04,	1.90E-04,	4.70E-05,	1.17E-05,	2.97E-06,	7.62E-07,	2.00E-07,	5.39E-08,	1.51E-08,	5.23E-09])
r1_data1=np.array([7.85E-04,	1.90E-04,	4.67E-05,	1.16E-05,	2.86E-06,	7.10E-07,	1.76E-07,	4.38E-08,	1.10E-08,	3.30E-09])


plt.figure('Graf_comr1',figsize=(12,8))
plt.style.use('fast')
plt.title('Perfil de $r_1$ vs distancia')
plt.plot(xi, r1_data, label="WMA t=0.1")
plt.plot(xi, r1_data1, label="WMA t=1.0")
plt.plot(xi,  r1_hist[9][:], "-o", label="this_code t=0.1")
plt.plot(xi,  r1_hist[99][:], "-s", label="this_code t=1.0")
plt.xlabel("Distancia x [$m$]")
plt.ylabel("$r_1$ [$mol/l/day$]")
plt.legend(loc=0)
plt.minorticks_on()
plt.grid(True,which='major', color='k', linestyle='--')
# plt.grid(True, which='minor', color='g', linestyle='--')


r2_data=np.array([2.75E-03,	1.13E-03,	4.84E-04,	2.12E-04,	9.76E-05,	4.99E-05,	3.00E-05,	2.15E-05,	1.79E-05,	1.66E-05])
r2_data1=np.array([2.94E-03,	1.21E-03,	5.27E-04,	2.39E-04,	1.12E-04,	5.46E-05,	2.76E-05,	1.47E-05,	8.62E-06,	6.14E-06])

plt.figure('Graf_comr2',figsize=(12,8))
plt.style.use('fast')
plt.title('Perfil de $r_2$ vs distancia')
plt.plot(xi, r2_data, label="WMA t=0.1")
plt.plot(xi, r2_data1, label="WMA t=1.0")
plt.plot(xi,  r2_hist[9][:], "-o", label="this_code t=0.1")
plt.plot(xi,  r2_hist[99][:], "-s", label="this_code t=1.0")
plt.xlabel("Distancia x [$m$]")
plt.ylabel("$r_2$ [$mol/l/day$]")
plt.legend(loc=0)
plt.minorticks_on()
plt.grid(True,which='major', color='k', linestyle='--')
# plt.grid(True, which='minor', color='g', linestyle='--')

plt.show()





tiempoComputo_final=time()                                                                         #Toma el valor de tiempo final de ejecución, para  calcuar el tiempo total de la simulación
tiempo_ejec_seg=(tiempoComputo_final-tiempoComputo_inicial)                                               #Calcula el tiempo total de simulación, pasado a segundos
tiempo_ejec_min=(tiempoComputo_final-tiempoComputo_inicial)/60                                            #Calcula el tiempo total de simulación, pasado a minutos

print("\nTiempo de Ejecución:",tiempo_ejec_seg, "[Segundos]","\t", tiempo_ejec_min, "[Minutos]\n")
print("|------------------------------------------------------------------------------|")
print("|------------------ FIN DE LA SIMULACIÓN WMA IMP  -------------------------|")
print("|------------------------------------------------------------------------------|")

# framesU1[0].save('U1.gif', format='GIF', append_images=framesU1[1:], save_all=True, duration=200, loop=0)  #Crea el gif de la presión con las imagenes guardadas en la lista de frames
# framesC1[0].save('C1.gif', format='GIF', append_images=framesC1[1:], save_all=True, duration=200, loop=0)  #Crea el gif de la presión con las imagenes guardadas en la lista de frames
# framesC2[0].save('C2.gif', format='GIF', append_images=framesC2[1:], save_all=True, duration=200, loop=0)  #Crea el gif de la presión con las imagenes guardadas en la lista de frames
# framesR1[0].save('R1.gif', format='GIF', append_images=framesR1[1:], save_all=True, duration=200, loop=0)  #Crea el gif de la presión con las imagenes guardadas en la lista de frames

