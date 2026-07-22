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


def monod_rates_denit(
    CH2O, O2, NO3,
    mu1=0.5, K_DOC1=1.0e-3, K_O2=1.0e-5,
    mu2=2.0, K_DOC2=1.4e-3, K_NO3=5.9e-3, I_O2=2.52e-5,
    eps=1e-30
):
    CH2O = max(float(CH2O), 0.0)
    O2   = max(float(O2), 0.0)
    NO3  = max(float(NO3), 0.0)

    rk1 = (
        mu1
        * CH2O / (K_DOC1 + CH2O + eps)
        * O2   / (K_O2   + O2   + eps)
    )

    rk2 = (
        mu2
        * I_O2 / (I_O2 + O2 + eps)
        * CH2O / (K_DOC2 + CH2O + eps)
        * NO3  / (K_NO3  + NO3  + eps)
    )

    return rk1, rk2


def kinetics_update_U_denit_implicit(
    u_transport,
    phi,
    dt_days,
    mu1=0.5, K_DOC1=1.0e-3, K_O2=1.0e-5,
    mu2=2.0, K_DOC2=1.4e-3, K_NO3=5.9e-3, I_O2=2.52e-5
):
    """
    Cinética implícita para el nuevo ejemplo de denitrificación.

    Variables:
        u1 = H - CO3
        u2 = Cl
        u3 = NO3
        u4 = N2
        u5 = CH2O
        u6 = HCO3 + CO3
        u7 = O2
    """

    u_transport = np.asarray(u_transport, dtype=float)

    def residual(u_new):
        u1, u2, u3, u4, u5, u6, u7 = u_new

        NO3  = max(u3, 0.0)
        CH2O = max(u5, 0.0)
        O2   = max(u7, 0.0)

        rk1, rk2 = monod_rates_denit(
            CH2O=CH2O,
            O2=O2,
            NO3=NO3,
            mu1=mu1, K_DOC1=K_DOC1, K_O2=K_O2,
            mu2=mu2, K_DOC2=K_DOC2, K_NO3=K_NO3, I_O2=I_O2
        )


        R1 = rk1
        R2 = rk2

        F = np.zeros(7)

        F[0] = u1 - (u_transport[0] + dt_days * ( R1 + 0.2 * R2 ))
        F[1] = u2 -  u_transport[1]
        F[2] = u3 - (u_transport[2] - dt_days * ( 0.8 * R2 ))
        F[3] = u4 - (u_transport[3] + dt_days * ( 0.4 * R2 ))
        F[4] = u5 - (u_transport[4] - dt_days * ( R1 + R2 ))
        F[5] = u6 - (u_transport[5] + dt_days * ( R1 + R2 ))
        F[6] = u7 - (u_transport[6] - dt_days * R1)

        return F

    sol = root(residual, u_transport, method="hybr", tol=1e-10)  #MINimization PACKage heredada de fortran, método hibrido de Powell 

    if not sol.success:
        raise RuntimeError(f"No convergió la cinética implícita: {sol.message}")

    u_new = sol.x

    # Protección numérica
    u_new[2] = max(u_new[2], 0.0)  # NO3
    u_new[3] = max(u_new[3], 0.0)  # N2
    u_new[4] = max(u_new[4], 0.0)  # CH2O
    u_new[6] = max(u_new[6], 0.0)  # O2

    # Tasas finales
    NO3  = u_new[2]
    CH2O = u_new[4]
    O2   = u_new[6]

    rk1, rk2 = monod_rates_denit(
        CH2O=CH2O,
        O2=O2,
        NO3=NO3,
        mu1=mu1, K_DOC1=K_DOC1, K_O2=K_O2,
        mu2=mu2, K_DOC2=K_DOC2, K_NO3=K_NO3, I_O2=I_O2
    )

    R1 = rk1
    R2 = rk2

    return u_new, R1, R2


def flash_equilibrium_denit_from_U(
    u,
    logK_bicarb=-10.3288
):
    """
    Flash de equilibrio para el nuevo ejemplo de denitrificación.

    Equilibrio:
        HCO3- = H+ + CO3--

    Ley de acción de masas:
        K = H * CO3 / HCO3

    Variables U:
        u1 = H - CO3
        u2 = Cl
        u3 = NO3
        u4 = N2
        u5 = CH2O
        u6 = HCO3 + CO3
        u7 = O2
    """

    u1, u2, u3, u4, u5, u6, u7 = map(float, u)

    K = 10.0 ** logK_bicarb

    # Resolver CO3 desde:
    # K = (u1 + CO3)*CO3/(u6 - CO3)
    # CO3^2 + (u1 + K)*CO3 - K*u6 = 0

    disc = (u1 + K)**2 + 4.0 * K * u6

    if disc < 0.0:
        raise RuntimeError("Discriminante negativo en flash bicarbonato-carbonato.")

    CO3 = (-(u1 + K) + np.sqrt(disc)) / 2.0

    H = u1 + CO3
    HCO3 = u6 - CO3

    if H <= 0.0 or HCO3 <= 0.0 or CO3 < 0.0:
        raise RuntimeError(
            f"Flash no físico: H={H:.6e}, HCO3={HCO3:.6e}, CO3={CO3:.6e}"
        )

    state = {
        "H": H,
        "pH": -np.log10(H),
        "Cl": u2,
        "NO3": u3,
        "N2": u4,
        "CH2O": u5,
        "HCO3": HCO3,
        "O2": u7,
        "CO3": CO3,
    }

    return state


def check_mass_balance_stoichiometry(tol=1e-12, verbose=True):
    """
    Verifica conservación elemental y de carga para las reacciones cinéticas:
        A_elem @ S_k.T = 0

    Especies:
        H+, Cl-, NO3-, N2(aq), CH2O(aq), HCO3-, O2(aq), CO3--, H2O
    """

    species = [
        "H+", "Cl-", "NO3-", "N2", "CH2O", "HCO3-", "O2", "CO3--", "H2O"
    ]

    elements = ["C", "H", "O", "N", "Cl", "charge"]

    S_k = np.array([
        # H+  Cl-  NO3-  N2   CH2O  HCO3-  O2   CO3--  H2O
        [1.0, 0.0,  0.0, 0.0, -1.0,  1.0, -1.0, 0.0,   0.0],
        [0.2, 0.0, -0.8, 0.4, -1.0,  1.0,  0.0, 0.0,   0.4]
    ], dtype=float)

    A_elem = np.array([
        # H+ Cl- NO3- N2 CH2O HCO3- O2 CO3-- H2O

        # C
        [0,  0,   0,  0,   1,    1,   0,   1,    0],

        # H
        [1,  0,   0,  0,   2,    1,   0,   0,    2],

        # O
        [0,  0,   3,  0,   1,    3,   2,   3,    1],

        # N
        [0,  0,   1,  2,   0,    0,   0,   0,    0],

        # Cl
        [0,  1,   0,  0,   0,    0,   0,   0,    0],

        # charge
        [1, -1,  -1,  0,   0,   -1,   0,  -2,    0],
    ], dtype=float)

    balance = A_elem @ S_k.T

    if verbose:
        print("\n--- Verificación estequiométrica: A_elem @ S_k.T ---")
        print("Especies:", species)
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
    

K_bicarb = 10**(-10.3288)

def compute_CO3_from_H_HCO3(H, HCO3, K=K_bicarb):
    return K * HCO3 / H



c_ini_primary = {
    "H":    1.0e-7,
    "Cl":   1.0e-16,
    "NO3":  1.0e-5,
    "N2":   1.0e-16,
    "CH2O": 1.0e-5,
    "HCO3": 1.0e-3,
    "O2":   1.0e-6
}

c_inf_primary = {
    "H":    1.0e-8,
    "Cl":   1.0e-3,
    "NO3":  1.0e-3,
    "N2":   1.0e-16,
    "CH2O": 2.0e-3,
    "HCO3": 5.3761e-4,
    "O2":   2.5295e-4
}

# ---------------------------------------------------------
# Componentes del dominio:
# u1 = H - CO3
# u2 = Cl
# u3 = NO3
# u4 = N2
# u5 = CH2O
# u6 = HCO3 + CO3
# u7 = O2
# ---------------------------------------------------------

u=np.zeros(7)
u_old=np.zeros(7)

r1=np.zeros(nx)
r2=np.zeros(nx)

CO3_ini = compute_CO3_from_H_HCO3(
    c_ini_primary["H"],
    c_ini_primary["HCO3"]
)

CO3_inf = compute_CO3_from_H_HCO3(
    c_inf_primary["H"],
    c_inf_primary["HCO3"]
)


u1_inf= c_inf_primary["H"]-CO3_inf
u2_inf= c_inf_primary["Cl"]
u3_inf= c_inf_primary["NO3"]
u4_inf= c_inf_primary["N2"]
u5_inf= c_inf_primary["CH2O"] 
u6_inf= c_inf_primary["HCO3"]+CO3_inf 
u7_inf= c_inf_primary["O2"]

u1_ini= c_ini_primary["H"]-CO3_ini
u2_ini= c_ini_primary["Cl"]
u3_ini= c_ini_primary["NO3"]
u4_ini= c_ini_primary["N2"]
u5_ini= c_ini_primary["CH2O"] 
u6_ini= c_ini_primary["HCO3"]+CO3_ini
u7_ini= c_ini_primary["O2"]

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
Cl_hist   = np.zeros((nstp, nx))
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
            # A[i][i]=   TC[i]+TW[i]    
            
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
            # AL[i][i]=   BETA_C+BETA_W

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
        
    # sys.exit()

    solcnsv_u1[ti+1][:] = u1_v[:]
    solcnsv_u2[ti+1][:] = u2_v[:]
    solcnsv_u3[ti+1][:] = u3_v[:]
    solcnsv_u4[ti+1][:] = u4_v[:]
    solcnsv_u5[ti+1][:] = u5_v[:]
    solcnsv_u6[ti+1][:] = u6_v[:]
    solcnsv_u7[ti+1][:] = u7_v[:]
    
  
    
    for i in range(0, nx):
        
        u[0]=u1_v[i]; 
        u[1]=u2_v[i]; 
        u[2]=u3_v[i]; 
        u[3]=u4_v[i]; 
        u[4]=u5_v[i]; 
        u[5]=u6_v[i]; 
        u[6]=u7_v[i]


        # state = flash_equilibrium_denit_from_U(u)

        # H_hist[ti][i]   = state["H"]  
        # pH_hist[ti][i]   = state["pH"]  
        # Cl_hist[ti][i]   = state["Cl"]
        # NO3_hist[ti][i]  = state["NO3"]
        # N2_hist[ti][i]   = state["N2"]
        # CH2O_hist[ti][i] = state["CH2O"]        
        # HCO3_hist[ti][i] = state["HCO3"]
        # O2_hist[ti][i]   = state["O2"]        
        # CO3_hist[ti][i]  = state["CO3"]


        u, r1[i], r2[i] = kinetics_update_U_denit_implicit(u, Por, dt)

        u1_k[i]=u[0]; 
        u2_k[i]=u[1]; 
        u3_k[i]=u[2]; 
        u4_k[i]=u[3]; 
        u5_k[i]=u[4]; 
        u6_k[i]=u[5]; 
        u7_k[i]=u[6];

        state = flash_equilibrium_denit_from_U(u)

        H_hist[ti][i]   = state["H"]  
        pH_hist[ti][i]   = state["pH"]  
        Cl_hist[ti][i]   = state["Cl"]
        NO3_hist[ti][i]  = state["NO3"]
        N2_hist[ti][i]   = state["N2"]
        CH2O_hist[ti][i] = state["CH2O"]        
        HCO3_hist[ti][i] = state["HCO3"]
        O2_hist[ti][i]   = state["O2"]        
        CO3_hist[ti][i]  = state["CO3"]
        

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
excel_file = "denit_2reacts_01.xlsx"

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



# plt.figure('Graf_ComCl',figsize=(12,8))
# plt.title('Perfil de $Cl-$ vs distancia')    
# plt.plot(xi, cl_data[3][:], label="WMA t=0.1")
# plt.plot(xi, cl_data[4][:], label="WMA t=1.0")
# plt.plot(xi,  solcnsk_u2[8][:], "k--o", label="this_code t=0.1")
# plt.plot(xi,  solcnsk_u2[70][:], "m--s", label="this_code t=0.1")
# plt.xlabel("Distancia x [$m$]")
# plt.ylabel("Cl [mol/l]")
# plt.legend(loc=0)
# plt.minorticks_on()
# plt.grid(True,which='major', color='k', linestyle='--')
# # plt.grid(True, which='minor', color='g', linestyle='--')



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
plt.title('Perfil de $u_5(CH2O)$ vs distancia')
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

