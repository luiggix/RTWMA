import os
import shutil
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from time import perf_counter
import flopy
import xmf6

## ---> Módulos para las diferentes etapas del cálculo numérico
from src_02 import gwf, gwf_api, lambdas, wma, vis

## ---> Las variables de entorno de leen del archivo 'env.json'
with open('../env.json', 'r', encoding='utf-8') as file:
    env = json.load(file)
env["ROOT_DIR"] = os.getcwd() # Agregamos el dir raíz

xmf6.nice_print("Environment variables", env)
    
# --- DEFINICIÓN DE RUTAS Y ARCHIVOS ---

## ---> Diccionario con información de rutas y archivos
paths = dict(
    # Ejecutable de Modflow
    mf6_exe = env["MF6EXE"],
    mf6_dll = env["MF6DLL"],
    
    # Directorio de trabajo para WMA
    io_wma = os.path.join(env["ROOT_DIR"], "io_wma"),
    
    # Ejecutable "TR_1D_oper.exe"
    tr1d_exe = env["RT_WMA"],
    
    # Nombre de los modelos y espacios de trabajo
    flow_name = "flow",
    flow_ws = "io_mf6/gwf_api_dfc_wma",
    
    # Directorio de datos de salida
    vis_ana = os.path.join(env["ROOT_DIR"], "vis_ana")
)

## ---> Archivo para almacenar las proporciones de mezcla 
paths["wma_lambdas_filename"] = os.path.join(paths["io_wma"], 
                                             "input", "gypsum_eq", 
                                             "WMA_lambdas_gypsum_eq.dat")

## ---> Archivo para almacenar las proporciones de mezcla 
paths["wma_discr_gypsum_eq"] = os.path.join(paths["io_wma"], 
                                            "input", "gypsum_eq", 
                                            "WMA_discr_gypsum_eq.dat")

## ---> Archivo requerido por "TR_1D_oper.exe"
paths["wma_workingDirectory_file"] = os.path.join(paths["io_wma"], "workingDirectory.txt")

## ---> Archivo de resultados del transporte reactivo
paths["gypsum_eq"] = os.path.join(paths["io_wma"], 'gypsum_eq.out')

## ---> Copia de los resultados del transporte reactivo para comparación
paths["gypsum_eq_copy"] = os.path.join(env["ROOT_DIR"], paths["vis_ana"], "gypsum_eq_gwf_api_dfc.out")

## ---> Archivo con datos para comparación
paths["excel_data"] = os.path.join(env["ROOT_DIR"], paths["vis_ana"], 'comparativa_02.xlsx')

xmf6.nice_print("Paths, filenames and more ...", paths)

# --- DATOS PARA LA SIMULACIÓN ---

## ---> Discretización espacial
L = 30 # Longitud del dominio
nlay = 1
nrow = 1
ncol = 30
delr = L / ncol # 1.0
delc = 1.0
top  = 1.0
botm = 0.0

dis = {
    'length_units' : "meters",
    'units' : "meters",
    'nlay': nlay, 
    'nrow': nrow, 
    'ncol': ncol,
    'delr': delr, 
    'delc': delc, 
    'top' : top, 
    'botm': botm 
}
xmf6.nice_print("Spatial discretization", dis)

## ---> Discretización del tiempo para el flujo 
tdis = {
    'units': "days",
    'nper' : 1,
    'perioddata': [(40.0, 1, 1.0)] #PERLEN, NSTP, TSMULT
}
xmf6.nice_print("Time discretization (flow)", tdis)

## ---> Condición inicial para el transporte
x_pulse = 10.5
i_pulse = int(x_pulse / delr) # 10

## ---> Arreglo para la condición inicial de c1
c1_ini = np.full((nlay,nrow,ncol), 1.0000329371) # En todo el dominio
c1_ini[0, 0, i_pulse] = 200.000000165  # Pulso en x_11

## ---> Arreglo para la condición inicial de c2
c2_ini = np.full((nlay,nrow,ncol), 3.294e-5) # En todo el dominio
c2_ini[0, 0, i_pulse] = 1.647e-7  # Pulso en x_11

## ---> Arreglo para la condición inicial de U
U_ini = c1_ini - c2_ini # En todo el dominio
U_ini[0, 0, i_pulse] = c1_ini[0, 0, i_pulse] - c2_ini[0, 0, i_pulse]  # Pulso en x_11
xmf6.nice_print("Array info: U")
xmf6.info_array(U_ini)

## ---> Concentración en la inyección
U_s = c1_ini[0, 0, 0] - c2_ini[0, 0, 0]

## ---> Diccionario para almacenar los parámetros físicos.
phys = dict(
    initial_head = 1.0,
    bc_head_t1 =  {
        "pname": "CHD-1", 
        "stress_period_data": [[(0, 0, dis['ncol'] - 1), 1.0],],
    },
    hydraulic_conductivity = 1.0, 
    specific_discharge = 0.2, 
    source_concentration = U_s,
    porosity = 0.5,
    initial_concentration = U_ini,
    longitudinal_dispersivity = 0.5,
    dispersion_coefficient = 0.2,
)

## ---> Agregamos la información del pozo
q = phys["specific_discharge"] * dis['delc'] * dis['delr'] * (dis['top'] - dis['botm'])
phys["well"] = {
    "pname": "WEL-1",
    "stress_period_data": [[(0, 0, 0), q, phys["source_concentration"]],],
    "auxiliary": ["CONCENTRATION"]
}

## ---> Requerido en la funcion lambdas.mixingRatios_dfc()
phys["decay_rate"] =  0.0

xmf6.nice_print("Physical parameters", phys)

# --- SIMULACIÓN DE FLUJO ---

## ---> Escritura de los archivos de entrada para la simulación de flujo.
xmf6.nice_print("Writing input files for GWF.")
o_sim, o_gwf = gwf.build(paths, tdis, phys, dis, silent = False) 

## ---> Ejecución de la simulación de flujo a través de la API
xmf6.nice_print("Initializing and running the API (GWF)")
head, qtotal = gwf_api.run(o_sim, paths)
qx = qtotal[:,0] # Descarga específica en dirección x

# --- ANÁLSIS DE RESULTADOS DEL FLUJO ---
    
## ---> Recuperamos las coordenadas del dominio
grid = o_gwf.modelgrid

## ---> Recuperamos las coordenadas de los centros de las celdas de la malla
x, y, z = o_gwf.modelgrid.xyzcellcenters
    
## ---> Tiempos calculados para el flujo.
times_h = np.array(o_gwf.output.head().get_times())

flow_data = dict(
    xcoord = [x],
    times_h = [times_h],
    head = [head],
    qx = [qx],
)
xmf6.nice_print("Head", flow_data)
    
# --- CÁLCULO DE LAS PROPORCIONES DE MEZCLA USANDO DFC ---

## ---> Pasos de tiempo para el cálculo del transporte reactivo
tdis = {
        'units': "days",
        'nper' : 1,
        'perioddata': [(40.0, 40, 1.0)]
}
xmf6.nice_print("Time discretization for WMA", tdis)

## ---> Cálculo de las lambdas 
xmf6.nice_print("Calculating 𝜆's")

## ---> Cálculo de las proporciones de mezcla
print("-> Calculating DFC coefficients and 𝜆's")
lambdas1D, mixingWaters = lambdas.mixingRatios_dfc(phys, o_gwf.modelgrid, tdis, head, qx)

## ---> Almacenamiento de las proporciones de mezcla 
print("-> Storing the mixing ratios")
lambdas.save_mixing(paths["wma_lambdas_filename"], lambdas1D, mixingWaters)
    
# --- CÁLCULO DEL TRANSPORTE REACTIVO ---

## ---> Ejecución de "TR_1D_oper.exe"
xmf6.nice_print("Executing TR_1D_oper.exe")
wma.run(paths)

## ---> Copiamos el resultado al directorio principal
shutil.copy2(paths["gypsum_eq"], paths["gypsum_eq_copy"])

# --- ANÁLISIS DE LOS RESULTADOS ---

## ---> Recuperamos los resultados del transporte reactivo
c_dfc = vis.data_recovery(paths["gypsum_eq_copy"])

## ---> Datos de la solución exacta
c1, c2, an_c1, an_c2 = vis.data_recovery_excel(paths["excel_data"])

## ---> Visualización
vis.latex(True)
vis.plot(x[0], (an_c1, an_c2), 
         (c_dfc,), ("s-",), ("DFC",))