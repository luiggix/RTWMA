import os, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flopy
import xmf6
from gypsum import gwf, gwt, gwt_api, lambdas, wma, vis

# Variables de entorno
import json
with open('../env.json', 'r', encoding='utf-8') as file:
    env = json.load(file)
env["ROOT_DIR"] = os.getcwd() # Agregamos el dir raíz

for d in env.items():
    print(d)
    
#
# --- DEFINICIÓN DE LAS RUTAS Y NOMBRES ---
#
paths = dict(
    # Ejecutable de Modflow
    mf6_exe = env["MF6EXE"],
    mf6_dll = env["MF6DLL"],
    #
    # Directorio de trabajo para WMA
    wma_working_dir = os.path.join(env["ROOT_DIR"], "rt_files"),
    #
    # Ejecutable "TR_1D_oper.exe"
    tr1d_exe = os.path.join(env["RT_BINARIES"], "TR_1D_oper.exe"),
    #
    # Nombre de los modelos y espacios de trabajo
    flow_name = "flow",
    flow_ws = "output_gwf_gwt_api_wma/gwf",
    tran_name = "transport",
    tran_ws = "output_gwf_gwt_api_wma/gwt",
    #
    # Directorio de datos de salida
    visual = os.path.join(env["ROOT_DIR"], "visual_analysis")
)

#
# Archivo para almacenar las proporciones de mezcla 
paths["wma_lambdas_filename"] = os.path.join(paths["wma_working_dir"], 
                                             "input", "gypsum_eq", 
                                             "WMA_lambdas_gypsum_eq.dat")
#
# Archivo requerido por "TR_1D_oper.exe"
paths["wma_workingDirectory_file"] = os.path.join(paths["wma_working_dir"], "workingDirectory.txt")
#
# Archivo de resultados del transporte reactivo
paths["gypsum_eq"] = os.path.join(paths["wma_working_dir"], 'gypsum_eq.out')
#
# Copia de los resultados del transporte reactivo para comparación
paths["gypsum_eq_gwt"] = os.path.join(env["ROOT_DIR"], paths["visual"], "gypsum_eq_gwt.out")
#
# Archivo con datos para comparación
paths["excel_data"] = os.path.join(env["ROOT_DIR"], paths["visual"], 'comparativa_02.xlsx')

xmf6.nice_print("Paths, filenames and more ...", paths)

# --- DATOS PARA LA SIMULACIÓN ---
#
# Discretización espacial
nlay = 1
nrow = 1
ncol = 30
delr = 1.0
delc = 1.0
top  = 1.0
botm = 0.0

dis = {
    'length_units' : "meters",
    'nlay': 1, 
    'nrow': 1, 
    'ncol': 30,
    'delr': 1.0, 
    'delc': 1.0, 
    'top' : 1.0, 
    'botm': 0.0 
}
xmf6.nice_print("Spatial discretization", dis)
#
# Discretización del tiempo para el flujo
tdis_f = {
    'units': "days",
    'nper' : 1,
    'perioddata': [(40.0, 1, 1.0)]
}
xmf6.nice_print("Time discretization (flow)", tdis_f)
#
# Discretización del tiempo para el transporte
tdis_t = {
        'units': "days",
        'nper' : 1,
        'perioddata': [(40.0, 40, 1.0)] #PERLEN, NSTP, TSMULT
    }
xmf6.nice_print("Time discretization (transport)", tdis_t)

# Arreglo para la condición inicial de c1
c1_ini = np.full((nlay,nrow,ncol), 1.0000329) # En todo el dominio
c1_ini[0, 0, 11] = 200.0  # Pulso en x_L

# Arreglo para la condición inicial de c2
c2_ini = np.full((nlay,nrow,ncol), 3.294e-5) # En todo el dominio
c2_ini[0, 0, 11] = 1.647e-7  # Pulso en x_L

# Arreglo para la condición inicial de U
U_ini = c1_ini - c2_ini # np.full((nlay,nrow,ncol), 1.0) # En todo el dominio
U_ini[0, 0, 11] = c1_ini[0, 0, 11] - c2_ini[0, 0, 11]  # Pulso en x_L
print("Array info: U")
xmf6.info_array(U_ini)

U_s = c1_ini[0, 0, 0] - c2_ini[0, 0, 0]

phys = dict(
    initial_head = 1.0,
    bc_head_t1 = [("CHD-1" , [(0, 0, dis['ncol'] - 1), 1.0])],
    hydraulic_conductivity = 1.0, 
    specific_discharge = 0.2, 
    source_concentration = U_s, # 1.0 o U_s 
    porosity = 0.5,
    initial_concentration = U_ini,
    bc_conc_t1 = [("CNC-1", [(0, 0, 0), U_s])],
    longitudinal_dispersivity = 0.5, 
    dispersion_coefficient = 0.2,
    decay_rate =  0.0,    
)
# Agregamos la información del pozo
q = phys["specific_discharge"] * dis['delc'] * dis['delr'] * dis['top']
phys["well"] = [("WEL-1", "AUX", "CONCENTRATION"), ((0, 0, 0), q, phys["source_concentration"])]

xmf6.nice_print("Physical parameters", phys)

#
# --- SIMULACIÓN DE FLUJO ---
xmf6.nice_print("Writing input files for GWF")
# Escritura de los archivos de entrada para la simulación.
o_sim_f, o_gwf = gwf.build(paths, tdis_f, phys, dis, silent = False)

xmf6.nice_print("Executing GWF")
# Ejecución de la simulación de flujo.
o_sim_f.run_simulation(silent=False)

# --- ANÁLSIS DE RESULTADOS DEL FLUJO ---

# Objeto para acceder a los resultados de la carga hidráulica.
o_head = o_gwf.output.head()

# Tiempos calculados para el flujo.
times_h = np.array(o_head.get_times())

# Recuperamos la carga hidráulica del paso 40.
head = o_head.get_data(totim=40)[0, 0, :]

# Recuperamos la descarga específica del paso 40.
budget = o_gwf.output.budget()
spdis = budget.get_data(totim=40, text="DATA-SPDIS")[0]
qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, o_gwf)

# Recuperamos las coordenadas del dominio
grid = o_gwf.modelgrid

# Recuperamos las coordenadas de los centros de las celdas de la malla
x = grid.xcellcenters[0] # centros de los volúmenes

flow_data = dict(
    xcoord = [x],
    times_h = [times_h],
    head = [head],
    qx = qx[:,0],
)
xmf6.nice_print("Head", flow_data)

# --- CÁLCULO DE LAS PROPORCIONES DE MEZCLA USANDO LOS COEF DE GWT ---
#
# Construcción de las matrices de transporte usando GWT + API
xmf6.nice_print("Writting input files for GWT")
# Escritura de los archivos de entrada para la simulación.
o_sim_t, o_gwt = gwt.build(paths, tdis_t, phys, dis, silent = False)

xmf6.nice_print("Initializing and running the API (GWT)")
# Ejecución de la simulación de transporte a través de la API
A, RHS, U = gwt_api.run(o_sim_t, paths)
#
xmf6.nice_print("Construction of 𝜆 matrices (GWT)")
# Cálculo de las proporciones de mezcla
lambdas.mixingRatios_gwt(dis, tdis_t, phys, A, RHS, U, paths, silent=True)

# --- CÁLCULO DEL TRANSPORTE REACTIVO ---
xmf6.nice_print("Executing TR_1D_oper.exe")

# Ejecución de "TR_1D_oper.exe"
wma.run(paths)

# Copiamos el resultado al directorio principal
shutil.copy2(paths["gypsum_eq"], paths["gypsum_eq_gwt"])

#
# - ANÁLISIS DE RESULTADOS
#
# Recuperamos los resultados del transporte reactivo
c_gwt = vis.data_recovery(paths["gypsum_eq_gwt"])
#
# Datos de la solución exacta
c1, c2, an_c1, an_c2 = vis.data_recovery_excel(paths["excel_data"])
#
# Visualización
vis.latex(True)
vis.plot(x, (an_c1, an_c2),
         (c_gwt,), ("s-",), ("GWT",))
