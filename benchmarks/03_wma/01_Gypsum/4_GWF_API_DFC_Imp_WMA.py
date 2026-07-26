import os, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flopy
import xmf6
from gypsum import gwf, gwf_api, lambdas, wma, vis

# --- DEFINICIÓN DE LAS RUTAS ---

paths = dict(
    # Ejecutable de Modflow: WINDOWS
    mf6_exe = r"C:\Users\luiggi\Documents\GitSites\mf6_tutorial\mf6\windows\mf6",
    mf6_dll = r"C:\Users\luiggi\Documents\GitSites\mf6_tutorial\mf6\windows\libmf6.dll",  
    #
    # Directorio de trabajo para WMA
    wma_working_dir = r"C:\Users\luiggi\Documents\GitSites\RTWMA\benchmarks\03_wma\RT-EXE",
    wma_wdname = "workingDirectory.txt",
    #
    # Archivo de salida de "TR_1D_oper.exe"
    gypsum_eq_dfc = 'gypsum_eq_dfc.out',
    excel_data = 'comparativa_02.xlsx',
    #
    # Nombre de los modelos y espacios de trabajo
    flow_name = "flow",
    flow_ws = "output_gwf_api_dfc_wma",
)
#
# Ejecutable "TR_1D_oper.exe"
paths["tr1d_exe"] = os.path.join(paths["wma_working_dir"], "TR_1D_oper.exe")
#
# Archivo para almacenar las proporciones de mezcla 
paths["wma_lambdas_filename"] = os.path.join(paths["wma_working_dir"], 
                                             "input", "gypsum_eq", 
                                             "WMA_lambdas_gypsum_eq.dat")
#
# Archivo de resultados del transporte reactivo
paths["gypsum_eq"] = os.path.join(paths["wma_working_dir"], 'gypsum_eq.out')
#
# Archivo requerido por "TR_1D_oper.exe"
paths["wma_workingDirectory_file"] = os.path.join(paths["wma_working_dir"], "workingDirectory.txt")

xmf6.nice_print("Paths, filenames and more ...", paths)

# --- DATOS PARA LA SIMULACIÓN ---

# Discretización espacial
nlay = 1
nrow = 1
ncol = 30
delr = 1.0
delc = 1.0
top  = 1.0
botm = 0.0

dis = {
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

# Discretización del tiempo para el flujo 
tdis = {
    'units': "days",
    'nper' : 1,
    'perioddata': [(40.0, 1, 1.0)] #PERLEN, NSTP, TSMULT
}
xmf6.nice_print("Time discretization (flow)", tdis)

# Arreglo para la condición inicial de c1
c1_ini = np.full((nlay,nrow,ncol), 1.0000329) # En todo el dominio
c1_ini[0, 0, 11] = 200.0  # Pulso en x_L

# Arreglo para la condición inicial de c2
c2_ini = np.full((nlay,nrow,ncol), 3.294e-5) # En todo el dominio
c2_ini[0, 0, 11] = 1.647e-7  # Pulso en x_L

# Arreglo para la condición inicial de U
U_ini = c1_ini - c2_ini # En todo el dominio
U_ini[0, 0, 11] = c1_ini[0, 0, 11] - c2_ini[0, 0, 11]  # Pulso en x_L
print("Array info: U")
xmf6.info_array(U_ini)

U_s = c1_ini[0, 0, 0] - c2_ini[0, 0, 0]

phys = dict(
    initial_head = 1.0,
    bc_head_t1 = [("CHD-1" , [(0, 0, dis['ncol'] - 1), 1.0])],
    hydraulic_conductivity = 1.0, 
    specific_discharge = 0.2, 
    source_concentration = U_s,
    porosity = 0.5,
    initial_concentration = U_ini,
    bc_conc_t1 = [("CNC-1", [(0, 0, 0), U_s])],
    longitudinal_dispersivity = 0.2,
    dispersion_coefficient = 0.2,
    decay_rate =  0.0,

)
# Agregamos la información del pozo
q = phys["specific_discharge"] * dis['delc'] * dis['delr'] * dis['top']
phys["well"] = [("WEL-1", "AUX", "CONCENTRATION"), ((0, 0, 0), q, phys["source_concentration"])]

xmf6.nice_print("Physical parameters", phys)

# --- SIMULACIÓN DE FLUJO ---
xmf6.nice_print("Writing input files for GWF")
# Escritura de los archivos de entrada para la simulación.
o_sim, o_gwf = gwf.build(paths, tdis, phys, dis, silent = False) 

xmf6.nice_print("Initializing and running the API (GWF)")
# Ejecución de la simulación de flujo a través de la API
head, q = gwf_api.run(o_sim, paths)

# --- ANÁLSIS DE RESULTADOS DEL FLUJO ---

# Recuperamos las coordenadas del dominio
grid = o_gwf.modelgrid

# Recuperamos las coordenadas de los centros de las celdas de la malla
x = grid.xcellcenters[0] # centros de los volúmenes

# Tiempos calculados para el flujo.
times_h = np.array(o_gwf.output.head().get_times())

flow_data = dict(
    xcoord = [x],
    times_h = [times_h],
    head = [head],
    qx = [q[:,0]],
)
xmf6.nice_print("Head", flow_data)

# --- CÁLCULO DE LAS PROPORCIONES DE MEZCLA USANDO DFC ---
#
# Cálculo de las lambdas ...
xmf6.nice_print("Construction of 𝜆 matrices (DFC)")

# Pasos de tiempo para el cálculo del transporte reactivo
tdis = {
        'units': "days",
        'nper' : 1,
        'perioddata': [(40.0, 40, 1.0)]
}
xmf6.nice_print("Time discretization for DFC", tdis)

# Cálculo de las proporciones de mezcla
lambdas1D, mixingWaters = lambdas.mixingRatios_dfc(phys, grid, tdis, head, q[:,0])

# Almacenamiento de las proporciones de mezcla 
print("-> Storing the mixing ratios")
lambdas.save_mixing(paths["wma_lambdas_filename"], lambdas1D, mixingWaters)

# --- CÁLCULO DEL TRANSPORTE REACTIVO ---
xmf6.nice_print("Executing TR_1D_oper.exe")

# Ejecución de "TR_1D_oper.exe"
wma.run(paths)

# Copiamos el resultado al directorio principal
shutil.copy2(paths["gypsum_eq"], paths["gypsum_eq_dfc"])

# --- ANÁLISIS DE LOS RESULTADOS ---
#
c_dfc = vis.data_recovery("gypsum_eq_dfc.out")
c1, c2, an_c1, an_c2 = vis.data_recovery_excel("comparativa_02.xlsx")

vis.latex(True)
vis.plot(x, (an_c1, an_c2), 
         (c_dfc,), ("s-",), ("DFC",))