import os, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flopy
import xmf6
import gwf_for_dfc as gwf
import WMA_1D as WMA # Calcula las proporciones de mezcla
import resultsFortran as rF  #importa los resultados del WMA
linea = 50*chr(0x2015)

#
# --- DEFINICIÓN DE LAS RUTAS Y NOMBRES ---
#
paths = dict(
    # Ejecutable de Modflow: WINDOWS
    mf6_exe = r"C:\Users\luiggi\Documents\GitSites\mf6_tutorial\mf6\windows\mf6",
    mf6_dll = r"C:\Users\luiggi\Documents\GitSites\mf6_tutorial\mf6\windows\libmf6.dll",
#    mf6_exe = r"../../../bin/macosarm/mf6",
#    mf6_dll = r"../../../bin/macosarm/libmf6.dylib",    
    #
    # Directorio de trabajo para WMA
    wma_working_dir = r"C:\Users\luiggi\Documents\GitSites\RTWMA\benchmarks\03_wma\RT-EXE",
#    wma_working_dir = r"../RT-EXE/",
    wma_wdname = "workingDirectory.txt",
    #
    # Archivo de salida de "TR_1D_oper.exe"
    tr1d_ofile='gypsum_eq.out',
    tr1d_ofile_dfc = 'gypsum_eq_dfc.out',
    excel_data = 'comparativa_02.xlsx',
    #
    # Nombre de los modelos y espacios de trabajo
    flow_name = "flow",
    flow_ws = "output_gwf_gwt_api_wma",
    tran_name = "transport",
    tran_ws = "output_gwf_gwt_api_wma"
)
#
# Ejecutable "TR_1D_oper.exe"
paths["tr1d_exe"] = os.path.join(paths["wma_working_dir"], "TR_1D_oper.exe")
#
# Archivo para almacenar las proporciones de mezcla 
paths["wma_lambdas_filename"] = os.path.join(paths["wma_working_dir"], 
                                             "input", "gypsum_eq", 
                                             "WMA_lambdas_gypsum_eq.dat")
xmf6.nice_print(paths, "Rutas y nombres de archivos")
#
# --- DATOS PARA LA SIMULACIÓN ---
#
# Parámetros para la discretización espacial
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
xmf6.nice_print(dis, "Spatial discretization")

# Discretización del tiempo para el flujo 
tdis = {
    'units': "days",
    'nper' : 1,
    'perioddata': [(40.0, 1, 1.0)] #PERLEN, NSTP, TSMULT
}
xmf6.nice_print(tdis, "Time discretization (flow)")

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
    dispersion_coefficient = 0.2, # 1.0 ???
    decay_rate =  0.0,

)
# Agregamos la información del pozo
q = phys["specific_discharge"] * dis['delc'] * dis['delr'] * dis['top']
phys["well"] = [("WEL-1", "AUX", "CONCENTRATION"), ((0, 0, 0), q, phys["source_concentration"])]

xmf6.nice_print(phys, "Physical parameters")
#
# --- SIMULACIÓN DE FLUJO ---
#
# Escritura de los archivos de entrada para la simulación.
o_sim, o_gwf = gwf.build(paths, tdis, phys, dis, silent = False) 

# Ejecución de la simulación de flujo.
o_sim.run_simulation(silent = False)
#
# --- ANALISIS DE RESULTADOS DE FLUJO ---
#
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

flow_dict = dict(
    xcoord = [x],
    times_h = [times_h],
    head = [head],
    qx = qx,
)
xmf6.nice_print(flow_dict, "Head")
#
# --- CÁLCULO DE LAMBDAS A PARTIR DE DFC ---
#
# Cálculo de las lambdas ...
print(linea)
print("- Calculando las 𝜆's")

tdis = {
        'units': "days",
        'nper' : 1,
        'perioddata': [(40.0, 40, 1.0)]
}
# sim_flow["tdis"]

lambdas1D, mixingWaters = WMA.mixingRatios1D(phys, grid, tdis, head, qx[0,0])#head[0][0][:], qx[0][0][:])
#print(lambdas1D)

# Archivo para almacenar las proporciones de mezcla 
#wma_lambdas_filename = os.path.join(working_dir, "input", "gypsum_eq", "WMA_lambdas_gypsum_eq.dat")

# Almacenamiento de las proporciones de mezcla 
print(f"- Escribiendo las 𝜆's")
WMA.save_mixing(paths["wma_lambdas_filename"], lambdas1D, mixingWaters)
print(linea)
#
# --- TRANSPORTE REACTIVO ---
#
# Creación del archivo "workingDirectory.txt" necesario para 
# la ejecución del programa "TR_1D_oper.exe"
with open(os.path.join(paths["wma_working_dir"], paths["wma_wdname"]), "w") as f:
    f.write(paths["wma_working_dir"])

# Ruta al ejecutable "TR_1D_oper.exe"
#tr1d_exe = os.path.join(paths["wma_working_dir"], "TR_1D_oper.exe")

# Ejecución de "TR_1D_oper.exe"
print(linea)
print("- Ejecutando TR_1D_oper.exe")
print(linea)
result = subprocess.run([paths["tr1d_exe"]], cwd = paths["wma_working_dir"], 
                        capture_output = True, text = True)

# Salida de "TR_1D_oper.exe"
print("Standard Output:", result.stdout)
print("Standard Error:", result.stderr)
print(linea)