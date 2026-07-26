import os, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flopy
import xmf6
from gypsum import gwf_exch, gwt_exch, gwf_gwt_exch_api, lambdas, wma, vis

#
# --- DEFINICIÓN DE LAS RUTAS Y NOMBRES ---
#
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
    gypsum_eq_exch = 'gypsum_eq_exch.out',
    excel_data = 'comparativa_02.xlsx',   # Datos para comparación
    #
    # Nombre de los modelos y espacios de trabajo
    sim_name = "flow_trans",
    sim_ws = "output_gwf_gwt_exch_api_wma",
    flow_name = "flow",
    tran_name = "transport",
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
#
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
    'perioddata': [(1.0, 1, 1.0)] #PERLEN, NSTP, TSMULT
}
xmf6.nice_print("Time discretization (flow)", tdis)

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
U_ini = c1_ini - c2_ini #np.full((nlay,nrow,ncol), 1.0) # En todo el dominio 
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
    bc_conc_t1 = [("CNC-1", [(0, 0, 0), 1.0])], # 1.0 o U_s  (no converge!)
    longitudinal_dispersivity = 0.2,
    dispersion_coefficient = 0.2,
    decay_rate =  0.0,    
)
# Agregamos la información del pozo
q = phys["specific_discharge"] * dis['delc'] * dis['delr'] * dis['top']
phys["well"] = [("WEL-1", "AUX", "CONCENTRATION"), ((0, 0, 0), q, phys["source_concentration"])]

xmf6.nice_print("Physical parameters", phys)

#
# --- SIMULACIÓN GWF-GWT exchange ---
# Componentes
# Simulación y discretización temporal. 
# Los objetos 'o_sim' y 'o_tdis' se comparten por ambos modelos.

xmf6.nice_print("GWF-GWT exchange model")
# Creación del objeto de la simulación de flujo
o_sim = flopy.mf6.MFSimulation(
    sim_name = paths["sim_name"], 
    sim_ws = paths["sim_ws"], 
    exe_name = paths["mf6_exe"], 
    version="mf6"
)

# Creación del objeto de la discretización del tiempo
o_tdis = flopy.mf6.ModflowTdis(
    o_sim,
    time_units = tdis["units"],
    nper = tdis["nper"],
    perioddata = tdis["perioddata"],
)

# -------------------------------------------

print("-> Function gwf_exch.build(...) :  flow model creation")
# Escritura de los archivos de entrada para la simulación.
o_gwf = gwf_exch.build(paths, o_sim, phys, dis, silent = False) 

print("-> Function gwt_exch.build(...) :  transport model creation")
# Escritura de los archivos de entrada para la simulación.
o_gwt = gwt_exch.build(paths, o_sim, phys, dis, silent = False) 

print("-> GWF-GWT exchange creation")
# Agregamos el objeto del intercambio entre los modelos.
o_gwfgwt = flopy.mf6.ModflowGwfgwt(
    o_sim, 
    exgtype="GWF6-GWT6", 
    exgmnamea=o_gwf.name, 
    exgmnameb=o_gwt.name,
    filename=f"{paths["sim_name"]}.gwfgwt",
)

xmf6.nice_print("Writing input files for the simulation")
# Escritura de los archivos de entrada para la simulación.
o_sim.write_simulation(silent = False)

xmf6.nice_print("Initializing and running the API (GWF-GWT exchange)")
print("- > Getting the coefficients ")
# Ejecución de la simulación con la API.
# Obtenemos la matriz y el RHS del transporte: "SLN_2"
A, RHS, U = gwf_gwt_exch_api.run(o_sim, paths, "SLN_2")

# --- CÁLCULO DE LAS PROPORCIONES DE MEZCLA USANDO LOS COEF DE GWT ---
#
# Cálculo de las proporciones de mezcla
lambdas.mixingRatios_gwt(dis, tdis_t, phys, A, RHS, U, paths, silent=True)

# --- CÁLCULO DEL TRANSPORTE REACTIVO ---
xmf6.nice_print("Executing TR_1D_oper.exe")

# Ejecución de "TR_1D_oper.exe"
wma.run(paths)

# Copiamos el resultado al directorio principal
shutil.copy2(paths["gypsum_eq"], paths["gypsum_eq_exch"])
#
# - ANÁLISIS DE RESULTADOS
#
# Recuperamos las coordenadas del dominio
grid = o_gwf.modelgrid
#
# Recuperamos las coordenadas de los centros de las celdas de la malla
x = grid.xcellcenters[0] # centros de los volúmenes
#
# Recuperamos los resultados del transporte reactivo
c_gwt = vis.data_recovery("gypsum_eq_exch.out")
#
# Datos de la solución exacta
c1, c2, an_c1, an_c2 = vis.data_recovery_excel("comparativa_02.xlsx")
#
# Visualización
vis.latex(True)
vis.plot(x, (an_c1, an_c2),
         (c_gwt,), ("s-",), ("GWF-GWT + API",))

