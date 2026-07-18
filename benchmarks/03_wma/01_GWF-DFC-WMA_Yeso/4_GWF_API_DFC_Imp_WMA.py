import os, sys, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flopy
import xmf6
import gwf_for_dfc as gwf
import gwf_api
import wma_1D as wma
import resultsFortran as rF 
linea = 50*chr(0x2015)

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
    tr1d_ofile='gypsum_eq.out',
    tr1d_ofile_dfc = 'gypsum_eq_dfc.out',
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
xmf6.nice_print(paths, "Rutas y nombres de archivos")

# --- DATOS PARA LA SIMULACIÓN ---

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

# --- SIMULACIÓN DE FLUJO ---
print(linea)
print("Escribiendo archivos de entrada para GWF")
print(linea)
# Escritura de los archivos de entrada para la simulación.
o_sim, o_gwf = gwf.build(paths, tdis, phys, dis, silent = False) 

print(linea)
print("Ejecutando GWF con la API")
# Ejecución de la simulación de flujo a través de la API
head, qx = gwf_api.run(o_sim, paths)

# --- ANÁLSIS DE RESULTADOS DEL FLUJO ---

# Recuperamos las coordenadas del dominio
grid = o_gwf.modelgrid

# Recuperamos las coordenadas de los centros de las celdas de la malla
x = grid.xcellcenters[0] # centros de los volúmenes

# Tiempos calculados para el flujo.
times_h = np.array(o_gwf.output.head().get_times())

flow_dict = dict(
    xcoord = [x],
    times_h = [times_h],
    head = [head],
    qx = [qx[:,0]],
)
xmf6.nice_print(flow_dict, "Head")

# --- CÁLCULO DE LAS PROPORCIONES DE MEZCLA USANDO DFC ---

print(linea)
print("Iniciando DFC ")
print(linea)

# --- Cálculo de las lambdas ... ---
print("- Discretización con DFC")
print("- Calculando las 𝜆's\n")
tdis = {
        'units': "days",
        'nper' : 1,
        'perioddata': [(40.0, 40, 1.0)]
}
xmf6.nice_print(tdis, "Discretización del tiempo para DFC")

lambdas1D, mixingWaters = wma.mixingRatios1D(phys, grid, tdis, head, qx[:, 0])
    #print(lambdas1D)

# --- Almacenamiento de las proporciones de mezcla --- 
print(f"\n - Escribiendo las 𝜆's")
wma.save_mixing(paths["wma_lambdas_filename"], lambdas1D, mixingWaters)

# --- CÁLCULO DEL TRANSPORTE REACTIVO ---

print(linea)
print("Iniciando WMA ".center(50))

# Creación del archivo "workingDirectory.txt" necesario para 
# la ejecución del programa "TR_1D_oper.exe"
with open(os.path.join(paths["wma_working_dir"], "workingDirectory.txt"), "w") as f:
    f.write(paths["wma_working_dir"])

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

# --- ANÁLISIS DE LOS RESULTADOS ---

# Tiempo seleccionado de hoja excel
tsel=10

# Archivo con los resultados de TR_1D_oper.exe
file_name='gypsum_eq.out'   
file_path = os.path.join(paths["wma_working_dir"], file_name)
Res_contrsns_ini, Res_contrsns_fin = rF.import_results(file_path)

# Lectura de datos de la tabla de excel
WMA_I=pd.read_excel('comparativa_02.xlsx', sheet_name='c_2')     
data_set_C2 = np.transpose(WMA_I.iloc[11:,5:95].to_numpy())

# Calculamos el RMSE
RMSE = np.linalg.norm(data_set_C2[tsel][:]-np.array(Res_contrsns_fin[1][:])) / np.sqrt(len(Res_contrsns_fin))

# --- Gráfica de los resultados
plt.figure(figsize=(6,4))
plt.plot(x, data_set_C2[tsel][:], 
         lw=0.5, ls="--", c="k", zorder=5)
plt.scatter(x, Res_contrsns_fin[1][:], 
            marker = "s", c = "mediumblue", s=30, label="TR_1D", zorder=5)
plt.scatter(x, data_set_C2[tsel][:], 
            marker = "o", c = "violet", s=15,label="Excel", zorder=5)
plt.title(f"t = {tsel} días, RMSE = {RMSE:5.3e}")
plt.xlabel("$x$ [$m$]")
plt.ylabel("$c_2 [mgr/cm^{3}]$")
plt.legend()
plt.minorticks_on()
plt.grid(True,which='major', color='darkgray', lw = 0.5)
plt.grid(True, which='minor', color='silver', lw = 0.25)
plt.show()
