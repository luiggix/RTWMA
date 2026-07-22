import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flopy
import xmf6
import gwf_for_dfc as gwf
import lambdas_dfc_1D as lambdas 
import wma_gypsum as wma
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
    flow_ws = "output_gwf_dfc_wma",
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

xmf6.nice_print(paths, "Rutas y nombres de archivos")

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
    dispersion_coefficient = 0.2,
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
print("Ejecutando GWF")
print(linea)
# Ejecución de la simulación de flujo.
o_sim.run_simulation(silent = False)
print(linea)

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
xmf6.nice_print(flow_data, "Head")

# --- CÁLCULO DE LAS PROPORCIONES DE MEZCLA USANDO DFC ---
#
# Cálculo de las lambdas ...
print(linea)
print("- Calculando las 𝜆's")

# Pasos de tiempo para el cálculo del transporte reactivo
tdis = {
        'units': "days",
        'nper' : 1,
        'perioddata': [(40.0, 40, 1.0)]
}
xmf6.nice_print(tdis, "Discretización del tiempo para DFC")

# Cálculo de las proporciones de mezcla
lambdas1D, mixingWaters = lambdas.mixingRatios1D(phys, grid, tdis, head, qx[0,0])

# Almacenamiento de las proporciones de mezcla 
print(f"- Escribiendo las 𝜆's")
lambdas.save_mixing(paths["wma_lambdas_filename"], lambdas1D, mixingWaters)

# --- CÁLCULO DEL TRANSPORTE REACTIVO ---
print(linea)
print("- Ejecutando TR_1D_oper.exe")
print(linea)

# Ejecución de "TR_1D_oper.exe"
wma.run(paths)

print(linea)

# --- ANÁLISIS DE LOS RESULTADOS ---
#
# Leemos el archivo con los resultados de TR_1D_oper.exe ---
with open(paths["gypsum_eq"], "r") as f:
        lines = f.readlines()
#
# Obtenemos la concentración de las dos especies
key_string = "Final concentration of aqueous species:"
i = xmf6.find_index(lines, key_string, 2)
c_final = [np.array(l.split(), dtype=float) for l in lines[i:i+2]]
#
# --- Leemos del archivo excel los datos para comparación ---
tsel=10 # Tiempo seleccionado de hoja excel
#
# Concentraciones de los dos especies al tiempo 'tsel'
c1_excel = pd.read_excel('comparativa_02.xlsx', sheet_name='c_1').iloc[11:,tsel+5].to_numpy()     
c2_excel = pd.read_excel('comparativa_02.xlsx', sheet_name='c_2').iloc[11:,tsel+5].to_numpy()
#
# Solución analítica de los dos especies al tiempo 'tsel'
an_c1 = pd.read_excel('comparativa_02.xlsx', sheet_name='An_C1').iloc[11:,6].to_numpy()
an_c2 = pd.read_excel('comparativa_02.xlsx', sheet_name='An_C2').iloc[11:,6].to_numpy()  
#
# Calculamos el RMSE (https://en.wikipedia.org/wiki/Root_mean_square_deviation)
RMSE1 = np.linalg.norm(an_c1 - c_final[0]) / np.sqrt(len(c_final[0]))
RMSE2 = np.linalg.norm(an_c2 - c_final[1]) / np.sqrt(len(c_final[1]))
#
# Calculamos el MAPE (https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)
MAPE1 = np.mean(np.abs((an_c1 - c_final[0]) / an_c1)) * 100
MAPE2 = np.mean(np.abs((an_c2 - c_final[1]) / an_c2)) * 100

# --- Visualización de los resultados
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7,6))
#
# Gráfica de la especie c1
ax[0].plot(x, an_c1, "o--", c = "black", lw=2.0, alpha=0.5, label="Exacta", zorder=4)
ax[0].plot(x, c_final[0], "s", c = "black", mec = "black", mfc="None", lw = 1.0, label="DFC", zorder=5)
ax[0].plot(x, c1_excel, ".", c = "dimgray", mec = "black", label="Excel", zorder=5)
ax[0].plot(x, c1_excel, "-", c = "black", lw=1.0, alpha=0.75, zorder=5)
ax[0].set_title(rf"MAPE={MAPE1:5.2f} \%, RMSE = {RMSE1:5.2e}", fontsize=10, loc="right")
ax[0].set_ylabel("$c_1$")
ax[0].minorticks_on()
ax[0].grid(True, which='minor', color='gainsboro', lw = 0.05, zorder=2)
ax[0].grid(True,which='major', color='gray', lw = 0.5, zorder=3)
ax[0].legend()
ax[0].spines.right.set_visible(False)
ax[0].spines.top.set_visible(False)
#
# Gráfica de la especie c2
ax[1].plot(x, an_c2, "o--", c = "black", lw=2.0, alpha=0.5, label="Exacta", zorder=4)
ax[1].plot(x, c_final[1], "s", c = "black", mec = "black", mfc="None", lw = 1.0, label="DFC", zorder=5)
ax[1].plot(x, c2_excel, ".", c = "dimgray", mec = "black", label="Excel", zorder=5)
ax[1].plot(x, c2_excel, "-", c = "black", lw=1.0, alpha=0.75, zorder=5)
ax[1].set_title(rf"MAPE={MAPE2:5.2f} \%, RMSE = {RMSE2:5.2e}", fontsize=10, loc="right")
ax[1].set_xlabel("$x$ (m)")
ax[1].set_ylabel("$c_2$")
ax[1].minorticks_on()
ax[1].grid(True, which='minor', color='gainsboro', lw = 0.05, zorder=2)
ax[1].grid(True,which='major', color='gray', lw = 0.5, zorder=3)
ax[1].legend(loc="lower right")
ax[1].spines.right.set_visible(False)
ax[1].spines.top.set_visible(False)

plt.tight_layout()
plt.show()

