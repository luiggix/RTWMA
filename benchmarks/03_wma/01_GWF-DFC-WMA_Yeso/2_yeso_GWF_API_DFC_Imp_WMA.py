import os, sys, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flopy
from modflowapi import ModflowApi
import xmf6
import WMA_1D as WMA
import resultsFortran as rF 
linea = 50*chr(0x2015)

# --- DEFINICIÓN DE LAS RUTAS ---

# Ejecutable de Modflow
mf6_exe = r"C:\Users\luiggi\Documents\GitSites\mf6_tutorial\mf6\windows\mf6"
mf6_dll = r"C:\Users\luiggi\Documents\GitSites\mf6_tutorial\mf6\windows\libmf6.dll"

# Directorio de trabajo para WMA
working_dir = r"C:\Users\luiggi\Documents\GitSites\RTWMA\benchmarks\03_wma\RT-EXE"

# Archivo para almacenar las proporciones de mezcla 
wma_lambdas_filename = os.path.join(working_dir, 
                                    "input", "gypsum_eq", 
                                    "WMA_lambdas_gypsum_eq.dat")

# Creación del archivo "workingDirectory.txt" necesario para 
# la ejecución del programa "TR_1D_oper.exe"
with open(os.path.join(working_dir, "workingDirectory.txt"), "w") as f:
    f.write(working_dir)

# Ejecutable "TR_1D_oper.exe"
tr1d_exe = os.path.join(working_dir, "TR_1D_oper.exe")

# --- TRANSPORTE REACTIVO ---
def reactive_transport_wma(tr1d_exe):
    # Ejecución de "TR_1D_oper.exe"
    print("\n- Ejecutando TR_1D_oper.exe")
    result = subprocess.run([tr1d_exe], cwd = working_dir, 
                            capture_output = True, text = True)
    
    # Output from the program
    print("Standard Output:", result.stdout)
    print("Standard Error:", result.stderr)

# --- DATOS PARA LA SIMULACIÓN ---
nlay = 1
nrow = 1
ncol = 30
delr = 1.0
delc = 1.0
top = 1.0 
botm = 0.0

init_conc = np.ones((nlay,nrow,ncol),dtype=float) * 1.0
init_conc[:,0,11]=199.999967

phys = dict(
    specific_discharge = 0.2,  # Specific discharge ($cm s^{-1}$)
    hydraulic_conductivity = 1.0,  # Hydraulic conductivity ($cm s^{-1}$)
    source_concentration = 1.0,  # Source concentration (unitless)
    porosity = 0.5,  # Porosity of mobile domain (unitless)
    initial_concentration = init_conc, # Initial concentration (unitless)
    longitudinal_dispersivity = 0.2,
    retardation_factor = 1.0,
    decay_rate =  0.0,
    dispersion_coefficient = 0.2
)
    
print(linea)
print("Datos".center(50))
print(linea)
xmf6.nice_print(phys, "Parámetros físicos")

# --- SIMULACIÓN DE FLUJO ---

# Nombre de la simulación de flujo 
sim_name = "flow"

# --- Diccionario para la inicialización de la simulación ---
sim_flow = dict(
    # Parámetros de la simulación (flopy.mf6.MFSimulation)
    init = {
        'sim_name' : sim_name,
#        'exe_name' : mf6_exe,
        'sim_ws' : "output"
    },
    
    # Parámetros para el tiempo (flopy.mf6.ModflowTdis)
    tdis = {
        'units': "days",
        'nper' : 1,
        'perioddata': [(40.0, 1, 1.0)]
    },
    
    # Parámetros para la solución numérica (flopy.mf6.ModflowIms)
    ims = {}
)

# --- Diccionario para la simulación de flujo ---
gwf_d = dict(
    # Parámetros para el modelo de flujo (flopy.mf6.ModflowGwf)
    gwf = { 
        'modelname': sim_name,
        'save_flows': True
    },
    
    # Parámetros para la discretización espacial (flopy.mf6.ModflowGwfdis)
    dis = {
        'length_units' : "meters",
        'nlay': nlay, 
        'nrow': nrow, 
        'ncol': ncol,
        'delr': delr, 
        'delc': delc, 
        'top' : top, 
        'botm': botm 
    },
    
    # Parámetros para las condiciones iniciales (flopy.mf6.ModflowGwfic)
    ic = {
        'strt': 1.0
    },
    
    # Parámetros para las condiciones de frontera (flopy.mf6.ModflowGwfchd)
    chd = {
        'stress_period_data': [[(0, 0, ncol - 1), 1.0]],     
    },
    
    # Parámetros para las propiedades de flujo (flopy.mf6.ModflowGwfnpf)
    npf = {
        'save_specific_discharge': True,
        'save_saturation': True,
        'icelltype':  0,
        'k': phys["hydraulic_conductivity"] 
    },
    
    # Parámetros para las propiedades de los pozos (flopy.mf6.ModflowGwfwel)
    well = {
        'stress_period_data': [[(0, 0, 0), 
                                phys["specific_discharge"] * delc * delr * top, 
                                phys["source_concentration"],]],
        'pname': "WEL-1",
        'auxiliary' : ["CONCENTRATION"],
    #    'save_flows': True
    },
    
    # Parámetros para almacenar y mostrar la salida de la simulación (flopy.mf6.ModflowGwfoc)
    oc = {
        'budget_filerecord': f"{sim_name}.bud",
        'head_filerecord': f"{sim_name}.hds",
        'saverecord': [("HEAD", "ALL"), ("BUDGET", "ALL")],
    }
)

xmf6.nice_print(sim_flow["tdis"], "Discretización del tiempo para GWF")

# --- Inicialización de la simulación ---
o_sim_f = xmf6.common.init_sim(silent = True, **sim_flow)
o_gwf, packages = xmf6.gwf.set_packages(o_sim_f, silent = True, **gwf_d)

# --- Escritura de archivos ---

print(linea)
print("Iniciando GWF con la API".center(50))
print(linea)
print("\n- Escribiendo archivos de entrada para GWF")
o_sim_f.write_simulation(silent = True)

# --- EJECUCIÓN CON LA API ---
# Rutas a la biblioteca compartida y al archivo de configuración
mf6_config_file = os.path.join(o_sim_f.sim_path, 'mfsim.nam')
print("\n- Shared library:", mf6_dll)
print("\n- Config file:", mf6_config_file)

# Objeto para acceder a toda la funcionalidad de la API
mf6 = ModflowApi(mf6_dll, working_directory=o_sim_f.sim_path)

# Inicialización del modelo
mf6.initialize(mf6_config_file)

# Obtenemos el tiempo actual y el tiempo final de la simulación
current_time = mf6.get_current_time()
end_time = mf6.get_end_time()

# Máximo número de iteraciones para el algorimo de solución numérica
max_iter = mf6.get_value(mf6.get_var_address("MXITER", "SLN_1"))

# --- Recuperamos datos de la malla ---
grid = o_gwf.modelgrid
x, y, z = grid.xyzcellcenters

print(f"\n ---> Tiempo actual  = {current_time}")
print(f" ---> Tiempo total   = {end_time}")
print(f" ---> Iter (solver) = {max_iter}")

tdis = {
        'units': "days",
        'nper' : 1,
        'perioddata': [(40.0, 40, 1.0)]
}

print()
print(linea)
print("Iniciando ciclo de la API para GWF".center(50))
print(linea)

while current_time < end_time:
    # Preparar el paso de tiempo 
    mf6.prepare_time_step(mf6.get_time_step())

    # Preparar la solución
    mf6.prepare_solve()

    print("\n ---> dt:", mf6.get_time_step(), ", t:", current_time, ", end_t:", end_time)

    # Ciclo del algoritmo numérico de solución
    kiter = 0
    while kiter < max_iter:
        print("\nkiter :", kiter)
        
        # Construir el sistema del problema y lo resolverlo
        has_converged = mf6.solve(1)
        
        if has_converged:
            print(f" ---> ¿Convergencia obtenida? : {has_converged}")
            break
        else:
            print(f" ---> ¿Convergencia obtenida? : {has_converged}")
            
        kiter += 1
        
    # Finalizamos la solución del paso de tiempo actual
    mf6.finalize_solve()

    # Finalizamos el paso de tiempo actual. 
    mf6.finalize_time_step()

    # Avanzamos en el tiempo
    current_time = mf6.get_current_time()

    if not has_converged:
        print("model did not converge")
        break

    # Recuperamos la solución obtenida por MF6 
    # (ojo: necesitamos hacer una copia del arreglo)
    head = np.copy(mf6.get_value_ptr(mf6.get_var_address("X", 'FLOW')))
    q = mf6.get_value(mf6.get_var_address("SPDIS", "FLOW", "NPF"))

    print("\n- Descarga específica", q[:,0].shape)
    print(q[:,0])
    print("\n- Carga hidráulica", head.shape)
    print(head)

    print()
    print("Iniciando DFC (dentro del ciclo)".center(50))
    print(linea)
    xmf6.nice_print(tdis, "Discretización del tiempo para DFC")

    # --- Cálculo de las lambdas ... ---
    print("\n- Discretización con DFC")
    print("\n- Calculando las 𝜆's\n")

    lambdas1D, mixingWaters = WMA.mixingRatios1D(phys, grid, tdis, head, q[:, 0])
    print(lambdas1D)

    # --- Almacenamiento de las proporciones de mezcla --- 
    print(f"\n - Escribiendo las 𝜆's")
    WMA.save_mixing(wma_lambdas_filename, lambdas1D, mixingWaters)
    print()

    # --- Cálculo del transporte reactivo
    print("Iniciando WMA (dentro del ciclo)".center(50))
    print(linea)
    reactive_transport_wma(tr1d_exe)
    print(linea)

    print(f"\n ---> t: {current_time}, end_t: {end_time}\n")

# Finalizamos la simulación completa
try:
    mf6.finalize()
    success = True
except:
    raise RuntimeError

print(linea)
print("Finalizando el ciclo de la API".center(50))
print(linea)

dummy = input("\n\n Teclear <ENTER> para continuar ...")

# Tiempo seleccionado de hoja excel
tsel=10

# Archivo con los resultados de TR_1D.exe
file_name='gypsum_eq.out'   
file_path = os.path.join(working_dir, file_name)
Res_contrsns_ini, Res_contrsns_fin = rF.import_results(file_path)

# Lectura de datos de la tabla de excel
WMA_I=pd.read_excel('comparativa_02.xlsx', sheet_name='c_2')     
data_set_C2 = np.transpose(WMA_I.iloc[11:,5:95].to_numpy())

# Calculamos el RMSE
RMSE = np.linalg.norm(data_set_C2[tsel][:]-np.array(Res_contrsns_fin[1][:])) / np.sqrt(len(Res_contrsns_fin))

# --- Gráfica de los resultados
plt.figure(figsize=(6,4))
plt.plot(x[0], data_set_C2[tsel][:], 
         lw=0.5, ls="--", c="k", zorder=5)
plt.scatter(x[0], Res_contrsns_fin[1][:], 
            marker = "s", c = "mediumblue", s=30, label="TR_1D", zorder=5)
plt.scatter(x[0], data_set_C2[tsel][:], 
            marker = "o", c = "violet", s=15,label="Excel", zorder=5)
plt.title(f"t = {tsel} días, RMSE = {RMSE:5.3e}")
plt.xlabel("$x$ [$m$]")
plt.ylabel("$c_2 [mgr/cm^{3}]$")
plt.legend()
plt.minorticks_on()
plt.grid(True,which='major', color='darkgray', lw = 0.5)
plt.grid(True, which='minor', color='silver', lw = 0.25)
plt.show()
