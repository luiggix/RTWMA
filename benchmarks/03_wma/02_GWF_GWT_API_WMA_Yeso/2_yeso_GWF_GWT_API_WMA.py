import os, sys, subprocess
import numpy as np
import matplotlib.pyplot as plt
import flopy
from modflowapi import ModflowApi
import xmf6
import WMA_1D as WMA # Calcula las proporciones de mezcla
import resultsFortran as rF 
linea = 50*chr(0x2015)

# --- DEFINICIÓN DE LAS RUTAS ---

# Ejecutable de Modflow
# WINDOWS
mf6_exe = r"C:\Users\luiggi\Documents\GitSites\mf6_tutorial\mf6\windows\mf6"
mf6_dll = r"C:\Users\luiggi\Documents\GitSites\mf6_tutorial\mf6\windows\libmf6.dll"

# MACOSARM
#mf6_exe = r"../../../bin/macosarm/mf6"
#mf6_dll = r"../../../bin/macosarm/libmf6.dylib"

# Directorio de trabajo para WMA
working_dir = r"C:\Users\luiggi\Documents\GitSites\RTWMA\benchmarks\03_wma\RT-EXE"
#working_dir = r"../RT-EXE"

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

# Datos de la discretización espacial 
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

xmf6.nice_print(phys, "Parámetros físicos")


# --- SIMULACIÓN DE FLUJO ---

# Nombre de la simulación de flujo 
sim_name = "flow"

# --- Diccionario para la inicialización de la simulación ---
sim_flow = dict(
    # Parámetros de la simulación (flopy.mf6.MFSimulation)
    init = {
        'sim_name' : sim_name,
        'exe_name' : mf6_exe,
        'sim_ws' : "output_1"
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

# --- Inicialización de la simulación ---
o_sim = xmf6.common.init_sim(silent = True, **sim_flow)
o_gwf, packages = xmf6.gwf.set_packages(o_sim, silent = True, **gwf_d)

# --- Escritura de archivos ---
print(linea)
print("- Escribiendo archivos de entrada para GWF")
o_sim.write_simulation(silent = True)

# --- Ejecución de la simulación ---
print("- Ejecutando GWF")
o_sim.run_simulation()

# --- Recuperamos los resultados de flujo de la simulación ---
head = xmf6.gwf.get_head(o_gwf)
qx, qy, qz, n_q = xmf6.gwf.get_specific_discharge(o_gwf, text="DATA-SPDIS")

print(linea)
print("Descarga específica", qx.shape)
print(qx)
print("Carga hidráulica", head.shape)
print(head)
print(linea)

# --- SIMULACIÓN DE TRANSPORTE ---

trans_sim_name = "transport"

# Parámetros de la simulación (flopy.mf6.MFSimulation)
init_t = {
    'sim_name' : trans_sim_name,
    'exe_name' : sim_flow['init']['exe_name'],
    'sim_ws' : sim_flow['init']['sim_ws'],
}

# Parámetros para el tiempo (flopy.mf6.ModflowTdis)
tdis_t = {
    'units': "days",
    'nper' : 1,
    'perioddata': [(40.0, 40, 1.0)] #PERLEN, NSTP, TSMULT
}

# Parámetros para la solución numérica (flopy.mf6.ModflowIms)
ims_t = {
    'linear_acceleration': "bicgstab"
}

# Parámetros para el modelo de transporte (flopy.mf6.ModflowGwt)
gwt = { 
    'modelname': init_t["sim_name"],
    'save_flows': True
}

# Parámetros para la discretización espacial (flopy.mf6.ModflowGwtdis)
# El mismo dis de flow

# Parámetros para las condiciones iniciales (flopy.mf6.ModflowGwtic)
ic_t = {
    'strt': phys['initial_concentration']
}

# Parámetros para MST (flopy.mf6.ModflowGwtmst)
mst ={
        "porosity": phys["porosity"]
}

# Parámetros para ADV (flopy.mf6.ModflowGwtadv)
adv = {
    "scheme" : "TVD" #"UPSTREAM"
}

# Parámetros para DSP (flopy.mf6.ModflowGwtdsp)
dsp = {
    "xt3d_off" : True,
    "alh" : phys["longitudinal_dispersivity"],
    "ath1" : phys["longitudinal_dispersivity"],
}

# Parámetros para FMI (flopy.mf6.ModflowGwtfmi)
fmi = {
    "packagedata" : [("GWFHEAD", f"{sim_flow['init']['sim_name']}.hds", None),
                     ("GWFBUDGET", f"{sim_flow['init']['sim_name']}.bud", None),
                    ]   
#    "packagedata" : [("GWFHEAD", f"{init_f['sim_name']}.hds", None),
#                     ("GWFBUDGET", f"{init_f['sim_name']}.bud", None),
#                    ]
}

# Parámetros para SSM (flopy.mf6.ModflowGwtssm)
ssm = {
    "sources" : [["WEL-1", "AUX", "CONCENTRATION"]]
}

# Parámetros para OBS (flopy.mf6.ModflowGwtobs)
obs = {
    "digits" : 10, 
    "print_input" : True, 
    "continuous" : {
        "transport_obs.csv": [
            ("X005", "CONCENTRATION", (0, 0, 5)),
            ("X011", "CONCENTRATION", (0, 0, 11)),
            ("X029", "CONCENTRATION", (0, 0, 29)),
        ],
    }
}

# Parámetros para almacenar y mostrar la salida de la simulación (flopy.mf6.ModflowGwtoc)
oc_t = {
    'budget_filerecord': f"{init_t['sim_name']}.cbc",
    'concentration_filerecord': f"{init_t['sim_name']}.ucn",
    'saverecord' : [("CONCENTRATION", "ALL"), ("BUDGET", "LAST")],
    'printrecord' : [("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
}

# --- Inicialización de la simulación ---
o_sim_t = xmf6.common.init_sim(silent = True, init = init_t, 
                               tdis = tdis_t, ims = ims_t)
o_gwt, packagest = xmf6.gwt.set_packages(o_sim_t, silent = True,
                                        gwt = gwt, 
                                        dis = gwf_d['dis'], ic = ic_t, 
                                        mst = mst, adv = adv, dsp = dsp, fmi = fmi, ssm = ssm, 
                                        oc = oc_t)

o_obs = xmf6.common.set_obs(o_gwt, obs, silent = True)

# --- Escritura de archivos ---
o_sim_t.write_simulation(silent = True)

# --- EJECUCIÓN CON LA API ---

# Rutas a la biblioteca compartida y al archivo de configuración
mf6_config_file = os.path.join(o_sim_t.sim_path, 'mfsim.nam')
print("Shared library:", mf6_dll)
print("Config file:", mf6_config_file)

# Objeto para acceder a toda la funcionalidad de la API
mf6 = ModflowApi(mf6_dll, working_directory=o_sim_t.sim_path)

# Inicialización del modelo
mf6.initialize(mf6_config_file)

# Obtenemos el tiempo actual y el tiempo final de la simulación
current_time = mf6.get_current_time()
end_time = mf6.get_end_time()

# Máximo número de iteraciones para el algorimo de solución numérica
max_iter = mf6.get_value(mf6.get_var_address("MXITER", "SLN_1"))

print(linea)
print("Iniciando la simulación")
print(f"Tiempo actual: {current_time}")
print(f"Tiempo final: {end_time}")
print(linea)

# --- Recuperamos datos de la malla ---
grid = o_gwf.modelgrid
x, y, z = grid.xyzcellcenters

# Define los tiempos de interés
times = [10.0, 20.0, 30.0, 40.0]
U = []

# Ciclo sobre tiempo
while current_time < end_time:
    # Obtenemos el paso de tiempo
    dt = mf6.get_time_step()
    print("dt:", dt, ", t:", current_time, ", end_t:", end_time, ", max_iter:", max_iter)

    # Preparar el objeto de la API para obtener la solución y
    # con el paso de tiempo
    mf6.prepare_time_step(dt)
    mf6.prepare_solve()
    
    # Ciclo del algoritmo numérico de solución
    kiter = 0
    while kiter < max_iter:
#        print("\nkiter :", kiter)
        
        # Construye el sistema del problema y lo resuelve
        has_converged = mf6.solve(1)
        
        if has_converged:
            print(f" ---> Convergencia obtenida en iter = {kiter} ")
            break
#        else:
#            print(f" ---> ¿Convergencia obtenida? : {has_converged}")
            
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

    # Obtenemos la solución obtenida por MF6 
    # (ojo: necesitamos hacer una copia del arreglo)
    if current_time in times:
        U.append(np.copy(mf6.get_value_ptr(mf6.get_var_address("X", 'TRANSPORT'))))
#    q = mf6.get_value(mf6.get_var_address("SPDIS", "FLOW", "NPF"))

    A, _, _, _ = xmf6.api.build_mat(mf6)
    RHS = mf6.get_value(mf6.get_var_address("RHS", 'SLN_1'))

#    print(A)
    # --- Cálculo y almacenamiento de las lambdas ... ---
#    lambdas1D, mixingWaters = WMA.mixingRatios1D(phys, grid, sim_flow["tdis"], head, q[:, 0])
#    WMA.save_mixing(wma_lambdas_filename, lambdas1D, mixingWaters)

    # --- Cálculo del transporte reactivo
#    reactive_transport_wma(tr1d_exe)

# Probar con 'SPDIS','SLN_1' o  'FLOW'
component = ['TRANSPORT/ADV','SLN_1']
for c in component:
    for label in mf6.get_input_var_names():
        if c in label:
            print(label, end = "\n")

# Finalizamos la simulación completa
try:
    mf6.finalize()
    success = True
except:
    raise RuntimeError

# Construcción de las matrices para generar las lambdas
D = np.identity(ncol) * phys["porosity"]
dt = tdis_t['perioddata'][0][0] / tdis_t['perioddata'][0][1]
invA = np.linalg.inv(D/dt-A)
LAMBDAS_IMP = invA.dot(D/dt)
QinvA = invA.dot(RHS)
lambdas1D = LAMBDAS_IMP

mixingWaters = np.zeros((ncol,ncol+2))
for j in range(0,ncol):
    for i in range (0,ncol+2):
        mixingWaters[j][i] = i+1
    
aux = np.linspace(3,ncol+2,ncol)
aux = aux[:, np.newaxis]
mixingWaters = np.concatenate((aux,  mixingWaters), axis=1)
mixingWaters = mixingWaters.astype(np.int32)

WMA.save_mixing(wma_lambdas_filename, lambdas1D, mixingWaters)

print(linea)
print("MATRIZ A")
print(A)
print(linea)
print("MATRIZ D")
print(D)
print(linea)
print("LAMBDAS_U")
print(LAMBDAS_IMP)
print(linea)
print("LAMBDAS_INF")
print(QinvA)
print(linea)
print("Finalizando la simulación")
print(linea)

# --- Cálculo del transporte reactivo
reactive_transport_wma(tr1d_exe)

# --- Recuperamos los resultados de flujo de la simulación ---
# TODO: checar si se puede recuperar la información del objeto o_gwf directamente.
head = xmf6.gwf.get_head(o_gwf)
qx, qy, qz, n_q = xmf6.gwf.get_specific_discharge(o_gwf, text="DATA-SPDIS")

# --- Recuperamos los resultados de la concentración de la simulación ---
#cobj = o_gwt.output.concentration()

# --- Recuperamos las coordenadas del dominio
#x, y, z = o_gwf.modelgrid.xyzcellcenters
#row_length = o_gwf.modelgrid.extent[1]

# --- Definición de la figura ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize =(6,5), 
                                    height_ratios=[0.1,1.0,1.5])

# --- Gráfica 1. Malla ---
pmv = flopy.plot.PlotMapView(o_gwf, ax=ax1)
pmv.plot_grid(colors='dimgray', lw=0.5)
ax1.set_yticks(ticks=[0, 1.0])#, fontsize=8)
ax1.set_title("Mesh")

# --- Gráfica 2. Carga hidráulica---
ax2.plot(x[0], head[0, 0], marker="o", lw =1.0, #mec="blue", mfc="none", 
         markersize="4", alpha = 0.75, label = 'Head')
ax2.set_xlim(0, 30)
ax2.set_ylabel("Head")
ax2.grid()

# --- Gráfica 3. Concentración ---

# Crear la gráfica en formato de líneas para ambos tiempos
for t, u in zip(times, U):
    ax3.plot(x[0], u, ls ="-", lw = 1.0, #c = "C1", 
             marker ="o", markersize="4", alpha = 0.75,
             label=f"t = {t} days", zorder=2)


# Obtención de los datos de concentración simulados para cada tiempo
#conc_t1 = cobj.get_data(totim=t1).flatten()
#conc_t2 = cobj.get_data(totim=t2).flatten()
#conc_t3 = cobj.get_data(totim=t3).flatten()
#conc_t4 = cobj.get_data(totim=t4).flatten()

# Crear la gráfica en formato de líneas para ambos tiempos
#ax3.plot(x[0], conc_t1, ls ="-", lw = 1.0, c = "C1", 
#         marker ="o", markersize="4", alpha = 0.75,
#         label=f"t = {t1} days", zorder=2)
#ax3.plot(x[0], conc_t2, ls ="-", lw = 1.0, c = "C2", 
#         marker="s", markersize="4", alpha = 0.75,
#         label=f"t = {t2} days", zorder=2)
#ax3.plot(x[0], conc_t3, ls ="-", lw = 1.0, c = "C3", 
#         marker="v", markersize="4", alpha = 0.75,
#         label=f"t = {t3} days", zorder=2)
#ax3.plot(x[0], conc_t4, ls ="-", lw = 1.0, c = "C4", 
#         marker="^", markersize="4", alpha = 0.75,
#         label=f"t = {t4} days", zorder=2)

ax3.set_xlabel("Distance (m)")
ax3.set_ylabel("Normalized Concentration")
ax3.set_xlim(0, 30)
ax3.grid()

# Ajusta el límite superior del eje Y basado en el valor máximo de ambos conjuntos de datos
max_y = max(U[0].max(), U[1].max(), U[2].max(), U[3].max())
ax3.set_ylim(0, max_y * 1.1)
ax3.legend(fontsize=7)

plt.tight_layout()
plt.show()

