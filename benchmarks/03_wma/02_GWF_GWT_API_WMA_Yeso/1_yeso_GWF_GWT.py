import os, sys, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flopy
import xmf6
line_size = 50

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
    initial_concentration = init_conc,  # Initial concentration (unitless)
    longitudinal_dispersivity  = 0.2,
    retardation_factor  = 1.0,
    decay_rate  = 0.0,
    dispersion_coefficient = 0.2
)
xmf6.nice_print(phys, "Parámetros físicos")

# --- SIMULACIÓN DE FLUJO ---

# Nombre de la simulación de flujo 
sim_name = "flow"

print(line_size * chr(0x2015))
print("Iniciando simulación de flujo")

# --- Diccionario para la inicialización de la simulación ---
sim_flow = dict(
    # Parámetros de la simulación (flopy.mf6.MFSimulation)
    init = {
        'sim_name' : sim_name,
        'exe_name' : r"C:\Users\luiggi\Documents\GitSites\mf6_tutorial\mf6\windows\mf6",
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
print(line_size * chr(0x2015))
print("- Escribiendo archivos de entrada para GWF")
o_sim.write_simulation(silent = True)

# --- Ejecución de la simulación ---
print("- Ejecutando GWF")
o_sim.run_simulation(silent = True)

# --- Recuperamos los resultados de flujo de la simulación ---
head = xmf6.gwf.get_head(o_gwf)
qx, qy, qz, n_q = xmf6.gwf.get_specific_discharge(o_gwf, text="DATA-SPDIS")

print(line_size * chr(0x2015))
print("Descarga específica", qx.shape)
print(qx)
print("Carga hidráulica", head.shape)
print(head)
print(line_size * chr(0x2015))

# --- SIMULACIÓN DE TRANSPORTE ---

trans_sim_name = "trans"

# ---------------
sim_trans = dict(
    # Parámetros de la simulación (flopy.mf6.MFSimulation)
    init = {
        'sim_name' : trans_sim_name,
        'exe_name' : sim_flow['init']['exe_name'],
        'sim_ws' : sim_flow['init']['sim_ws'],
    },
    
    # Parámetros para el tiempo (flopy.mf6.ModflowTdis)
    tdis = {
        'units': "days",
        'nper' : 1,
        'perioddata': [(40.0, 40, 1.0)]
    },
    
    # Parámetros para la solución numérica (flopy.mf6.ModflowIms)
    ims = {
        'linear_acceleration': "bicgstab"
    }
)

# ---------------
gwt_d = dict(
    # Parámetros para el modelo de transporte (flopy.mf6.ModflowGwt)
    gwt = { 
        'modelname': trans_sim_name,
        'save_flows': True
    },
    
    # Parámetros para la discretización espacial (flopy.mf6.ModflowGwtdis)
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
    
    # Parámetros para las condiciones iniciales (flopy.mf6.ModflowGwtic)
    ic = {
        'strt': phys["initial_concentration"],
    },
    
    # Parámetros para MST (flopy.mf6.ModflowGwtmst)
    mst = {
        "porosity": phys["porosity"]
    },
    
    # Parámetros para ADV (flopy.mf6.ModflowGwtadv)
    adv = {
        "scheme" : "TVD" #"UPSTREAM"
    },
    
    # Parámetros para DSP (flopy.mf6.ModflowGwtdsp)
    dsp = {
        "xt3d_off" : True,
        "alh" : phys["longitudinal_dispersivity"],
        "ath1" : phys["longitudinal_dispersivity"],
    },
    
    # Parámetros para FMI (flopy.mf6.ModflowGwtfmi)
    fmi = {
        "packagedata" : [("GWFHEAD", f"{sim_flow['init']['sim_name']}.hds", None),
                         ("GWFBUDGET", f"{sim_flow['init']['sim_name']}.bud", None),
                        ]
    },
    
    # Parámetros para SSM (flopy.mf6.ModflowGwtssm)
    ssm = {
        "sources" : [["WEL-1", "AUX", "CONCENTRATION"]]
    },
    
    # Parámetros para OBS (flopy.mf6.ModflowGwtobs)
    obs = {
        "digits" : 10, 
        "print_input" : True, 
        "continuous" : {
            "transporte.obs.csv": [
                ("X005", "CONCENTRATION", (0, 0, 0)),
                ("X011", "CONCENTRATION", (0, 0, 11)),
                ("X305", "CONCENTRATION", (0, 0, 29)),
            ],
        }
    },
    
    # Parámetros para almacenar y mostrar la salida de la simulación (flopy.mf6.ModflowGwtoc)
    oc = {
        'budget_filerecord': f"{trans_sim_name}.bud",
        'concentration_filerecord': f"{trans_sim_name}.ucn",
        'saverecord' : [("CONCENTRATION", "ALL"), ("BUDGET", "LAST")],
        'printrecord' : [("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
    },
)

# --- Inicialización de la simulación ---
o_sim_t = xmf6.common.init_sim(silent = True, **sim_trans)
o_gwt, packages = xmf6.gwt.set_packages(o_sim_t, silent = True, **gwt_d)

o_obs = xmf6.common.set_obs(o_gwt, gwt_d['obs'], silent = True)

print("-"*50)
# --- Escritura de archivos ---
print(line_size * chr(0x2015))
print("- Escribiendo archivos de entrada para GWT")
o_sim_t.write_simulation(silent = True)

# --- Ejecución de la simulación ---
print("- Ejecutando GWF")
o_sim_t.run_simulation(silent = True)

# --- Recuperamos los resultados de transporte de la simulación ---
cobj = o_gwt.output.concentration()
times = [10.0, 20.0, 30.0, 40.0]
U = [cobj.get_data(totim=times[0]).flatten(),
     cobj.get_data(totim=times[1]).flatten(),
     cobj.get_data(totim=times[2]).flatten(),
     cobj.get_data(totim=times[3]).flatten()]
     
print(line_size * chr(0x2015))
print("Concentración")
for i, u in zip(times, U):
    print(i, u.shape)
    print(u)
print(line_size * chr(0x2015))

# --- Recuperamos las coordenadas del dominio
grid = o_gwf.modelgrid
x, y, z = grid.xyzcellcenters
row_length = o_gwf.modelgrid.extent[1]

# --- Definición de la figura ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize =(6,5), height_ratios=[0.1,1,1.5])

# --- Gráfica 1. Malla ---
pmv = flopy.plot.PlotMapView(o_gwf, ax=ax1)
pmv.plot_grid(colors='dimgray', lw=0.5)
ax1.set_yticks(ticks=[0, 0.1])#, fontsize=8)
ax1.set_title("Mesh")

# --- Gráfica 2. Carga hidráulica---
ax2.plot(x[0], head[0, 0], marker="o", lw =1.0, #mec="blue", mfc="none", 
         markersize="4", alpha = 0.75, label = 'Head')
ax2.set_xlim(0, 30)
ax2.set_ylabel("Head")
ax2.grid()

# --- Gráfica 3. Concentración ---
for t, u in zip(times, U):
    ax3.plot(x[0], u, ls ="-", lw = 1.0, #c = "C1", 
             marker ="o", markersize="4", alpha = 0.75,
             label=f"t = {t} days", zorder=2)
ax3.set_xlim(0, 30)
ax3.set_ylabel("U")
ax3.grid()

plt.legend()
plt.tight_layout()
plt.show()

