## ---> Bibliotecas que se usarán en este ejemplo
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import flopy
from src_gypsum import gwf_exch as gwf
from src_gypsum import gwt_exch as gwt
from src_gypsum import vis
import xmf6

## ---> Variables de entorno
with open('../env.json', 'r', encoding='utf-8') as file:
    env = json.load(file)
env["ROOT_DIR"] = os.getcwd() # Agregamos el dir raíz

xmf6.nice_print("Environment variables", env)

## ---> Diccionario con información de rutas y archivos
paths = dict(
    # Ejecutable de MODFLOW 6
    mf6_exe = env["MF6EXE"],
    #
    # Nombre de los modelos y espacios de trabajo
    sim_name = "flow_trans",
    sim_ws = "io_mf6/Uexch",
    flow_name = "flow",
    tran_name = "transport",
)
xmf6.nice_print("Paths, files and more ...", paths)

## ---> Discretización espacial
L = 30 # Longitud del dominio
nlay = 1
nrow = 1
ncol = 30
delr = L / ncol # 1.0
delc = 1.0
top  = 1.0
botm = 0.0

## ---> Diccionario con información de la discretización espacial
dis = {
    'length_units' : "meters",
    'nlay': nlay, 
    'nrow': nrow, 
    'ncol': ncol,
    'delr': delr, 
    'delc': delc, 
    'top' : top, 
    'botm': botm 
}
xmf6.nice_print("Spatial discretization", dis)

## ---> Diccionario con información de la discretización del tiempo para la simulación 
tdis = {
    'units': "days",
    'nper' : 1,
    'perioddata': [(40.0, 40, 1.0)] #PERLEN, NSTP, TSMULT
}
xmf6.nice_print("Time discretization (flow)", tdis)


## ---> Condición inicial para el transporte
x_pulse = 10.5
i_pulse = int(x_pulse / delr)
print("x_p = ", i_pulse)

## ---> Arreglo para la condición inicial de c1
c1_ini = np.full((nlay,nrow,ncol), 1.0000329371) # En todo el dominio
c1_ini[0, 0, i_pulse] = 200.0000001647  # Pulso en x_11

## ---> Arreglo para la condición inicial de c2
c2_ini = np.full((nlay,nrow,ncol), 3.29371e-5) # En todo el dominio
c2_ini[0, 0, i_pulse] = 1.647e-7  # Pulso en x_11

## ---> Arreglo para la condición inicial de U
U_ini = c1_ini - c2_ini # En todo el dominio
U_ini[0, 0, i_pulse] = c1_ini[0, 0, i_pulse] - c2_ini[0, 0, i_pulse]  # Pulso en x_11
xmf6.nice_print("Array info: U")
xmf6.info_array(U_ini)

U_s = c1_ini[0, 0, 0] - c2_ini[0, 0, 0]

## ---> Diccionario para los parámetros físicos
phys = dict(
    initial_head = 1.0,
    bc_head_t1 = {
        "pname": "CHD-1", 
        "stress_period_data": [[(0, 0, dis['ncol'] - 1), 1.0],],
    },
    hydraulic_conductivity = 1.0, 
    specific_discharge = 0.2, 
    source_concentration = U_s,
    porosity = 0.5,
    initial_concentration = U_ini,
    longitudinal_dispersivity = 0.5, # 0.2 o 0.5?
)

## ---> Agregamos la información del pozo
q = phys["specific_discharge"] * dis['delc'] * dis['delr'] * (dis['top'] - dis['botm'])
phys["well"] = {
    "pname": "WEL-1",
    "stress_period_data": [[(0, 0, 0), q, phys["source_concentration"]],],
    "auxiliary": ["CONCENTRATION"]
}

phys["SSM"] = [(phys["well"]["pname"], 
               "AUX", 
               phys["well"]["auxiliary"][0]),]

xmf6.nice_print("Physical parameters", phys)

# --- COMPONENTES ---

## ---> Simulación y discretización temporal. 
## Los objetos 'o_sim' y 'o_tdis' son compartidos por ambos modelos.

## ---> Creación del objeto de la simulación de flujo
o_sim = flopy.mf6.MFSimulation(
    sim_name = paths["sim_name"], 
    sim_ws = paths["sim_ws"], 
    exe_name = paths["mf6_exe"], 
    version="mf6"
)

## ---> Creación del objeto de la discretización del tiempo
o_tdis = flopy.mf6.ModflowTdis(
    o_sim,
    time_units = tdis["units"],
    nper = tdis["nper"],
    perioddata = tdis["perioddata"],
)

## ---> Construcción del modelo de flujo
xmf6.nice_print("Function gwf.build(...) :  flow model creation")
o_gwf = gwf.build(paths, o_sim, phys, dis, silent = False) 

## ---> Construcción del modelo de transporte
xmf6.nice_print("Function gwt.build(...) :  transport model creation")
o_gwt = gwt.build(paths, o_sim, phys, dis, silent = False) 

## ---> Agregamos el objeto del intercambio entre los modelos.
xmf6.nice_print("GWF-GWT exchange creation")
o_gwfgwt = flopy.mf6.ModflowGwfgwt(
    o_sim, 
    exgtype="GWF6-GWT6", 
    exgmnamea=o_gwf.name, 
    exgmnameb=o_gwt.name,
    filename=f"{paths["sim_name"]}.gwfgwt",
)

## ---> Escritura de los archivos de entrada para la simulación.
xmf6.nice_print("Writing input files for the simulation")
o_sim.write_simulation(silent = False)

## ---> Ejecución de la simulación.
xmf6.nice_print("Executing the simulation")
o_sim.run_simulation(silent = False)
