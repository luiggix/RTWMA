## ---> Bibliotecas que se usarán en este ejemplo
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import flopy
from src_gypsum import gwf, gwt
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
    flow_name = "flow",
    flow_ws = "io_mf6/U/gwf",
    tran_name = "transport",
    tran_ws = "io_mf6/U/gwt"
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

## ---> Diccionario con información de la discretización del tiempo para el flujo 
tdis_f = {
    'units': "days",
    'nper' : 1,
    'perioddata': [(40.0, 1, 1.0)] #PERLEN, NSTP, TSMULT
}
xmf6.nice_print("Time discretization (flow)", tdis_f)

## ---> Diccionario con información de la discretización del tiempo para el transporte 
tdis_t = {
    'units': "days",
    'nper' : 1,
    'perioddata': [(40.0, 40, 1.0)] #PERLEN, NSTP, TSMULT
}
xmf6.nice_print("Time discretization (transport)", tdis_t)

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

## ---> Construcción del modelo de flujo y escritura de los archivos de entrada.
xmf6.nice_print("Function gwf.build(...) :  flow model creation")
o_sim, o_gwf = gwf.build(paths, tdis_f, phys, dis, silent = False) 

## ---> Ejecución de la simulación de flujo.
xmf6.nice_print("Flow simulation execution")
o_sim.run_simulation(silent = False)

## ---> Construcción del modelo de transporte y escritura de los archivos de entrada.
xmf6.nice_print("Function gwt.build(...) :  transport model creation")
o_sim_t, o_gwt = gwt.build(paths, tdis_t, phys, dis, silent = False) 

## ---> Ejecución de la simulación de flujo.
xmf6.nice_print("Transport simulation execution")
o_sim_t.run_simulation(silent = False)
