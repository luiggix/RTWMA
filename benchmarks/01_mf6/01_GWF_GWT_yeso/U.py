import os
import numpy as np
import matplotlib.pyplot as plt
import flopy
import gwf_gypsum as gwf
import gwt_gypsum as gwt
#import vis
import xmf6

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
xmf6.nice_print("Spatial discretization", dis)

# Discretización del tiempo para el flujo 
tdis_f = {
    'units': "days",
    'nper' : 1,
    'perioddata': [(40.0, 1, 1.0)] #PERLEN, NSTP, TSMULT
}
xmf6.nice_print("Time discretization (flow)", tdis_f)

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
U_ini = c1_ini - c2_ini # En todo el dominio
U_ini[0, 0, 11] = c1_ini[0, 0, 11] - c2_ini[0, 0, 11]  # Pulso en x_L
xmf6.nice_print("Array info: U")
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
    dispersion_coefficient = 1.0 
)
# Agregamos la información del pozo
q = phys["specific_discharge"] * dis['delc'] * dis['delr'] * dis['top']
phys["well"] = [("WEL-1", "AUX", "CONCENTRATION"), ((0, 0, 0), q, phys["source_concentration"])]

xmf6.nice_print("Physical parameters", phys)

paths = dict(
    # Ejecutable de MODFLOW 6
    mf6_exe = r"C:\Users\luiggi\Documents\GitSites\mf6_tutorial\mf6\windows\mf6",
    #
    # Nombre de los modelos y espacios de trabajo
    flow_name = "flow",
    flow_ws = "output_U/gwf",
    tran_name = "transport",
    tran_ws = "output_U/gwt"
)
xmf6.nice_print("Paths, files and more ...", paths)

xmf6.nice_print("Function gwf.build(...) :  flow model creation")
# Escritura de los archivos de entrada para la simulación.
o_sim, o_gwf = gwf.build(paths, tdis_f, phys, dis, silent = False) 

xmf6.nice_print("Flow simulation execution")
# Ejecución de la simulación de flujo.
o_sim.run_simulation(silent = False)

xmf6.nice_print("Function gwt.build(...) :  transport model creation")
# Escritura de los archivos de entrada para la simulación.
o_sim_t, o_gwt = gwt.build(paths, tdis_t, phys, dis, silent = False) 

xmf6.nice_print("Transport simulation execution")
# Ejecución de la simulación de flujo.
o_sim_t.run_simulation(silent = False)
