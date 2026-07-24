import os
import numpy as np
import xmf6
from gypsum import gwf, gwt, api, wma, plot
linea = 50*chr(0x2015)
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
    tr1d_ofile='gypsum_eq.out',           # Resultados del transporte reactivo 
    tr1d_ofile_dfc = 'gypsum_eq_dfc.out', # Resultados del transporte reactivo DFC
    excel_data = 'comparativa_02.xlsx',   # Datos para comparación
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
    'length_units' : "meters",
    'nlay': 1, 
    'nrow': 1, 
    'ncol': 30,
    'delr': 1.0, 
    'delc': 1.0, 
    'top' : 1.0, 
    'botm': 0.0 
}
xmf6.nice_print(dis, "Spatial discretization")
#
# Discretización del tiempo para el flujo
tdis_f = {
    'units': "days",
    'nper' : 1,
    'perioddata': [(40.0, 1, 1.0)]
}
xmf6.nice_print(dis, "Time discretization (flow)")
#
# Discretización del tiempo para el transporte
tdis_t = {
        'units': "days",
        'nper' : 1,
        'perioddata': [(40.0, 40, 1.0)] #PERLEN, NSTP, TSMULT
    }
xmf6.nice_print(tdis_t, "Time discretization (transport)")

# <> -----------------------------------------------------------------------
# Arreglo para la condición inicial de c1
c1_ini = np.full((nlay,nrow,ncol), 1.0000329) # En todo el dominio
c1_ini[0, 0, 11] = 200.0  # Pulso en x_L

# Arreglo para la condición inicial de c2
c2_ini = np.full((nlay,nrow,ncol), 3.294e-5) # En todo el dominio
c2_ini[0, 0, 11] = 1.647e-7  # Pulso en x_L

# Arreglo para la condición inicial de U
U_ini = np.full((nlay,nrow,ncol), 1.0) # c1_ini - c2_ini # En todo el dominio
U_ini[0, 0, 11] = c1_ini[0, 0, 11] - c2_ini[0, 0, 11]  # Pulso en x_L
print("Array info: U")
xmf6.info_array(U_ini)
U_s = c1_ini[0, 0, 0] - c2_ini[0, 0, 0]
# <> -----------------------------------------------------------------------

phys = dict(
    initial_head = 1.0,
    bc_head_t1 = [("CHD-1" , [(0, 0, dis['ncol'] - 1), 1.0])],
    hydraulic_conductivity = 1.0, 
    specific_discharge = 0.2, 
    source_concentration = 1.0, # U_s
    porosity = 0.5,
    initial_concentration = U_ini,
    bc_conc_t1 = [("CNC-1", [(0, 0, 0), 1.0])], #U_s])],
    longitudinal_dispersivity = 0.2,
    dispersion_coefficient = 1.0, # 0.2
    decay_rate =  0.0,    
)
# Agregamos la información del pozo
q = phys["specific_discharge"] * dis['delc'] * dis['delr'] * dis['top']
phys["well"] = [("WEL-1", "AUX", "CONCENTRATION"), ((0, 0, 0), q, phys["source_concentration"])]

xmf6.nice_print(phys, "Parámetros físicos")

#
# Concentración inicial
#init_conc = np.ones((dis['nlay'],dis['nrow'],dis['ncol']),dtype=float) * 1.0
#init_conc[:,0,11]=199.999967
#
# Parámetros físicos.
#phys = dict(
#    specific_discharge = 0.2,  # Specific discharge ($cm s^{-1}$)
#    hydraulic_conductivity = 1.0,  # Hydraulic conductivity ($cm s^{-1}$)
#    source_concentration = 1.0,  # Source concentration (unitless)
#    porosity = 0.5,  # Porosity of mobile domain (unitless)
#    initial_concentration = init_conc, # Initial concentration (unitless)
#    longitudinal_dispersivity = 0.2,
#    retardation_factor = 1.0,
#    decay_rate =  0.0,
#    dispersion_coefficient = 0.2
#)
#xmf6.nice_print(phys, "Parámetros físicos")
#
#
__dummy = input("\nTeclea [ENTER] para continuar\n") 
#
# - SIMULACIÓN DE FLUJO CON GWF ---
#
# --- Construcción de la simulación de flujo
#o_sim_f 
o_sim_f, o_gwf = gwf.build(paths, tdis_f, phys, dis, silent = False) #gwf.build(phys, dis, paths)
#(paths, tdis, phys, dis, silent = False):
#
# --- Ejecución de la simulación ---
print("- Ejecutando GWF")
o_sim_f.run_simulation(silent=False)
#
__dummy = input("\nTeclea [ENTER] para continuar\n")
#
# - SIMULACIÓN DE TRANSPORTE CON GWT + API ---
#
#o_sim_t = gwt.build(phys, dis, paths)
o_sim_t, o_gwt = gwt.build(paths, tdis_t, phys, dis, silent = False)
#
# --- Ejecución de la API ---
A, RHS, U = api.run(o_sim_t, paths)
#
__dummy = input("\nTeclea [ENTER] para continuar\n") 
#
# - CÁLCULO DE LAS RAZONES DE MEZCLA (WMA)
#
# --- Usando DFC
DFC = input("Calcular con DFC?")

if DFC == "S":
    print(linea)
    print("- Calculando las proporciones de mezcla con DFC")
    o_gwf = o_sim_f.get_model(flow_name)
    grid = o_gwf.modelgrid
    head = xmf6.gwf.get_head(o_gwf)
    qx, qy, qz, n_q = xmf6.gwf.get_specific_discharge(o_gwf, text="DATA-SPDIS")
    print(head[0,0])
    print(qx[0,0])
    mixingRatios, mixingWaters = wma.build_dfc(phys, grid, tdis, head[0,0], qx[0,0])
    wma.save_mixing(paths, mixingRatios, mixingWaters)
    wma.reactive_transport_wma(paths)
    file_name = os.path.join(paths["wma_working_dir"], tr1d_ofile)
    file_dfc = os.path.join(paths["wma_working_dir"], tr1d_ofile_dfc)
    print("- Renombrando archivo:")
    print(f"  Archivo original: {file_name}")
    print(f"  Archivo nuevo   : {file_dfc}")
    os.rename(file_name, file_dfc)
#
__dummy = input("\nTeclea [ENTER] para continuar\n") 
#
print(linea)
#
# --- Usando GWT + API
wma.build(dis, tdis_t, phys, A, RHS, U, paths, silent=True)
#
__dummy = input("\nTeclea [ENTER] para continuar\n") 
#
# - CÁLCULO DE TRANSPORTE REACTIVO
#
wma.reactive_transport_wma(paths)
#
__dummy = input("\nTeclea [ENTER] para continuar\n") 
#
# - ANÁLISIS DE RESULTADOS
#
plot.show(o_sim_f, o_sim_t, paths)

print(linea)
print("- FIN DE LA SIMULACIÓN -")
print(linea)
