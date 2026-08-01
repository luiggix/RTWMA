import os
from modflowapi import ModflowApi
import numpy as np
import xmf6

def run(o_sim, paths, IMS = "SLN_1"):
  
    # Rutas a la biblioteca compartida y al archivo de configuración
    mf6_config_file = os.path.join(o_sim.sim_path, 'mfsim.nam')
    print("-> Shared library:", paths["mf6_dll"])
    print("-> Config file:", mf6_config_file)

    # Objeto para acceder a toda la funcionalidad de la API
    mf6 = ModflowApi(paths["mf6_dll"], working_directory=o_sim.sim_path)
    
    # Inicialización del modelo
    mf6.initialize(mf6_config_file)

    # Obtenemos el tiempo actual y el tiempo final de la simulación
    current_time = mf6.get_current_time()
    end_time = mf6.get_end_time()

    # Máximo número de iteraciones para el algorimo de solución numérica
    max_iter = mf6.get_value(mf6.get_var_address("MXITER", IMS))

    # Obtenemos el paso de tiempo
    dt = mf6.get_time_step()
    print("dt:", dt, ", t:", current_time, ", end_t:", end_time, ", max_iter:", max_iter)
    
    # Preparar el objeto de la API para obtener la solución y con el paso de tiempo
    mf6.prepare_time_step(dt)

    sln_gwf = "SLN_1"
    sln_gwt = "SLN_2"

    print("-> Initializing the simulation with the API (one time step for each model)")
    for sol_id in range(1, mf6.get_subcomponent_count() + 1):
        mf6.prepare_solve(sol_id)
        mf6.solve(sol_id) 
        mf6.finalize_solve(sol_id)

    # Finalizamos el paso de tiempo actual. 
    mf6.finalize_time_step()
    
    U = np.copy(mf6.get_value_ptr(mf6.get_var_address("X", "TRANSPORT")))

    # Obtenemos la matriz del sistema (ojo: usamos la función build_mat()
    # para reconstruir la matriz en formato denso, pues está en CRS.
    A, _, _, _ = xmf6.api.build_mat(mf6, IMS)
    
    # Obtenemos el lado derecho del sistema
    RHS = mf6.get_value(mf6.get_var_address("RHS", IMS))
    

    # Finalizamos la simulación
    try:
        mf6.finalize()
        success = True
    except:
        raise RuntimeError

    return A, RHS, U
    