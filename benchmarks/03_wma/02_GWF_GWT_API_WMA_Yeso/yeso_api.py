import os
from modflowapi import ModflowApi
import numpy as np
import xmf6
linea = 50*chr(0x2015)

# --- EJECUCIÓN CON LA API ---

def build(o_sim, mf6_dll, VARNAME = "TRANSPORT"):

    print(linea)
    print("- Incializando la API")
    
    # Rutas a la biblioteca compartida y al archivo de configuración
    mf6_config_file = os.path.join(o_sim.sim_path, 'mfsim.nam')
    print("Shared library:", mf6_dll)
    print("Config file:", mf6_config_file)

    # Objeto para acceder a toda la funcionalidad de la API
    mf6 = ModflowApi(mf6_dll, working_directory=o_sim.sim_path)
    
    # Inicialización del modelo
    mf6.initialize(mf6_config_file)
    
    # Obtenemos el tiempo actual y el tiempo final de la simulación
    current_time = mf6.get_current_time()
    end_time = mf6.get_end_time()
    
    # Máximo número de iteraciones para el algorimo de solución numérica
    max_iter = mf6.get_value(mf6.get_var_address("MXITER", "SLN_1"))
    
    print(linea)
    print("- Iniciando la simulación con la API")
    print(f"  Tiempo actual: {current_time}")
    print(f"  Tiempo final: {end_time}")
    print(linea)
    
    # Obtenemos el paso de tiempo
    dt = mf6.get_time_step()
    print("dt:", dt, ", t:", current_time, ", end_t:", end_time, ", max_iter:", max_iter)
    
    # Preparar el objeto de la API para obtener la solución y con el paso de tiempo
    mf6.prepare_time_step(dt)
    mf6.prepare_solve()
        
    # Ciclo del algoritmo numérico de solución
    kiter = 0
    while kiter < max_iter:        
        # Construye el sistema del problema y lo resuelve
        has_converged = mf6.solve(1)
            
        if has_converged:
            print(f" ---> Convergencia obtenida en iter = {kiter} ")
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
        print("  Model did not converge!")
    
    # Obtenemos la solución obtenida por MF6 
    # (ojo: necesitamos hacer una copia del arreglo)
    U = np.copy(mf6.get_value_ptr(mf6.get_var_address("X", VARNAME)))

    # Obtenemos la matriz del sistema (ojo: usamos la función build_mat()
    # para reconstruir la matriz en formato denso, pues está en CRS.
    A, _, _, _ = xmf6.api.build_mat(mf6)
    
    # Obtenemos el lado derecho del sistema
    RHS = mf6.get_value(mf6.get_var_address("RHS", 'SLN_1'))
    
    # Finalizamos la simulación
    try:
        mf6.finalize()
        success = True
    except:
        raise RuntimeError

    return A, RHS, U