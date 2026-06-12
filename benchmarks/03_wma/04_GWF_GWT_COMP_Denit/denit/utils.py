import os
import numpy as np

# Encuentra el índice donde esta la cadena (ks) y salta (sK) al renglón donde comienza la info
find_index = lambda ls, ks, sk: [n for n, l in enumerate(ls) if ks in l][0] + sk
    
def get_head(o_sim_f):
# --- Recuperamos el nombre y el objeto del modelo de flujo
    flow_name = o_sim_f.model_names[0]
    o_gwf = o_sim_f.get_model(flow_name)
    #
    # --- Recuperamos los resultados de flujo de la simulación ---
    head = o_gwf.output.head().get_data() #(mflay=0)
    #
    # --- Budget ---
    #bud = o_gwf.output.budget()
    #print(bud.get_unique_record_names())
    #flow_data = bud.get_data(idx=idx, full3D=True)
    #
    # --- Coodenadas ---
    x, y, z = o_gwf.modelgrid.xyzcellcenters
    return o_gwf, head, x, y, z

def get_conc(o_sim_t):
    # --- Recuperamos el nombre y el objeto del modelo de transporte
    tran_name = o_sim_t.model_names[0]
    o_gwt = o_sim_t.get_model(tran_name)

    # --- Recuperamos la lista de tiempos del modelo de transporte
    U_obj = o_gwt.output.concentration()
    time = U_obj.get_times()
    # --- Recuperamos los resultados de la concentración de la simulación ---
    # --- Ojo, usamos solo el primer tiempo
    return U_obj.get_data(totim=time[0])[0,0]

def initial_conditions(paths, ncomp):
    if not os.path.isfile(paths["u_init"]):
        print(f"-> File {paths["u_init"]} does not exist")
        print(f"-> Generating {paths["u_init"]}: ... \n {paths["command"]}")
        ret = os.system(paths["command"])
    else:
        print(f"-> File {paths["u_init"]} already exist")
    #
    # Leer información de "u_init.dat"
    with open(paths["u_init"], "r") as f:
        lines = f.readlines()
    #
    # Leer nombres de las componentes
    key_string = "Aqueous components:"
    icn = find_index(lines, key_string, 1)
    comp_names = [l.split("= ")[1][:-1] for l in lines[icn:icn+ncomp]]
    #
    # Concentraciones iniciales de cada componente
    key_string = "Aqueous component concentrations in the domain:"
    icc = find_index(lines, key_string, 1)
    comp_conc_init = [np.array(l.split(), dtype=float) for l in lines[icc:icc+ncomp]]
    #
    # Concentración de cada componente que proviene del exterior
    key_string = "Aqueous component concentrations of external waters:"
    iew = find_index(lines, key_string, 2)
    comp_ext_water = [float(cw) for cw in lines[iew:iew+ncomp]]
    
    return comp_names, comp_conc_init, comp_ext_water