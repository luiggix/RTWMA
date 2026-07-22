import os
import numpy as np

# Encuentra el índice donde esta la cadena (ks) y salta (sk) al renglón donde comienza la info
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

def initial_conditions(paths, spec = False, comp = False):
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
    #
    # Diccionario para las componentes
    components = {}
    species = {}
    #
    # Número de componentes
    ncomp = int(lines[0].split(": ")[1].strip())
    #
    # Número de especies (¿DEBERÍA ESTAR EL NÚMERO DE ESPECIES?)
    nspec = int(lines[0].split(": ")[1].strip()) + 1
    #
    if comp:
        #
        # Leer nombres de las componentes
        key_string = "Aqueous components:"
        i = find_index(lines, key_string, 1)
        components["names"] = [l.split("= ")[1].strip() for l in lines[i:i+ncomp]]
        #
        # Concentraciones iniciales de cada componente
        key_string = "Aqueous component concentrations in the domain:"
        i = find_index(lines, key_string, 1)
        components["init_conc"] = [np.array(l.split(), dtype=float) for l in lines[i:i+ncomp]]
        #
        # Concentración de cada componente que proviene del exterior
        key_string = "Aqueous component concentrations of external waters:"
        i = find_index(lines, key_string, 2)
        components["ext_water"] = [float(l) for l in lines[i:i+ncomp]]
        
    if spec:
        #
        # Leer nombres de las especies
        key_string = "Aqueous variable activity species:"
        i = find_index(lines, key_string, 1)
        species["names"] = [l.split(": ")[1].strip() for l in lines[i:i+nspec]]
        #
        # Concentraciones iniciales de cada componente
        key_string = "Aqueous variable activity species concentrations in the domain:"
        i = find_index(lines, key_string, 1)
        species["init_conc"] = [np.array(l.split(), dtype=float) for l in lines[i:i+nspec]]
        #
        # Concentración de cada componente que proviene del exterior
        key_string = "Aqueous variable activity species concentrations of external waters:"
        i = find_index(lines, key_string, 2)
        species["ext_water"] = [float(sw) for sw in lines[i:i+nspec]]
        
    return components, species