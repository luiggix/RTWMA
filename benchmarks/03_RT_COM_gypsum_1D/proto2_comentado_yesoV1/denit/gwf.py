import flopy
import xmf6
linea = 50*chr(0x2015)

def build(paths, tdis, phys, dis, m, silent = False):
    # --- COMPONENTES ---

    # Creación del objeto de la simulación de flujo
    o_sim = flopy.mf6.MFSimulation(
        sim_name = paths["flow_name"], 
        sim_ws = paths["flow_ws"], 
        exe_name = paths["mf6_exe"], 
        version="mf6"
    )
    
    # Creación del objeto de la discretización del tiempo
    o_tdis = flopy.mf6.ModflowTdis(
        o_sim,
        time_units = tdis["units"],
        nper = tdis["nper"],
        perioddata = tdis["perioddata"],
    )
    
    # Creación del objeto de la solución del sistema lineal
    o_ims = flopy.mf6.ModflowIms(
        o_sim,
        complexity="SIMPLE",
        outer_dvclose=1e-6,
        inner_dvclose=1e-6,
    )
    
    # Creación del objeto del modelo de flujo
    o_gwf = flopy.mf6.ModflowGwf(
        o_sim, 
        modelname = paths["flow_name"], 
        save_flows = True
    )
    
    # --- PAQUETES ---
    
    # Agregamos el paquete DIS para la discretización espacial estructurada
    o_dis = flopy.mf6.ModflowGwfdis(
        o_gwf,
        nlay = dis["nlay"], 
        nrow = dis["nrow"], 
        ncol = dis["ncol"],
        delr = dis["delr"], 
        delc = dis["delc"], 
        top  = dis["top"], 
        botm = dis["botm"],
    )
    
    # Agregamos el paquete NPF para las prop. del flujo
    o_npf = flopy.mf6.ModflowGwfnpf(
        o_gwf,
        k = phys["hydraulic_conductivity"], 
        icelltype=0, 
        save_specific_discharge = True,
        save_saturation=True
    )
    
    # Agregamos el paquete IC para las condiciones iniciales
    o_ic = flopy.mf6.ModflowGwfic(
        o_gwf, 
        strt = phys["initial_head"]
    )
    
    # Agregamos el paquete CHD para definir la carga fija (condición de frontera)
    o_chd = flopy.mf6.ModflowGwfchd(
        o_gwf,
        stress_period_data = [phys["bc_head_t1"][0][1],],
        pname = phys["bc_head_t1"][0][0],
    )
    
    # Agregamos el paquete WEL para definir un pozo de inyección
    # Pozo de inyeccion en x1 (con concentracion auxiliar para el acople con GWT)
    o_wel = flopy.mf6.ModflowGwfwel(
        o_gwf,
        stress_period_data = [phys[f"well_c{m+1}"][1],],
        auxiliary = [phys[f"well_c{m+1}"][0][2]],
        pname = phys[f"well_c{m+1}"][0][0],
    )
    
    # Agregamos el paquete OC para almacenar la salida
    oc_gwf = flopy.mf6.ModflowGwfoc(
        o_gwf,
        head_filerecord=f"{paths["flow_name"]}.hds",
        budget_filerecord=f"{paths["flow_name"]}.bud",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    # Escritura de los archivos de entrada para la simulación.
    o_sim.write_simulation(silent = silent)
    
    return o_sim, o_gwf
    
def buildX(phys, dis, m, paths, silent = True):
    # --- Diccionario para la inicialización de la simulación ---
    sim_flow = dict(
        # Parámetros de la simulación (flopy.mf6.MFSimulation)
        init = {
            'sim_name' : paths["flow_name"],
            'exe_name' : paths["mf6_exe"],
            'sim_ws' : paths["flow_ws"]
        },
    
        # Parámetros para el tiempo (flopy.mf6.ModflowTdis)
        tdis = {
            'units': "days",
            'nper' : 1,
            'perioddata': phys["perioddata"]
        },
    
        # Parámetros para la solución numérica (flopy.mf6.ModflowIms)
        ims = {}
    )

    # --- Diccionario para la simulación de flujo ---
    area = dis['delc'] * (dis['top'] - dis['botm'])
    gwf_d = dict(
        # Parámetros para el modelo de flujo (flopy.mf6.ModflowGwf)
        gwf = { 
            'modelname': paths["flow_name"],
            'save_flows': True
        },
    
        # Parámetros para la discretización espacial (flopy.mf6.ModflowGwfdis)
        dis = dis,
    
        # Parámetros para las condiciones iniciales (flopy.mf6.ModflowGwfic)
        ic = {
            'strt': 1.0
        },
    
        # Parámetros para las condiciones de frontera (flopy.mf6.ModflowGwfchd)
        chd = {
            'stress_period_data': [[(0, 0, dis['ncol'] - 1), 1.0]]  #h=1.0
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
                                    phys["inflow"] * area, 
                                    phys["source_concentration"][m],]],
            'pname': "WEL-1",
            'auxiliary' : ["CONCENTRATION"],
#            'save_flows': True
        },

        # Parámetros para almacenar y mostrar la salida de la simulación (flopy.mf6.ModflowGwfoc)
        oc = {
            'budget_filerecord': f"{paths["flow_name"]}.bud",
            'head_filerecord': f"{paths["flow_name"]}.hds",
            'saverecord': [("HEAD", "ALL"), ("BUDGET", "ALL")],
        }
    )

    # --- Inicialización de la simulación ---
    o_sim = xmf6.common.init_sim(silent = True, **sim_flow)
    o_gwf, packages = xmf6.gwf.set_packages(o_sim, silent = True, **gwf_d)

    # --- Escritura de archivos ---
    if not silent:
        print(linea)
        print("- Escribiendo archivos de entrada para GWF")
    
    o_sim.write_simulation(silent = silent)

    return(o_sim)



    