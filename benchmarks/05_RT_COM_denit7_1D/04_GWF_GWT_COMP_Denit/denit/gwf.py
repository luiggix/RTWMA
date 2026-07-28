import xmf6
linea = 50*chr(0x2015)

def build(phys, dis, m, paths, silent = True):
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
            'stress_period_data': [[(0, 0, dis['ncol'] - 1), 1.0]]  
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
                                    phys["s_source_concentration"][m],]],
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

#    print(linea)
#    print(m)
#    print(phys["specific_discharge"] * volume)
#    print(phys["source_concentration"][m])
#    print(linea)

    o_sim.write_simulation(silent = silent)

    return(o_sim)



    