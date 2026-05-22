import xmf6
linea = 50*chr(0x2015)

def build(phys, dis, mf6_exe, tran_name, tran_ws):
    # --- Diccionario para la inicialización de la simulación ---
    sim_trans = dict(
        # Parámetros de la simulación (flopy.mf6.MFSimulation)
        init = {
            'sim_name' : tran_name,
            'exe_name' : mf6_exe,
            'sim_ws' : tran_ws,
        },

        # Parámetros para el tiempo (flopy.mf6.ModflowTdis)
        tdis = {
            'units': "days",
            'nper' : 1,
            'perioddata': [(40.0, 40, 1.0)] #PERLEN, NSTP, TSMULT
        },

        # Parámetros para la solución numérica (flopy.mf6.ModflowIms)
        ims = {
            'linear_acceleration': "bicgstab"
        }
    )

    # --- Diccionario para la simulación de flujo ---
    gwt_d = dict(
        # Parámetros para el modelo de transporte (flopy.mf6.ModflowGwt)
        gwt = { 
            'modelname': tran_name,
            'save_flows': True
        },
    
        # Parámetros para la discretización espacial (flopy.mf6.ModflowGwtdis)
        # El mismo dis de flow
        dis = dis,
        
        # Parámetros para las condiciones iniciales (flopy.mf6.ModflowGwtic)
        ic = {
            'strt': phys['initial_concentration']
        },

        # Parámetros para MST (flopy.mf6.ModflowGwtmst)
        mst ={
                "porosity": phys["porosity"]
        },

        # Parámetros para ADV (flopy.mf6.ModflowGwtadv)
        adv = {
            "scheme" : "UPSTREAM" #"CENTRAL"# "UPSTREAM" #"TVD" #
        },

        # Parámetros para DSP (flopy.mf6.ModflowGwtdsp)
        dsp = {
            "xt3d_off" : True,
            "alh" : phys["longitudinal_dispersivity"],
            "ath1" : phys["longitudinal_dispersivity"],
        },

        # Parámetros para FMI (flopy.mf6.ModflowGwtfmi)
        fmi = {
            "packagedata" : [("GWFHEAD", "flow.hds", None),
                             ("GWFBUDGET", "flow.bud", None),
                            ]   
        },

        # Parámetros para SSM (flopy.mf6.ModflowGwtssm)
        ssm = {
            "sources" : [["WEL-1", "AUX", "CONCENTRATION"]]
        },

        # Parámetros para OBS (flopy.mf6.ModflowGwtobs)
        obs = {
            "digits" : 10, 
            "print_input" : True, 
            "continuous" : {
                "transport_obs.csv": [
                    ("X005", "CONCENTRATION", (0, 0, 5)),
                    ("X011", "CONCENTRATION", (0, 0, 11)),
                    ("X029", "CONCENTRATION", (0, 0, 29)),
                ],
            }
        },

        # Parámetros para almacenar y mostrar la salida de la simulación (flopy.mf6.ModflowGwtoc)
        oc = {
            'budget_filerecord': f"{tran_name}.cbc",
            'concentration_filerecord': f"{tran_name}.ucn",
            'saverecord' : [("CONCENTRATION", "ALL"), ("BUDGET", "LAST")],
            'printrecord' : [("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
        },
    )
    
    # --- Inicialización de la simulación ---
    o_sim = xmf6.common.init_sim(silent = True, **sim_trans)
    o_gwt, packagest = xmf6.gwt.set_packages(o_sim, silent = True, **gwt_d)
    o_obs = xmf6.common.set_obs(o_gwt, gwt_d['obs'], silent = True)

    # --- Escritura de archivos ---
    print(linea)
    print("- Escribiendo archivos de entrada para GWT")
    o_sim.write_simulation(silent = True)

    return(o_sim)