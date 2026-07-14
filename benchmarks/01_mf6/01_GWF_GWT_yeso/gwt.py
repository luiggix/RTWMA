import os
import flopy

def build(paths, tdis, phys, dis, silent = False):
    # --- COMPONENTES ---

    # Creación del objeto de la simulación de flujo
    o_sim = flopy.mf6.MFSimulation(
        sim_name = paths["tran_name"], 
        sim_ws = paths["tran_ws"], 
        exe_name = paths["mf6_exe"], 
        version="mf6"
    )
    
    # Creación del objeto de la discretización del tiempo
    # OJO: aquí cambio el número de pasos de simulación.
    o_tdis = flopy.mf6.ModflowTdis(
        o_sim,
        time_units = tdis["units"],
        nper = tdis["nper"],
        perioddata = tdis["perioddata"],
    )
    
    # Creación del objeto de la solución del sistema lineal
    # En este caso se requiere de un "solver" para sistemas no simétricos
    o_ims = flopy.mf6.ModflowIms(
        o_sim,
        complexity="SIMPLE",
        linear_acceleration="BICGSTAB",
        outer_dvclose=1e-8,
        inner_dvclose=1e-8,
    )
    
    # Creación del objeto del modelo de transporte
    o_gwt = flopy.mf6.ModflowGwt(
        o_sim, 
        modelname = paths["tran_name"], 
        save_flows = True
    )
    
    # --- PAQUETES ---
    
    # Agregamos el paquete DIS para la discretización espacial estructurada
    o_dis = flopy.mf6.ModflowGwfdis(
        o_gwt,
        nlay = dis["nlay"], 
        nrow = dis["nrow"], 
        ncol = dis["ncol"],
        delr = dis["delr"], 
        delc = dis["delc"], 
        top  = dis["top"], 
        botm = dis["botm"],
    )
    
    # Agregamos el paquete IC para las condiciones iniciales
    # Condicion inicial: c2(x,t0) = 3.294e-05 en todo el dominio,
    # excepto en x12 donde c2(x_12,t0) = 1.647e-07
    o_ic = flopy.mf6.ModflowGwtic(
        o_gwt, 
        strt = phys["initial_concentration"]
    )
    
    # Agregamos el paquete ADV para seleccionar el esquema de advección
    o_adv = flopy.mf6.ModflowGwtadv(
        o_gwt, 
        scheme = "TVD"
    )
    
    # Agregamos el paquete DSP para seleccionar el modelo de dispersión
    o_dsp = flopy.mf6.ModflowGwtdsp(
        o_gwt, 
        xt3d_off = True,
        alh = phys["longitudinal_dispersivity"], 
        ath1 = phys["longitudinal_dispersivity"], 
    )
    
    # Agregamos el paquete MST para definir la porosidad
    o_mst = flopy.mf6.ModflowGwtmst(
        o_gwt, 
        porosity = phys["porosity"]
    )
    
    # Agregamos el paquete CNC para definir una concentracion fija 
    # c1(x1,t) = 3.294e-05
    o_cnc = flopy.mf6.ModflowGwtcnc(
        o_gwt,
        stress_period_data = [phys["bc_conc_t1"][0][1]],
        pname = phys["bc_conc_t1"][0][0],
    )
    
    # Agregamos el paquete FMI para enlazar la solución del flujo con la de transporte
    path_flow = os.path.join(os.getcwd(), paths["flow_ws"], paths["flow_name"])
    o_fmi = flopy.mf6.ModflowGwtfmi(
        o_gwt, 
        packagedata = [("GWFHEAD", f"{path_flow}.hds", None),
                       ("GWFBUDGET", f"{path_flow}.bud", None),
                      ],
        flow_imbalance_correction=True
    )
    
    # Agregamos el paquete SSM para vincular la concentración auxiliar del 
    # pozo WEL-1 con el transporte
    o_ssm = flopy.mf6.ModflowGwtssm(
        o_gwt, 
        sources = [phys["well"][0]]
    )
    
    # Agregamos el paquete OC para almacenar la salida
    oc_gwt = flopy.mf6.ModflowGwtoc(
        o_gwt,
        concentration_filerecord=f"{paths["tran_name"]}.ucn",
        budget_filerecord=f"{paths["tran_name"]}.cbc",
        saverecord=[("CONCENTRATION", "ALL"), ("BUDGET", "ALL")],
    )

    # Escritura de los archivos de entrada para la simulación.
    o_sim.write_simulation(silent = silent)
    
    return o_sim, o_gwt