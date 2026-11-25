import os
import flopy

def build(o_sim, paths, phys, num):
    """
    Construye el modelo de flujo.

    Parameters
    ----------
    o_sim : Objeto principal de la simulación a la que se agregará el
        modelo de flujo. 
    paths : dict
        Diccionario con las rutas y archivos usados en la simulación.
    phys : dict
        Diccionario con los parámetros físicos de la simulación.
    num : dict
        Diccionario con los parámetros numéricos de la simulación.

    Returns
    -------
    flopy.mf6.ModflowGwf
        Objeto "o_gwf" que representa el modelo de flujo.
    """
    ## ---> Creación del objeto del modelo de flujo
    o_gwf = flopy.mf6.ModflowGwf(
        o_sim, 
        modelname = paths["flow_name"], 
        save_flows = True
    )

    ## ---> Creación y registro del objeto de la solución del sistema lineal
    o_ims_f = flopy.mf6.ModflowIms(
        o_sim,
        filename=f"{o_gwf.name}.ims",
        complexity = num["ims_flow"]["complexity"],
        outer_dvclose = num["ims_flow"]["outer_dvclose"],
        inner_dvclose = num["ims_flow"]["outer_dvclose"],

    )
    o_sim.register_ims_package(o_ims_f, [o_gwf.name])
    
    # --- PAQUETES ---
    
    ## ---> Agregamos el paquete DIS para la discretización espacial estructurada
    o_dis = flopy.mf6.ModflowGwfdis(
        o_gwf,
        length_units = num["dis"]["length_units"],
        nlay = num["dis"]["nlay"], 
        nrow = num["dis"]["nrow"], 
        ncol = num["dis"]["ncol"],
        delr = num["dis"]["delr"], 
        delc = num["dis"]["delc"], 
        top  = num["dis"]["top"], 
        botm = num["dis"]["botm"],
    )
    
    ## ---> Agregamos el paquete NPF para las prop. del flujo
    o_npf = flopy.mf6.ModflowGwfnpf(
        o_gwf,
        k = phys["hydraulic_conductivity"], 
        icelltype = 0, 
        save_specific_discharge = True,
        save_saturation = True
    )
    
    ## ---> Agregamos el paquete IC para las condiciones iniciales
    o_ic = flopy.mf6.ModflowGwfic(
        o_gwf, 
        strt = phys["initial_head"]
    )
    
    ## ---> Agregamos el paquete CHD para definir la carga fija (condición de frontera)
    o_chd = flopy.mf6.ModflowGwfchd(
        o_gwf,
        stress_period_data = phys["bc_head_t1"]["stress_period_data"],
        pname = phys["bc_head_t1"]["pname"],
    )
    
    ## ---> Agregamos el paquete WEL para definir un pozo de inyección
    ## Pozo de inyeccion en x1 (con concentracion auxiliar para el acople con GWT)
    o_wel = flopy.mf6.ModflowGwfwel(
        o_gwf,
        stress_period_data = phys["well"]["stress_period_data"], 
        auxiliary = phys["well"]["auxiliary"],
        pname = phys["well"]["pname"],
    )

    ## ---> Agregamos el paquete STO para definir el caso estacionario
    o_sto = flopy.mf6.ModflowGwfsto(
        o_gwf,
        steady_state=True,
    )
    
    ## ---> Agregamos el paquete OC para almacenar la salida
    oc_gwf = flopy.mf6.ModflowGwfoc(
        o_gwf,
        head_filerecord=f"{paths["flow_name"]}.hds",
        budget_filerecord=f"{paths["flow_name"]}.bud",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
  
    return o_gwf