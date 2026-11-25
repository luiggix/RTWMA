import os
import flopy

def build(o_sim, paths, phys, num, exch = False):
    """
    Construye el modelo de transporte.

    Parameters
    ----------
    o_sim : Objeto principal de la simulación a la que se agregará el
        modelo de transporte. 
    paths : dict
        Diccionario con las rutas y archivos usados en la simulación.
    phys : dict
        Diccionario con los parámetros físicos de la simulación.
    num : dict
        Diccionario con los parámetros numéricos de la simulación.
    exch : bool
        Indica si hay intercambio (True) o no. Valor por omisión False (no hay intercambio)

    Returns
    -------
    flopy.mf6.ModflowGwt
        Objeto "o_gwt" que representa el modelo de transporte.
    """
    ## ---> Creación del objeto del modelo de transporte
    o_gwt = flopy.mf6.ModflowGwt(
        o_sim, 
        modelname = paths["tran_name"], 
        save_flows = True
    )

    ## ---> Creación y registro del objeto de la solución del sistema lineal
    o_ims_t = flopy.mf6.ModflowIms(
        o_sim,
        filename=f"{o_gwt.name}.ims",
        complexity = num["ims_tran"]["complexity"],
        linear_acceleration = num["ims_tran"]["linear_acceleration"],
        outer_dvclose = num["ims_tran"]["outer_dvclose"],
        inner_dvclose = num["ims_tran"]["inner_dvclose"],
    )
    o_sim.register_ims_package(o_ims_t, [o_gwt.name])
    
    # --- PAQUETES ---
    
    ## ---> Agregamos el paquete DIS para la discretización espacial estructurada
    o_dis = flopy.mf6.ModflowGwtdis(
        o_gwt,
        length_units = num["dis"]["length_units"],
        nlay = num["dis"]["nlay"], 
        nrow = num["dis"]["nrow"], 
        ncol = num["dis"]["ncol"],
        delr = num["dis"]["delr"], 
        delc = num["dis"]["delc"], 
        top  = num["dis"]["top"], 
        botm = num["dis"]["botm"],
    )
    
    ## ---> Agregamos el paquete IC para las condiciones iniciales
    o_ic = flopy.mf6.ModflowGwtic(
        o_gwt, 
        strt = phys["initial_concentration"]
    )
    
    ## ---> Agregamos el paquete ADV para seleccionar el esquema de advección
    o_adv = flopy.mf6.ModflowGwtadv(
        o_gwt, 
        scheme = num["adv_scheme"]
    )
    
    ## ---> Agregamos el paquete DSP para seleccionar el modelo de dispersión
    o_dsp = flopy.mf6.ModflowGwtdsp(
        o_gwt, 
        xt3d_off = True,
        alh = phys["longitudinal_dispersivity"], 
        ath1 = phys["longitudinal_dispersivity"], 
    )
    
    ## ---> Agregamos el paquete MST para definir la porosidad
    o_mst = flopy.mf6.ModflowGwtmst(
        o_gwt, 
        porosity = phys["porosity"]
    )

    ## ---> Cuando no hay intercambio se agrega FMI.
    if not exch: 
        ## ---> Agregamos el paquete FMI para enlazar la solución del flujo con la de transporte
        path_flow = os.path.join(os.getcwd(), paths["flow_ws"], paths["flow_name"])
        o_fmi = flopy.mf6.ModflowGwtfmi(
            o_gwt, 
            packagedata = [("GWFHEAD", f"{path_flow}.hds", None),
                           ("GWFBUDGET", f"{path_flow}.bud", None),
                          ],
            flow_imbalance_correction = True
        )
    
    ## ---> Agregamos el paquete SSM para vincular la concentración auxiliar del 
    ##      pozo WEL-1 con el transporte
    o_ssm = flopy.mf6.ModflowGwtssm(
        o_gwt, 
        sources = phys["SSM"]
    )
    
    ## ---> Agregamos el paquete OC para almacenar la salida
    oc_gwt = flopy.mf6.ModflowGwtoc(
        o_gwt,
        concentration_filerecord=f"{paths["tran_name"]}.ucn",
        budget_filerecord=f"{paths["tran_name"]}.cbc",
        saverecord=[("CONCENTRATION", "ALL"), ("BUDGET", "ALL")],
    )
    
    return o_gwt