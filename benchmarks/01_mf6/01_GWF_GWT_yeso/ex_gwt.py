import os
import flopy

def build(paths, o_sim, phys, dis, silent = False):
    # --- COMPONENTES ---
    
    # Creación del objeto del modelo de transporte
    o_gwt = flopy.mf6.ModflowGwt(
        o_sim, 
        modelname = paths["tran_name"], 
        save_flows = True
    )
    
    # Definición del modelo de solución. Se realiza en este punto porque primero
    # se requiere definir el objeto de flujo 'o_gwt' para conocer el nombre y
    # posteriormente hacer el "registro" del modelo de solución en el objeto 'o_sim'.
    o_ims_t = flopy.mf6.ModflowIms(
        o_sim,
        filename=f"{o_gwt.name}.ims",
        complexity="SIMPLE",
        linear_acceleration="BICGSTAB",   # requerido: la matriz de GWT es asimetrica
        outer_dvclose=1e-8,
        inner_dvclose=1e-8,
    )
    o_sim.register_ims_package(o_ims_t, [o_gwt.name])
    
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
    # Condicion inicial: U(x,t0) en todo el dominio,
    # excepto en x12 donde U(x_12,t0) tiene un valor diferente.
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
    
    # Agregamos el paquete CNC para definir una concentracion fija igual a U(x1,t)
    o_cnc = flopy.mf6.ModflowGwtcnc(
        o_gwt,
        stress_period_data = [phys["bc_conc_t1"][0][1]],
        pname = phys["bc_conc_t1"][0][0],
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

    return o_gwt