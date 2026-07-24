import os
import flopy

def build(paths, o_sim, phys, dis, silent = False):
    # --- COMPONENTES ---
    
    # Creación del objeto del modelo de flujo
    o_gwf = flopy.mf6.ModflowGwf(
        o_sim, 
        modelname = paths["flow_name"], 
        save_flows = True
    )
    
    # Definición del modelo de solución. Se realiza en este punto porque primero
    # se requiere definir el objeto de flujo 'o_gwf' para conocer el nombre y
    # posteriormente hacer el "registro" del modelo de solución en el objeto 'o_sim'.
    o_ims_f = flopy.mf6.ModflowIms(
        o_sim,
        filename=f"{o_gwf.name}.ims",
        complexity="SIMPLE",
        outer_dvclose=1e-6,
        inner_dvclose=1e-6,
    )
    o_sim.register_ims_package(o_ims_f, [o_gwf.name])
    
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
        stress_period_data = [phys["well"][1],],
        auxiliary = [phys["well"][0][2]],
        pname = phys["well"][0][0],
    )
    
    o_sto = flopy.mf6.ModflowGwfsto(
        o_gwf,
        steady_state=True,
    )
    
    # Agregamos el paquete OC para almacenar la salida
    oc_gwf = flopy.mf6.ModflowGwfoc(
        o_gwf,
        head_filerecord=f"{paths["flow_name"]}.hds",
        budget_filerecord=f"{paths["flow_name"]}.bud",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    return o_gwf