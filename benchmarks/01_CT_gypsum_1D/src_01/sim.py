import os
import flopy

def init(paths, num, model = "sim"):
    """
    Inicializa la simulación y agrega la discretización del tiempo.

    Parameters
    ----------
    paths : dict
        Diccionario con las rutas y archivos usados en la simulación.
    num : dict
        Diccionario con los parámetros numéricos de la simulación.
    model : str
        Define si se trata del modelo de flujo "flow", del modelo de 
        transporte "tran", o de una simulación general "sim" donde se
        aplica un intercambio (este es el valor por omisión).

    Returns
    -------
    flopy.mf6.MFSimulation
        Objeto "o_sim" que representa la simulación inicializada. 
    """
    name = model + "_name"
    ws = model + "_ws"
    
    ## ---> Creación del objeto de la simulación de flujo
    o_sim = flopy.mf6.MFSimulation(
        sim_name = paths[name], 
        sim_ws = paths[ws], 
        exe_name = paths["mf6_exe"], 
        version="mf6"
    )
    
    ## ---> Creación del objeto de la discretización del tiempo
    o_tdis = flopy.mf6.ModflowTdis(
        o_sim,
        time_units = num["tdis"]["units"],
        nper = num["tdis"]["nper"],
        perioddata = num["tdis"]["perioddata"],
    )

    return o_sim