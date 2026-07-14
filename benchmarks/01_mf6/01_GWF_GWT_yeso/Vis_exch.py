import numpy as np
import matplotlib.pyplot as plt
import flopy
import xmf6

def data_recovery(o_gwf, o_gwt):
    # Objeto para acceder a los resultados de la carga hidráulica.
    o_head = o_gwf.output.head()
    
    # Tiempos calculados para el flujo.
    times_h = np.array(o_head.get_times())
    
    # Recuperamos la carga hidráulica
    head = o_head.get_data(totim=40)[0, 0, :]
    
    # Recuperamos la descarga específica
    budget = o_gwf.output.budget()
    spdis = budget.get_data(totim=40, text="DATA-SPDIS")[0]
    qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, o_gwf)
    
    # Recuperamos las coordenadas de los centros de las celdas de la malla
    x, y, z = o_gwf.modelgrid.xyzcellcenters
    
    flow_dict = dict(
        xcoord = x,
        times_h = [times_h],
        head = [head],
        qx = qx,
    )
    xmf6.nice_print(flow_dict, "Carga hidráulica")
    
    # Objeto para recuperar los resultados del transporte
    o_conc = o_gwt.output.concentration()
    
    # Recuperamos los pasos de tiempo calculados
    times_c = np.array(o_conc.get_times())
    
    # Recuperamos la información del último paso de tiempo
    U_40 = o_conc.get_data(totim=times_c[-1]).flatten()
    
    # Diccionario para imprimir la información en pantalla
    tran_dict = dict(
        xcoord = x,
        times = [times_c],
        conc = [U_40]
    )
    xmf6.nice_print(tran_dict, "Concentración")

    return x, head, qx, qy, o_conc, times_c

def plot(o_gwf, x, head, qx, qy, o_conc, times_c):
    # --- Definición de la figura. Se definen tres gráficas 
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex = True, figsize =(6,5),
                                        height_ratios=[0.1, 0.5, 0.5])
    
    # --- Gráfica 1. Carga hidráulica y descarga específica sobre la malla
    ax1.set_aspect('equal')
    pmv = flopy.plot.PlotMapView(model = o_gwf, ax = ax1)
    pmv.plot_grid(colors = 'k', lw = 0.5, ls="-")
    pmv.plot_array(head, cmap = "viridis", alpha=0.5)
    pmv.plot_vector(qx, qy, scale=40, pivot="mid", width=0.004, normalize=True, color="k")
    
    # --- Gráfica 2. Carga hidráulica vs posición
    ax2.plot(x[0], head, marker="o", lw =1.0, mec="k", mfc="C0", 
             markersize="4", alpha = 1.0, label = 'Head')
    ax2.set_xlim(0, 30)
    ax2.set_ylabel("$h$ (m)")
    ax2.grid()
    
    max_y = 0
    # --- Gráfica 3. Concentración para diferentes pasos de tiempo
    for t in times_c[9::10]:
        U = o_conc.get_data(totim=t).flatten()
        ax3.plot(x[0], U, ls ="-", lw = 1.0, 
             marker ="o", markersize="4", alpha = 0.75,
             label=f"t = {t} days", zorder=2)
        max_y = max(max_y, U.max())
    
    ax3.set_ylim(0, max_y * 1.1)
    ax3.set_xlabel("$x$ (m)")
    ax3.set_ylabel("$c_1$")
    ax3.legend(fontsize=7)
    ax3.grid()
    
    plt.tight_layout()
    plt.show()