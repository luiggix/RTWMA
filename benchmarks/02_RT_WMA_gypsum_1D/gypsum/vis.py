import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import flopy
import xmf6

def latex(on):
    if on:
        # Usamos LaTeX para los tectos de la figura.
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif", 
            "font.serif": ["Computer Modern Roman"],
        })
    else:
        plt.rcParams.update({
            "text.usetex": False,
        })
        
def data_recovery(filename):
    #
    # --- Leemos el archivo con los resultados de TR_1D_oper.exe ---
    with open(filename, "r") as f:
            lines = f.readlines()
    #
    # Obtenemos la concentración de las dos especies
    key_string = "Final concentration of aqueous species:"
    i = xmf6.find_index(lines, key_string, 2)
    conc = [np.array(l.split(), dtype=float) for l in lines[i:i+2]]

    return conc
    
def data_recovery_excel(filename, tsel = 10):
    #
    # Concentraciones de los dos especies al tiempo 'tsel'
    c1 = pd.read_excel(filename, sheet_name='c_1').iloc[11:,tsel+5].to_numpy()     
    c2 = pd.read_excel(filename, sheet_name='c_2').iloc[11:,tsel+5].to_numpy()
    #
    # Solución analítica de los dos especies al tiempo 'tsel'
    an_c1 = pd.read_excel(filename, sheet_name='An_C1').iloc[11:,6].to_numpy()
    an_c2 = pd.read_excel(filename, sheet_name='An_C2').iloc[11:,6].to_numpy()

    return c1, c2, an_c1, an_c2
    
def data_recovery_exch(o_gwf, o_gwt):
    # Objeto para acceder a los resultados de la carga hidráulica.
    o_head = o_gwf.output.head()
    
    # Tiempos calculados para el flujo.
    times_h = np.array(o_head.get_times())
    print(times_h)
    
    # Recuperamos la carga hidráulica del paso 40.
    head = o_head.get_data(totim=times_h[-1])[0, 0, :]
    
    # Recuperamos la descarga específica del paso 40.
    budget = o_gwf.output.budget()
    spdis = budget.get_data(totim=times_h[-1], text="DATA-SPDIS")[0]
    qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, o_gwf)
    
    # Recuperamos las coordenadas de los centros de las celdas de la malla
    x, y, z = o_gwf.modelgrid.xyzcellcenters
    
    flow_dict = dict(
        xcoord = x,
        times_h = [times_h],
        head = [head],
        qx = qx,
    )
    xmf6.nice_print("Carga hidráulica", flow_dict)
    
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
    xmf6.nice_print("Concentración", tran_dict)

    return x, head, qx, qy, o_conc, times_c
    
def plot(x, data_exa, data_num, markers, labels, savefig=False):
    # --- Visualización de los resultados
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7,6))
    plt.suptitle("Species concentration, t = $10$ days", x=0.085, ha="left")
    #
    # Gráfica 1
    ax[0].plot(x, data_exa[0], "o--", c = "black", lw=2.0, alpha=0.5, label="Exact", zorder=4)
    for y, m, l in zip(data_num, markers, labels):
        ax[0].plot(x, y[0], m, c = "black", mec = "black", mfc="None", lw = 1.0, label=l, zorder=5)

#    ax[0].set_title(rf"MAPE={MAPE1:5.2f} \%, RMSE = {RMSE1:5.2e}", fontsize=10, loc="right")
    ax[0].set_ylabel("$c_1$")
    ax[0].minorticks_on()
    ax[0].grid(True, which='minor', color='gainsboro', lw = 0.05, zorder=2)
    ax[0].grid(True,which='major', color='gray', lw = 0.5, zorder=3)
    ax[0].legend()
    ax[0].spines.right.set_visible(False)
    ax[0].spines.top.set_visible(False)
    #
    # Gráfica 2
    ax[1].plot(x, data_exa[1], "o--", c = "black", lw=2.0, alpha=0.5, label="Exact", zorder=4)
    for y, m, l in zip(data_num, markers, labels):
        ax[1].plot(x, y[1], m, c = "black", mec = "black", mfc="None", lw = 1.0, label=l, zorder=5)
#    ax[1].set_title(rf"MAPE={MAPE2:5.2f} \%, RMSE = {RMSE2:5.2e}", fontsize=10, loc="right")
    ax[1].set_xlabel("$x$ (m)")
    ax[1].set_ylabel("$c_2$")
    ax[1].minorticks_on()
    ax[1].grid(True, which='minor', color='gainsboro', lw = 0.05, zorder=2)
    ax[1].grid(True,which='major', color='gray', lw = 0.5, zorder=3)
    ax[1].legend(loc="lower right")
    ax[1].spines.right.set_visible(False)
    ax[1].spines.top.set_visible(False)
    
    plt.tight_layout()
    if savefig:
        plt.savefig("gypsum.pdf")
    plt.show()

def plot_exch(o_gwf, x, head, qx, qy, o_conc, times_c, savefig=False):
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
    ax2.plot(x[0], head, marker="o", lw =1.0, c = "dimgray", label = 'Head', 
             mec="black", mfc="black", markersize="5", alpha = 0.75, )
    ax2.set_xlim(0, 30)
    ax2.set_ylabel("$h$ (m)")
    ax2.grid()
    
    max_y = 0
    # --- Gráfica 3. Concentración para diferentes pasos de tiempo
    marker = ["o", "s", "v", "^"]
    if len(times_c) > 1:
        for i, t in enumerate(times_c[9::10]):
            U = o_conc.get_data(totim=t).flatten()
            ax3.plot(x[0], U, ls ="-", lw = 1.0, label=f"t = {t} days", zorder=2,
                 marker = marker[i], markersize="4", alpha = 0.75)
            max_y = max(max_y, U.max())
    else:
        U = o_conc.get_data(totim=times_c[-1]).flatten()
        ax3.plot(x[0], U, ls ="-", lw = 1.0, label=f"t = {1} days", zorder=2,
                 marker = marker[0], markersize="4", alpha = 0.75)
        max_y = max(max_y, U.max())
    
    ax3.set_ylim(0, max_y * 1.1)
    ax3.set_xlabel("$x$ (m)")
    ax3.set_ylabel("$c_1$")
    ax3.legend(fontsize=7)
    ax3.grid()
    
    plt.tight_layout()
    if savefig:
        plt.savefig("gypsum.pdf")
    plt.show()
