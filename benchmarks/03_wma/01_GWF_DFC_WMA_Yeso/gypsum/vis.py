import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xmf6

def data_recovery(filename, excel):
    # --- Leemos el archivo con los resultados de TR_1D_oper.exe ---
    with open(filename, "r") as f:
            lines = f.readlines()
    #
    # Obtenemos la concentración de las dos especies
    key_string = "Final concentration of aqueous species:"
    i = xmf6.find_index(lines, key_string, 2)
    c_final = [np.array(l.split(), dtype=float) for l in lines[i:i+2]]
    #
    # --- Leemos del archivo excel los datos para comparación ---
    tsel=10 # Tiempo seleccionado de hoja excel
    #
    # Concentraciones de los dos especies al tiempo 'tsel'
    c1_excel = pd.read_excel(excel, sheet_name='c_1').iloc[11:,tsel+5].to_numpy()     
    c2_excel = pd.read_excel(excel, sheet_name='c_2').iloc[11:,tsel+5].to_numpy()
    #
    # Solución analítica de los dos especies al tiempo 'tsel'
    an_c1 = pd.read_excel(excel, sheet_name='An_C1').iloc[11:,6].to_numpy()
    an_c2 = pd.read_excel(excel, sheet_name='An_C2').iloc[11:,6].to_numpy()
    #
    # Calculamos el RMSE (https://en.wikipedia.org/wiki/Root_mean_square_deviation)
    RMSE1 = np.linalg.norm(an_c1 - c_final[0]) / np.sqrt(len(c_final[0]))
    RMSE2 = np.linalg.norm(an_c2 - c_final[1]) / np.sqrt(len(c_final[1]))
    print(f"RMSE c1: {RMSE1:5.2e} \t RMSE c2: {RMSE2:5.2e}")
    #
    # Calculamos el MAPE (https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)
    MAPE1 = np.mean(np.abs((an_c1 - c_final[0]) / an_c1)) * 100
    MAPE2 = np.mean(np.abs((an_c2 - c_final[1]) / an_c2)) * 100
    print(f"MAPE c1: {MAPE1:5.2f}% \t MAPE c2: {MAPE2:5.2f}")

    return c_final, c1_excel, c2_excel, an_c1, an_c2, RMSE1, RMSE2, MAPE1, MAPE2


def plot(x, c_final, c1_excel, c2_excel, an_c1, an_c2, RMSE1, RMSE2, MAPE1, MAPE2, label1, label2, latex = False):
    if latex:
        # Usamos LaTeX para los tectos de la figura.
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",  # Uses standard Computer Modern font
            "font.serif": ["Computer Modern Roman"],
        })
    # --- Visualización de los resultados
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7,6))
    #
    # Gráfica de la especie c1
    ax[0].plot(x, an_c1, "o--", c = "black", lw=2.0, alpha=0.5, label="Exacta", zorder=4)
    ax[0].plot(x, c_final[0], "s-", c = "black", mec = "black", mfc="None", lw = 1.0, label=label1, zorder=5)
    ax[0].plot(x, c1_excel, ".-", c = "black", mec = "black", mfc="None", lw = 1.0,label=label2, zorder=5)
    ax[0].set_title(rf"MAPE={MAPE1:5.2f} \%, RMSE = {RMSE1:5.2e}", fontsize=10, loc="right")
    ax[0].set_ylabel("$c_1$")
    ax[0].minorticks_on()
    ax[0].grid(True, which='minor', color='gainsboro', lw = 0.05, zorder=2)
    ax[0].grid(True,which='major', color='gray', lw = 0.5, zorder=3)
    ax[0].legend()
    ax[0].spines.right.set_visible(False)
    ax[0].spines.top.set_visible(False)
    #
    # Gráfica de la especie c2
    ax[1].plot(x, an_c2, "o--", c = "black", lw=2.0, alpha=0.5, label="Exacta", zorder=4)
    ax[1].plot(x, c_final[1], "s-", c = "black", mec = "black", mfc="None", lw = 1.0, label=label1, zorder=5)
    ax[1].plot(x, c2_excel, ".-", c = "black", mec = "black", mfc="None", lw = 1.0, label=label2, zorder=5)
    ax[1].set_title(rf"MAPE={MAPE2:5.2f} \%, RMSE = {RMSE2:5.2e}", fontsize=10, loc="right")
    ax[1].set_xlabel("$x$ (m)")
    ax[1].set_ylabel("$c_2$")
    ax[1].minorticks_on()
    ax[1].grid(True, which='minor', color='gainsboro', lw = 0.05, zorder=2)
    ax[1].grid(True,which='major', color='gray', lw = 0.5, zorder=3)
    ax[1].legend(loc="lower right")
    ax[1].spines.right.set_visible(False)
    ax[1].spines.top.set_visible(False)
    
    plt.tight_layout()
    #plt.savefig("yeso.pdf")
    plt.show()