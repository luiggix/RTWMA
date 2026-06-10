import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flopy
#import resultsFortran as rF 
import xmf6
import os
    
def show_f(o_gwf, head, x, y, z): 
    #
    # --- Definición de la figura ---
    fig, ax = plt.subplots(2, 1, figsize =(6,3), 
                                        height_ratios=[0.1,1.0])
    #
    # --- Gráfica 1. Malla ---
    pmv = flopy.plot.PlotMapView(o_gwf, ax=ax[0])
    pmv.plot_grid(colors='dimgray', lw=0.5)
    ax[0].set_yticks(ticks=[0, 0.1])#, fontsize=8)
    ax[0].set_title("Mesh")
    
    # --- Gráfica 2. Carga hidráulica---
    ax[1].plot(x[0], head[0, 0], marker="o", lw =1.0, #mec="blue", mfc="none", 
             markersize="4", alpha = 0.75, label = 'Head')
    ax[1].set_xlim(0, 1.0)
    ax[1].set_ylabel("Head")
    ax[1].grid()

def show_t(U, m, x, y, z):
    # --- Gráficas de las concentraciones ---
    plt.figure(figsize=(6,2))
    plt.plot(x[0], U, ls ="-", lw = 1.0, c = f"C{m}", 
                 marker ="o", markersize="4", alpha = 0.75,
                 label=f"t = {0} days", zorder=2)
#    plt.xlabel("Distance (m)")
    plt.ylabel(f"Concentration - C{m+1}")
    plt.gca().yaxis.set_major_formatter('{x:6.3e}') 
    plt.xlim(0, 1.0)
    plt.grid()
    plt.show()
