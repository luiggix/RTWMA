import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flopy
#import resultsFortran as rF 
import xmf6
import os

def show_f(o_gwf, head, x, y, z, ncol, delr): 
    lx = ncol*delr
    #
    # --- Definición de la figura ---
    fig, ax = plt.subplots(2, 1, figsize =(6,3), 
                                        height_ratios=[0.1,1.0])
    #
    # --- Gráfica 1. Malla ---
    pmv = flopy.plot.PlotMapView(o_gwf, ax=ax[0])
    pmv.plot_grid(colors='dimgray', lw=0.5)
    ax[0].set_yticks(ticks=[0, lx])#, fontsize=8)  #ax[0].set_yticks(ticks=[0, 1.0])#, fontsize=8) 
    ax[0].set_title("Mesh")
    
    # --- Gráfica 2. Carga hidráulica---
    ax[1].plot(x[0], head[0, 0], marker="o", lw =1.0, #mec="blue", mfc="none", 
             markersize="4", alpha = 0.75, label = 'Head')
    ax[1].set_xlim(0, lx) #ax[1].set_xlim(0, 1.0)
    ax[1].set_ylabel("Head")
    ax[1].grid()
    ax[1].set_title("Carga hidráulica")

def show_t(U, labels, x, y, z, ncol, delr, yscale = "linear", title=None,
           time=None):
    lx = ncol*delr
    ncomp = len(labels)
    # --- Gráficas de las concentraciones ---
    plt.figure(figsize=(6,3))
    titulo = ""
    
    if title is not None:
        titulo += title
    
    if time is not None:
        titulo += f"  (time = {time:.2f} d)"
    
    if titulo:
        plt.title(titulo)
        
    for m in range(0,ncomp):
        plt.plot(x[0], U[m], ls ="-", lw = 1.0, c = f"C{m}", 
                 marker ="o", markersize="4", alpha = 0.75,
                 label=labels[m], zorder=2)
    plt.xlabel("Distance (m)")
    plt.ylabel(f"Concentration")
#    plt.gca().yaxis.set_major_formatter('{x:5.2e}') 
    plt.yscale(yscale)
    plt.xlim(0, lx) #plt.xlim(0, 1.0)
    plt.legend(loc='best', bbox_to_anchor=(0.95, 0.5, 0.5, 0.5))
    plt.grid()
    plt.show()
