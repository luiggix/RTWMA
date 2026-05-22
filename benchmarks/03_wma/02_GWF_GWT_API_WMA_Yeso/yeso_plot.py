import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flopy
#import resultsFortran as rF 
import xmf6
import os

def show(o_sim_f, o_sim_t, wma_working_dir):

    # --- Recuperamos el nombre y el objeto del modelo de flujo
    flow_name = o_sim_f.model_names[0]
    o_gwf = o_sim_f.get_model(flow_name)

    # --- Recuperamos el nombre y el objeto del modelo de transporte
    tran_name = o_sim_t.model_names[0]
    o_gwt = o_sim_t.get_model(tran_name)

    # --- Recuperamos la lista de tiempos del modelo de transporte
    U_obj = o_gwt.output.concentration()
    time = U_obj.get_times()
    
    # --- Recuperamos los resultados de flujo de la simulación ---
#    head = xmf6.gwf.get_head(o_gwf)
#    qx, qy, qz, n_q = xmf6.gwf.get_specific_discharge(o_gwf, text="DATA-SPDIS")
    head = o_gwf.output.head().get_data() #(mflay=0)

    # --- Recuperamos los resultados de la concentración de la simulación ---
    # --- Ojo, usamos solo el primer tiempo
#    U = xmf6.gwt.get_concentration(o_sim_t, time[0])
    U = U_obj.get_data(totim=time[0])[0,0]
    
    # --- Recuperamos las coordenadas del dominio
    x, y, z = o_gwf.modelgrid.xyzcellcenters
    row_length = o_gwf.modelgrid.extent[1]
    
    # --- Definición de la figura ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize =(6,5), 
                                        height_ratios=[0.1,1.0,1.5])
    
    # --- Gráfica 1. Malla ---
    pmv = flopy.plot.PlotMapView(o_gwf, ax=ax1)
    pmv.plot_grid(colors='dimgray', lw=0.5)
    ax1.set_yticks(ticks=[0, 1.0])#, fontsize=8)
    ax1.set_title("Mesh")
    
    # --- Gráfica 2. Carga hidráulica---
    ax2.plot(x[0], head[0, 0], marker="o", lw =1.0, #mec="blue", mfc="none", 
             markersize="4", alpha = 0.75, label = 'Head')
    ax2.set_xlim(0, 30)
    ax2.set_ylabel("Head")
    ax2.grid()
    
    # --- Gráfica 3. Concentración ---
    ax3.plot(x[0], U, ls ="-", lw = 1.0, #c = "C1", 
                 marker ="o", markersize="4", alpha = 0.75,
                 label=f"t = {0} days", zorder=2)
    ax3.set_xlabel("Distance (m)")
    ax3.set_ylabel("Normalized Concentration")
    ax3.set_xlim(0, 30)
    ax3.grid()
    
    plt.tight_layout()
    plt.show()
    
    #dummy = input("\n\n Teclear <ENTER> para continuar ...")
    
    # --- ANÁLISIS DE LOS RESULTADOS ---

    # Tiempo seleccionado de hoja excel
    tsel=10
    
    # Archivo con los resultados de TR_1D.exe
    file_name='gypsum_eq.out'   
    file_path = os.path.join(wma_working_dir, file_name)
    Res_contrsns_ini, Res_contrsns_fin = import_results(file_path)

    file_path_dfc = os.path.join(wma_working_dir, 'gypsum_eq_dfc.out' )
    Res_contrsns_ini_dfc, Res_contrsns_fin_dfc = import_results(file_path_dfc)

    # Lectura de datos de la tabla de excel
    WMA_I=pd.read_excel('comparativa_02.xlsx', sheet_name='c_2')     
    data_set_C2 = np.transpose(WMA_I.iloc[11:,5:95].to_numpy())
    
    # Calculamos el RMSE
    RMSE = np.linalg.norm(data_set_C2[tsel][:]-np.array(Res_contrsns_fin[1][:])) / np.sqrt(len(Res_contrsns_fin))
    
    # --- Gráfica de los resultados
    plt.figure(figsize=(6,4))
    plt.plot(x[0], data_set_C2[tsel][:], 
             lw=0.5, ls="--", c="k", zorder=5)
    plt.scatter(x[0], Res_contrsns_fin[1][:], 
                marker = "s", c = "mediumblue", s=30, label="TR_1D", zorder=3)
    plt.scatter(x[0], Res_contrsns_fin_dfc[1][:], 
                marker = "*", c = "black", s=40, label="TR_1D_DFC", zorder=4)
    plt.scatter(x[0], data_set_C2[tsel][:], 
                marker = "o", c = "violet", s=15,label="Excel", zorder=5)
    plt.title(f"t = {tsel} días, RMSE = {RMSE:5.3e}")
    plt.xlabel("$x$ [$m$]")
    plt.ylabel("$c_2 [mgr/cm^{3}]$")
    plt.legend()
    plt.minorticks_on()
    plt.grid(True,which='major', color='darkgray', lw = 0.5)
    plt.grid(True, which='minor', color='silver', lw = 0.25)
    plt.show()

def import_results(file_name):

    # Initializing variables
    integration_method = ""
    number_of_targets = 0
    time_step = 0.0
    final_time = 0.0
    matrix_data = []
    mtrz_contrsns_ini = []
    mtrz_contrsns_fin = []
    
    # Reading the .out file and importing data
    with open(file_name, 'r') as file:
        for line in file:
            
            # Clearing blank lines before and after text
            clean_line = line.strip()
            
            # Searching for specific information
            if "Integration method" in clean_line:
                integration_method = clean_line.split(":")[-1].strip()
            elif "Number of targets" in clean_line:
                number_of_targets = int(clean_line.split(":")[-1].strip())
            elif "Time step" in clean_line:
                next(file)
                time_step = float(next(file).strip())
            elif "Final time" in clean_line:
                next(file)
                final_time = float(next(file).strip())
            elif "Dimension + Mixing ratios (by rows)" in clean_line:
                next(file)
                
                # Starting to save the results matrix
                for line in file:
                    
                    # If the line is not empty
                    if line.strip():  
                        numbers = list(map(float, line.split()))
                        matrix_data.append(numbers)
                    
                    # Exit loop if an empty line is found
                    else:
                        break
                    
            elif "Initial concentration of aqueous species" in clean_line:
                next(file)
                
                # Starting to save the results matrix
                for line in file:
                    
                    # If the line is not empty
                    if line.strip():  
                        numbers = list(map(float, line.split()))
                        mtrz_contrsns_ini.append(numbers)
                    
                    # Exit loop if an empty line is found
                    else:
                        break
                    
            elif "Final concentration of aqueous species" in clean_line:
                next(file)
                
                # Starting to save the results matrix
                for line in file:
                    
                    # If the line is not empty
                    if line.strip():  
                        numbers = list(map(float, line.split()))
                        mtrz_contrsns_fin.append(numbers)
                    
                    # Exit loop if an empty line is found
                    else:
                        break
    return (mtrz_contrsns_ini, mtrz_contrsns_fin)
