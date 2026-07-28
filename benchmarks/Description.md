# Ejemplos de simulación de flujo (**GWF**, **DFC**), transporte (**GWT**, **DFC**) y transporte reactivo (**WMA**).

* **01_CT_gypsum_1D**. Ejemplos de simulación de flujo y transporte conservativo utilizando GWF y GWT.
    - $c_1 = SO_4^{2-}$ (`01_flow_tran_gypsum_c1.ipynb`)
    - $c_2 = Ca^{2+}$ (`02_flow_tran_gypsum_c2.ipynb`)
    - $U = c_1 - c_2$ (`03_flow_tran_gypsum_U.ipynb`, `03_flow_tran_gypsum_U.ipynb`)
    - $U = c_1 - c_2$ con intercambio entre el flujo y transporte (`05_flow_tran_gypsum_Uexch.ipynb`, `06_flow_tran_gypsum_Uexch_vis.ipynb`)
    - 
* **02_RT_WMA_gypsum_1D**.
    - `1_GWF_DFC_WMA_Imp.ipynb`. Solución del ejemplo del Yeso con transporte reactivo. El flujo se calcula con GWT, mientras que el transporte reactivo se calcula con WMA usando el programa REMIX. Para obtener las proporciones de mezcla se utilizan diferencias finitas centradas (DFC) para generar los coeficientes de la ecuación de transporte conservativa, en conjunto con un desarrollo en python para generar las matrices correspondientes. En esta notebook, además de resolver el problema, contiene toda la explicación paso a paso de la solución. Se considera solo el caso **implícito** del WMA.
    - `2_Optimized_Code.ipynb`. Se resuelve el mismo problema resuelto en la notebook **1_GWF_DFC_WMA_Imp.ipynb** como sigue:
        + **3_GWF_DFC_WMA_Imp.py**. Contiene todo el código en un solo archivo que se puede ejecutar en línea de comando.
        + **4_GWF_API_DFC_Imp_WMA.py**. Se usa la API de MODFLOW para tomar la información del flujo directamente de memoria y usar esta información en los procesos del transporte reactivo. La implementación de la solución del flujo con GWF y la API se puede ver en el archivo `gypsum/gwf_api.py`.
        + **5_GWF_GWT_API_WMA.py**. Se resuelve el mismo problema del yeso descrito en **1_GWF_DFC_WMA_Imp.ipynb** pero en este caso se generan los coeficientes de la matriz del transporte conservativo usando GWT y la API; posteriormente se utiliza la API para extraer dichos coeficientes del sistema lineal generado por GWT; estos coeficientes son usados para generar las matrices de las proporciones de mezcla. La carpeta `gypsum/` contiene módulos separados para la solución del flujo (`gwf.py`), la solución del transporte conservativo (un solo paso para generar las matrices) (`gwt.py`); la extracción de la información del sistema lineal con la API (`gwt_api.py`), la construcción de las matrices de las proporciones de mezcla (`lambdas.py`) y el cálculo del transporte reactivo (`wma.py`) y el análisis y visualización de resultados (`vis.py`).

 
* **03_RT_COM_gypsum_1D**.
    - `00_analytical_solution`. Comparación de la solución analítica (utilizando IA y de De Simoni, et.al. 2005) para el yeso con dos especies químicas, utilizando los datos generador por Jesús y Jordi, y el grupo de MMC, se usa el enfoque de componentes.
