# Ejemplos de simulación de flujo (**GWF**, **DFC**), transporte (**GWT**, **DFC**) y transporte reactivo (**WMA**).

* **01_GWF_DFC_WMA_Yeso**.
    - `1_GWF_DFC_WMA_Imp.py`. Solución del ejemplo del Yeso con transporte reactivo. El flujo se calcula con GWT, mientras que el transporte reactivo se calcula con WMA usando el programa REMIX. Para obtener las proporciones de mezcla se utilizan diferencias finitas centradas (DFC) para generar los coeficientes de la ecuación de transporte conservativa, en conjunto con un desarrollo en python para generar las matrices correspondientes (WMA_1D.py). La notebook `1_GWF_DFC_WMA_Imp.ipynb` además de resolver el mismo problema, contiene toda la explicación paso a paso de la solución. Se considera solo el caso **implícito** del WMA.
    - `2_GWF_API_DFC_WMA_Imp.py`, se resuelve el mismo problema del yeso, pero en este caso se utiliza la API de Modflow para resolver el flujo con GWF y obtener la solución de memoria.

* **02_GWF_GWT_API_WMA_Yeso**.
    - `1_GWF_GWT_API_WMA.py`. Se resuelve el problema del yeso descrito en **01_GWF_DFC_WMA_Yeso** pero en este caso se generan los coeficientes de la matriz del transporte conservativo usando GWT y la API; posteriormente se utiliza la API para extraer dichos coeficientes del sistema lineal generado por GWT; estos coeficientes son usados para generar las matrices de las proporciones de mezcla. La carpeta `gypsum/` contiene módulos separados para la solución del flujo (`gwf.py`), la solución del transporte conservativo (un solo paso para generar las matrices) (`gwt.py`); la extracción de la información del sistema lineal con la API (`api.py`), la construcción de las matrices de las proporciones de mezcla (`wma.py`) y el análisis y visualización de resultados (`plot.py`).
    - `vis.ipynb` notebook para realizar el análisis y visualización de resultados una vez que se han hecho los cálculos.
 
* **03_GWF_GWT_WMA_COMPONENTES_Yeso_Sol_Analitica**. Comparación de la solución analítica (utilizando IA y de De Simoni, et.al. 2005) para el yeso con dos especies químicas, utilizando los datos generador por Jesús y Jordi, y el grupo de MMC, se usa el enfoque de componentes.
