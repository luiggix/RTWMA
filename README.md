# Soluciones basadas en la naturaleza a la contaminación del agua subterránea por especies de nitrógeno (SECIHTI: CF-2023-G-904)

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![NumPy Badge](https://img.shields.io/badge/numpy-2.4.6-013243?logo=numpy&logoColor=white)](https://numpy.org/devdocs/release/2.4.6-notes.html)
[![Matplotlib Badge](https://custom-icon-badges.demolab.com/badge/Matplotlib-3.11.1-71D291?logo=matplotlib&logoColor=fff)](https://matplotlib.org)
[![Pandas](https://img.shields.io/badge/pandas-2.3.3-%23150458?style=flat&logo=pandas&logoColor=white)](https://pypi.org/project/pandas/2.3.3/)
[![Flopy Badge](https://custom-icon-badges.demolab.com/badge/Flopy-3.10.0-ffb338?logo=flopy&logoColor=fff)](https://github.com/modflowpy/flopy)
[![XMF6 Badge](https://custom-icon-badges.demolab.com/badge/xmf6-0.1.0-1E90FF?logo=flopy&logoColor=fff)](https://github.com/luiggix/xmf6)

**Responsable**.

* Dra. Graciela Herrera Zamarrón, IGEF-UNAM.

## Descripción.
La contaminación de acuíferos por nitrato y amonio constituye un problema ambiental y de salud
pública de alta relevancia en México y a escala global. Este proyecto, financiado por la SECIHTI a
través del programa Ciencia de Frontera (CF-2023-G-904), tiene como objetivo desarrollar soluciones
basadas en la naturaleza para la remoción *in situ* de especies de nitrógeno en aguas subterráneas,
mediante el aprovechamiento de procesos hidrogeoquímicos naturales. Para alcanzar este objetivo
se conformó un equipo interdisciplinario e interinstitucional de carácter binacional que investiga los
mecanismos de transformación del amonio y nitrato en el subsuelo, evalúa métodos de remoción
mediante experimentación y modelación matemática, y desarrolla herramientas computacionales
para la simulación del transporte reactivo en medios porosos. 

En este repositorio se depositan todas las herramientas de software así como los ejemplos para la
simulación de la remoción de nitrato y amonio en diferentes sistemas acuíferos de interés. Los ejemplos de transporte reacivo se desarrollan usando el enfoque de mezcla de aguas ([WMA](https://doi.org/10.1016/j.advwatres.2022.104131) por sus siglas en inglés).

# Organización de repositorio.

**Responsable del repositorio y organización del software.**

* Dr. Luis Miguel de la Cruz Salas, IGEF-UNAM.

## Colaboradores académicos

* Dr. Jesús Carrera Ramírez, IDAEA-CSIC
* Dr. Iván Contreras Trejo, IGF-UNAM
* Dr. Guillermo de Jesús Hernández García, IGF-UNAM
* Dr. Mario Hernández Hernández, IGF-UNAM
* Dr. Eric Morales Casique, IGl-UNAM
* Dr. Leonardo Teja Juárez, FI-UNAM
* Dr. Norberto Carmen Vera Guzmán, IGF-UNAM

## Estudiantes.
* Jordi Petchamé Guerrero, Doctorado en Ingeniería del Terreno, Multiscale Characterization Of
GroUndwater and Solutes discharge into The Ocean (MuchoGusto), Instituto de Diagnóstico
Ambiental y Estudios del Agua (IDAEA-CSIC), España.

* José Armando Torres López, Maestría en Ingeniería Exploración y Explotación de Recursos
Naturales, Programa de Maestría y Doctorado en Ingeniería, UNAM.

## Estructura del repositorio.

El repositorio ha sido configurado usando la herramienta [uv](https://docs.astral.sh/uv/) la cual permite configurar y reproducir el ambiente del proyecto 
de una manera fácil y eficiente. El desarrollo de los ejemplos se basa en [MODFLOW 6](https://www.usgs.gov/software/modflow-6-usgs-modular-hydrologic-model), 
[FloPy](https://github.com/modflowpy/flopy) y códigos propios desarrollados en Python y Fortran. La estructura de directorios del repositorio es como sigue:


* **benchmarks**. Contiene ejemplos desarrollados por todo el grupo de trabajo.
* **bin**. Contiene los ejecutables de MODFLOW 6 para diferentes sistemas operativos.
	- **linux**. Sistema operativo Linux.
	- **macos**. Sistema operativo MacOS, procesador Intel.
	- **macosarm**.  Sistema operativo MacOS, procesador ARM64.
	- **windows**. Sistema operativo Windows 64 bits.
	- **doc**. Documentación de MODFLOW 6.
* **src**. Fuentes de código común para todos los ejemplos.
* **README.md**. Este archivo.
* **LICENSE.txt**. Licencia de uso del software.
* **pyproject.toml**. Archivo de configuración del proyecto.
* **uv.sync**. Archivo para reproducir el ambiente del proyecto en una computadora local.

Los ejemplos desarrollados se encuentran en la carpeta **benchmarks**. Se ha definido una notación para los nombres de las subcarpetas y de los archivos de los ejemplos, la cual permite identificar de que se trata cada ejemplo, y se describe en la Tabla 1.

*Tabla 1. Descripción de la notación de los nombres de archivos.*

|Cadena|Descripción|
|---|---|
|`CT`| Ejemplo de transporte conservativo (*Conservative Transport*)|
|`RT`| Ejemplo de transporte reactivo (*Reactive Transport*)|
|`WMA`| Ejemplo de transporte reactivo resuelto con el enfoque de mezcla de aguas.|
|`COM`| Ejemplo de transporte reactivo resuelto con el enfoque de componentes.|
|`GWF`| Ejemplo de flujo resuelto con GWF de MODFLOW 6.|
|`GWT`| Ejemplo de transporte resuelto con GWT de MODFLOW 6.|
|`EXCH`| Ejemplo de flujo y transporte que utiliza un intercambio GWFGWT de MODFLOW 6.|
|`API`| Ejemplo que utiliza la API de MODFLOW 6.|
|`DFC`| Ejemplos que resuelve flujo y/o transporte usando un software basado en Diferencias Finitas Centrales escrito en Python.|

### Descripción de los ejemplos.

En la carpeta **benchmarks** se tiene la siguiente estructura de archivos:

* **figures**. Figuras compartidas por todos los ejemplos.
* **Desciption.md**. Descripción corta de cada ejemplo.
* **env.json**. Configuración de variables de ambiente locales.
* **00_RT_EXE**. Archivos ejecutables para realizar la simulación de transporte reactivo con el enfoque de mezclas y con el enfoque de componentes.


Los carpetas **01_...** en adelante contienen cada una de ellas un ejemplo. Dentro de estas carpetas se tiene también una estructura que permite organizar el código de una manera clara. Esta estructura se describe en la tabla 2.

*Tabla 2. Descripción de las subcarpetas de cada ejemplo.*

|Subcarpeta|Descripción|
|---|---|
|**io_mf6**| Archivos de entrada y salida de MODFLOW 6. |
|**nr_analysis**| Archivos de resultados para un análisis posterior y visualización.|
|**rt_workingDir**| Directorio de trabajo para la simulación de transporte reactivo. Contiene entre otras cosas bases de datos químicas, archivos de entrada y de salida.|
|**src_**| Códigos en Python (`.py`) que permiten separar la implementación en módulos para una mejor organización. El nombre se completa con una palabra del ejemplo en cuestión (p ej. **src_gypsum**).|


**Nota.** Los ejemplos pueden contener las cuatro carpetas mencionadas en la tabla 2 o un subconjunto de ellas, dependiendo de las necesidades de cada ejemplo.

Los ejemplos dentro de cada carpeta se organizan en jupyter notebooks (`.ipynb`) o archivos de Python (`.py`) que se ejecutan en línea de comandos. 

# Enlaces a otros repositorios.

El código fuente del software que realiza la simulación de transporte reactivo se puede obtener del siguiente repositorio (mediante solicitud a los autores):

* [REMIX](https://github.com/jordipg10/interfaz_remix)
