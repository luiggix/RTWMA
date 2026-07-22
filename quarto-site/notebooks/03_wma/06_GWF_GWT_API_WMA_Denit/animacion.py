import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from denit import utils
plt.style.use('ggplot')

nspec = 8

#
# Leer información de "u_init.dat"
with open("remix/u_init.dat", "r") as f:
    spec_info = f.readlines()
key_string = "Aqueous variable activity species:"
isp = utils.find_index(spec_info, key_string, 1)
spec_names = [l.split(": ")[1].strip() for l in spec_info[isp:isp+nspec]]
print(spec_names)

with open("final_species.dat", "r") as f:
    datos = f.readlines()
      
steps = len(datos) // nspec

#
# Leer información del excel
denit_excel = pd.read_excel("remix/denit_2reacts_corregido.xlsx", 
                            sheet_name = "results", usecols=range(0,11), 
                            skiprows=4, header=None)
spe_names = [c.strip() for c in denit_excel.iloc[2:10,0]]
idx = 2
skip = 15
times = ["0.00d", "0.01d", "0.02d", "0.10d", "1.00d", "5.00d"]
ranges = [(idx + i*skip, idx + i*skip + nspec) for i in range(0, 6)]
print(ranges)

s_excel = []
for it in range(0,6):
    iini = ranges[it][0]
    ifin = ranges[it][1]
    s_excel.append(denit_excel.iloc[iini:ifin, 1:].to_numpy(dtype=float))

#
#
# 
s = int(input("Specie: "))
print(f"{spec_names[s]}")
cmax = max([float(l) for l in datos[s].split()][1:])
cmin = min([float(l) for l in datos[s].split()][1:])
for ts in range(1, steps):
    temp_max = max([float(l) for l in datos[s + ts * nspec].split()][1:])
    temp_min = min([float(l) for l in datos[s + ts * nspec].split()][1:])
    cmax = cmax if cmax > temp_max else temp_max
    cmin = cmin if cmin < temp_min else temp_min

ofs = (cmax - cmin)*0.10

x = np.linspace(0, 1.0, 10)

#
# Paso 1. Definición de la figura
#
fig = plt.figure(figsize=(5,3))           # Figura
ax = plt.axes(xlim=(-0.01, 1.01), ylim=(cmin-ofs, cmax+ofs)) # Ejes
ax.set_title(f"Specie: {spec_names[s]}, t = {0:4.2f}d")
ax.set_yscale("linear")
#
# Paso 2. Graficación del primer estado de la curva
#
for i, t in enumerate(times):
    ax.plot(x, s_excel[i][s], lw = 1.0, label=f"{times[i]}", zorder = 5)
    
l = plt.plot(x, [float(l) for l in datos[s].split()][1:], ".-k",)
#
# Paso 3. Definición de una función para actualizar los datos 
#
def plotLine(i, linea, datos):
    linea.set_ydata([float(l) for l in datos[s + i * nspec].split()][1:]) # cambia los datos en la dirección y
    ax.set_title(f"Specie: {spec_names[s]}, t = {float(datos[s + i * nspec].split()[0]):4.2f}")

#
# Paso 4. Uso de la función FuncAnimation() para crear la animación
#
anim = FuncAnimation(fig,             # Figura
                     plotLine,        # Función que cambia los datos
                     fargs=(l[0], datos), # Argumentos de la funcion plotLine()
                     interval=10,    # Intervalo entre cuadros [ms]
                     frames=steps,       # Total de cuadros
                     repeat=True)     # Animación en un ciclo

plt.show()