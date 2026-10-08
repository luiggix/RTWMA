import os
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import flopy


# ============================================================
# 1. PARÁMETROS FÍSICOS
# ============================================================

L = 5.0                  # longitud [m]
u = 1.0                  # velocidad [m/d]
D = 0.2                  # dispersión/difusión [m2/d]
km = 0.5                 # transferencia [m/d]

M = 1.0                  # masa del pulso
x0 = L / 3.0             # posición del pulso

t_final = 1.5


# ============================================================
# 2. DISCRETIZACIÓN
# ============================================================

N = 100

dx = L / N

x = (
    np.arange(N) + 0.5
) * dx


# ============================================================
# 3. PASO DE TIEMPO
# ============================================================

# Para transporte explícito:
#
# CFL = u dt/dx
#
# Fo = D dt/dx²
#
# MODFLOW 6 utilizará su solucionador implícito,
# pero mantenemos pasos relativamente pequeños para
# actualizar la frontera Robin.

dt_adv = dx / u
dt_diff = dx**2 / (2.0 * D)

dt = 0.25 * min(
    dt_adv,
    dt_diff
)

nsteps = int(
    np.ceil(t_final / dt)
)

dt = t_final / nsteps


# ============================================================
# 4. TIEMPOS DE SALIDA
# ============================================================

times_to_save = np.array([
    0.0,
    0.10,
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50
])


# ============================================================
# 5. CARPETA DE TRABAJO
# ============================================================

workspace = Path(
    "mf6_point_pulse"
)

if workspace.exists():
    shutil.rmtree(workspace)

workspace.mkdir(
    parents=True
)


# ============================================================
# 6. EJECUTABLE MODFLOW 6
# ============================================================

mf6_exe = "/Users/luiggi/GitSites/00_NEAT/RTWMA/bin/macosarm/mf6"


# ============================================================
# 7. SIMULACIÓN
# ============================================================

sim = flopy.mf6.MFSimulation(
    sim_name="point_pulse",
    version="mf6",
    exe_name=mf6_exe,
    sim_ws=str(workspace)
)


# ============================================================
# 8. TDIS
# ============================================================

# Un stress period por paso temporal.
#
# Esto permite modificar la concentración de la frontera
# Robin en cada paso.

perioddata = [
    (dt, 1, 1.0)
    for _ in range(nsteps)
]

tdis = flopy.mf6.ModflowTdis(
    sim,
    time_units="DAYS",
    nper=nsteps,
    perioddata=perioddata
)


# ============================================================
# 9. SOLVER IMS
# ============================================================

ims = flopy.mf6.ModflowIms(
    sim,
    print_option="SUMMARY",

    outer_dvclose=1e-8,
    outer_maximum=100,

    inner_dvclose=1e-8,
    inner_maximum=100,

    linear_acceleration="BICGSTAB"
)


# ============================================================
# 10. MODELO GWF
# ============================================================

#gwf = flopy.mf6.ModflowGwf(
#    sim,
#    modelname="gwf",
#    save_flows=True
#)

gwf = flopy.mf6.ModflowGwf(
    sim,
    modelname="gwf",
    model_nam_file="gwf.nam",
    model_rel_path="gwf",
    save_flows=True
)


# ============================================================
# 11. DISCRETIZACIÓN GWF
# ============================================================

dis_gwf = flopy.mf6.ModflowGwfdis(
    gwf,

    nlay=1,
    nrow=1,
    ncol=N,

    delr=dx,
    delc=1.0,

    top=1.0,
    botm=0.0
)


# ============================================================
# 12. CONDICIÓN INICIAL DE FLUJO
# ============================================================

ic_gwf = flopy.mf6.ModflowGwfic(
    gwf,
    strt=5.0
)


# ============================================================
# 13. PROPIEDADES HIDRÁULICAS
# ============================================================

# K = 1
#
# Con:
#
# h_left  = 5
# h_right = 0
#
# el gradiente hidráulico es:
#
# dh/dx = -1
#
# y por tanto:
#
# q = -K dh/dx = 1

npf = flopy.mf6.ModflowGwfnpf(
    gwf,

    icelltype=0,

    k=1.0,

    save_specific_discharge=True
)


# ============================================================
# 14. CONDICIONES DE CARGA
# ============================================================

# Izquierda:
# h = 5
#
# Derecha:
# h = 0

chd_data = {
    0: [
        ((0, 0, 0), 5.0),
        ((0, 0, N - 1), 0.0)
    ]
}

chd = flopy.mf6.ModflowGwfchd(
    gwf,
    stress_period_data=chd_data,
    save_flows=True
)


# ============================================================
# 15. STORAGE GWF
# ============================================================

sto = flopy.mf6.ModflowGwfsto(
    gwf,

    iconvert=0,

    ss=1.0e-6,

    sy=0.2,

    steady_state={
        0: True
    },

    transient={
        1: True
    },

    save_flows=True
)


# ============================================================
# 16. OUTPUT GWF
# ============================================================

oc_gwf = flopy.mf6.ModflowGwfoc(
    gwf,

    head_filerecord="gwf.hds",

    budget_filerecord="gwf.bud",

    saverecord=[
        ("HEAD", "ALL"),
        ("BUDGET", "ALL")
    ]
)


# ============================================================
# 17. MODELO GWT
# ============================================================

#gwt = flopy.mf6.ModflowGwt(
#    sim,
#    modelname="gwt"
#)

gwt = flopy.mf6.ModflowGwt(
    sim,
    modelname="gwt",
    model_nam_file="gwt.nam",
    model_rel_path="gwt",
)

# Registrar IMS para GWT
sim.register_ims_package(
    ims,
    [gwt.name]
)


# ============================================================
# 18. DISCRETIZACIÓN GWT
# ============================================================

dis_gwt = flopy.mf6.ModflowGwtdis(
    gwt,

    nlay=1,
    nrow=1,
    ncol=N,

    delr=dx,
    delc=1.0,

    top=1.0,
    botm=0.0
)


# ============================================================
# 19. CONDICIÓN INICIAL: PULSO PUNTUAL
# ============================================================

C_initial = np.zeros(N)

i0 = np.argmin(
    np.abs(x - x0)
)

C_initial[i0] = M / dx


ic_gwt = flopy.mf6.ModflowGwtic(
    gwt,
    strt=C_initial
)


# ============================================================
# 20. ADVECCIÓN
# ============================================================

adv = flopy.mf6.ModflowGwtadv(
    gwt,

    scheme="UPSTREAM"
)


# ============================================================
# 21. DISPERSIÓN
# ============================================================

dsp = flopy.mf6.ModflowGwtdsp(
    gwt,

    xt3d_off=True,

    alh=D,
    ath1=D,
    atv=D
)


# ============================================================
# 22. MASS STORAGE
# ============================================================

mst = flopy.mf6.ModflowGwtmst(
    gwt,

    porosity=1.0
)


# ============================================================
# 23. CONDICIÓN ROBIN
# ============================================================

# Aquí está la parte importante.
#
# En cada paso:
#
#     J = km(C_inf - Cb)
#
# y
#
#     J = u Cb
#         - D (C1-Cb)/(dx/2)
#
# de donde:
#
# Cb =
#
#     alpha*C1 + km*Cinf
#     -------------------
#     u + alpha + km
#
# con
#
# alpha = 2D/dx
#
# Como MODFLOW 6 CNC fija la concentración de la celda
# de frontera, actualizaremos Cb antes de cada paso.
#
# El primer paso utiliza C1(t=0).


def robin_concentration(
    C1,
    u,
    D,
    dx,
    km,
    Cinf=0.0
):

    alpha = (
        2.0 * D / dx
    )

    return (
        alpha * C1
        +
        km * Cinf
    ) / (
        u
        + alpha
        + km
    )


# Valor inicial de la frontera
Cb_initial = robin_concentration(
    C_initial[0],
    u,
    D,
    dx,
    km,
    0.0
)


# ============================================================
# 24. CNC
# ============================================================

cnc_data = {
    0: [
        ((0, 0, 0), Cb_initial)
    ]
}

cnc = flopy.mf6.ModflowGwtcnc(
    gwt,

    maxbound=1,

    stress_period_data=cnc_data,

    save_flows=True
)


# ============================================================
# 25. FMI
# ============================================================

# GWT obtiene los flujos del modelo GWF.

fmi = flopy.mf6.ModflowGwtfmi(
    gwt,

    flow_imbalance_correction=True,

    packagedata=[
        ("GWFHEAD", "./mf6_point_pulse/gwf/gwf.hds"),
        ("GWFBUDGET", "./mf6_point_pulse/gwf/gwf.bud")
    ]
)


# ============================================================
# 26. OUTPUT GWT
# ============================================================

oc_gwt = flopy.mf6.ModflowGwtoc(
    gwt,

    concentration_filerecord="gwt.ucn",

    budget_filerecord="gwt.bud",

    saverecord=[
        ("CONCENTRATION", "ALL"),
        ("BUDGET", "ALL")
    ]
)


# ============================================================
# 27. ESCRIBIR ARCHIVOS
# ============================================================

sim.write_simulation()


# ============================================================
# 28. EJECUTAR MODFLOW 6
# ============================================================

print()
print("Ejecutando MODFLOW 6...")

success, buff = sim.run_simulation(
    silent=False
)

if not success:

    print("\n".join(buff))

    raise RuntimeError(
        "MODFLOW 6 no terminó correctamente."
    )


print(
    "MODFLOW 6 terminó correctamente."
)