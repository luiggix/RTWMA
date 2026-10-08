import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. PARÁMETROS DEL PROBLEMA
# ============================================================

L = 5.0  # longitud del dominio
u = 1.0  # velocidad de advección
D = 0.2  # coeficiente de difusión
km = 0.5  # coeficiente de transferencia en Robin

# Pulso inicial
C0 = 1.0
x0 = L / 3.0  # posición inicial del pulso
sigma0 = 0.15  # ancho inicial

# Tiempo
t_final = 1.5

# Para guardar soluciones
times_to_save = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50]


# ============================================================
# 2. PULSO INICIAL
# ============================================================


def initial_condition(x):
    """
    Pulso gaussiano centrado en x0.
    """
    return C0 * np.exp(-((x - x0) ** 2) / (2.0 * sigma0**2))


# ============================================================
# 3. SOLUCIÓN ANALÍTICA
# ============================================================


def analytical_solution(x, t):
    """
    Solución analítica para una gaussiana que se transporta
    y difunde en un dominio infinito:

        C_t + u C_x = D C_xx

    C(x,0) = C0 exp[-(x-x0)^2/(2 sigma0^2)]

    La solución es:

        C(x,t) =
            C0 * sigma0 / sigma(t)
            * exp[-(x-xc(t))^2/(2 sigma(t)^2)]

    donde

        xc(t) = x0 + u*t
        sigma(t)^2 = sigma0^2 + 2*D*t
    """

    sigma2 = sigma0**2 + 2.0 * D * t
    sigma = np.sqrt(sigma2)

    xc = x0 + u * t

    return C0 * sigma0 / sigma * np.exp(-((x - xc) ** 2) / (2.0 * sigma2))


# ============================================================
# 4. FLUJO INTERIOR
# ============================================================


def interior_flux(C_left, C_right, u, D, dx):
    """
    Flujo total:

        J = u C - D dC/dx

    Para la advección se utiliza upwind de primer orden.

    Para u > 0:

        C_upwind = C_left

    por lo que:

        J = u C_left
            - D (C_right - C_left)/dx
    """

    advective_flux = u * C_left

    diffusive_flux = -D * (C_right - C_left) / dx

    return advective_flux + diffusive_flux


# ============================================================
# 5. CONDICIÓN ROBIN EN x = 0
# ============================================================


def robin_boundary_flux(C1, u, D, dx, km):
    """
    Condición de frontera:

        J(0) = km (C_inf - Cb)

    En este ejemplo:

        C_inf = 0

    por lo que:

        J(0) = -km Cb

    El flujo físico en la frontera es:

        J(0) = u Cb - D dC/dx

    Aproximando el gradiente entre la frontera y
    el centro del primer volumen:

        dC/dx = (C1 - Cb)/(dx/2)

    entonces:

        J = u Cb
            - D (C1 - Cb)/(dx/2)

    Resolviendo para Cb:

        Cb =
        [2D/dx C1]
        --------------------------------
        [u + 2D/dx + km]

    """

    alpha = 2.0 * D / dx

    Cb = alpha * C1 / (u + alpha + km)

    J = -km * Cb

    return J


# ============================================================
# 6. FLUJO EN LA FRONTERA DERECHA
# ============================================================


def outlet_flux(C_last, u):
    """
    Condición de salida:

        dC/dx = 0

    Por tanto:

        J = u C

    """

    return u * C_last


# ============================================================
# 7. CÁLCULO DE TODOS LOS FLUJOS
# ============================================================


def calculate_fluxes(C, u, D, dx, km):
    """
    Calcula los flujos en todas las caras.

    Para N volúmenes:

        C[0] ... C[N-1]

    existen N+1 caras:

        J[0]     frontera izquierda
        J[1]     cara 1-2
        ...
        J[N-1]   cara N-1 - N
        J[N]     frontera derecha
    """

    N = len(C)

    J = np.zeros(N + 1)

    # --------------------------------------------------------
    # Frontera izquierda: Robin
    # --------------------------------------------------------

    J[0] = robin_boundary_flux(C[0], u, D, dx, km)

    # --------------------------------------------------------
    # Caras interiores
    # --------------------------------------------------------

    for i in range(N - 1):
        J[i + 1] = interior_flux(C[i], C[i + 1], u, D, dx)

    # --------------------------------------------------------
    # Frontera derecha: outflow
    # --------------------------------------------------------

    J[N] = outlet_flux(C[N - 1], u)

    return J


# ============================================================
# 8. UN PASO TEMPORAL
# ============================================================


def time_step(C, dt, dx, u, D, km):

    J = calculate_fluxes(C, u, D, dx, km)

    # Balance de masa:

    # dx dC_i/dt =
    #       J_{i-1/2} - J_{i+1/2}

    C_new = C - (dt / dx) * (J[1:] - J[:-1])

    return C_new


# ============================================================
# 9. SOLVER DE VOLÚMENES FINITOS
# ============================================================


def solve_fvm(N, t_final, u, D, km):

    dx = L / N

    # Centros de los volúmenes de control
    x = (np.arange(N) + 0.5) * dx

    # --------------------------------------------------------
    # Condición inicial
    # --------------------------------------------------------

    C = initial_condition(x)

    # --------------------------------------------------------
    # Paso de tiempo
    #
    # Condiciones aproximadas de estabilidad:
    #
    # CFL = u dt/dx <= 1
    #
    # difusión:
    #
    # D dt/dx^2 <= 1/2
    #
    # Usamos un factor de seguridad.
    # --------------------------------------------------------

    dt_adv = dx / u
    dt_diff = dx**2 / (2.0 * D)

    dt = 0.4 * min(dt_adv, dt_diff)

    # Número de pasos
    n_steps = int(np.ceil(t_final / dt))

    # Ajustamos dt para terminar exactamente
    # en t_final
    dt = t_final / n_steps

    # --------------------------------------------------------
    # Tiempos que queremos guardar
    # --------------------------------------------------------

    saved = {}

    t = 0.0

    # Guardamos condición inicial
    saved[0.0] = C.copy()

    next_save_index = 1

    # --------------------------------------------------------
    # Integración temporal
    # --------------------------------------------------------

    for n in range(n_steps):
        C = time_step(C, dt, dx, u, D, km)

        t += dt

        # Guardar solución si hemos alcanzado
        # uno de los tiempos solicitados
        while (
            next_save_index < len(times_to_save) and t >= times_to_save[next_save_index]
        ):
            saved[times_to_save[next_save_index]] = C.copy()

            next_save_index += 1

    return x, saved, dt


# ============================================================
# 10. ERROR L2
# ============================================================


def calculate_L2_error(x, C_num, t):

    C_exact = analytical_solution(x, t)

    dx = x[1] - x[0]

    error = np.sqrt(dx * np.sum((C_num - C_exact) ** 2))

    norm = np.sqrt(dx * np.sum(C_exact**2))

    relative_error = error / norm

    return error, relative_error


# ============================================================
# 11. SOLUCIÓN PRINCIPAL
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # Malla principal
    # --------------------------------------------------------

    N = 100

    x, solutions, dt = solve_fvm(N, t_final, u, D, km)

    print("=" * 60)
    print("SIMULACIÓN 1D DE ADVECCIÓN-DIFUSIÓN")
    print("=" * 60)

    print(f"L        = {L}")
    print(f"u        = {u}")
    print(f"D        = {D}")
    print(f"km       = {km}")
    print(f"N        = {N}")
    print(f"dx       = {L / N:.5f}")
    print(f"dt       = {dt:.6e}")
    print(f"x0       = {x0}")
    print(f"sigma0   = {sigma0}")
    print(f"t_final  = {t_final}")

    # --------------------------------------------------------
    # CFL
    # --------------------------------------------------------

    dx = L / N

    CFL = u * dt / dx
    Fo = D * dt / dx**2

    print()
    print(f"CFL = {CFL:.4f}")
    print(f"Fo  = {Fo:.4f}")

    # --------------------------------------------------------
    # Comparación numérica / analítica
    # --------------------------------------------------------

    print()
    print("Errores:")
    print("-" * 60)

    for t in times_to_save[:-1]:
        print(t)
        C_num = solutions[t]

        error, relative_error = calculate_L2_error(x, C_num, t)

        print(f"t = {t:5.2f}  L2 = {error:.6e}  error relativo = {relative_error:.6e}")

    # ========================================================
    # 12. GRÁFICA: EVOLUCIÓN DEL PULSO
    # ========================================================

    plt.figure(figsize=(10, 6))

    for t in times_to_save[:-1]:
        C_num = solutions[t]

        plt.plot(x, C_num, label=f"FV t={t:g}")

        C_exact = analytical_solution(x, t)

        plt.plot(x, C_exact, "--", linewidth=1.5, label=f"Analítica t={t:g}")

    plt.xlabel("x")
    plt.ylabel("Concentración C")
    plt.title("Advección-difusión: volúmenes finitos vs solución analítica")

    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()

    # ========================================================
    # 13. GRÁFICAS INDIVIDUALES
    # ========================================================

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)

    axes = axes.ravel()

    for ax, t in zip(axes, times_to_save):
        C_num = solutions[t]

        C_exact = analytical_solution(x, t)

        ax.plot(x, C_num, label="FV")

        ax.plot(x, C_exact, "--", label="Analítica")

        ax.set_title(f"t = {t:g}")

        ax.grid(True, alpha=0.3)

        ax.set_xlabel("x")
        ax.set_ylabel("C")

    axes[0].legend()

    fig.suptitle("Evolución del pulso")

    plt.tight_layout()
    plt.show()

    # ========================================================
    # 14. COMPARACIÓN DE MALLAS
    # ========================================================

    mesh_sizes = [25, 50, 100, 200]

    t_compare = 1.0

    plt.figure(figsize=(10, 6))

    print()
    print("=" * 60)
    print("CONVERGENCIA CON EL REFINAMIENTO DE MALLA")
    print("=" * 60)

    for Nmesh in mesh_sizes:
        x_mesh, sol_mesh, dt_mesh = solve_fvm(Nmesh, t_final, u, D, km)

        C_mesh = sol_mesh[t_compare]

        C_exact = analytical_solution(x_mesh, t_compare)

        error, relative_error = calculate_L2_error(x_mesh, C_mesh, t_compare)

        print(
            f"N = {Nmesh:4d}   "
            f"dx = {L / Nmesh:.5f}   "
            f"dt = {dt_mesh:.3e}   "
            f"L2 = {error:.5e}   "
            f"rel = {relative_error:.5e}"
        )

        plt.plot(x_mesh, C_mesh, label=f"FV N={Nmesh}")

    # Solución analítica sobre una malla muy fina
    x_fine = np.linspace(0, L, 1000)

    C_exact_fine = analytical_solution(x_fine, t_compare)

    plt.plot(x_fine, C_exact_fine, "k--", linewidth=2, label="Analítica")

    plt.xlabel("x")
    plt.ylabel("Concentración C")

    plt.title(f"Influencia del refinamiento de malla (t={t_compare})")

    plt.grid(True, alpha=0.3)

    plt.legend()
    plt.tight_layout()
    plt.show()

    # ========================================================
    # 15. POSICIÓN DEL MÁXIMO
    # ========================================================

    print()
    print("=" * 60)
    print("POSICIÓN Y AMPLITUD DEL PULSO")
    print("=" * 60)

    print(
        f"{'t':>8}"
        f"{'x_max FV':>15}"
        f"{'x_max analítica':>18}"
        f"{'C_max FV':>15}"
        f"{'C_max analítica':>18}"
    )

    for t in times_to_save[:-1]:
        C_num = solutions[t]

        i_max = np.argmax(C_num)

        x_max_num = x[i_max]
        C_max_num = C_num[i_max]

        xc_exact = x0 + u * t

        sigma2 = sigma0**2 + 2 * D * t

        C_max_exact = C0 * sigma0 / np.sqrt(sigma2)

        print(
            f"{t:8.2f}"
            f"{x_max_num:15.4f}"
            f"{xc_exact:18.4f}"
            f"{C_max_num:15.4f}"
            f"{C_max_exact:18.4f}"
        )
