import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import brentq
from scipy.integrate import simpson


# ============================================================
# 1. PARÁMETROS DEL PROBLEMA
# ============================================================

L = 5.0
u = 1.0
D = 0.2
km = 0.5

# Concentración de fondo
# Para este ejemplo:
# C_inf = 0
#
# Esto permite comparar directamente el pulso.
C_inf = 0.0


# ============================================================
# 2. PULSO INICIAL
# ============================================================

C0 = 1.0

# El pulso comienza en un tercio del dominio
x0 = L / 3.0

sigma0 = 0.15


def initial_condition(x):

    return C0 * np.exp(
        -(x - x0)**2
        / (2.0 * sigma0**2)
    )


# ============================================================
# 3. TIEMPO
# ============================================================

t_final = 1.5

times_to_save = np.array([
    0.0,
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50
])


# ============================================================
# 4. SOLUCIÓN ANALÍTICA DE DOMINIO FINITO
# ============================================================

class FiniteDomainAnalyticalSolution:

    def __init__(
        self,
        L,
        u,
        D,
        km,
        initial_condition,
        n_modes=1000
    ):

        self.L = L
        self.u = u
        self.D = D
        self.km = km

        self.beta = u / (2.0 * D)

        self.h0 = (
            u / 2.0 + km
        ) / D

        self.hL = (
            u / (2.0 * D)
        )

        self.n_modes = n_modes

        # ----------------------------------------------------
        # Encontrar eigenvalues k_n
        # ----------------------------------------------------

        self.k = self.find_eigenvalues(
            n_modes
        )

        # ----------------------------------------------------
        # Eigenvalues temporales
        # ----------------------------------------------------

        self.lambda_n = (
            D * self.k**2
            + u**2 / (4.0 * D)
        )

        # ----------------------------------------------------
        # Calcular coeficientes A_n
        # ----------------------------------------------------

        self.A = self.calculate_coefficients(
            initial_condition
        )


    # ========================================================
    # ECUACIÓN TRASCENDENTAL
    # ========================================================

    def eigenvalue_function(self, k):

        h0 = self.h0
        hL = self.hL
        L = self.L

        return (
            (h0 + hL)
            * k
            * np.cos(k * L)
            +
            (h0 * hL - k**2)
            * np.sin(k * L)
        )


    # ========================================================
    # BÚSQUEDA DE LOS k_n
    # ========================================================

    def find_eigenvalues(self, n_modes):

        # Los eigenvalues están aproximadamente
        # alrededor de n*pi/L.

        # Usamos una malla suficientemente fina
        # para localizar cambios de signo.

        k_max = (
            n_modes + 2
        ) * np.pi / self.L

        # Resolución elevada para no perder raíces
        n_scan = max(
            200000,
            1000 * n_modes
        )

        k_values = np.linspace(
            1e-10,
            k_max,
            n_scan
        )

        f_values = np.array([
            self.eigenvalue_function(k)
            for k in k_values
        ])

        roots = []

        for i in range(
            len(k_values) - 1
        ):

            f1 = f_values[i]
            f2 = f_values[i + 1]

            if f1 * f2 < 0.0:

                root = brentq(
                    self.eigenvalue_function,
                    k_values[i],
                    k_values[i + 1]
                )

                # Evitar duplicados
                if (
                    len(roots) == 0
                    or abs(root - roots[-1]) > 1e-8
                ):
                    roots.append(root)

            if len(roots) >= n_modes:
                break

        roots = np.array(roots)

        if len(roots) < n_modes:

            raise RuntimeError(
                f"Solo se encontraron "
                f"{len(roots)} eigenvalues "
                f"de {n_modes} solicitados."
            )

        return roots


    # ========================================================
    # EIGENFUNCIONES
    # ========================================================

    def eigenfunction(
        self,
        x,
        k
    ):

        return (
            np.cos(k * x)
            +
            self.h0 / k
            * np.sin(k * x)
        )


    # ========================================================
    # COEFICIENTES A_n
    # ========================================================

    def calculate_coefficients(
        self,
        initial_condition
    ):

        # Malla fina para las integrales
        x_int = np.linspace(
            0.0,
            self.L,
            20000
        )

        C_initial = initial_condition(
            x_int
        )

        # phi(x,0)
        #
        # C = exp(beta*x) phi
        #
        # phi = exp(-beta*x) C

        phi_initial = (
            np.exp(-self.beta * x_int)
            * C_initial
        )

        A = np.zeros(
            len(self.k)
        )

        for n, k in enumerate(
            self.k
        ):

            X = self.eigenfunction(
                x_int,
                k
            )

            numerator = simpson(
                phi_initial * X,
                x=x_int
            )

            denominator = simpson(
                X**2,
                x=x_int
            )

            A[n] = (
                numerator
                / denominator
            )

        return A


    # ========================================================
    # SOLUCIÓN ANALÍTICA
    # ========================================================

    def __call__(
        self,
        x,
        t
    ):

        x = np.asarray(x)

        # Matriz:
        #
        # filas -> posiciones x
        # columnas -> modos

        X = np.zeros(
            (len(x), len(self.k))
        )

        for n, k in enumerate(
            self.k
        ):

            X[:, n] = self.eigenfunction(
                x,
                k
            )

        temporal = (
            self.A
            * np.exp(
                -self.lambda_n * t
            )
        )

        phi = X @ temporal

        C = (
            np.exp(self.beta * x)
            * phi
        )

        return C


# ============================================================
# 5. FLUJO INTERIOR FV
# ============================================================

def interior_flux(
    C_left,
    C_right,
    u,
    D,
    dx
):

    # Advección upwind
    J_adv = u * C_left

    # Difusión centrada
    J_diff = (
        -D
        * (C_right - C_left)
        / dx
    )

    return J_adv + J_diff


# ============================================================
# 6. FLUJO ROBIN
# ============================================================

def robin_boundary_flux(
    C1,
    u,
    D,
    dx,
    km,
    C_inf=0.0
):

    # Distancia entre frontera y centro
    # del primer volumen:
    #
    # dx/2

    alpha = (
        2.0 * D / dx
    )

    # Flujo:
    #
    # J = u Cb
    #     - alpha(C1-Cb)
    #
    # y
    #
    # J = km(Cinf-Cb)

    Cb = (
        alpha * C1
        +
        km * C_inf
    ) / (
        u
        + alpha
        + km
    )

    J = km * (
        C_inf - Cb
    )

    return J


# ============================================================
# 7. FLUJO DE SALIDA
# ============================================================

def outlet_flux(
    C_last,
    u
):

    # dC/dx = 0
    #
    # J = u C

    return u * C_last


# ============================================================
# 8. TODOS LOS FLUJOS
# ============================================================

def calculate_fluxes(
    C,
    u,
    D,
    dx,
    km,
    C_inf=0.0
):

    N = len(C)

    J = np.zeros(
        N + 1
    )

    # --------------------------------------------------------
    # Frontera izquierda
    # --------------------------------------------------------

    J[0] = robin_boundary_flux(
        C[0],
        u,
        D,
        dx,
        km,
        C_inf
    )

    # --------------------------------------------------------
    # Caras interiores
    # --------------------------------------------------------

    for i in range(N - 1):

        J[i + 1] = interior_flux(
            C[i],
            C[i + 1],
            u,
            D,
            dx
        )

    # --------------------------------------------------------
    # Frontera derecha
    # --------------------------------------------------------

    J[N] = outlet_flux(
        C[N - 1],
        u
    )

    return J


# ============================================================
# 9. PASO TEMPORAL FV
# ============================================================

def time_step(
    C,
    dt,
    dx,
    u,
    D,
    km,
    C_inf=0.0
):

    J = calculate_fluxes(
        C,
        u,
        D,
        dx,
        km,
        C_inf
    )

    C_new = (
        C
        -
        dt / dx
        * (
            J[1:]
            -
            J[:-1]
        )
    )

    return C_new


# ============================================================
# 10. SOLVER FV
# ============================================================

def solve_fvm(
    N,
    t_final,
    u,
    D,
    km,
    C_inf=0.0
):

    dx = L / N

    # Centros de volumen
    x = (
        np.arange(N) + 0.5
    ) * dx

    # Condición inicial
    C = initial_condition(x)

    # --------------------------------------------------------
    # Paso temporal
    # --------------------------------------------------------

    dt_adv = dx / abs(u)

    dt_diff = (
        dx**2
        / (2.0 * D)
    )

    # Factor de seguridad
    dt = (
        0.4
        * min(
            dt_adv,
            dt_diff
        )
    )

    n_steps = int(
        np.ceil(
            t_final / dt
        )
    )

    dt = (
        t_final
        / n_steps
    )

    # --------------------------------------------------------
    # Soluciones
    # --------------------------------------------------------

    solutions = {}

    solutions[0.0] = C.copy()

    t = 0.0

    next_save = 1

    for n in range(
        n_steps
    ):

        C = time_step(
            C,
            dt,
            dx,
            u,
            D,
            km,
            C_inf
        )

        t += dt

        while (
            next_save
            < len(times_to_save)
            and
            t >= times_to_save[next_save]
        ):

            solutions[
                times_to_save[next_save]
            ] = C.copy()

            next_save += 1

    return (
        x,
        solutions,
        dt
    )


# ============================================================
# 11. ERROR L2
# ============================================================

def calculate_L2_error(
    x,
    C_num,
    C_exact
):

    dx = (
        x[1] - x[0]
    )

    error = np.sqrt(
        dx
        * np.sum(
            (
                C_num
                -
                C_exact
            )**2
        )
    )

    norm = np.sqrt(
        dx
        * np.sum(
            C_exact**2
        )
    )

    relative_error = (
        error / norm
    )

    return (
        error,
        relative_error
    )


# ============================================================
# 12. EJECUCIÓN PRINCIPAL
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Malla FV
    # --------------------------------------------------------

    N = 100

    # --------------------------------------------------------
    # Solución analítica
    #
    # Usamos bastantes modos para representar
    # correctamente el pulso inicial.
    # --------------------------------------------------------

    print(
        "Construyendo solución analítica..."
    )

    analytical = (
        FiniteDomainAnalyticalSolution(
            L=L,
            u=u,
            D=D,
            km=km,
            initial_condition=initial_condition,
            n_modes=500
        )
    )

    print(
        f"Se utilizaron "
        f"{len(analytical.k)} modos."
    )

    print()
    print("Primeros eigenvalues k_n:")

    for i, k in enumerate(
        analytical.k[:10]
    ):

        print(
            f"k[{i}] = {k:.8f}"
        )


    # --------------------------------------------------------
    # Solución FV
    # --------------------------------------------------------

    x, solutions, dt = solve_fvm(
        N,
        t_final,
        u,
        D,
        km,
        C_inf
    )

    dx = L / N

    CFL = (
        u * dt / dx
    )

    Fo = (
        D * dt / dx**2
    )


    # ========================================================
    # INFORMACIÓN
    # ========================================================

    print()
    print("=" * 70)
    print("PROBLEMA TRANSITORIO 1D")
    print("=" * 70)

    print(
        f"L       = {L}"
    )

    print(
        f"u       = {u}"
    )

    print(
        f"D       = {D}"
    )

    print(
        f"km      = {km}"
    )

    print(
        f"N       = {N}"
    )

    print(
        f"dx      = {dx}"
    )

    print(
        f"dt      = {dt:.6e}"
    )

    print(
        f"CFL     = {CFL:.5f}"
    )

    print(
        f"Fo      = {Fo:.5f}"
    )

    print(
        f"x0      = {x0}"
    )

    print(
        f"sigma0  = {sigma0}"
    )


    # ========================================================
    # 13. COMPARACIÓN NUMÉRICA VS ANALÍTICA
    # ========================================================

    print()
    print("=" * 70)
    print("ERROR DE LA SOLUCIÓN DE VOLÚMENES FINITOS")
    print("=" * 70)

    print(
        f"{'t':>8}"
        f"{'L2':>18}"
        f"{'Error relativo':>20}"
    )

    for t in times_to_save[:-1]:

        C_num = solutions[t]

        C_exact = analytical(
            x,
            t
        )

        error, relative_error = (
            calculate_L2_error(
                x,
                C_num,
                C_exact
            )
        )

        print(
            f"{t:8.2f}"
            f"{error:18.6e}"
            f"{relative_error:20.6e}"
        )


    # ========================================================
    # 14. COMPARACIÓN GRÁFICA
    # ========================================================

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(15, 7),
        sharex=True,
        sharey=True
    )

    axes = axes.ravel()

    for ax, t in zip(
        axes,
        times_to_save[1:-1]
    ):

        C_num = solutions[t]

        C_exact = analytical(
            x,
            t
        )

        ax.plot(
            x,
            C_num,
            label="FV"
        )

        ax.plot(
            x,
            C_exact,
            "--",
            label="Analítica"
        )

        ax.set_title(
            f"t = {t:.2f}"
        )

        ax.set_xlabel("x")
        ax.set_ylabel("C")

        ax.grid(
            True,
            alpha=0.3
        )

    # El último subplot no se utiliza
    axes[-1].axis("off")

    axes[0].legend()

    fig.suptitle(
        "Advección-difusión en dominio finito\n"
        "Volúmenes finitos vs solución analítica"
    )

    plt.tight_layout()
    plt.show()


    # ========================================================
    # 15. COMPARACIÓN EN UN TIEMPO PARTICULAR
    # ========================================================

    t_compare = 1.0

    C_num = solutions[
        t_compare
    ]

    C_exact = analytical(
        x,
        t_compare
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        x,
        C_num,
        "o-",
        markersize=3,
        label="Volúmenes finitos"
    )

    plt.plot(
        x,
        C_exact,
        "--",
        linewidth=2,
        label="Analítica dominio finito"
    )

    plt.xlabel("x")
    plt.ylabel("Concentración C")

    plt.title(
        f"Comparación a t = {t_compare}"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()
    plt.show()


    # ========================================================
    # 16. COMPARACIÓN DE MALLAS
    # ========================================================

    mesh_sizes = [
        25,
        50,
        100,
        200
    ]

    print()
    print("=" * 70)
    print("CONVERGENCIA CON EL REFINAMIENTO")
    print("=" * 70)

    print(
        f"{'N':>8}"
        f"{'dx':>12}"
        f"{'L2':>18}"
        f"{'Error relativo':>20}"
    )

    plt.figure(
        figsize=(10, 6)
    )

    for Nmesh in mesh_sizes:

        x_mesh, sol_mesh, dt_mesh = (
            solve_fvm(
                Nmesh,
                t_final,
                u,
                D,
                km,
                C_inf
            )
        )

        C_mesh = sol_mesh[
            t_compare
        ]

        C_exact = analytical(
            x_mesh,
            t_compare
        )

        error, relative_error = (
            calculate_L2_error(
                x_mesh,
                C_mesh,
                C_exact
            )
        )

        print(
            f"{Nmesh:8d}"
            f"{L/Nmesh:12.5f}"
            f"{error:18.6e}"
            f"{relative_error:20.6e}"
        )

        plt.plot(
            x_mesh,
            C_mesh,
            label=f"FV N={Nmesh}"
        )


    # Analítica sobre malla fina
    x_fine = np.linspace(
        0,
        L,
        1000
    )

    C_exact_fine = analytical(
        x_fine,
        t_compare
    )

    plt.plot(
        x_fine,
        C_exact_fine,
        "k--",
        linewidth=2,
        label="Analítica"
    )

    plt.xlabel("x")
    plt.ylabel("C")

    plt.title(
        f"Convergencia de la solución "
        f"(t={t_compare})"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()
    plt.show()


    # ========================================================
    # 17. VERIFICACIÓN DE LAS CONDICIONES DE FRONTERA
    # ========================================================

    print()
    print("=" * 70)
    print("VERIFICACIÓN DE LA SOLUCIÓN ANALÍTICA")
    print("=" * 70)

    print()
    print(
        "Parámetros de la transformación:"
    )

    print(
        f"beta = {analytical.beta:.6f}"
    )

    print(
        f"h0   = {analytical.h0:.6f}"
    )

    print(
        f"hL   = {analytical.hL:.6f}"
    )


    # ========================================================
    # 18. MASA TOTAL
    # ========================================================

    print()
    print("=" * 70)
    print("MASA TOTAL")
    print("=" * 70)

    for t in times_to_save:

        C_num = solutions[t]

        C_exact = analytical(
            x,
            t
        )

        mass_num = np.sum(
            C_num
        ) * dx

        mass_exact = np.trapz(
            C_exact,
            x
        )

        print(
            f"t={t:5.2f}   "
            f"M_FV={mass_num:.8e}   "
            f"M_analítica={mass_exact:.8e}"
        )