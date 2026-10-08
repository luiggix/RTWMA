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

# Masa total del pulso puntual
M = 1.0

# Posición del pulso
x0 = L / 3.0

# Tiempo final
t_final = 1.5

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
# 2. SOLUCIÓN INICIAL PUNTUAL
# ============================================================

def initial_point_pulse(x, dx):
    """
    Representación FV de:

        C(x,0) = M delta(x-x0)

    La masa total es M.

    La celda que contiene x0 recibe:

        C = M/dx
    """

    C = np.zeros_like(x)

    i0 = np.argmin(
        np.abs(x - x0)
    )

    C[i0] = M / dx

    return C


# ============================================================
# 3. SOLUCIÓN ANALÍTICA DE DOMINIO FINITO
#
#    Green's function para:
#
#       C_t + u C_x = D C_xx
#
#    BC izquierda:
#
#       u C - D C_x = -km C
#
#    BC derecha:
#
#       C_x = 0
#
# ============================================================

class FiniteDomainPointPulse:

    def __init__(
        self,
        L,
        u,
        D,
        km,
        x0,
        M=1.0,
        n_modes=500
    ):

        self.L = L
        self.u = u
        self.D = D
        self.km = km
        self.x0 = x0
        self.M = M

        # ----------------------------------------------------
        # Transformación:
        #
        # C = exp(beta*x) phi
        # ----------------------------------------------------

        self.beta = (
            u / (2.0 * D)
        )

        # ----------------------------------------------------
        # Condiciones de frontera transformadas
        # ----------------------------------------------------

        self.h0 = (
            u / 2.0 + km
        ) / D

        self.hL = (
            u / (2.0 * D)
        )

        self.n_modes = n_modes

        # ----------------------------------------------------
        # Eigenvalues espaciales
        # ----------------------------------------------------

        self.k = self.find_eigenvalues(
            n_modes
        )

        # ----------------------------------------------------
        # Eigenvalues temporales
        # ----------------------------------------------------

        self.lambda_n = (
            D * self.k**2
            +
            u**2 / (4.0 * D)
        )

        # ----------------------------------------------------
        # Coeficientes de la delta
        # ----------------------------------------------------

        self.A = (
            self.calculate_point_coefficients()
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
    # ENCONTRAR k_n
    # ========================================================

    def find_eigenvalues(self, n_modes):

        # Aproximación del máximo requerido
        k_max = (
            n_modes + 2
        ) * np.pi / self.L

        # Malla para encontrar cambios de signo
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

                if (
                    len(roots) == 0
                    or
                    abs(
                        root - roots[-1]
                    ) > 1e-8
                ):

                    roots.append(root)

            if len(roots) >= n_modes:
                break

        roots = np.asarray(
            roots
        )

        if len(roots) < n_modes:

            raise RuntimeError(
                f"Se encontraron "
                f"{len(roots)} eigenvalues, "
                f"pero se solicitaron "
                f"{n_modes}."
            )

        return roots


    # ========================================================
    # EIGENFUNCIÓN
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
    # NORMA DE LA EIGENFUNCIÓN
    # ========================================================

    def eigenfunction_norm(
        self,
        k
    ):

        x = np.linspace(
            0.0,
            self.L,
            20000
        )

        X = self.eigenfunction(
            x,
            k
        )

        return simpson(
            X**2,
            x=x
        )


    # ========================================================
    # COEFICIENTES PARA DELTA DE DIRAC
    # ========================================================

    def calculate_point_coefficients(self):

        A = np.zeros(
            len(self.k)
        )

        # Para:

        # C(x,0) = M delta(x-x0)

        # tenemos:

        # phi(x,0)
        # =
        # exp(-beta*x0)
        # M delta(x-x0)

        # Por tanto:

        # A_n =
        #
        # M exp(-beta*x0) X_n(x0)
        # -------------------------
        # integral X_n^2 dx

        exp_factor = (
            self.M
            * np.exp(
                -self.beta * self.x0
            )
        )

        for n, k in enumerate(
            self.k
        ):

            X0 = self.eigenfunction(
                self.x0,
                k
            )

            norm = (
                self.eigenfunction_norm(
                    k
                )
            )

            A[n] = (
                exp_factor
                * X0
                / norm
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

        phi = np.zeros(
            len(x)
        )

        for n, k in enumerate(
            self.k
        ):

            X = self.eigenfunction(
                x,
                k
            )

            phi += (
                self.A[n]
                * X
                * np.exp(
                    -self.lambda_n[n]
                    * t
                )
            )

        C = (
            np.exp(
                self.beta * x
            )
            * phi
        )

        return C


# ============================================================
# 4. FLUJO INTERIOR
# ============================================================

def interior_flux(
    C_left,
    C_right,
    u,
    D,
    dx
):

    # Advección upwind
    J_adv = (
        u * C_left
    )

    # Difusión centrada
    J_diff = (
        -D
        * (C_right - C_left)
        / dx
    )

    return (
        J_adv + J_diff
    )


# ============================================================
# 5. FRONTERA ROBIN
# ============================================================

def robin_boundary_flux(
    C1,
    u,
    D,
    dx,
    km
):

    # C_inf = 0

    alpha = (
        2.0 * D / dx
    )

    # Cb =
    #
    # alpha C1
    # -------------------
    # u + alpha + km

    Cb = (
        alpha * C1
        /
        (
            u
            + alpha
            + km
        )
    )

    # J = -km Cb

    return (
        -km * Cb
    )


# ============================================================
# 6. FRONTERA DERECHA
# ============================================================

def outlet_flux(
    C_last,
    u
):

    # C_x = 0

    return (
        u * C_last
    )


# ============================================================
# 7. TODOS LOS FLUJOS
# ============================================================

def calculate_fluxes(
    C,
    u,
    D,
    dx,
    km
):

    N = len(C)

    J = np.zeros(
        N + 1
    )

    # Frontera izquierda
    J[0] = robin_boundary_flux(
        C[0],
        u,
        D,
        dx,
        km
    )

    # Caras interiores
    for i in range(
        N - 1
    ):

        J[i + 1] = interior_flux(
            C[i],
            C[i + 1],
            u,
            D,
            dx
        )

    # Frontera derecha
    J[N] = outlet_flux(
        C[N - 1],
        u
    )

    return J


# ============================================================
# 8. PASO TEMPORAL
# ============================================================

def time_step(
    C,
    dt,
    dx,
    u,
    D,
    km
):

    J = calculate_fluxes(
        C,
        u,
        D,
        dx,
        km
    )

    return (
        C
        -
        dt / dx
        * (
            J[1:]
            -
            J[:-1]
        )
    )


# ============================================================
# 9. SOLVER FVM
# ============================================================

def solve_fvm(
    N,
    t_final,
    u,
    D,
    km
):

    dx = L / N

    # Centros de celda
    x = (
        np.arange(N) + 0.5
    ) * dx

    # --------------------------------------------------------
    # Pulso puntual
    # --------------------------------------------------------

    C = initial_point_pulse(
        x,
        dx
    )

    # --------------------------------------------------------
    # Paso temporal
    # --------------------------------------------------------

    dt_adv = (
        dx / abs(u)
    )

    dt_diff = (
        dx**2
        / (2.0 * D)
    )

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
        t_final / n_steps
    )

    # --------------------------------------------------------
    # Guardar soluciones
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
            km
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
# 10. ERROR L2
# ============================================================

def L2_error(
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

    relative = (
        error / norm
    )

    return (
        error,
        relative
    )


# ============================================================
# 11. EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # MALLA
    # ========================================================

    N = 100

    dx = L / N


    # ========================================================
    # SOLUCIÓN ANALÍTICA
    # ========================================================

    print(
        "Construyendo solución analítica..."
    )

    analytical = (
        FiniteDomainPointPulse(
            L=L,
            u=u,
            D=D,
            km=km,
            x0=x0,
            M=M,
            n_modes=500
        )
    )

    print(
        f"Modos utilizados: "
        f"{len(analytical.k)}"
    )


    # ========================================================
    # SOLUCIÓN FV
    # ========================================================

    x, solutions, dt = solve_fvm(
        N,
        t_final,
        u,
        D,
        km
    )


    # ========================================================
    # INFORMACIÓN
    # ========================================================

    CFL = (
        u * dt / dx
    )

    Fo = (
        D * dt / dx**2
    )

    i0 = np.argmin(
        np.abs(x - x0)
    )

    print()
    print("=" * 70)
    print("TRANSPORTE DE UN PULSO PUNTUAL")
    print("=" * 70)

    print(
        f"L          = {L}"
    )

    print(
        f"u          = {u}"
    )

    print(
        f"D          = {D}"
    )

    print(
        f"km         = {km}"
    )

    print(
        f"N          = {N}"
    )

    print(
        f"dx         = {dx}"
    )

    print(
        f"dt         = {dt:.6e}"
    )

    print(
        f"CFL        = {CFL:.5f}"
    )

    print(
        f"Fo         = {Fo:.5f}"
    )

    print(
        f"x0         = {x0:.6f}"
    )

    print(
        f"centro celda = {x[i0]:.6f}"
    )

    print(
        f"C inicial  = {M/dx:.6f}"
    )


    # ========================================================
    # COMPROBAR MASA INICIAL
    # ========================================================

    C_initial = solutions[0.0]

    mass_initial = (
        np.sum(C_initial)
        * dx
    )

    print()
    print(
        f"Masa inicial FV = "
        f"{mass_initial:.12f}"
    )


    # ========================================================
    # ERROR
    # ========================================================

    print()
    print("=" * 70)
    print("COMPARACIÓN FV vs SOLUCIÓN ANALÍTICA")
    print("=" * 70)

    print(
        f"{'t':>8}"
        f"{'L2':>18}"
        f"{'Error relativo':>20}"
        f"{'Masa FV':>18}"
        f"{'Masa exacta':>18}"
    )

    for t in times_to_save[1:-1]:

        C_num = solutions[t]

        C_exact = analytical(
            x,
            t
        )

        error, relative = L2_error(
            x,
            C_num,
            C_exact
        )

        mass_num = (
            np.sum(C_num)
            * dx
        )

        mass_exact = np.trapezoid(
            C_exact,
            x
        )

        print(
            f"{t:8.2f}"
            f"{error:18.6e}"
            f"{relative:20.6e}"
            f"{mass_num:18.6e}"
            f"{mass_exact:18.6e}"
        )


    # ========================================================
    # EVOLUCIÓN DEL PULSO
    # ========================================================

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(15, 7),
        sharex=True,
        sharey=False
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
            "o-",
            markersize=2,
            label="FV"
        )

        ax.plot(
            x,
            C_exact,
            "--",
            linewidth=2,
            label="Analítica"
        )

        ax.axvline(
            x0 + u * t,
            linestyle=":",
            alpha=0.7,
            label="x0 + ut"
            if t == 0
            else None
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

    axes[0].legend()

    fig.suptitle(
        "Pulso puntual: "
        "volúmenes finitos vs solución analítica"
    )

    plt.tight_layout()
    plt.show()


    # ========================================================
    # COMPARACIÓN DETALLADA
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
        markersize=4,
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
    plt.ylabel("C")

    plt.title(
        f"Pulso puntual a t={t_compare}"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()
    plt.show()


    # ========================================================
    # CONVERGENCIA DE MALLA
    # ========================================================

    mesh_sizes = [
        25,
        50,
        100,
        200
    ]

    print()
    print("=" * 70)
    print("CONVERGENCIA DE MALLA")
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
                km
            )
        )

        C_mesh = sol_mesh[
            t_compare
        ]

        C_exact = analytical(
            x_mesh,
            t_compare
        )

        error, relative = L2_error(
            x_mesh,
            C_mesh,
            C_exact
        )

        print(
            f"{Nmesh:8d}"
            f"{L/Nmesh:12.5f}"
            f"{error:18.6e}"
            f"{relative:20.6e}"
        )

        plt.plot(
            x_mesh,
            C_mesh,
            label=f"FV N={Nmesh}"
        )


    # Analítica en malla fina
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
        f"Convergencia del pulso puntual "
        f"(t={t_compare})"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()
    plt.show()