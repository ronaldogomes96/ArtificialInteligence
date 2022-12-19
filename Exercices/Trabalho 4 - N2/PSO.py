# ---Algoritmo PSO -------
import math
import numpy as np

a_lo = -5
a_up = 12
b_lo = 5
b_up = 12
n_points = 1000


def f(a, b):
    return 20 + (a ** 2) + (b ** 2) - (10 * (np.cos(2 * a * math.pi) + np.cos(2 * b * math.pi)))


n_iterations = 50


def run_PSO(n_particles=5, omega=0.5, phi_p=0.5, phi_g=0.7):
    """ Algortimo PSO para a definição de uma função.
    Params:
        omega = 0.5  # Pesos das particulas (intercial)
        phi_p = 0.1  # melhor peso de particua
        phi_g = 0.1  # Peso global

    """
    global a_best_p_global, a_particles, b_best_p_global, b_particles, y_particles, u_particles, v_particles

    # Nota: estamos usando variáveis ​​globais para facilitar o uso de widgets interativos
    # Este código funcionará bem sem o global (e na verdade será mais seguro)

    ## Inicialização:

    a_particles = np.zeros((n_particles, n_iterations))
    a_particles[:, 0] = np.random.uniform(a_lo, a_up, size=n_particles)
    b_particles = np.zeros((n_particles, n_iterations))
    b_particles[:, 0] = np.random.uniform(a_lo, a_up, size=n_particles)

    a_best_particles = np.copy(a_particles[:, 0])
    b_best_particles = np.copy(b_particles[:, 0])

    y_particles = f(a_particles[:, 0], b_particles[:, 0])
    y_best_global = np.min(y_particles[:])
    index_best_global = np.argmin(y_particles[:])
    a_best_p_global = np.copy(a_particles[index_best_global, 0])
    b_best_p_global = np.copy(b_particles[index_best_global, 0])

    # unidades de velocidade são [Comprimento / iteração]
    a_velocity_lo = a_lo - a_up
    a_velocity_up = a_up - a_lo
    b_velocity_lo = b_lo - b_up
    b_velocity_up = b_up - b_lo

    ua_particles = np.zeros((n_particles, n_iterations))
    ua_particles[:, 0] = 0.1 * np.random.uniform(a_velocity_lo, a_velocity_up, size=n_particles)
    ub_particles = np.zeros((n_particles, n_iterations))
    ub_particles[:, 0] = 0.1 * np.random.uniform(b_velocity_lo, b_velocity_up, size=n_particles)

    v_particles = np.zeros((n_particles, n_iterations))  # Necessário para traçar a velocidade como vetores

    # Inicio do PSO

    iteration = 1
    while iteration <= n_iterations - 1:
        for i in range(n_particles):
            a_p = a_particles[i, iteration - 1]
            b_p = b_particles[i, iteration - 1]

            ua_p = ua_particles[i, iteration - 1]
            ub_p = ub_particles[i, iteration - 1]

            a_best_p = a_best_particles[i]
            b_best_p = b_best_particles[i]

            r_p = np.random.uniform(0, 1)
            r_g = np.random.uniform(0, 1)

            ua_p_new = omega * ua_p + \
                       phi_p * r_p * (a_best_p - a_p) + \
                       phi_g * r_g * (a_best_p_global - a_p)
            ub_p_new = omega * ub_p + \
                       phi_p * r_p * (b_best_p - b_p) + \
                       phi_g * r_g * (b_best_p_global - b_p)

            a_p_new = a_p + ua_p_new
            b_p_new = b_p + ub_p_new

            if not a_lo <= a_p_new <= a_up:
                a_p_new = a_p  # ignorar a nova posição, está fora do domínio
                ua_p_new = 0
            if not b_lo <= b_p_new <= b_up:
                b_p_new = b_p  # ignorar a nova posição, está fora do domínio
                ub_p_new = 0

            a_particles[i, iteration] = np.copy(a_p_new)
            ua_particles[i, iteration] = np.copy(ua_p_new)
            b_particles[i, iteration] = np.copy(b_p_new)
            ub_particles[i, iteration] = np.copy(ub_p_new)

            y_p_best = f(a_best_p, b_best_p)
            y_p_new = f(a_p_new, b_p_new)

            if y_p_new < y_p_best:
                a_best_particles[i] = np.copy(a_p_new)
                b_best_particles[i] = np.copy(b_p_new)

                y_p_best_global = f(a_best_p_global, b_best_p_global)

                if y_p_new < y_p_best_global:
                    a_best_p_global = a_p_new
                    b_best_p_global = b_p_new

        iteration = iteration + 1

    # Plot da convergencia

    y_particles = f(a_particles, b_particles)
    y_particles_best_hist = np.min(y_particles, axis=0)
    y_particles_worst_hist = np.max(y_particles, axis=0)

    y_best_global = np.min(y_particles[:])
    index_best_global = np.argmin(y_particles[:])

    print("Melhor: ", y_best_global)

    return


run_PSO()
