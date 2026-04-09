import os
import urllib.request
import uaibot as ub
import numpy as np
import matplotlib.pyplot as plt

def set_configuration_speed(robot, q_dot, t, dt):
    q_next = robot.q + q_dot*dt
    robot.add_ani_frame(time = t+dt, q = q_next)


def draw_pc(path, sim, color="white", radius = 0.02):
    sl = [ ]
    for htm in path:
        sl.append( htm[ 0 : 3 , 3] ) 
    pc = ub.PointCloud(size = radius, color = color, points = sl)
    sim.add(pc)


def salvar_caminho(caminho, arquivo):
    base_dir = os.path.dirname(__file__)
    arquivo = os.path.join(base_dir, arquivo)
    with open(arquivo, "w") as f:
        for matriz in caminho:
            np.savetxt(f, matriz, delimiter=",", fmt="%.4f")


def carregar_caminho(arquivo):
    base_dir = os.path.dirname(__file__)
    arquivo = os.path.join(base_dir, arquivo)
    caminho = []
    with open(arquivo, "r") as f:
        linhas = f.readlines()
        for i in range(0, len(linhas), 6):
            matriz = np.loadtxt(linhas[i:i+6], delimiter=",")
            caminho.append(matriz)
    return caminho


def plot_vector_list(data_list, t, title, file_name, labels=None):
    if len(data_list) == 0:
        return

    processed = []
    for d in data_list:
        d = np.asarray(d).squeeze()

        if d.ndim == 0:
            d = d.reshape(1)   # escalar -> vetor de dimensão 1
        elif d.ndim != 1:
            raise ValueError("Cada elemento deve ser escalar ou vetor 1D após squeeze.")

        processed.append(d)

    # checa se todos têm mesma dimensão
    n = processed[0].shape[0]
    for d in processed:
        if d.shape[0] != n:
            raise ValueError("Todos os elementos de data_list devem ter a mesma dimensão.")

    # checa compatibilidade com tempo
    if len(t) != len(processed):
        raise ValueError("len(t) deve ser igual a len(data_list).")

    data = np.vstack(processed)   # shape = (N, n)

    plt.figure(figsize=(10, 6))
    for i in range(n):
        label = labels[i] if labels is not None else f'x_{i+1}'
        plt.plot(t, data[:, i], label=label)

    plt.title(title)
    plt.xlabel('Tempo (s)')
    plt.ylabel('Valor')
    plt.grid(True)

    if n > 1:
        plt.legend()

    plt.tight_layout()

    base_dir = os.path.dirname(__file__)
    save_path = os.path.join(base_dir, file_name)
    plt.savefig(save_path)
    plt.close()

    print(f"Plot salvo em: {save_path}")
 
    
def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi


def smooth_path(path, alpha=0.6 , iterations=300):
    """
    path: lista ou array de shape (N,6) -> [x y z r p y]
    alpha: intensidade da suavização
    iterations: quantas vezes aplicar
    """

    path = np.asarray(path, dtype=float)

    # força formato (N,6)
    if path.ndim == 1:
        path = path.reshape(-1, 6)
    if path.shape[0] == 6 and path.shape[1] != 6:
        path = path.T

    N = path.shape[0]
    if N < 3:
        return path.copy()

    new = path.copy()

    for _ in range(iterations):

        for i in range(1, N-1):

            # linear
            new[i,0:3] += alpha * (new[i-1,0:3] + new[i+1,0:3] - 2*new[i,0:3])

            # angular
            diff_prev = wrap_angle(new[i-1,3:] - new[i,3:])
            diff_next = wrap_angle(new[i+1,3:] - new[i,3:])

            new[i,3:] += alpha * (diff_prev + diff_next)
            new[i,3:] = wrap_angle(new[i,3:])

    return new



   
    