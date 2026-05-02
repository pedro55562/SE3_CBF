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


def carregar_htm(nome_arquivo):
    pasta_script = os.path.dirname(os.path.abspath(__file__))
    caminho_arquivo = os.path.join(pasta_script, nome_arquivo)

    htms = []
    matriz_atual = []

    with open(caminho_arquivo, 'r') as f:
        for linha in f:
            linha = linha.strip()
            
            if linha == "":
                if matriz_atual:
                    htms.append(np.matrix(matriz_atual))
                    matriz_atual = []
            else:
                valores = [float(v) for v in linha.split()]
                matriz_atual.append(valores)

        if matriz_atual:
            htms.append(np.matrix(matriz_atual))

    return np.array(htms)

def plot_vector_list(
    data_list,
    t,
    file_name,
    labels=None,
    xlabel='Time (s)',
    ylabel='Value',
    title=None,
    figsize=(6.0, 3.6),
    linewidth=1.2,
    dpi=300,
    show_grid=True,
    show_plot=False
):
    if len(data_list) == 0:
        return
    
    processed = []
    for d in data_list:
        d = np.asarray(d).squeeze()

        if d.ndim == 0:
            d = d.reshape(1)
        elif d.ndim != 1:
            raise ValueError("Each element must be scalar or 1D after squeeze.")

        processed.append(d)

    n = processed[0].shape[0]
    for d in processed:
        if d.shape[0] != n:
            raise ValueError("All elements in data_list must have the same dimension.")

    if len(t) != len(processed):
        raise ValueError("len(t) must match len(data_list).")

    data = np.vstack(processed)

    # estilo tipo MATLAB / paper
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'lines.linewidth': linewidth,
        'axes.linewidth': 1.0,
    })

    fig, ax = plt.subplots(figsize=figsize)

    # plots
    for i in range(n):
        label = labels[i] if labels is not None else f'$x_{i+1}$'
        ax.plot(t, data[:, i], label=label)

    # título
    if title is not None:
        ax.set_title(title)

    # labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # grid (leve)
    if show_grid:
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)

    # === FULL BOX (estilo MATLAB) ===
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color('black')

    # ticks nos 4 lados, para dentro
    ax.tick_params(
        direction='in',
        top=True,
        right=True,
        length=4,
        width=0.8,
        colors='black'
    )

    # legenda
    if labels is not None and n > 1:
        ax.legend(frameon=False)

    fig.tight_layout()

    base_dir = os.path.dirname(__file__)
    save_path = os.path.join(base_dir, file_name)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    if show_plot:
        plt.show()

    plt.close(fig)

    print(f"Plot saved at: {save_path}") 




def plot_dist_min(
    data_list,
    t,
    file_name,
    labels=None,
    xlabel='Time (s)',
    ylabel='Value',
    title=None,
    figsize=(6.0, 3.6),
    linewidth=1.2,
    dpi=300,
    show_grid=True,
    show_plot=False
):
    if len(data_list) == 0:
        return
    
    processed = []
    for d in data_list:
        d = np.asarray(d).squeeze()

        if d.ndim == 0:
            d = d.reshape(1)
        elif d.ndim != 1:
            raise ValueError("Each element must be scalar or 1D after squeeze.")

        processed.append(d)

    n = processed[0].shape[0]
    for d in processed:
        if d.shape[0] != n:
            raise ValueError("All elements in data_list must have the same dimension.")

    if len(t) != len(processed):
        raise ValueError("len(t) must match len(data_list).")

    data = np.vstack(processed)

    # estilo MATLAB / paper
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'lines.linewidth': linewidth,
        'axes.linewidth': 1.0,
    })

    fig, ax = plt.subplots(figsize=figsize)

    # curvas
    for i in range(n):
        label = labels[i] if labels is not None else f'$x_{i+1}$'
        ax.plot(t, data[:, i], label=label)

    # linha de segurança
    safety_distance = 0.01
    ax.axhline(
        y=safety_distance,
        linestyle='--',
        linewidth=1.2,
        color='red'
    )

    # anotação discreta
    ax.text(
        t[-1],
        safety_distance,
        ' safety limit',
        color='red',
        fontsize=9,
        va='bottom',
        ha='right'
    )

    # títulos e eixos
    if title is not None:
        ax.set_title(title)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # grid leve
    if show_grid:
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)

    # === FULL BOX (MATLAB style) ===
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color('black')

    # ticks nos 4 lados
    ax.tick_params(
        direction='in',
        top=True,
        right=True,
        length=4,
        width=0.8,
        colors='black'
    )

    # legenda (somente se necessário)
    if labels is not None and n > 1:
        ax.legend(frameon=False)

    fig.tight_layout()

    base_dir = os.path.dirname(__file__)
    save_path = os.path.join(base_dir, file_name)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    if show_plot:
        plt.show()

    plt.close(fig)

    print(f"Plot saved at: {save_path}")
    
def plot_u_xi(
    u_list,
    xi_list,
    t,
    file_name,
    labels_u=None,
    labels_xi=None,
    xlabel='Time (s)',
    ylabel_u=r'$u$',
    ylabel_xi=r'$\xi$',
    title_u='Control Input',
    title_xi='System Twist',
    figsize=(6.0, 5.5),
    linewidth=1.2,
    dpi=300,
    show_grid=True,
    show_plot=False
):
    if len(u_list) == 0 or len(xi_list) == 0:
        return

    def process(data_list):
        processed = []
        for d in data_list:
            d = np.asarray(d).squeeze()
            if d.ndim == 0:
                d = d.reshape(1)
            elif d.ndim != 1:
                raise ValueError("Each element must be scalar or 1D.")
            processed.append(d)

        n = processed[0].shape[0]
        for d in processed:
            if d.shape[0] != n:
                raise ValueError("Dimension mismatch in data_list.")

        return np.vstack(processed), n

    data_u, n_u = process(u_list)
    data_xi, n_xi = process(xi_list)

    if len(t) != data_u.shape[0] or len(t) != data_xi.shape[0]:
        raise ValueError("Time vector size mismatch.")

    # estilo global
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'lines.linewidth': linewidth,
        'axes.linewidth': 1.0,
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # ===== Plot u =====
    for i in range(n_u):
        label = labels_u[i] if labels_u is not None else f'$u_{i+1}$'
        ax1.plot(t, data_u[:, i], label=label)

    ax1.set_ylabel(ylabel_u)
    if title_u:
        ax1.set_title(title_u)

    # ===== Plot xi =====
    for i in range(n_xi):
        label = labels_xi[i] if labels_xi is not None else f'$x_{i+1}$'
        ax2.plot(t, data_xi[:, i], label=label)

    ax2.set_ylabel(ylabel_xi)
    ax2.set_xlabel(xlabel)
    if title_xi:
        ax2.set_title(title_xi)

    # ===== estilo comum =====
    for ax in [ax1, ax2]:
        if show_grid:
            ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)

        # full box
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color('black')

        ax.tick_params(
            direction='in',
            top=True,
            right=True,
            length=4,
            width=0.8,
            colors='black'
        )

    # legenda (opcional)
    if labels_u is not None and n_u > 1:
        ax1.legend(loc='upper right', frameon=False)

    if labels_xi is not None and n_xi > 1:
        ax2.legend(loc='upper right', frameon=False)

    fig.tight_layout()

    base_dir = os.path.dirname(__file__)
    save_path = os.path.join(base_dir, file_name)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    if show_plot:
        plt.show()

    plt.close(fig)

    print(f"Plot saved at: {save_path}")        

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



   
    