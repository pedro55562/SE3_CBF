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

def plot_twist(data_list, t, title, file_name):
    if len(data_list) == 0:
        return

    data = np.hstack([d.reshape(6,1) for d in data_list]).T

    plt.figure(figsize=(10,6))
    for i in range(6):
        plt.plot(t, data[:, i], label=f'xi_{i+1}')

    plt.title(title)
    plt.xlabel('Tempo (s)')
    plt.ylabel('Valor')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


    base_dir = os.path.dirname(__file__)
    save_path = os.path.join(base_dir, file_name)
    plt.savefig(save_path)
    plt.close()

    print(f"Plot salvo em: {save_path}")