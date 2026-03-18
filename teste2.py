import uaibot as ub
import numpy as np
import matplotlib.pyplot as plt
import os
from setup import *
import itertools
from scipy.interpolate import CubicSpline
from scipy.interpolate import splprep, splev


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

def save_q_dot_plot(q_dot_list, dt=0.01, file_name='q_dot_plot.png', save_dir=None):
    """Salva um gráfico de q_dot ao longo do tempo na pasta do script.

    q_dot_list: lista de vetores q_dot (cada um tem formato (n,1) ou (n,))
    dt: passo de tempo entre amostras
    file_name: nome do arquivo de saída (PNG)
    save_dir: diretório onde salvar (por padrão, a pasta do script)
    """
    if len(q_dot_list) == 0:
        return None

    # Converter para matriz (timesteps, n_joints)
    cols = [np.asarray(q).reshape(-1, 1) for q in q_dot_list]
    data = np.hstack(cols).T

    t = np.arange(data.shape[0]) * dt

    plt.figure(figsize=(10, 6))
    for j in range(data.shape[1]):
        plt.plot(t, data[:, j], label=f'xid{j+1}')

    plt.xlabel('Tempo (s)')
    plt.ylabel('xid ')
    plt.title('xid por tempo')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()

    if save_dir is None:
        save_dir = os.path.dirname(__file__)

    path = os.path.join(save_dir, file_name)
    plt.savefig(path)
    plt.close()
    print(f'Gráfico salvo em: {path}')



robot = ub.Robot.create_rigid_body_se3()
sim = ub.Simulation(background_color = 'black')
sim.add([robot])


 
objss = []
#Loop the link for data//
for link in robot.links:
    #Loop the data for collision objects
    for col_obj_data in link.col_objects:
        #The actual UAIBot object (ub.Box, ub.Cylinder, etc...)    
        obj = col_obj_data[0]
        col_obj = obj
        objss.append(obj)
sim.add(objss)

htm_target = ub.Utils.trn([0, 2.5, 1]) * robot.fkm() * ub.Utils.rot(axis=[4,3,-2], angle= 77 * np.pi / 180) 
frame_target = ub.Frame(htm=htm_target)
sim.add([frame_target])

parede_frente = ub.Box(htm = ub.Utils.trn([0, 2, 0.8]) ,   width=3, depth=0.1, height = 1.9  ,color='red')
piso = ub.Box(htm = ub.Utils.trn([0, 0, -.35]) ,width=7, depth=7  , height = 0.05  ,color='red')
teto = ub.Box(htm = ub.Utils.trn([0, 0, 1.45]) ,width=7, depth=7  , height = 0.05  ,color='red',opacity=0.3)
parede_fundo = ub.Box(htm = ub.Utils.trn([0, 3.5, 0.8]) ,   width=7, depth=0.1, height = 1.9  ,color='red',opacity=0.3)
parede_lateral = ub.Box(htm = ub.Utils.trn([-1.5, 2.75, 0.8]) * ub.Utils.rotz(np.pi/2) ,   width=1.5, depth=0.1, height = 1.9  ,color='red')
known_obs = [parede_frente, piso, teto, parede_fundo, parede_lateral]


parede_sup = ub.Box(htm = ub.Utils.trn([1.3, 2.75, 1.3]) * ub.Utils.rotz(np.pi/2) ,   width=1.5, depth=0.1, height = 1  ,color='cyan')
parede_sup_lat = ub.Box(htm = ub.Utils.trn([1.3, 3.2, 0.8]) * ub.Utils.rotz(np.pi/2) ,   width=1, depth=0.1, height = 1.9  ,color='cyan')

unknown_obs = [parede_sup, parede_sup_lat] 
#unknown_obs = []
all_obs = known_obs + unknown_obs
sim.add(all_obs)
 


gerar_novo_caminho = False
simular_movimento = not gerar_novo_caminho 


caminho_arquivo = "ultimo_caminho.txt"

def salvar_caminho(caminho, arquivo):
    with open(arquivo, "w") as f:
        for matriz in caminho:
            np.savetxt(f, matriz, delimiter=",", fmt="%.4f")

def carregar_caminho(arquivo):
    caminho = []
    with open(arquivo, "r") as f:
        linhas = f.readlines()
        for i in range(0, len(linhas), 6):  # Cada matriz tem 6 linhas
            matriz = np.loadtxt(linhas[i:i+6], delimiter=",")
            caminho.append(matriz)
    return caminho

if gerar_novo_caminho:
    q_goal = robot.ikm(htm_tg=htm_target, obstacles=known_obs, no_tries=2000, no_iter_max=4000)
    success1, path1, iterations1, num_tries1, planning_time1 = robot.runSE3RRT(q0=robot.q0, q_goal=[q_goal], obstacles=known_obs)
    path1 = smooth_path(path=path1)
    draw_pc(pathhh_=path1, robot=robot, sim=sim, color="magenta", radius = 0.03)
    print(len(path1))
    print(planning_time1)
    salvar_caminho(path1, caminho_arquivo)
else:
    path1 = carregar_caminho(caminho_arquivo)
    draw_pc(pathhh_=path1, robot=robot, sim=sim, color="magenta", radius = 0.03)



htm_path = []    
for qc in path1:
    fkm = robot.fkm(q=qc)
    htm_path.append(fkm)


i=0
dt = 0.01
t=0



kt1 =  1           
kt2 = .4           
kt3 = .4          
kn1 = 0.1          
kn2 = 0.1

param_eta = 1
param_obs_delta = 0.02

idx =0
xid_list = []
print(len(htm_path))
atingiu = False
if simular_movimento:
    r , Jr = robot.task_function(htm_tg = htm_target)
    while np.linalg.norm(r) > 0.05 and t < 40:
        t = i*dt


        if idx > int( 0.9 * len(htm_path)):
            if not atingiu:
                print("atingiu: ", idx , " em t ", t)
                atingiu = True
            kt1 = .5           
            kt2 = .3           
            kt3 = .3          
            kn1 = 0.9        
            kn2 = 0.9
            
        jac_geo, fkm = robot.jac_geo()
        xid, dist, idx = robot.vector_field_SE3(
            
            state=fkm,            
            curve=htm_path,       

            kt1 = kt1,              
            kt2 = kt2,              
            kt3 = kt3,             
            
            kn1 = kn1,             
            kn2 = kn2,  
            ds  = dt ,
            delta = 1e-3,
            
        )
        xid = np.matrix(xid).T
        xid[0 : 3 , :] = xid[0 : 3 , :] +   ub.Utils.S(xid[3 : 6 , :]) * fkm[0 : 3 , -1]


        jac, htm = robot.jac_geo()
        
        s = htm[0:3,-1]
        v = jac[0:3,:]
        w = jac[3:6,:]

        Ad_obj = np.zeros((0, 6))
        Bd_obj = np.zeros((0, 1))
        k=0     
        for ob in all_obs:
            k+=1
                
            point_robot, point_obs, dist, _ = col_obj.compute_dist(ob , h =  0.05, eps = 0.02)



            jac_dist = ((point_robot - point_obs).T * v + np.cross((point_robot - s ).T, (point_robot - point_obs).T)  * w) / (dist + 1e-6)

            Ad_obj = np.vstack((Ad_obj, jac_dist))
            Bd_obj = np.vstack((Bd_obj, -param_eta*(dist-param_obs_delta)))

        eps = 1e-3

        H = 2*(eps)*np.eye(6) + 2* jac.T * jac 
        f = -2*  jac.T * xid
        
        try:
            u = ub.Utils.solve_qp(
                H=H,
                f=f,
                A=Ad_obj,
                b=Bd_obj
                )
        except:
            print("tempo: ", t)
            sim.save(address="/home/pedro/code_robot/SE3_CBF/",file_name="teste_SE3")
            break

        xid_list.append(u)
        set_configuration_speed(robot, u, t, dt)
        r , Jr = robot.task_function(htm_tg = htm_target)
        i+=1
        
        
save_q_dot_plot(xid_list, dt=dt, file_name='xid_plot.png')
sim.save(address="/home/pedro/code_robot/SE3_CBF/",file_name="teste_SE3")
