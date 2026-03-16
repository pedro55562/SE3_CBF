import uaibot as ub
import numpy as np
import matplotlib.pyplot as plt
import os
from setup import *
import itertools


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

htm_target = ub.Utils.trn([0, 2.5, 0.7]) * robot.fkm() * ub.Utils.rot(axis=[4,3,-2], angle= 77 * np.pi / 180) 

frame_target = ub.Frame(htm=htm_target)
sim.add([frame_target])

box1 = ub.Box(htm = ub.Utils.trn([-0.5, 2, 0.8]) ,   width=3, depth=0.1, height = 1.6  ,color='red')
#box2 = ub.Box(htm = ub.Utils.trn([0, 0, -.15]) ,width=7, depth=7  , height = 0.05  ,color='red')
#box3 = ub.Box(htm = ub.Utils.trn([0, 0, 1.45]) ,width=7, depth=7  , height = 0.05  ,color='red',opacity=0.4)
#box4 = ub.Box(htm = ub.Utils.trn([0, 3.5, 0.8]) ,   width=7, depth=0.1, height = 1.6  ,color='red')


known_obs = [box1]



unknown_obs = []
all_obs = known_obs + unknown_obs
sim.add(all_obs)

bola1 = ub.Ball(radius=0.02, color = 'white')
bola2 = ub.Ball(radius=0.02, color = 'blue')
sim.add([bola1, bola2])


# Flag para gerar novo caminho
# Se True, gera um novo caminho e salva em um arquivo
# Se False, carrega o último caminho salvo

gerar_novo_caminho = False

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
    draw_pc(pathhh_=path1, robot=robot, sim=sim)
    print(len(path1))
    print(planning_time1)
    salvar_caminho(path1, caminho_arquivo)
else:
    path1 = carregar_caminho(caminho_arquivo)
    draw_pc(pathhh_=path1, robot=robot, sim=sim)



htm_path = []    
for qc in path1:
    fkm = robot.fkm(q=qc)
    htm_path.append(fkm)



i=0
dt = 0.01
t=0

xid_list = []

r , Jr = robot.task_function(htm_tg = htm_target)
while np.linalg.norm(r) > 0.025 and t < 50:
    t = i*dt


    jac_geo, fkm = robot.jac_geo()
    xid, dist, idx = robot.vector_field_SE3(
        
        state=fkm,            
        curve=htm_path,       

        kt1=0.85,              
        kt2=1.0,              
        kt3=1.0,             
        
        kn1=1.0/7,             
        kn2=1.0/7,  
        ds = dt,
        delta = 1e-2,
        
    )
    xid = np.matrix(xid).T
    xid[0 : 3 , :] = xid[0 : 3 , :] +   ub.Utils.S(xid[3 : 6 , :]) * fkm[0 : 3 , -1]


    jac, htm = robot.jac_geo()
    
    s = htm[0:3,-1]
    v = jac[0:3,:]
    w = jac[3:6,:]

    Ad_obj = np.zeros((0, 6))
    Bd_obj = np.zeros((0, 1))
    param_eta = 1.3
    param_obs_delta = 0.5
    k=0     
    for ob in all_obs:
        k+=1
            
        point_robot, point_obs, dist, _ = col_obj.compute_dist(ob, h =  0.05, eps = 0.02)



        jac_dist = ((point_robot - point_obs).T * v + np.cross((point_robot - s ).T, (point_robot - point_obs).T)  * w) / (dist + 1e-6)

        Ad_obj = np.vstack((Ad_obj, jac_dist))
        Bd_obj = np.vstack((Bd_obj, -param_eta*(dist-param_obs_delta)))
        if k ==1:
            bola1.add_ani_frame(time=t, htm=ub.Utils.trn(point_robot.flatten()))
            bola2.add_ani_frame(time=t, htm=ub.Utils.trn(point_obs.flatten()))

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
        sim.save(address="/home/pedro/code/SE3_CBF/",file_name="teste_SE3")
        break

    xid_list.append(u)
    #qdot = ub.Utils.dp_inv(jac_geo, 1e-3) @ u.reshape(6,1)
    set_configuration_speed(robot, u, t, dt)
    r , Jr = robot.task_function(htm_tg = htm_target)
    i+=1

save_q_dot_plot(xid_list, dt=dt, file_name='xid_plot.png')
sim.save(address="/home/pedro/code/SE3_CBF/",file_name="teste_SE3")
