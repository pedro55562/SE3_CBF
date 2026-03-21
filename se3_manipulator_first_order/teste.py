import uaibot as ub
import numpy as np
import matplotlib.pyplot as plt
import os
from setup import *


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
        plt.plot(t, data[:, j], label=f'q{j+1}')

    plt.xlabel('Tempo (s)')
    plt.ylabel('q_dot (rad/s)')
    plt.title('q_dot por tempo')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()

    if save_dir is None:
        save_dir = os.path.dirname(__file__)

    path = os.path.join(save_dir, file_name)
    plt.savefig(path)
    plt.close()
    print(f'Gráfico salvo em: {path}')


# 420 450 1250 695 740 820 2400 900 910 920 930 940 960 970 1130 1160 1170 1260 1270 1170 
# 1350 1360 1700 1800 1900 2000 2050 2100 2150 2200 2250 2300 2400 2500 2550
robot, sim, all_obs, q0, htm_tg, htm_base = setup_motion_planning_simulation( 420, use_pc=False)

success, path, iterations, num_of_tries, planning_time = robot.runSE3RRT(
    q0=q0,
    htm=htm_base,
    q_goal=None,
    htm_tg=htm_tg,
    obstacles=all_obs,
    no_iter_max=2000,
    n_tries=10,
    goal_tolerance=0.15,
    goal_bias=0.35,
    step_size_min=0.2,
    step_size_max=1.5,
    usemultthread=True,
)
print(success)
print(len(path))




q_dot_list = []
     
if success:
    htm_path = []
    htm_d_path = []
    
    for qc in path:
        fkm = robot.fkm(q=qc)
        htm_path.append(fkm)

    
    t = 0
    i = 0
    dt = 0.01
    while t < 20:
        jac_geo, fkm = robot.jac_geo()

        r , Jr = robot.task_function(htm_tg = htm_tg, htm = htm_base)
        if (np.linalg.norm(r) < 0.01 ):
            print("Chegou ao objetivo!")
            break
        
        xid, dist, idx = robot.vector_field_SE3(
            
            state=fkm,            
            curve=htm_path,       

            kt1=1.0,              
            kt2=1.0,              
            kt3=1.0,             
            
            kn1=1.0,             
            kn2=1.0,  
            ds = dt,
            delta = 1e-2,
              
        )
        xid = np.matrix(xid).T

        xid[0 : 3 , :] = xid[0 : 3 , :] +   ub.Utils.S(xid[3 : 6 , :]) * fkm[0 : 3 , -1]
        
        qdot = ub.Utils.dp_inv(jac_geo, 1e-3) @ xid.reshape(6,1)
        
        q_dot_list.append(xid)
        
        set_configuration_speed(robot=robot, q_dot=qdot, t=t, dt=dt)
        
        i += 1
        t = i * dt
    

    
    save_q_dot_plot(q_dot_list, dt=dt)
    draw_balls(pathhh_ = htm_path, sim=sim)

sim.save(address="/home/pedro/code/SE3_CBF/",file_name="teste")


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
        plt.plot(t, data[:, j], label=f'q{j+1}')

    plt.xlabel('Tempo (s)')
    plt.ylabel('q_dot (rad/s)')
    plt.title('q_dot por tempo')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()

    if save_dir is None:
        save_dir = os.path.dirname(__file__)

    path = os.path.join(save_dir, file_name)
    plt.savefig(path)
    plt.close()
    print(f'Gráfico salvo em: {path}')




# python setup.py build_ext --inplace
# pip install -e .