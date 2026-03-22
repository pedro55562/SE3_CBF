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

def eval_xid_from_state(state_htm, robot, htm_path, kt1, kt2, kt3, kn1, kn2, dt):
    xid, dist, idx = robot.vector_field_SE3(
        state=state_htm,
        curve=htm_path,
        kt1=kt1,
        kt2=kt2,
        kt3=kt3,
        kn1=kn1,
        kn2=kn2,
        ds=dt,
        delta=1e-3,
    )
    xid = np.asarray(xid, dtype=float).reshape(6, 1)
    xid[0:3, :] = xid[0:3, :] + ub.Utils.S(xid[3:6, :]) * state_htm[0:3, -1]
    return xid, dist, idx

def compute_obstacle_distance_gradient(robot_ob, ob, Jv, Jw, s, h, eps):      
    point_robot, point_obs, dist, _ = robot_ob.compute_dist(ob , h =  h, eps = eps)
    jac_dist = ((point_robot - point_obs).T * Jv + np.cross((point_robot - s ).T, (point_robot - point_obs).T)  * Jw) / dist
    return dist, jac_dist, point_robot, point_obs


##############################
#     Robot Initialization   #
##############################

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
robot_ob = objss[0]

##############################



######################################
#     Workspace & Obstacle Setup     #
######################################

htm_target = ub.Utils.trn([0, 2.5, 1]) * robot.fkm() * ub.Utils.rot(axis=[4,3,-2], angle= 77 * np.pi / 180) 
frame_target = ub.Frame(htm=htm_target)
sim.add([frame_target])

piso = ub.Box(htm = ub.Utils.trn([0, 0, -.35]) ,width=7, depth=7  , height = 0.05  ,color='red')
teto = ub.Box(htm = ub.Utils.trn([0, 0, 1.45]) ,width=7, depth=7  , height = 0.05  ,color='red',opacity=0.3)
parede_frente = ub.Box(htm = ub.Utils.trn([0, 2, 0.8]) ,   width=3, depth=0.1, height = 1.9  ,color='red')
parede_fundo = ub.Box(htm = ub.Utils.trn([0, 3.5, 0.8]) ,   width=7, depth=0.1, height = 1.9  ,color='red',opacity=0.3)
parede_lateral = ub.Box(htm = ub.Utils.trn([-1.5, 2.75, 0.8]) * ub.Utils.rotz(np.pi/2) ,   width=1.5, depth=0.1, height = 1.9  ,color='red')
parede_sup = ub.Box(htm = ub.Utils.trn([1.3, 2.75, 1.3]) * ub.Utils.rotz(np.pi/2) ,   width=1.5, depth=0.1, height = 1  ,color='cyan')
parede_sup_lat = ub.Box(htm = ub.Utils.trn([1.3, 3.2, 0.8]) * ub.Utils.rotz(np.pi/2) ,   width=1, depth=0.1, height = 1.9  ,color='cyan')

unknown_obs = [parede_sup, parede_sup_lat] 
known_obs = [parede_frente, piso, teto, parede_fundo, parede_lateral]
all_obs = known_obs + unknown_obs
sim.add(all_obs)
#####################################


    
##############################
#     Path Planning          #
##############################

caminho_arquivo = "ultimo_caminho.txt"

gerar_novo_caminho = False
simular_movimento = not gerar_novo_caminho 
if gerar_novo_caminho:
    q_goal = robot.ikm(htm_tg=htm_target, obstacles=known_obs, no_tries=2000, no_iter_max=4000)
    success1, c_space_path, iterations1, num_tries1, planning_time1 = robot.runSE3RRT(q0=robot.q0, q_goal=[q_goal], obstacles=known_obs)
    c_space_path = smooth_path(path=c_space_path)
    salvar_caminho(c_space_path, caminho_arquivo)
    
else:
    c_space_path = carregar_caminho(caminho_arquivo)
    
htm_path = []
for qc in c_space_path:
    htm_path.append(robot.fkm(q=qc))
draw_pc(path=htm_path, sim=sim, color="white", radius = 0.02)

##############################



##############################
#     Control Parameters     #
##############################

kt1 = 1.0        
kt2 = 0.4           
kt3 = 0.4         
kn1 = 0.1          
kn2 = 0.1

param_eta = 1.0
param_obs_delta = 0.01

param_eta = 1.0
param_obs_delta = 0.01

eps = 1e-3
##############################



################################
#     Simulation Settings      #
################################

tmax = 20
dt = 0.01
idx =0
xid_list = []
t_list   = []

##############################


atingiu = False
if simular_movimento:
    for i in range(0, int (tmax/dt)):
        t = i*dt
        
        
        #################################
        #   Gain adjustment near goal   #
        #################################
        
        if idx > int( 0.9 * len(htm_path)):
            if not atingiu:
                print("atingiu: ", idx , " em t ", t)
                atingiu = True
            kt1 = .5           
            kt2 = .3           
            kt3 = .3          
            kn1 = 0.9        
            kn2 = 0.9
    
    
    
        ##########################################
        #   Reference twist from path tracking   #
        ##########################################
        
        curr_jac, curr_fkm = robot.jac_geo()
        xid, dist, idx = eval_xid_from_state(
            state_htm=curr_fkm,
            robot=robot,
            htm_path=htm_path,
            kt1=kt1,
            kt2=kt2,
            kt3=kt3,
            kn1=kn1,
            kn2=kn2,
            dt=dt
        )

        ###############################
        #    Build CBF constraints    #
        ###############################
        
        s = curr_fkm[0:3,-1]
        Jv = curr_jac[0:3,:]
        Jw = curr_jac[3:6,:]
        
        Ad_obj = np.zeros((0, 6))
        Bd_obj = np.zeros((0, 1))
        for ob in all_obs:
            dist, jac_dist, point_robot, point_obs = compute_obstacle_distance_gradient(robot_ob, ob, Jv, Jw, s, 0.05, 0.02)
            
            Ad_obj = np.vstack((Ad_obj, jac_dist))
            Bd_obj = np.vstack((Bd_obj, -param_eta*(dist-param_obs_delta)))

        ######################
        #   QP formulation   #
        ######################
        
        H =  2*( eps*np.eye(6) + curr_jac.T @ curr_jac)
        f = -2*curr_jac.T @ xid
        try:
            u = ub.Utils.solve_qp(
                H=H,
                f=f,
                A=Ad_obj,
                b=Bd_obj
            )
        except:
            print("\n QP Falhou!  ")
            print("Tempo: ", t)
            sim.save(address="/home/pedro/code_robot/SE3_CBF/Teste Drone - 1st order/",file_name="teste_SE3")
            break
        
        
        # Apply control
        xid_list.append(u)
        t_list.append(t)
        set_configuration_speed(robot, u, t, dt)
        
# Results
plot_twist(data_list = xid_list, t = t_list , title = "xi", file_name='xi_plot.png')
sim.save(address="/home/pedro/code_robot/SE3_CBF/se3_rigid_body_first_order/",
         file_name="se3_first_order_cbf_obstacle_avoidance"
         )
