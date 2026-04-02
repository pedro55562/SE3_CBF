import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import uaibot as ub
from scipy.linalg import logm
from scipy.interpolate import CubicSpline, splprep, splev
from scipy.linalg import expm

from setup import *


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

def compute_distance_gradient(robot_ob, ob, curr_state, curr_jac, dist_param_h, dist_param_eps):    
    s =  curr_state[0:3,-1]
    Jv = curr_jac[0:3,:]
    Jw = curr_jac[3:6,:]  
    
    point_robot, point_obs, dist, _ = robot_ob.compute_dist(ob , h =  dist_param_h, eps = dist_param_eps)
    jac_dist = ((point_robot - point_obs).T * Jv + np.cross((point_robot - s ).T, (point_robot - point_obs).T)  * Jw) /(dist + 1e-6)
    return dist, jac_dist, point_robot, point_obs

def propagate_htm(htm, xi, dt_step):
    p = htm[0:3, 3].reshape(3, 1)
    R = htm[0:3, 0:3]

    v = np.asarray(xi[0:3]).reshape(3, 1)
    w = np.asarray(xi[3:6]).reshape(3, 1)

    R_next = expm(ub.Utils.S(w) * dt_step) @ R
    p_next = p + v * dt_step

    htm_next = np.eye(4)
    htm_next[0:3, 0:3] = R_next
    htm_next[0:3, 3] = p_next.flatten()

    return np.matrix(htm_next)

def compute_xi_dot(robot, curr_state, htm_path, xi,kt1, kt2, kt3, kn1, kn2, Kv):
    
        xid, dist, idx = eval_xid_from_state(
        state_htm=curr_state,
        robot=robot,
        htm_path=htm_path,
        kt1=kt1,
        kt2=kt2,
        kt3=kt3,
        kn1=kn1,
        kn2=kn2,
        dt=dt
        )

        # Numerical approximation of reference twist derivative ( xid_dot )
        
        htm_plus =  propagate_htm(curr_state, xid,  dt)
        htm_minus = propagate_htm(curr_state, xid, -dt)
        
        xid_plus, _, _ = eval_xid_from_state(
            state_htm=htm_plus,
            robot=robot,
            htm_path=htm_path,
            kt1=kt1,
            kt2=kt2,
            kt3=kt3,
            kn1=kn1,
            kn2=kn2,
            dt=dt,
        )

        xid_minus, _, _ = eval_xid_from_state(
            state_htm=htm_minus,
            robot=robot,
            htm_path=htm_path,
            kt1=kt1,
            kt2=kt2,
            kt3=kt3,
            kn1=kn1,
            kn2=kn2,
            dt=dt,
        )

        xid_dot = (xid_plus - xid_minus) / (2.0 * dt)
        gamma, _ = modulation(curr_state, htm_path[-1], gain=1, lam=20)

        return  (gamma*(xid_dot - Kv * (xi - xid)), dist, idx)

def compute_distance_hessian(robot_ob, ob, xi, curr_state, curr_jac, dist_param_h, dist_param_eps):
    
    htm_plus =  propagate_htm(curr_state, xi,  dt)
    htm_minus = propagate_htm(curr_state, xi, -dt)
    
    _, jac_dist_plus , _, _  = compute_distance_gradient(robot_ob, ob, htm_plus , curr_jac, dist_param_h, dist_param_eps)
    _, jac_dist_minus, _, _  = compute_distance_gradient(robot_ob, ob, htm_minus, curr_jac, dist_param_h, dist_param_eps)
    
    return (jac_dist_plus - jac_dist_minus)/(2*dt)
    
def compute_Jg_dot(robot, qdot, dt):    
    jac_plus  , _ = robot.jac_geo(q = robot.q + qdot*dt)
    jac_minus , _ = robot.jac_geo(q = robot.q - qdot*dt)

    return (jac_plus - jac_minus)/(2*dt)

def d_se3(H: np.ndarray, H_target: np.ndarray) -> float:
    """
    Distância em SE(3) dada por:
        || log( H^{-1} H_target ) ||_F

    Parâmetros
    ----------
    H : np.ndarray
        Matriz homogênea 4x4 do estado atual.
    H_target : np.ndarray
        Matriz homogênea 4x4 do estado alvo.

    Retorna
    -------
    float
        Valor da distância.
    """
    H = np.asarray(H, dtype=float)
    H_target = np.asarray(H_target, dtype=float)

    if H.shape != (4, 4):
        raise ValueError("H deve ser uma matriz 4x4.")
    if H_target.shape != (4, 4):
        raise ValueError("H_target deve ser uma matriz 4x4.")

    H_rel = np.linalg.inv(H) @ H_target
    log_H_rel = logm(H_rel)   # principal matrix logarithm

    # Se aparecer pequena parte imaginária numérica, descarta se for desprezível
    if np.iscomplexobj(log_H_rel):
        if np.allclose(log_H_rel.imag, 0.0, atol=1e-10):
            log_H_rel = log_H_rel.real
        else:
            raise ValueError(
                "O logaritmo matricial resultou em parte imaginária não desprezível."
            )

    return float(np.linalg.norm(log_H_rel, ord='fro'))

def modulation(H: np.ndarray,
                         H_target: np.ndarray,
                         gain: float,
                         lam: float) -> float:
    d = d_se3(H, H_target)
    return (1.0 - np.exp(-lam * d) * (1.0 + lam * d)) , d

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

htm_target = ub.Utils.trn([-.4, 2.6, 1]) * robot.fkm() * ub.Utils.rot(axis=[4,3,-2], angle= 77 * np.pi / 180) 
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
simular_movimento =  not gerar_novo_caminho 
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

kt1 = 1.4    
kt2 = 0.30          
kt3 = 0.30         
kn1 = 0.10          
kn2 = 0.10

Kv = 20

param_eta =  1.8
param_obs_delta = 0.02

eps = 1e-3


u_max = np.array([
    [5],   # vx  [m/s]
    [5],   # vy  [m/s]
    [5],   # vz  [m/s]
    [5],   # wx  [rad/s]
    [5],   # wy  [rad/s]
    [5]    # wz  [rad/s]
])

u_min = -u_max


# generalized distance parameters

use_generalized_distance = True

if use_generalized_distance:
    dist_param_h   = 0.05
    dist_param_eps = 0.03
else:
    dist_param_h   = 0
    dist_param_eps = 0

##############################



################################
#     Simulation Settings      #
################################

tmax = 35
dt = 0.01
idx =0

xi_list = []
xi_dot_list = []
t_list   = []

xi   = np.zeros((6, 1))
qdot = np.zeros((6, 1))
##############################


atingiu = False
if simular_movimento:
    for i in range(0, int (tmax/dt)):
        t = i*dt
        
        
        #################################
        #   Gain adjustment near goal   #
        #################################
        
        if idx > int( 0.9 * len(htm_path)):
            # gamma, dist = modulation(curr_state, htm_path[-1], gain=1, lam=15)
            # print("==============================================")
            # print("tempo: ", t)
            # print("distancia ao alvo: ", dist)
            # print("gamma: ", gamma)
            if not atingiu:
                print("atingiu: ", idx , " em t ", t)
                atingiu = True
                kt1 = .5
                kt2 = 1    
                kt3 = 10
                kn1 = 3
                kn2 = 2

    
    
    
        ##########################################
        #   Reference twist from path tracking   #
        ##########################################
        
        curr_jac, curr_state = robot.jac_geo()

        xi_dot, dist, idx = compute_xi_dot(robot, curr_state, htm_path, xi, kt1, kt2, kt3, kn1, kn2, Kv)        
        
        ###############################
        #    Build CBF constraints    #
        ###############################
        
        
        Ad_obj = np.zeros((0, 6))
        Bd_obj = np.zeros((0, 1))
        for ob in all_obs:
            dist, jac_dist, point_robot, point_obs = compute_distance_gradient(robot_ob, ob, curr_state, curr_jac, dist_param_h, dist_param_eps)

            hess_dist = compute_distance_hessian(robot_ob, ob, xi, curr_state, curr_jac, dist_param_h, dist_param_eps)
            b = - (hess_dist @ qdot) -2*param_eta* (jac_dist @ qdot) - (param_eta**2)*(dist - param_obs_delta)

            Ad_obj = np.vstack((Ad_obj, jac_dist ))
            Bd_obj = np.vstack((Bd_obj, b.item() ))


        u_min = u_min.reshape(-1, 1)
        u_max = u_max.reshape(-1, 1)

        A_u = np.vstack((
            np.eye(6),
            -np.eye(6)
        ))

        b_u = np.vstack((
            u_min,
            -u_max
        ))

        Ad_obj = np.vstack((Ad_obj, A_u))
        Bd_obj = np.vstack((Bd_obj, b_u))  
        ######################
        #   QP formulation   #
        ######################
        
        
        H = 2*(curr_jac.T @ curr_jac + eps*np.identity(6))
        f = 2*curr_jac.T@( compute_Jg_dot(robot,qdot,dt)  @ qdot - xi_dot)
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
            sim.save(address="/home/pedro/code/SE3_CBF/se3_rigid_body_second_order",
                    file_name="se3_second_order_cbf_obstacle_avoidance"
                    )
            break
        
        
        #############################
        #       Apply control       #
        #############################
        qdot = qdot + u*dt
        xi = curr_jac @ qdot
        
        xi_dot_list.append(u)# compute_Jg_dot(robot,qdot,dt)  @ qdot + curr_jac @ u)
        xi_list.append(qdot)
        t_list.append(t)
        set_configuration_speed(robot, qdot, t, dt)


# Results
plot_twist(data_list = xi_list,     t = t_list , title = "Twist xi",                  file_name='xi_plot.png')
plot_twist(data_list = xi_dot_list, t = t_list , title = "Twist Acceleration xi_dot", file_name='xi_dot_plot.png')

sim.save(address="/home/pedro/code/SE3_CBF/se3_rigid_body_second_order",
         file_name="se3_second_order_cbf_obstacle_avoidance"
         )
