import uaibot as ub
import numpy as np
import matplotlib.pyplot as plt
import os
from setup import *
import itertools
from scipy.interpolate import CubicSpline
from scipy.interpolate import splprep, splev
from scipy.linalg import logm


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
    
    gamma = modulation(state_htm, htm_path[-1], gain=1, lam=5)
    
    return gamma * xid, dist, idx

def compute_obstacle_distance_gradient(robot_ob, ob, Jv, Jw, s, h, eps):      
    point_robot, point_obs, dist, _ = robot_ob.compute_dist(ob , h =  h, eps = eps)
    jac_dist = ((point_robot - point_obs).T * Jv + np.cross((point_robot - s ).T, (point_robot - point_obs).T)  * Jw) / (dist + 1e-5)
    return dist, jac_dist, point_robot, point_obs

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
    return (1.0 - np.exp(-lam * d) * (1.0 + lam * d))
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

dist = 4
step = 0.01

htm_start = robot.fkm()
htm_path = []
for i in range(0, int(dist/step)):
    alpha = i / (dist/step)
    htm_path.append(ub.Utils.trn([alpha * dist, 0, 0]) * robot.fkm())

htm_target = htm_path[-1]
draw_pc(path=htm_path, sim=sim, color="white", radius = 0.02)

frame_target = ub.Frame(htm=htm_target)
sim.add([frame_target])

###############################



##############################
#     Control Parameters     #
##############################

kt1 = 0.8        
kt2 = 0.3           
kt3 = 0.3         
kn1 = 0.1         
kn2 = 0.1

param_eta = 1.5
param_obs_delta = 0.02

eps = 1e-3

xi_max = np.array([
    [0.8],   # vx  [m/s]
    [0.8],   # vy  [m/s]
    [0.8],   # vz  [m/s]
    [0.8],   # wx  [rad/s]
    [0.8],   # wy  [rad/s]
    [1.0]    # wz  [rad/s]
])

xi_min = -xi_max
##############################



################################
#     Simulation Settings      #
################################

tmax = 30
dt = 0.01
idx =0
xid_list = []
t_list   = []

##############################


atingiu = False
simular_movimento = True
if simular_movimento:
    for i in range(0, int (tmax/dt)):
        t = i*dt
        
        
        #################################
        #   Gain adjustment near goal   #
        #################################
        

        # if idx > int( 0.92 * len(htm_path)):
        #     if not atingiu:
        #         print("atingiu: ", idx , " em t ", t)
        #         atingiu = True
        #     kt1 = .8           
        #     kt2 = .4           
        #     kt3 = .4          
        #     kn1 = 1.3        
        #     kn2 = 1.3
    
    
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

        u = ub.Utils.dp_inv_solve(A=curr_jac, b=xid, eps = eps)
        
        # Apply control
        xid_list.append(u)
        t_list.append(t)
        set_configuration_speed(robot, u, t, dt)
        
# Results
plot_twist(data_list = xid_list, t = t_list , title = "xi", file_name='xi_plot.png')
sim.save(address="/home/pedro/code/SE3_CBF/se3_rigid_body_first_order/",
         file_name="se3_teste"
         )
