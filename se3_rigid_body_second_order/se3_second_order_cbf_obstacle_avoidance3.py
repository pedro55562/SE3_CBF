import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import uaibot as ub

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
        return xid_dot - Kv * (xi - xid), dist, idx

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
#

# pose inicial do robô
htm_start = robot.fkm()

# alvo: deslocamento moderado, para frente e um pouco para o lado / alto
htm_target = ub.Utils.trn([0.2, 2.8, 1.0]) * htm_start * ub.Utils.rotx(np.pi/10) * ub.Utils.rotz(np.pi/8)
frame_target = ub.Frame(htm=htm_target)
sim.add([frame_target])

# ambiente
piso = ub.Box(
    htm=ub.Utils.trn([0, 0, -0.35]),
    width=7, depth=7, height=0.05,
    color='red'
)

teto = ub.Box(
    htm=ub.Utils.trn([0, 0, 1.55]),
    width=7, depth=7, height=0.05,
    color='red', opacity=0.25
)

# caixa que bloqueia o caminho reto entre início e alvo
# ela fica aproximadamente no corredor entre os dois pontos
caixa_bloqueio = ub.Box(
    htm=ub.Utils.trn([0.0, 1.45, 0.75]),
    width=0.9, depth=0.9, height=1.4,
    color='orange', opacity=0.85
)

# paredes opcionais só para dar contexto visual
parede_esq = ub.Box(
    htm=ub.Utils.trn([-1.8, 1.8, 0.75]) * ub.Utils.rotz(np.pi/2),
    width=2.8, depth=0.08, height=1.6,
    color='red', opacity=0.25
)

parede_dir = ub.Box(
    htm=ub.Utils.trn([1.8, 1.8, 0.75]) * ub.Utils.rotz(np.pi/2),
    width=2.8, depth=0.08, height=1.6,
    color='red', opacity=0.25
)

all_obs = [piso, teto, caixa_bloqueio, parede_esq, parede_dir]
sim.add(all_obs)

######################################
# Smooth Non-Straight Geometric Path
######################################

# ponto inicial e final
p0 = htm_start[0:3, 3]
pf = htm_target[0:3, 3]

# waypoints escolhidos manualmente para forçar um desvio em torno da caixa
# caminho com curvatura suave e comprimento médio
control_points = np.array([
    p0,
    p0 + np.array([0.00, 0.45, 0.05]),
    p0 + np.array([0.85, 1.05, 0.20]),   # vai para a direita
    p0 + np.array([1.05, 1.85, 0.25]),   # contorna a caixa
    p0 + np.array([0.55, 2.35, 0.12]),   # volta em direção ao alvo
    pf
])

def catmull_rom_spline(points, samples_per_segment=80):
    """
    Interpolação Catmull-Rom para gerar caminho suave passando pelos waypoints.
    Alta amostragem via samples_per_segment.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)

    # extensão das pontas para tratar início/fim
    ext = np.vstack([points[0], points, points[-1]])

    curve = []

    for i in range(1, n):
        P0 = ext[i - 1]
        P1 = ext[i]
        P2 = ext[i + 1]
        P3 = ext[i + 2]

        for j in range(samples_per_segment):
            t = j / samples_per_segment
            t2 = t * t
            t3 = t2 * t

            pt = 0.5 * (
                (2 * P1)
                + (-P0 + P2) * t
                + (2*P0 - 5*P1 + 4*P2 - P3) * t2
                + (-P0 + 3*P1 - 3*P2 + P3) * t3
            )
            curve.append(pt)

    curve.append(points[-1])
    return np.array(curve)

# alta amostragem
curve_points = catmull_rom_spline(control_points, samples_per_segment=100)

######################################
# Build HTM Path
######################################

# vamos manter orientação constante só para testar a geometria do caminho.
# depois, se quiser, dá para fazer a orientação "olhar" para a tangente do caminho.
R0 = htm_start[0:3, 0:3]

htm_path = []
for p in curve_points:
    H = np.eye(4)
    H[0:3, 0:3] = R0
    H[0:3, 3] = p
    htm_path.append(H)

# desenha o caminho interpolado
draw_pc(path=htm_path, sim=sim, color="white", radius=0.02)


##############################
#     Control Parameters     #
##############################

kt1 = 1.8     
kt2 = 0.30          
kt3 = 0.30         
kn1 = 0.20          
kn2 = 0.20

Kv = 20

param_eta =  1.3
param_obs_delta = 0.02

eps = 1e-3



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

tmax = 30
dt = 0.01
idx =0

xi_list = []
xi_dot_list = []
t_list   = []

xi   = np.zeros((6, 1))
qdot = np.zeros((6, 1))
##############################

simular_movimento = False
atingiu = False
if simular_movimento:
    for i in range(0, int (tmax/dt)):
        t = i*dt
        
        
        #################################
        #   Gain adjustment near goal   #
        #################################
        
        if idx > int( 0.94 * len(htm_path)):
            if not atingiu:
                print("atingiu: ", idx , " em t ", t)
                atingiu = True
            # kt1 = 0.40         
            # kt2 = 0.80          
            # kt3 = 0.80          
            # kn1 = 1.2      
            # kn2 = 1.2
            # param_eta =  0.3
    
    
    
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
            sim.save(address="/home/pedro/code/SE3_CBF/se3_rigid_body_second_order/",
                    file_name="se3_second_order_cbf_obstacle_avoidance"
                    )
            break
        
        
        #############################
        #       Apply control       #
        #############################
        qdot = qdot + u*dt
        xi = curr_jac*qdot
        
        xi_dot_list.append(xi_dot)
        xi_list.append(xi)
        t_list.append(t)
        set_configuration_speed(robot, qdot, t, dt)


# Results
plot_twist(data_list = xi_list,     t = t_list , title = "Twist xi",                  file_name='xi_plot.png')
plot_twist(data_list = xi_dot_list, t = t_list , title = "Twist Acceleration xi_dot", file_name='xi_dot_plot.png')

sim.save(address="/home/pedro/code/SE3_CBF/se3_rigid_body_second_order/",
         file_name="se3_second_order_cbf_obstacle_avoidance"
         )
