import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import uaibot as ub

from scipy.interpolate import CubicSpline, splprep, splev
from scipy.linalg import expm

from setup import *
from aux_functions import *

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
    xid[0:3, :] = xid[0:3, :] + ub.Utils.S(xid[3:6, :]) @ state_htm[0:3, -1].reshape(3, 1)
    
    alpha = modulation(state_htm, htm_path[-1], lam = 22)
    
    return xid * alpha, dist, idx



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
    
        # Reference twist
        xid, dist, idx = eval_xid_from_state(
            state_htm=curr_state,
            robot=robot,
            htm_path=htm_path,
            kt1=kt1,
            kt2=kt2,
            kt3=kt3,
            kn1=kn1,
            kn2=kn2,
            dt=dt,
        )

        # Numerical approximation of reference twist derivative 
        htm_plus = propagate_htm(curr_state, xi, dt)
        htm_minus = propagate_htm(curr_state, xi, -dt)

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




        return  ((xid_dot - Kv * (xi - xid)), dist, idx)

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




def modulation(H: np.ndarray, H_target: np.ndarray, lam: float) -> float:
    d = np.linalg.norm(log_SE3(ub.Utils.inv_htm(H) @ H_target))
    return (1.0 - np.exp(-lam * d) * (1.0 + lam * d))

def plot_rotation_components(htm_path):
    """
    htm_path: list of 3x4 numpy arrays (homogeneous transform matrices without last row)
    """

    n = len(htm_path)
    indices = np.arange(n)

    # Prepare storage for the 9 components
    R_components = np.zeros((n, 3, 3))

    for i, H in enumerate(htm_path):
        R_components[i] = H[0:3, 0:3]

    # Create 9 subplots (3x3 grid)
    fig, axes = plt.subplots(3, 3, figsize=(10, 8))

    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            ax.plot(indices, R_components[:, row, col])
            ax.set_title(f"R[{row},{col}]")
            ax.set_xlabel("i")
            ax.set_ylabel("value")
            ax.grid(True)

    plt.tight_layout()
    plt.show()


##############################
#     Robot Initialization   #
##############################

robot = ub.Robot.create_rigid_body_se3()
sim = ub.Simulation(background_color="black")
sim.add([robot])

collision_objects = []

for link in robot.links:
    for col_obj_data in link.col_objects:
        obj = col_obj_data[0]
        collision_objects.append(obj)
robot_ob = collision_objects[0]

sim.add(collision_objects)

######################################
#     Workspace & Obstacle Setup     #
######################################

htm_target = ub.Utils.trn([-.9, 2.6, .85]) * robot.fkm() * ub.Utils.rot(axis=[4,3,-2], angle= 77 * np.pi / 180) 
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
simular_movimento =  True 
if gerar_novo_caminho:
    c_space_path1 = carregar_caminho(caminho_arquivo)
    q_goal = robot.ikm(htm_tg=htm_target, obstacles=known_obs, no_tries=2000, no_iter_max=4000)
    success1, path2, iterations1, num_tries1, planning_time1 = robot.runSE3RRT(q0=c_space_path1[-1], q_goal=[q_goal], obstacles=all_obs)
    path2 = [np.asarray(x).reshape(-1) for x in path2]
    c_space_path1 = c_space_path1 + path2
    print(c_space_path1[0])
    c_space_path1 = smooth_path(path= c_space_path1)
    salvar_caminho(c_space_path1, caminho_arquivo)
    
else:
    c_space_path1 = carregar_caminho(caminho_arquivo)
    
htm_path = []
for qc in c_space_path1:
    htm_path.append(robot.fkm(q=qc))
draw_pc(path=htm_path, sim=sim, color="white", radius = 0.02)

##############################


##############################
#     Control Parameters     #
##############################

kt1 = 1.5
kt2 = .9        
kt3 = .2
       
kn1 = 0.1
kn2 = 0.1

Kv = 10.0


param_eta =  1.5
param_obs_delta = 0.01

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
#     Simulation Settings    #
##############################

dt = 0.01
t_max = 25.0

xi_list = []
xi_dot_list = []
time_list = []
error = []
last_err = 1
idx = 0
##############################
#      Simulation Loop       #
##############################
foi = True
path_followed = []
if simular_movimento:
    xi = np.zeros((6, 1))
    qdot = np.zeros((6, 1))

    for k in range(int(t_max / dt)):
        if last_err < 0.025:
            print("last_err : ", last_err)
            break
        
        t = k * dt
        
        if idx > 0.8 * len(htm_path):
            if foi:
                print("foiii: " + str(t))
                kt1 = 1.6
                kt2 = .7      
                kt3 = 1
                    
                kn1 = 3
                kn2 = 2.3   
                foi = False   
        
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
            sim.save(address="/home/pedro/code_robot/SE3_CBF/se3_rigid_body_second_order",
                    file_name="se3_second_order_cbf_obstacle_avoidance"
                    )
            break


        #############################
        #       Apply control       #
        #############################
        qdot = qdot + u*dt
        xi = curr_jac @ qdot 

        set_configuration_speed(robot, qdot, t, dt)
       
        # Store some useful data
        xi_list.append(xi)
        xi_dot_list.append(compute_Jg_dot(robot,qdot,dt) @ qdot + curr_jac @ u)
        time_list.append(t)
        path_followed.append(curr_state)
        error.append(np.linalg.norm(log_SE3(ub.Utils.inv_htm(robot.fkm()) @ htm_path[-1])))
        last_err = error[-1]


##############################
#          Results           #
##############################

if len(path_followed) > 0:
    draw_pc(path_followed, sim, "magenta", 0.01)

plot_vector_list(xi_list, time_list, "Twist xi", "xi_plot.png")
plot_vector_list(xi_dot_list, time_list, "Twist Acceleration xi_dot", "xi_dot_plot.png")
plot_vector_list(error, time_list, "error d = ||Log(H^(-1) H_tg)||F", "error.png")

sim.save(
    address="/home/pedro/code_robot/SE3_CBF/se3_rigid_body_second_order",
    file_name="se3_teste",
)