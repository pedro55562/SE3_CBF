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
robot.add_ani_frame(time=0, q=[0.3, 2, -2, 0, np.pi/2, 0])
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

##############################
#     Path Planning          #
##############################

height = .3
side = 4.0
step = 0.01

center_xy = (0, 0)
cx, cy = center_xy
center = np.array([cx, cy])

h = side / 2

v1 = np.array([cx - h, cy - h])
v2 = np.array([cx + h, cy - h])
v3 = np.array([cx + h, cy + h])
v4 = np.array([cx - h, cy + h])

vertices = [v1, v2, v3, v4, v1]

htm_path = []

for i in range(len(vertices) - 1):
    p_start = vertices[i]
    p_end = vertices[i + 1]

    dist = np.linalg.norm(p_end - p_start)
    num_points = max(2, int(dist / step))

    for j in range(num_points):
        alpha = j / num_points
        p_xy = (1 - alpha) * p_start + alpha * p_end

        p = np.array([p_xy[0], p_xy[1], height])

        #d = center - p_xy
        #yaw = np.arctan2(d[1], d[0])
        #htm = ub.Utils.trn(p) @ ub.Utils.rotz(yaw) @ ub.Utils.rotx(np.pi)
        
        htm = ub.Utils.trn(p) @ ub.Utils.rotx(np.pi)
       
        htm_path.append(htm)
draw_pc(path=htm_path, sim=sim, color="white", radius = 0.03)

##############################


######################################
#     Workspace & Obstacle Setup     #
######################################

obstacles = []

obstacles.append(
    ub.Box(
        htm=ub.Utils.trn([-2.0, 0.0, height]), 
        width=0.6,    
        depth=0.2,    
        height=0.6,   
        color='red',
        opacity=0.3
    )
)

obstacles.append(
    ub.Box(
        htm=ub.Utils.trn([2.0, 0.0, -.3]),  
        width=2,
        depth=0.2,
        height= 1.5,   
        color='blue',
        opacity=0.3
    )
)


# gap_width = 0.6      
# box_width = 2    
# length = .2        
# height_box = 2     

# x_obs = -2.0         
# y_obs = -1
# obstacles.append(
#     ub.Box(
#         htm=ub.Utils.trn([ - (gap_width/2 + box_width/2) + x_obs, y_obs, height ]),
#         width=box_width,
#         depth=length,
#         height=height_box,
#         color='red',
#         opacity=0.3
#     )
# )

# obstacles.append(
#     ub.Box(
#         htm=ub.Utils.trn([ + (gap_width/2 + box_width/2) + x_obs, y_obs, height ]),
#         width=box_width,
#         depth=length,
#         height=height_box,
#         color='red',
#         opacity=0.3
#     )
# )

all_obs = obstacles 
sim.add(all_obs)
#####################################


##############################
#     Control Parameters     #
##############################

kt1 = 1.30
kt2 = 0.95         
kt3 = 0.2
       
kn1 = 0.20          
kn2 = 0.10

Kv = 20

param_eta =  0.95
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

tmax = 50
dt = 0.01
idx =0

xi_list = []
xi_dot_list = []
t_list   = []

xi   = np.zeros((6, 1))
qdot = np.zeros((6, 1))
##############################

simular_movimento = True
if simular_movimento:
    for i in range(0, int (tmax/dt)):
        t = i*dt
        
    
        ##########################################
        #   Reference twist from path tracking   #
        ##########################################
        
        curr_jac, curr_state = robot.jac_geo()

        xid_dot, dist, idx = compute_xi_dot(robot, curr_state, htm_path, xi, kt1, kt2, kt3, kn1, kn2, Kv)        
        
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
        f = 2*curr_jac.T@( compute_Jg_dot(robot,qdot,dt)  @ qdot - xid_dot)
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
            sim.save(address="/home/pedro/code_robot/SE3_CBF/se3_rigid_body_second_order/",
                    file_name="se3_second_order_cbf_obstacle_avoidance"
                    )
            break
        
        
        #############################
        #       Apply control       #
        #############################
        qdot = qdot + u*dt
        xi = curr_jac*qdot
        
        xi_dot_list.append(xid_dot)
        xi_list.append(xi)
        t_list.append(t)
        set_configuration_speed(robot, qdot, t, dt)


# Results
plot_twist(data_list = xi_list,     t = t_list , title = "Twist xi",                  file_name='xi_plot.png')
plot_twist(data_list = xi_dot_list, t = t_list , title = "Twist Acceleration xi_dot", file_name='xi_dot_plot.png')

sim.save(address="/home/pedro/code_robot/SE3_CBF/se3_rigid_body_second_order/",
         file_name="se3_second_order_cbf_obstacle_avoidance"
         )
