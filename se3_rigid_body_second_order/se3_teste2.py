import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import uaibot as ub
from scipy.interpolate import CubicSpline, splprep, splev
from scipy.linalg import expm
from setup import *
from aux_functions import *

import jax
import jax.numpy as jnp


def smf(x, order, h):
    x = jnp.where(x < 0.0, 0.0, x)

    def case_h_zero():
        if order == 2:
            return jnp.ones_like(x)
        elif order == 1:
            return x
        elif order == 0:
            return 0.5 * x * x
        else:
            raise ValueError("Invalid order")

    def case_h_nonzero():
        if order == 2:
            return 1.0 - (x + 1.0) ** (-1.0 / h)
        elif order == 1:
            return x - (h / (h - 1.0)) * ((x + 1.0) ** (1.0 - 1.0 / h) - 1.0)
        elif order == 0:
            return (
                0.5 * x * x
                - (h / (h - 1.0))
                * ((h / (2.0 * h - 1.0)) * ((x + 1.0) ** (2.0 - 1.0 / h) - 1.0) - x)
            )
        else:
            raise ValueError("Invalid order")

    return jax.lax.cond(h == 0.0, case_h_zero, case_h_nonzero)

def transform_point(htm, point):
    pc = htm[:3, 3]
    Q = htm[:3, :3]
    pt = Q.T @ (point - pc)
    return pt, Q, pc

def projection_box(point, lx, ly, lz, htm, h, eps):
    pt, Q, pc = transform_point(htm, point)
    x, y, z = pt

    Gx = smf(x - lx / 2, 0, h) + smf(-x - lx / 2, 0, h)
    Gy = smf(y - ly / 2, 0, h) + smf(-y - ly / 2, 0, h)
    Gz = smf(z - lz / 2, 0, h) + smf(-z - lz / 2, 0, h)
    G = Gx + Gy + Gz

    dGx = smf(x - lx / 2, 1, h) - smf(-x - lx / 2, 1, h)
    dGy = smf(y - ly / 2, 1, h) - smf(-y - ly / 2, 1, h)
    dGz = smf(z - lz / 2, 1, h) - smf(-z - lz / 2, 1, h)

    cr = 1.2 * (lx * lx / 4 + ly * ly / 4 + lz * lz / 4)
    R = 0.5 * (x * x + y * y + z * z - cr)

    sigma2 = jnp.maximum(1 - 2 * eps, 0.0)

    F = eps * R
    dFx, dFy, dFz = eps * x, eps * y, eps * z

    M = jnp.sqrt(F * F + sigma2 * G * G + 1e-12)

    ex = dFx + (F * dFx + sigma2 * G * dGx) / M
    ey = dFy + (F * dFy + sigma2 * G * dGy) / M
    ez = dFz + (F * dFz + sigma2 * G * dGz) / M

    pi = jnp.array([x - ex, y - ey, z - ez])

    proj = Q @ pi + pc
    return proj

def projection_cylinder(point, radius, height, htm, h, eps):
    pt, Q, pc = transform_point(htm, point)
    x, y, z = pt

    radius_xy = jnp.sqrt(x * x + y * y + 1e-12)
    delta_r = radius_xy - radius

    Gr = smf(delta_r, 0, h)
    Gz = smf(z - height / 2, 0, h) + smf(-z - height / 2, 0, h)
    G = Gr + Gz

    dGr = smf(delta_r, 1, h)
    dGz = smf(z - height / 2, 1, h) - smf(-z - height / 2, 1, h)

    cr = 1.2 * (2 * radius * radius + height * height / 4)
    R = 0.5 * (x * x + y * y + z * z - cr)

    sigma2 = jnp.maximum(1 - 2 * eps, 0.0)

    F = eps * R
    dFx, dFy, dFz = eps * x, eps * y, eps * z

    M = jnp.sqrt(F * F + sigma2 * G * G + 1e-12)

    # gradiente radial
    nx = x / radius_xy
    ny = y / radius_xy

    ex = dFx + (F * dFx + sigma2 * G * dGr * nx) / M
    ey = dFy + (F * dFy + sigma2 * G * dGr * ny) / M
    ez = dFz + (F * dFz + sigma2 * G * dGz) / M

    pi = jnp.array([x - ex, y - ey, z - ez])

    proj = Q @ pi + pc
    return proj

@jax.jit
def jacobian_box(point, lx, ly, lz, htm, h, eps):
    return jax.jacfwd(projection_box, argnums=0)(
        point, lx, ly, lz, htm, h, eps
    )

@jax.jit
def jacobian_cylinder(point, radius, height, htm, h, eps):
    return jax.jacfwd(projection_cylinder, argnums=0)(
        point, radius, height, htm, h, eps
    )

def jacobian_projection_dispatch(obs, point, h, eps):
    cls_name = obs.__class__.__name__

    if cls_name == 'Box':
        return jacobian_box(
            point=point,
            lx=obs.width,
            ly=obs.depth,
            lz=obs.height,
            htm=obs.htm,
            h=h,
            eps=eps
        )

    elif cls_name == 'Cylinder':
        return jacobian_cylinder(
            point=point,
            radius=obs.radius,
            height=obs.height,
            htm=obs.htm,
            h=h,
            eps=eps
        )

    else:
        raise ValueError(f"Unsupported obstacle type: {cls_name}")

# =========================

def eval_xid_from_state(state_htm, htm_path, kt1, kt2, kt3, kn1, kn2, dt):
    xid, dist, idx = ub.Robot.vector_field_SE3(
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
    
    alpha = modulation(state_htm, htm_path[-1], lam = lambdaa)
    
    return xid * alpha, dist, idx

def compute_distance_gradient(robot_ob, ob, curr_state, curr_jac, dist_param_h, dist_param_eps):    
    s =  curr_state[0:3,-1]
    Jv = curr_jac[0:3,:]
    Jw = curr_jac[3:6,:]  
    no_iter_max = 3000
    tol         = 5e-6
    point_robot, point_obs, dist, _ = robot_ob.compute_dist(ob , h =  dist_param_h, eps = dist_param_eps, tol = tol, no_iter_max = no_iter_max)
    jac_dist = ((point_robot - point_obs).T * Jv + np.cross((point_robot - s ).T, (point_robot - point_obs).T)  * Jw) /(dist + 1e-6)
    return dist, jac_dist, point_robot, point_obs

def compute_distance_gradient2( ob, curr_state, dist_param_h, dist_param_eps):  
    global robot_body  
    robot_ob = ub.Cylinder(htm=curr_state, radius=robot_body.radius, height=robot_body.height)
    s =  curr_state[0:3,-1]
    Jv = np.hstack([np.identity(3), np.zeros((3,3))])
    Jw = np.hstack([np.zeros((3,3)),  np.identity(3)])
    no_iter_max = 3000
    tol         = 5e-6
    point_robot, point_obs, dist, _ = robot_ob.compute_dist(ob , h =  dist_param_h, eps = dist_param_eps, tol = tol, no_iter_max = no_iter_max)
    jac_dist = ((point_robot - point_obs).T @ Jv + np.cross((point_robot - s ).T, (point_robot - point_obs).T)  @ Jw) /(dist + 1e-6)
    return dist, np.matrix(jac_dist), point_robot, point_obs

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

def compute_ud( curr_state, htm_path, xi, kt1, kt2, kt3, kn1, kn2, Kv):
    
        # Reference twist
        xid, dist, idx = eval_xid_from_state(
            state_htm=curr_state,
            htm_path=htm_path,
            kt1=kt1,
            kt2=kt2,
            kt3=kt3,
            kn1=kn1,
            kn2=kn2,
            dt=dt_num,
        )

        # Numerical approximation of reference twist derivative 
        htm_plus = propagate_htm(curr_state, xid, dt_num)
        htm_minus = propagate_htm(curr_state, xid, -dt_num)

        xid_plus, _, _ = eval_xid_from_state(
            state_htm=htm_plus,
            htm_path=htm_path,
            kt1=kt1,
            kt2=kt2,
            kt3=kt3,
            kn1=kn1,
            kn2=kn2,
            dt=dt_num,
        )

        xid_minus, _, _ = eval_xid_from_state(
            state_htm=htm_minus,
            htm_path=htm_path,
            kt1=kt1,
            kt2=kt2,
            kt3=kt3,
            kn1=kn1,
            kn2=kn2,
            dt=dt_num,
        )

        xid_dot = (xid_plus - xid_minus) / (2.0 * dt_num)




        return  ((xid_dot - Kv * (xi - xid)), dist, idx)

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

def compute_distance_hessian_analytical(robot_ob, ob, xi, curr_state, dist_param_h, dist_param_eps):


    dist, jac_dist, a_star, b_star = compute_distance_gradient2(
         ob, curr_state, dist_param_h, dist_param_eps
    )


    dist = float(dist)

    J_Pi_A = np.array(
        jacobian_projection_dispatch(robot_ob, jnp.asarray(b_star).reshape(3,), dist_param_h, dist_param_eps)
    )
    J_Pi_B = np.array(
        jacobian_projection_dispatch(ob, jnp.asarray(a_star).reshape(3,), dist_param_h, dist_param_eps)
    )
    v = np.matrix(xi[0:3 , -1 ]).reshape(3, 1)
    w = np.matrix(xi[3:6 , -1 ]).reshape(3, 1)
    sA = np.matrix(curr_state[0:3, -1])
    psi_A_a = v + ub.Utils.S(w) @ (a_star - sA)
    psi_A_b = v + ub.Utils.S(w) @ (b_star - sA)

    dPi_A_dt = psi_A_a - J_Pi_A @ psi_A_b

    M = np.linalg.inv(np.identity(3) - J_Pi_A * J_Pi_B)
    a_star_dot = M @ dPi_A_dt
    b_star_dot = J_Pi_B * a_star_dot
     
    d_DeltaL_v = (a_star_dot - b_star_dot).T
    d_DeltaL_w =(ub.Utils.S((b_star_dot - v)) @ (a_star - b_star) +  ub.Utils.S((b_star - sA)) @ (a_star_dot - b_star_dot)).T
    d_DeltaL_L = np.hstack((d_DeltaL_v, d_DeltaL_w))

    d_L_dt = (d_DeltaL_L - (jac_dist @ xi) * jac_dist) /(dist + 1e-6)
     

    return d_L_dt



##############################
#     Robot Initialization   #
##############################

sim = ub.Simulation.create_sim_hill()
robot_body = ub.Cylinder(htm= ub.Utils.trn([0 , 0, 0]) * ub.Utils.roty(np.pi), 
                        name="robot_body", 
                        radius=0.3, 
                        height=0.17, 
                        color="cyan", 
                        opacity=0.55)  
robot_3d_model = ub.Model3D(
        url='https://cdn.jsdelivr.net/gh/pedro55562/SE3_CBF_ASSETS@main/TEMA12_DRONA6.obj',
        scale=0.0009, 
        mesh_material=ub.MeshMaterial.create_rough_metal()
        )
robot_frame = ub.Frame(size=0.10)
robot_rigid_3d = ub.RigidObject(list_model_3d=[robot_3d_model],htm=ub.Utils.trn([0 , 0, -.05]) * ub.Utils.roty(np.pi))

robot_UAV = ub.Group(list_of_objects=[robot_body, robot_rigid_3d, robot_frame], htm=ub.Utils.trn([0 , 0, .1])*ub.Utils.roty(np.pi) )
sim.add([robot_UAV])

######################################
#     Workspace & Obstacle Setup     #
######################################


texture_steel = ub.Texture(
            url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/Textures/rough_metal.jpg',
            wrap_s='RepeatWrapping', wrap_t='RepeatWrapping', repeat=[4, 4])

texture_gold = ub.Texture(
            url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/Textures/gold_metal.png',
            wrap_s='RepeatWrapping', wrap_t='RepeatWrapping', repeat=[4, 4])


material_steel= ub.MeshMaterial(metalness=0.7, clearcoat=1, roughness=0.5, normal_scale=[0.5, 0.5], texture_map=texture_steel)
material_gold= ub.MeshMaterial(metalness=0.7, clearcoat=1, roughness=0.5, normal_scale=[0.5, 0.5], texture_map=texture_gold)

material_metal  = ub.MeshMaterial.create_rough_metal()
material_wood   = ub.MeshMaterial.create_wood()
material_colored = ub.MeshMaterial.create_colored_metal(color='red')


piso = ub.Box(htm = ub.Utils.trn([0, 0, -.2]) ,width=7, depth=7, 
                        height = 0.05, mesh_material= material_wood)

teto = ub.Box(htm = ub.Utils.trn([0, 0, 1.74]) ,width=7, depth=7, 
                        height = 0.05, mesh_material= material_wood)

parede_frente = ub.Box(htm = ub.Utils.trn([0, 2, 0.8]) ,   width=3, depth=0.1, 
                        height = 1.9, mesh_material = material_wood)

parede_fundo = ub.Box(htm = ub.Utils.trn([0, 3.5, 0.8]) ,   width=7, depth=0.1, 
                        height = 1.9, mesh_material = material_wood)

parede_lateral = ub.Box(htm = ub.Utils.trn([-1.5, 2.75, 0.8]) * ub.Utils.rotz(np.pi/2) ,   width=1.5, depth=0.1, 
                        height = 1.9, mesh_material = material_wood)

parede_sup = ub.Box(htm = ub.Utils.trn([1.3, 2.42, 1.37]) * ub.Utils.rotz(np.pi/2) ,width=.75, depth=0.1, 
                    height = .95, mesh_material=material_steel)

parede_inf = ub.Box(htm = ub.Utils.trn([1.3, 2.42, -.5]) * ub.Utils.rotz(np.pi/2) ,width=.75, depth=0.1, 
                    height = .95, mesh_material=material_steel)

parede_sup_lat = ub.Box(htm = ub.Utils.trn([1.3, 3.16, 0.8]) * ub.Utils.rotz(np.pi/2) ,width=.74, depth=0.1, 
                    height = 1.9, mesh_material=material_steel)

pilar = ub.Cylinder(htm=ub.Utils.trn([1.35, 1, 1]),height=2, radius=.05, mesh_material=material_steel)

unknown_obs = [parede_sup, parede_sup_lat, pilar] 
known_obs = [parede_frente, piso, teto, parede_fundo, parede_lateral, parede_inf]
all_obs = known_obs + unknown_obs
sim.add(all_obs)
#####################################


##############################
#     Path Planning          #
##############################
     
htm_path = carregar_htm('caminho.txt')

htm_target = htm_path[-1]
frame_target = ub.Frame(htm=htm_target)
sim.add([frame_target])
draw_pc(path=htm_path, sim=sim, color="white", radius = 0.02)
##############################


##############################
#     Control Parameters     #
##############################
dt = 0.005
dt_num = 0.085
t_max = 60.0

kt1 = 9.3
kt2 = .4      
kt3 = 1
       
kn1 = .2
kn2 = .13

lambdaa = 15

Kv = 20.0


param_eta =  1.2
param_obs_delta = 0.01

eps = 1e-3

u_max = np.array([
    [2.5],   # vx
    [2.5],   # vy 
    [2.5],   # vz  
    [2.5],   # wx  
    [2.5],   # wy  
    [2.5]    # wz  
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

xi_list = []
xi_dot_list = []
time_list = []
error = []
last_err = 1
idx = 0

ball_tr = ub.Ball(htm = np.identity(4), radius=0.02, color="cyan")
sim.add([ball_tr])


##############################
#      Simulation Loop       #
##############################

simular_movimento = True

H = np.matrix(robot_UAV.htm)
foi = True
path_followed = []
if simular_movimento:
    xi = np.zeros((6, 1))
                       
    for k in range(int(t_max / dt)):
        if last_err < 0.025:
            print("last_err : ", last_err)
            break
        
        t = k * dt
        
        if idx > 0.7 * len(htm_path):
            if foi:
                print("foiii: " + str(t))
                # kt1 = 6.75
                # kt2 = .9
                # kt3 = 1                                 
                # lambdaa = 16
                # kn1 = .63
                # kn2 = .41
                
                kt1 = 5.65
                kt2 = .8
                kt3 = 1                                 
                lambdaa = 24
                kn1 = .85
                kn2 = .69
                foi = False

        ##########################################
        #   Reference twist from path tracking   #
        ##########################################

        ud, dist, idx = compute_ud(H, htm_path, xi, kt1, kt2, kt3, kn1, kn2, Kv) 


        ###############################
        #    Build CBF constraints    #
        ###############################
        
        
        Ad_obj = np.zeros((0, 6))
        Bd_obj = np.zeros((0, 1))
        for ob in all_obs:
            dist, jac_dist, point_robot, point_obs = compute_distance_gradient2( ob, H, dist_param_h, dist_param_eps)
            hess_dist_ana_1 = compute_distance_hessian_analytical(robot_body, ob, xi, H, dist_param_h, dist_param_eps)
            

            b = - (hess_dist_ana_1 * xi) -2*param_eta* (jac_dist @ xi) - (param_eta**2)*(dist - param_obs_delta)
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
        
        
        H_qp =  2*np.identity(6)
        f    = -2*ud
        try:
            u = ub.Utils.solve_qp(
                H=H_qp,
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
        xi = xi + u*dt

        H = propagate_htm(H, xi, dt)
        
        robot_UAV.add_ani_frame(time = t, htm = H)
        ball_tr.add_ani_frame(time = t, htm=htm_path[idx])
        
        # Store some useful data
        xi_list.append(xi)
        xi_dot_list.append(u)
        time_list.append(t)
        path_followed.append(H)
        error.append(np.linalg.norm(log_SE3(ub.Utils.inv_htm(H) @ htm_target)))
        last_err = error[-1]


##############################
#          Results           #
##############################
if len(path_followed) > 0:
    # draw_pc(path_followed, sim, "magenta", 0.01)
    print("last_err last msm : ", error[-1])

plot_vector_list(
    xi_list,
    time_list,
    file_name="xi_plot.png",
    labels=[r'$v_x$', r'$v_y$', r'$v_z$', r'$\omega_x$', r'$\omega_y$', r'$\omega_z$'],
    xlabel='Time (s)',
    ylabel=r'$\xi$',
    show_plot=False,
    title='System Twist'
)

plot_vector_list(
    xi_dot_list,
    time_list,
    file_name="u_plot.png",
    labels=[r'$\dot{v}_x$', r'$\dot{v}_y$', r'$\dot{v}_z$', r'$\dot{\omega}_x$', r'$\dot{\omega}_y$', r'$\dot{\omega}_z$'],
    xlabel='Time (s)',
    ylabel=r'$u$',
    show_plot=False,
    title='Control Input'
)

plot_vector_list(
    error,
    time_list,
    file_name="error.png",
    labels=[r'$d$'],
    xlabel='Time (s)',
    ylabel=r'$d$',
    show_plot=False,
    title='Pose Error'
)

sim.save(
    address="/home/pedro/code_robot/SE3_CBF/se3_rigid_body_second_order",
    file_name="se3_teste",
)