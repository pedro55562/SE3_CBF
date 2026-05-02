import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import uaibot as ub
from scipy.interpolate import CubicSpline, splprep, splev
from scipy.linalg import expm
from setup import *
from aux_functions import *
from scipy.spatial.transform import Rotation as R, Slerp

def upsample_transformations(T_list, factor):
    """
    Upsample de uma lista de matrizes homogêneas (4x4).

    Args:
        T_list (list of np.matrix or np.ndarray): lista de matrizes 4x4
        factor (int): número de interpolações entre cada par

    Returns:
        list of np.ndarray: nova lista upsampled
    """

    T_list = [np.asarray(T) for T in T_list]
    result = []

    for i in range(len(T_list) - 1):
        T0 = T_list[i]
        T1 = T_list[i + 1]

        # Separar rotação e translação
        R0, t0 = T0[:3, :3], T0[:3, 3]
        R1, t1 = T1[:3, :3], T1[:3, 3]

        # Converter rotações para quaternions
        rot_seq = R.from_matrix([R0, R1])
        key_times = [0, 1]
        slerp = Slerp(key_times, rot_seq)

        # Gerar interpolações
        times = np.linspace(0, 1, factor + 2)

        for t in times[:-1]:  # evita duplicar o último
            # Interpolar rotação
            R_interp = slerp([t]).as_matrix()[0]

            # Interpolar translação
            t_interp = (1 - t) * t0 + t * t1

            # Montar matriz homogênea
            T_interp = np.eye(4)
            T_interp[:3, :3] = R_interp
            T_interp[:3, 3] = t_interp

            result.append(T_interp)

    # Adicionar última transformação
    result.append(T_list[-1])

    return result

def cmpt_lambda_terms(objA,objB,H,xi,h,eps):
    
    #Extract:
    sA = H[0:3,-1]
    vA = xi[0:3,-1]
    wA = xi[3:6,-1]
    
    #Put objA in the pose, temporarily
    htm_A = np.matrix(objA.htm)
    objA.set_ani_frame(H)
    
    #Compute dist
    a_star, b_star, lambda_AB, _ = objA.compute_dist(obj = objB, h = h, eps = eps, tol=1e-6, no_iter_max=6000)
    
    #Compute D_xi_lambda_AB
    D_xi_lambda_AB = np.matrix(np.zeros((1,6)))
    D_xi_lambda_AB[:,0:3] = (a_star-b_star).T/(1e-6+lambda_AB)
    D_xi_lambda_AB[:,3:6] = (ub.Utils.S(b_star-sA)*(a_star-b_star)).T/(1e-6+lambda_AB)
     
    #Compute (d/dt) D_xi_lambda_AB
    psi_A_a = vA + ub.Utils.S(wA)*(a_star-sA)
    psi_A_b = vA + ub.Utils.S(wA)*(b_star-sA)
    J_Pi_A = np.matrix(objA.cpp_obj.projection_jacobian(b_star,h,eps))
    J_Pi_B = np.matrix(objB.cpp_obj.projection_jacobian(a_star,h,eps))
    d_Pi_A_dt = psi_A_a - J_Pi_A*psi_A_b

        
    a_star_dot = np.linalg.inv(np.identity(3)-J_Pi_A*J_Pi_B)*d_Pi_A_dt
    b_star_dot = J_Pi_B*a_star_dot
    
    d_D_xi_Lambda_AB_dt = np.matrix(np.zeros((1,6)))
    d_D_xi_Lambda_AB_dt[:,0:3] = (a_star_dot-b_star_dot).T
    d_D_xi_Lambda_AB_dt[:,3:6] = (ub.Utils.S(b_star-sA)*(a_star_dot-b_star_dot)+ub.Utils.S(b_star_dot-vA)*(a_star-b_star)).T
    
    d_D_xi_lambda_AB_dt = (d_D_xi_Lambda_AB_dt - (D_xi_lambda_AB*xi)*D_xi_lambda_AB)/(1e-6+lambda_AB)

    #Put back the initial pose of the object
    objA.set_ani_frame(htm_A)
    
    return lambda_AB, D_xi_lambda_AB, d_D_xi_lambda_AB_dt
    
def cmpt_control(H,xi,obj_robot,list_obs,u_d,h=0.05,eps=0.01,eta=0.3,lambda_min=0.01,xi_lim=0.08):
    falhou = False
    A = np.matrix(np.zeros((0,6)))
    b = np.matrix(np.zeros((0,1)))
    
    #Add the constraints for obstacle avoidance
    for obs in list_obs:
        lambda_RO, D_xi_lambda_RO, d_D_xi_lambda_RO_dt = cmpt_lambda_terms(obj_robot,obs,H,xi,h,eps)
        
        ff = -d_D_xi_lambda_RO_dt*xi
        d_lambda_RO_dt = D_xi_lambda_RO*xi
        b_temp = ff - 2*eta*d_lambda_RO_dt-eta*eta*(lambda_RO-lambda_min)
        
        A = np.vstack([A,D_xi_lambda_RO])
        b = np.vstack([b,b_temp])
        
        
    #Add limits for xi
    A = np.vstack([A,np.identity(6)])
    b = np.vstack([b,-xi_lim*np.ones((6,1))])
    A = np.vstack([A,-np.identity(6)])
    b = np.vstack([b,-xi_lim*np.ones((6,1))]) 
    
    #Solve the QP
    try:
        u = ub.Utils.solve_qp(2*np.identity(6),-2*u_d,A,b)   
    except:
        falhou = True
        print("\n QP Falhou!  ")
        print("Tempo: ", t)
        sim.save(address=os.path.dirname(__file__),
                file_name="se3_teste"
                )
        return 0, falhou


    return u, falhou
        
def g_func(s, K=1.0):
    s = np.asarray(s)
    return np.matrix(-K * s / (np.sqrt(np.abs(s)) + 1e-6))

def cmpt_control_reactive(H, xi, H_d, kc=0.5):
#Spatial acceleration to reach a constant target pose Hd    
    
    def ext(M):
        return np.matrix(np.diag(M)).T
    
    s = H[0:3,-1]
    s_d = H_d[0:3,-1]
    Q = H[0:3,0:3]
    Q_d = H_d[0:3,0:3]
    
    v = xi[0:3,-1]
    w = xi[3:6,-1]
    Sx = ub.Utils.S([1,0,0])
    Sy = ub.Utils.S([0,1,0])
    Sz = ub.Utils.S([0,0,1])
    Sw = ub.Utils.S(w)
    
    r = np.matrix(np.zeros((6,1)))
    r[0:3,-1] = s-s_d
    r[3:6,-1] = ext(np.identity(3)-Q_d.T*Q)
    
    # #(d/dt) r = D_xi_r*xi
    D_xi_r = np.matrix(np.zeros((6,6)))
    D_xi_r[0:3,:] = np.hstack([np.identity(3), np.zeros((3,3))])
    ax = ext(-Q_d.T*Sx*Q)
    ay = ext(-Q_d.T*Sy*Q)
    az = ext(-Q_d.T*Sz*Q)
    D_xi_r[3:6,:] = np.hstack([np.zeros((3,3)), ax, ay, az]) 
    
    #d_D_xi_r_dt = (d/dt) D_xi_r
    d_D_xi_r_dt = np.matrix(np.zeros((6,6)))
    dax = ext(-Q_d.T*Sx*Sw*Q)
    day = ext(-Q_d.T*Sy*Sw*Q)
    daz = ext(-Q_d.T*Sz*Sw*Q)
    d_D_xi_r_dt[3:6,:] = np.hstack([np.zeros((3,3)), dax, day, daz]) 
    
    #Compute u_d
    # g_r = g_func(r, K=kc)

    u_d = ub.Utils.dp_inv(D_xi_r)*(-d_D_xi_r_dt*xi-2*kc*D_xi_r*xi-kc*kc*r) 

    return u_d, r


# =========================

def eval_xid_from_state(state_htm, htm_path, xi, kt1, kt2, kt3, kn1, kn2, dt):
    
    xid, tangent, normal, dist, idx = ub.Robot.vector_field_SE3(
        state=state_htm,
        curve=htm_path,
        kt1=kt1,
        kt2=kt2,
        kt3=kt3,
        kn1=kn1,
        kn2=kn2,
        ds=dt,
        delta=1e-2,
    )
    
    xid = np.asarray(xid, dtype=float).reshape(6, 1)
    xid[0:3, :] = xid[0:3, :] + ub.Utils.S(xid[3:6, :]) @ state_htm[0:3, -1].reshape(3, 1)
    
    alpha = modulation(state_htm, htm_path[-1], lam = lambdaa)
    
    return xid * alpha, dist, idx

    # xid = np.matrix(xid)
    # alpha = modulation(state_htm, htm_path[-1], lam = lambdaa)    

    # xid[0:3, :] = xid[0:3,-1] + ub.Utils.S(xid[3:6, :-1]) @ state_htm[0:3, -1]
    
    # return alpha*xid, dist, idx

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
            xi=xi,
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
            xi =xi,
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
            xi = xi,
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
    d = np.linalg.norm(log_SE3(ub.Utils.inv_htm(np.matrix(H)) @ H_target))
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

def butter3_tustin_coeffs(wc, dt):
    K = 2.0 / dt

    A0 = K**3 + 2*wc*K**2 + 2*wc**2*K + wc**3
    A1 = -3*K**3 - 2*wc*K**2 + 2*wc**2*K + 3*wc**3
    A2 =  3*K**3 - 2*wc*K**2 - 2*wc**2*K + 3*wc**3
    A3 = -K**3 + 2*wc*K**2 - 2*wc**2*K + wc**3

    B0 = wc**3
    B1 = 3*wc**3
    B2 = 3*wc**3
    B3 = wc**3

    a1 = A1 / A0
    a2 = A2 / A0
    a3 = A3 / A0

    b0 = B0 / A0
    b1 = B1 / A0
    b2 = B2 / A0
    b3 = B3 / A0

    return b0, b1, b2, b3, a1, a2, a3


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
robot_body_copy = robot_body.copy() 
######################################
#     Workspace & Obstacle Setup     #
######################################


# texture_steel = ub.Texture(
#             url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/Textures/rough_metal.jpg',
#             wrap_s='RepeatWrapping', wrap_t='RepeatWrapping', repeat=[4, 4])

# material_steel= ub.MeshMaterial(metalness=0.7, clearcoat=1, roughness=0.5, normal_scale=[0.5, 0.5], texture_map=texture_steel)

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
                    height = .95, mesh_material=material_metal)

parede_inf = ub.Box(htm = ub.Utils.trn([1.3, 2.42, -.5]) * ub.Utils.rotz(np.pi/2) ,width=.75, depth=0.1, 
                    height = .95, mesh_material=material_metal)

parede_sup_lat = ub.Box(htm = ub.Utils.trn([1.3, 3.16, 0.8]) * ub.Utils.rotz(np.pi/2) ,width=.74, depth=0.1, 
                    height = 1.9, mesh_material=material_metal)

pilar = ub.Cylinder(htm=ub.Utils.trn([1.35, 1, 1]),height=2, radius=.05, mesh_material=material_metal)

unknown_obs = [parede_sup, parede_sup_lat, pilar] 
known_obs = [parede_frente, piso, teto, parede_fundo, parede_lateral, parede_inf]
all_obs = known_obs + unknown_obs
sim.add(all_obs)
#####################################


##############################
#     Path Planning          #
##############################
     
htm_path = carregar_htm('caminho.txt')

# print(len(htm_path))
# htm_path = upsample_transformations(htm_path, 1)
# print(len(htm_path))

htm_target = np.matrix(htm_path[-1])
frame_target = ub.Frame(htm=htm_target)
sim.add([frame_target])
draw_pc(path=htm_path, sim=sim, color="white", radius = 0.02)
##############################
# 0.005

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

Kv = 20

wc = 30 * 2 * np.pi
alpha = np.exp(- wc * dt)

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
ud_list = []
v_dot_list = []
w_dot_list = []
time_list = []
error = []
last_err = 1
idx = 0

ball_tr = ub.Ball(htm = np.identity(4), radius=0.02, color="cyan")
sim.add([ball_tr])


##############################
#      Simulation Loop       #
##############################

b0, b1, b2, b3, a1, a2, a3 = butter3_tustin_coeffs(wc=wc, dt=dt)

simular_movimento = True
falhou = False
H = np.matrix(robot_UAV.htm)
final = True
path_followed = []
r = 10
idx = 0
temp = True

target_s = htm_target[0:3, 3]
if simular_movimento:
    xi = np.matrix(np.zeros((6,1)))
    uds1 = np.matrix(np.zeros((6,1)))
    uds2 = np.matrix(np.zeros((6,1)))
    uds3 = np.matrix(np.zeros((6,1)))
    
    
    
    s = H[0:3 , 3] 
    for k in range(int(t_max / dt)):
        if last_err < 0.025:
            print("last d(H) : ", last_err)
            break
        
        t = k * dt
        
        if idx > 0.72 * len(htm_path):
            if final:
                print("Final: ",t)
                final = False
            # Ganhos originais
            # kt1 = 9.3
            # kt2 = .9      
            # kt3 = 1                    
            # kn1 = .25
            # kn2 = .16        
            
            kt1 = 6
            kt2 = .9
            kt3 = 1                                 
            lambdaa = 7
            kn1 = .45
            kn2 = .35 
        

            # erro 0.030
            # kt1 = 6
            # kt2 = .9
            # kt3 = 1                                 
            # lambdaa = 7
            # kn1 = .45
            # kn2 = .35 
            
            # ganhos salvos( waypoint )
            # kt1 = 6.5
            # kt2 = .9
            # kt3 = 1                                 
            # lambdaa = 6
            # kn1 = .35
            # kn2 = .25  
                

        ##########################################
        #   Reference twist from path tracking   #
        ##########################################

        ud, dist, idx = compute_ud(H, htm_path, xi, kt1, kt2, kt3, kn1, kn2, Kv) 
        ud_list.append(ud)
        if k > 3 :
            uds0 = -a1*uds1 - a2*uds2 - a3*uds3 + b0 * ud_list[-1] + b1 * ud_list[-2] + b2 * ud_list[-3] + b3 * ud_list[-4]
            uds3 = np.matrix(uds2)
            uds2 = np.matrix(uds1)
            uds1 = np.matrix(uds0)
        else:
            uds0 = ud
            
            
        
        
        # ud, r = cmpt_control_reactive(H, xi, htm_target, kc=0.5)
        
        u, falhou = cmpt_control(H ,xi, robot_body_copy, all_obs, ud, dist_param_h, dist_param_eps, param_eta, param_obs_delta,xi_lim=1)        
        if falhou:
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
        v_dot_list.append(np.linalg.norm(u[0:3,-1]))
        w_dot_list.append(np.linalg.norm(u[3:6,-1]))
        time_list.append(t)
        path_followed.append(H)
        error.append(np.linalg.norm(log_SE3(ub.Utils.inv_htm(H) @ htm_target)))
        last_err = error[-1]
        
##############################
#          Results           #
##############################
if len(path_followed) > 0:
    draw_pc(path_followed, sim, "magenta", 0.01)
    # print("last_err last msm : ", error[-1])

sim.save(
    address=os.path.dirname(__file__),
    file_name="se3_teste",
)


print(min(error))

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
    w_dot_list,
    time_list,
    file_name="wdot_plot.png",
    labels=[r'$\dot{v}_x$', r'$\dot{v}_y$', r'$\dot{v}_z$', r'$\dot{\omega}_x$', r'$\dot{\omega}_y$', r'$\dot{\omega}_z$'],
    xlabel='Time (s)',
    ylabel=r'$u$',
    show_plot=False,
    title='wdot_plot'
)

plot_vector_list(
    v_dot_list,
    time_list,
    file_name="vdot_plot.png",
    labels=[r'$\dot{v}_x$', r'$\dot{v}_y$', r'$\dot{v}_z$', r'$\dot{\omega}_x$', r'$\dot{\omega}_y$', r'$\dot{\omega}_z$'],
    xlabel='Time (s)',
    ylabel=r'$u$',
    show_plot=False,
    title='vdot_plot'
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
