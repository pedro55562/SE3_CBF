import uaibot as ub
import numpy as np
import matplotlib.pyplot as plt
import os
from setup import *
import itertools
from scipy.interpolate import CubicSpline
from scipy.interpolate import splprep, splev
from scipy.linalg import expm


def draw_pc2(pathhh_, sim, color="white", radius = 0.01):
    sl = [ ]
    for htm in pathhh_:
        sl.append( htm[ 0 : 3 , 3] ) 
    pc = ub.PointCloud(size = radius, color = color, points = sl)
    sim.add(pc)


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

def plot_twist(data_list, title, dt, time,file_name):
    if len(data_list) == 0:
        return

    data = np.hstack([d.reshape(6,1) for d in data_list]).T
    t = time

    plt.figure(figsize=(10,6))
    for i in range(6):
        plt.plot(t, data[:, i], label=f'xi_{i+1}')

    plt.title(title)
    plt.xlabel('Tempo (s)')
    plt.ylabel('Valor')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join("/home/pedro/code_robot/SE3_CBF/", file_name)
    plt.savefig(save_path)
    plt.close()

    print(f"Plot salvo em: {save_path}")

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

Altura = 1
Raio   = 1.5
Passo  = .25
dist   = 0

kt1 = 2      
kt2 = 1           
kt3 = 1          
kn1 = 1          
kn2 = 1

dt = 0.01
i =0

Kv = 10.0
dt_num = 0.01

xi_list = []
xi_dot_list = []
time = []

H_inicial = ub.Utils.trn([0, Raio, Altura]) * ub.Utils.rotx(np.pi) 
htm_path = []
size = int (2 * np.pi /(Passo * (np.pi / 180)))
for i in range(0,size):
    htm_path.append( ub.Utils.trn([0, dist,0])* ub.Utils.rotz(i/size * 2*np.pi) * H_inicial)

draw_pc2(htm_path,sim,"white",0.03)


path_followed = []
simular_movimento =True 
if simular_movimento:
    
    xi = np.zeros((6,1))
    qdot = np.zeros((6,1))
    for k in range(0 , int (30/dt)):    
        t = k * dt    
        
        jac_geo, fkm = robot.jac_geo()
        path_followed.append(fkm)
        
        xid, dist, idx = eval_xid_from_state(
            state_htm=fkm,
            robot=robot,
            htm_path=htm_path,
            kt1=kt1,
            kt2=kt2,
            kt3=kt3,
            kn1=kn1,
            kn2=kn2,
            dt=dt
        )

        H_plus  = propagate_htm(fkm, xi,  dt_num)
        H_minus = propagate_htm(fkm, xi, -dt_num)

        xid_plus, _, _ = eval_xid_from_state(
            state_htm=H_plus,
            robot=robot,
            htm_path=htm_path,
            kt1=kt1,
            kt2=kt2,
            kt3=kt3,
            kn1=kn1,
            kn2=kn2,
            dt=dt
        )
        xid_minus, _, _ = eval_xid_from_state(
            state_htm=H_minus,
            robot=robot,
            htm_path=htm_path,
            kt1=kt1,
            kt2=kt2,
            kt3=kt3,
            kn1=kn1,
            kn2=kn2,
            dt=dt
        )
        xid_dot = (xid_plus - xid_minus) / (2.0 * dt_num)
        xi_dot = xid_dot - Kv * (xi - xid)
        
        xi = xi + xi_dot * dt
        qdot =  ub.Utils.dp_inv(jac_geo, 1e-3) @ xi.reshape(6,1)
        
        xi_list.append(xi)
        xi_dot_list.append(xi_dot)
        time.append(t)
        set_configuration_speed(robot, qdot, t, dt)
        
draw_pc2(path_followed,sim,"magenta",0.01)

plot_twist(xi_list, "Twist xi (real)", dt, time,"xi_plot.png")
plot_twist(xi_dot_list, "Aceleracao xi_dot (Omega)", dt, time,"xi_dot_plot.png")
     
sim.save(address="/home/pedro/code_robot/SE3_CBF/",file_name="teste_SE3")