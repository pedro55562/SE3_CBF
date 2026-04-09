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
    
    alpha = modulation(state_htm, htm_path[-1], lam = 7)
    
    return xid * alpha, dist, idx


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

sim.add(collision_objects)

##############################
#      Reference Path        #
##############################

dist = 4
step = 0.001

htm_start = robot.fkm()
htm_path = []
for i in range(0, int(dist/step)):
    alpha = i / (dist/step)
    htm_path.append(ub.Utils.trn([alpha * dist, 0, 0]) * robot.fkm())

htm_target = htm_path[-1]
draw_pc(path=htm_path, sim=sim, color="white", radius = 0.02)

frame_target = ub.Frame(htm=htm_target)
sim.add([frame_target])

# plot_rotation_components(htm_path)

##############################
#     Control Parameters     #
##############################

kt1 = 8
kt2 = .7
kt3 = .7
kn1 = 1.0
kn2 = 1.0

Kv = 5.0

##############################
#     Simulation Settings    #
##############################

dt = 0.01
dt_num = 0.01
t_max = 15.0

xi_list = []
xi_dot_list = []
time_list = []
error = []
last_err = 1
##############################
#      Simulation Loop       #
##############################

path_followed = []
simular_movimento = True

if simular_movimento:
    xi = np.zeros((6, 1))
    qdot = np.zeros((6, 1))

    for k in range(int(t_max / dt)):
        if last_err < 0.01:
            break
        
        t = k * dt

        # Current robot state
        jac_geo, fkm = robot.jac_geo()
        
        # Reference twist
        xid, dist, idx = eval_xid_from_state(
            state_htm=fkm,
            robot=robot,
            htm_path=htm_path,
            kt1=kt1,
            kt2=kt2,
            kt3=kt3,
            kn1=kn1,
            kn2=kn2,
            dt=dt,
        )

        # Numerical approximation of reference twist derivative ( xid_dot )
        htm_plus = propagate_htm(fkm, xi, dt_num)
        htm_minus = propagate_htm(fkm, xi, -dt_num)

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

        xid_dot = (xid_plus - xid_minus) / (2.0 * dt_num)



        # Second-order tracking
        xi_dot = xid_dot - Kv * (xi - xid)


        xi = xi + xi_dot * dt

        qdot = ub.Utils.dp_inv(jac_geo, 1e-3) @ xi
        set_configuration_speed(robot, qdot, t, dt)
        
        # Store some useful data
        xi_list.append(xi)
        xi_dot_list.append(xi_dot)
        time_list.append(t)
        path_followed.append(fkm)
        error.append(np.linalg.norm(log_SE3(ub.Utils.inv_htm(fkm) @ htm_path[-1])))
        last_err = error[-1]


##############################
#          Results           #
##############################

draw_pc(path_followed, sim, "magenta", 0.01)

plot_vector_list(xi_list, time_list, "Twist xi", "xi_plot.png")
plot_vector_list(xi_dot_list, time_list, "Twist Acceleration xi_dot", "xi_dot_plot.png")
plot_vector_list(error, time_list, "error d = ||Log(H^(-1) H_tg)||F", "error.png")

sim.save(
    address="/home/pedro/code_robot/SE3_CBF/se3_rigid_body_second_order",
    file_name="se3_teste",
)