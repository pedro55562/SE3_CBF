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
    xid[0:3, :] = xid[0:3, :] + ub.Utils.S(xid[3:6, :]) @ state_htm[0:3, -1].reshape(3, 1)

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

height = 1.0
radius = 1.5
step_deg = 0.25
offset_y = 0.0

htm_initial = ub.Utils.trn([0, radius, height]) * ub.Utils.rotx(np.pi)

htm_path = []
num_points = int(2 * np.pi / (step_deg * (np.pi / 180)))

for i in range(num_points):
    htm = (
        ub.Utils.trn([0, offset_y, 0])
        * ub.Utils.rotz(i / num_points * 2 * np.pi)
        * htm_initial
    )
    htm_path.append(htm)

draw_pc(htm_path, sim, "white", 0.03)

##############################
#     Control Parameters     #
##############################

kt1 = 2.0
kt2 = 1.0
kt3 = 1.0
kn1 = 1.0
kn2 = 1.0

Kv = 10.0

##############################
#     Simulation Settings    #
##############################

dt = 0.01
dt_num = 0.01
t_max = 15.0

xi_list = []
xi_dot_list = []
time_list = []

##############################
#      Simulation Loop       #
##############################

path_followed = []
simular_movimento = True

if simular_movimento:
    xi = np.zeros((6, 1))
    qdot = np.zeros((6, 1))

    for k in range(int(t_max / dt)):
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

        # Store and apply control
        xi_list.append(xi)
        xi_dot_list.append(xi_dot)
        time_list.append(t)
        path_followed.append(fkm)
        
        set_configuration_speed(robot, qdot, t, dt)

##############################
#          Results           #
##############################

draw_pc(path_followed, sim, "magenta", 0.01)

plot_twist(xi_list, time_list, "Twist xi", "xi_plot.png")
plot_twist(xi_dot_list, time_list, "Twist Acceleration xi_dot", "xi_dot_plot.png")

sim.save(
    address="/home/pedro/code_robot/SE3_CBF/se3_rigid_body_second_order/",
    file_name="se3_second_order_free_space",
)