import uaibot as ub
import numpy as np
import matplotlib.pyplot as plt
from setup import *


class ReactiveControl:
    def __init__(self, robot, obstacles, htm_tg, 
                 dj_lim=np.deg2rad(2), d_lim=0.01, dlim_auto=0.002,
                 eta_obs=1.0, eta_auto=0.6, eta_joint=0.6,
                 eps=1e-3, kp=1.0):

        self.robot = robot
        self.obstacles = obstacles
        self.htm_tg = htm_tg
        self.dj_lim = dj_lim
        self.d_lim = d_lim
        self.dlim_auto = dlim_auto
        self.eta_obs = eta_obs
        self.eta_auto = eta_auto
        self.eta_joint = eta_joint
        self.eps = eps
        self.kp = kp

        self.u = robot.q * 0
        self.r , _ = self.robot.task_function(htm_tg=self.htm_tg, q=self.robot.q)
        self.n = len(self.robot.links)
    
    def fun_F(self, r):
        f = np.matrix(r)
        for j in range(np.shape(r)[0]):
            f[j, 0] = np.sign(r[j, 0]) * np.sqrt(np.abs(r[j, 0]))
        return f

    def compute_control(self, qq=None):
        u = self.robot.q * 0

        q = self.robot.q if qq is None else qq
        r, Jr = self.robot.task_function(htm_tg=self.htm_tg, q=q)

        q_min = self.robot.joint_limit[:, 0]
        q_max = self.robot.joint_limit[:, 1]

        Aj_min = np.identity(self.n)
        Aj_max = -np.identity(self.n)

        Bj_min = -self.eta_joint * ((q - q_min) - self.dj_lim)
        Bj_max = -self.eta_joint * ((q_max - q) - self.dj_lim)

        Ad_obj = np.zeros((0,  self.n))
        Bd_obj = np.zeros((0, 1))

        for ob in self.obstacles:
            ds = self.robot.compute_dist(ob, q=q)
            Ad_obj = np.vstack((Ad_obj, ds.jac_dist_mat))
            Bd_obj = np.vstack((Bd_obj, ds.dist_vect - self.d_lim))
        Bd_obj = -self.eta_obs * Bd_obj

        dist_auto = self.robot.compute_dist_auto(q=q)
        A_auto = dist_auto.jac_dist_mat
        B_auto = -self.eta_auto * (dist_auto.dist_vect - self.dlim_auto)

        A = np.vstack((Aj_min, Aj_max, Ad_obj, A_auto))
        b = np.vstack((Bj_min, Bj_max, Bd_obj, B_auto))

        H = 2 * (Jr.T @ Jr + self.eps * np.identity(self.n))
        f = self.kp * Jr.T @ self.fun_F(r)

        u = ub.Utils.solve_qp(
            np.array(H, dtype=np.float64),
            np.array(f, dtype=np.float64),
            np.array(A, dtype=np.float64),
            np.array(b, dtype=np.float64)
        )
        self.u = u
        self.r = r





def reactive_test(index):
    dt = 0.01                          
    t = 0
    qdot_hist = []                     
    r_hist = []                       
    t_hist = []                        
    
    robot, sim, all_obs, q0, htm_tg, htm_base = setup_motion_planning_simulation(index)
    
    controller = ReactiveControl(
        robot=robot,
        obstacles=all_obs,
        htm_tg=htm_tg,
        dj_lim=np.deg2rad(2),
        d_lim=0.01,
        dlim_auto=0.002,
        eta_obs=1.0,
        eta_auto=0.6,
        eta_joint=0.6,
        eps=1e-3,
        kp=1.0
    )

    for i in range(1500):
        controller.compute_control()
        set_configuration_speed(robot, controller.u , t, dt)
        
        qdot_hist.append(np.degrees(controller.u))
        r_hist.append(controller.r)
        t_hist.append(t)
        
        t = dt * i

    r_norm = [np.linalg.norm(r) for r in r_hist]

    #store_info(sim, qdot_hist, r_norm, r_hist, t_hist, "benchmark" + str(index))
    sim.save(address="/home/pedro/code/SE3_CBF/",file_name="testee_reactive")

def test(): 
    scenario_index = np.random.uniform(0, 2599, size=0).astype(int)
    scenario_index = np.append(scenario_index, [420]) 

    for index in scenario_index:
        print(index)
        reactive_test(index)



test()