import uaibot as ub
import numpy as np
import matplotlib.pyplot as plt
import os
from setup import *
import itertools


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
        objss.append(obj)
sim.add(objss)


htm_target = ub.Utils.trn([0, 2.5, 0.7]) * robot.fkm() * ub.Utils.rot(axis=[4,3,-2], angle= 77 * np.pi / 180) 

frame_target = ub.Frame(htm=htm_target)
sim.add([frame_target])

box = ub.Box(htm = ub.Utils.trn([0, 2, 0]) ,width=3, depth=0.1, height=4,)
all_obs = []
all_obs.append(box)

sim.add(all_obs)



q_goal = robot.ikm(htm_tg=htm_target, obstacles=all_obs, no_tries = 2000, no_iter_max=4000)

success1, path1, iterations1, num_tries1, planning_time1 = robot.runSE3RRT(q0 = robot.q0, q_goal=[q_goal], obstacles=all_obs)
draw_pc(pathhh_=path1,robot = robot, sim=sim)
print(len(path1))
print(planning_time1)






i=0
dt = 0.01
t=0

while np.linalg.norm(robot.q - q_goal) > 0.005 and t < 10:
   t = i*dt
   q_dot = vector_field(robot=robot, curve=path1)
   set_configuration_speed(robot, q_dot, t, dt)
   i+=1



sim.save(address="/home/pedro/code/SE3_CBF/",file_name="teste_SE3")
