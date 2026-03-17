import uaibot as ub
import numpy as np
import matplotlib.pyplot as plt
import os
from setup import *
import itertools
from scipy.interpolate import CubicSpline
from scipy.interpolate import splprep, splev



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


qdot = np.array([
    [0.0],
    [0.0],
    [0.0],
    [0.1],
    [0.0],
    [0.0]
])

dt = 0.1
for i in range(0, int (15/dt)):
    set_configuration_speed(robot=robot, q_dot=qdot,t=i*dt,dt=dt)
    print(i*dt)
    print(robot.q[3])
    print('-------')

sim.save(address="/home/pedro/code_robot/SE3_CBF/",file_name="teste_or")
