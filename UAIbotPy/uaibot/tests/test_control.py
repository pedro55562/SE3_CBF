import uaibot as ub
import numpy as np
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import sys
import os



#sim = ub.Demo.constrained_control_demo_1()


# Get the point cloud data
url = "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/PointCloud/data_wall_with_hole.npy"
wallpoints = np.load(BytesIO(requests.get(url).content))
pc = ub.PointCloud(points = wallpoints, size=0.02, color='cyan')


#Create the other objects and set up the simulation
point_link = ub.Ball(color='red',radius=0.02)
point_obj = ub.Ball(color='blue',radius=0.02)
robot = ub.Robot.create_franka_emika_3()
htm_des = ub.Utils.trn([0.45,0.31,0.7])*ub.Utils.rotx(-np.pi/2)
frame_des = ub.Frame(htm_des, size=0.1)
sim = ub.Simulation.create_sim_grid([pc, robot, point_link, point_obj, frame_des])
sim.set_parameters(width=500,height=500,load_screen_color='#191919',background_color='#191919')
d_safe_jl = (np.pi/180)*5
robot.add_ani_frame(0,[0.0, 0.0, 0.0, -np.pi*4/180-2*d_safe_jl, 0.0, 2*d_safe_jl, 0.0])

#Main loop
t = 0
dt= 0.01
cont = True

min_dobs = 1e6
min_dauto = 1e6
min_djoint = 1e6
for i in range(1000):
    
    #Get current configuration
    qr = robot.q
    
    #Get smooth distance structure from robot to point cloud
    qdot, ep, eo, dobs, dauto, djoint =robot.constrained_control(htm_tg=htm_des, q=qr, obstacles=pc)

    #Integrate joint velocity
    qr += qdot*dt
    
    min_dobs = min(min_dobs,dobs)
    min_dauto = min(min_dauto,dauto)
    min_djoint = min(min_djoint,djoint)
    
  

    
    sys.stdout.write("\rep = "+str(round(1000*ep,2))+", eo = "+str(round(eo,2))+", dobs = "+str(round(min_dobs,4))+", dauto = "+str(round(min_dauto,4))+", djoint = "+str(round(min_djoint,4)))
    sys.stdout.flush()
    
    #Update animation frames and time
    robot.add_ani_frame(t,qr)        
    t+=dt
    
    
current_folder = os.path.dirname(os.path.abspath(__file__))
sim.save(current_folder,"control")
