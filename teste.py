import uaibot as ub
import numpy as np
import matplotlib.pyplot as plt
import os

# https://sketchfab.com/3d-models/hexacopter-drone-eca2ef59d0714f42984b87b91bbabb62


# https://cdn.jsdelivr.net/gh/pedro55562/SE3_CBF_ASSETS@main/hexacopter_3d.obj

robot = ub.Robot.create_rigid_body_se3() #htm= ub.Utils.trn( [ 0, 0, 0] ) )
sim = ub.Simulation(background_color = 'black')
sim.add([robot])


objss = []
#Loop the link for data
for link in robot.links:
    #Loop the data for collision objects
    for col_obj_data in link.col_objects:
        #The actual UAIBot object (ub.Box, ub.Cylinder, etc...)    
        obj = col_obj_data[0]
        objss.append(obj)
sim.add(objss)

t  = 0
dt = 0.01

xid = 0.3 * np.array([[0],[0],[0],[1],[0],[0]])

for i in range( 0, int(10/dt) ):
    t = i * dt  
    jac_geo , fkm = robot.jac_geo()

    qdot = ub.Utils.dp_inv_solve(jac_geo, xid ,1e-3)
    print(qdot)
    q_next = robot.q + qdot*dt
    robot.add_ani_frame(time=t , q=q_next)

sim.save(address="/home/pedro/code/SE3_CBF/",file_name="teste")