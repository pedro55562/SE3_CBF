import uaibot as ub
import numpy as np
import matplotlib.pyplot as plt
import os

# https://sketchfab.com/3d-models/hexacopter-drone-eca2ef59d0714f42984b87b91bbabb62

robot = ub.Robot.create_rigid_body_se3()
sim = ub.Simulation(background_color = 'black')

model_drone = ub.Model3D(
    
    'https://cdn.jsdelivr.net/gh/pedro55562/SE3_CBF_ASSETS@main/hexacopter_3d.obj',
    scale=1, 
    mesh_material= ub.MeshMaterial(color='red')
    )
    
drone_body = ub.RigidObject(list_model_3d=[model_drone],htm=ub.Utils.trn([0,0,1.5]) * ub.Utils.rotx(np.pi/2) )



sim.add([drone_body, robot])
sim.save(address="/home/pedro/code_robot/SE3_CBF/",file_name="teste")