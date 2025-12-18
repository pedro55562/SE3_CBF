from utils_test import *
import uaibot as ub
import time
import sys
import matplotlib.pyplot as plt
import matplotlib
import os

print("Testing free configuration")

robot = ub.Robot.create_magician_e6()
style = "top:" + str(0.9 * 720) + "px;right:" + str(0) + "px;width:" + str(
    960) + "px;position:absolute;text-align:center;color:white;background-color:#222224;font-smooth:always;font-family:arial"


htmldiv = ub.HTMLDiv(style = style)
projball1 = ub.Ball(radius=0.02,color='blue')
projball2 = ub.Ball(radius=0.02,color='red')

sim = ub.Simulation.create_sim_grid([robot, htmldiv, projball1, projball2])

Nobj = 10
Ntry = 500

for link in robot.links:
    for col_obj in link.col_objects:
        sim.add(col_obj[0])


all_obj = []
for i in range(Nobj):
    cont = True
    
    while cont:
        obj = generate_rand_object(color='red', size=0.2, spread=0.4)
        isfree, message, info = robot.check_free_config(q=robot.q, obstacles=[obj])
        cont = not isfree
    
    all_obj.append(obj)
        
    sim.add(obj)

sum_time = 0
max_time = 0
    
for i in range(Ntry):
    qmin = robot.joint_limit[:,0]
    qmax = robot.joint_limit[:,1]
    
    qnew = np.matrix(0*qmin)
    
    n = np.shape(qnew)[0]
    
    for j in range(n):
        qnew[j] = qmin[j]+np.random.uniform()*(qmax[j]-qmin[j])
        
    robot.add_ani_frame(i,qnew)
    robot.update_col_object(i)
    
    start = time.time() 
    isfree, message, info = robot.check_free_config(q=qnew, obstacles=all_obj)
    end =  time.time() 
    elapsed = end - start
    sum_time+=elapsed
    max_time=max(max_time,elapsed) 
    
    min_dist = 1e6
    for obj in all_obj:
        dr = robot.compute_dist(q=qnew, obj=obj).get_closest_item()
        if dr.distance < min_dist:
            min_dist = dr.distance
            point_obj = dr.point_object
            point_link = dr.point_link
        
    projball1.add_ani_frame(i,ub.Utils.trn(point_obj))
    projball2.add_ani_frame(i,ub.Utils.trn(point_link))
    
    htmldiv.add_ani_frame(i,message)
    
    sys.stdout.write("\rCompleted: "+str(round(100*i/Ntry))+"%")
    sys.stdout.flush()
    
    
    
print("\nAverage time took to check free configuration: "+str(round(1000*sum_time/Ntry,2))+" ms")
print("Max time took to check free configuration: "+str(round(1000*max_time,2))+" ms")

print("Test was a success!")   


current_folder = os.path.dirname(os.path.abspath(__file__))        
sim.save(current_folder,"test_free_config")
