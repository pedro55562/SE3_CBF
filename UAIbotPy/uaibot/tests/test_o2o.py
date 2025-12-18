from utils_test import *
import uaibot as ub
import time
import sys
import os

print("Running Object to Object Distance Test")

t = 0

projball1 = ub.Ball(radius=0.02,color='blue')
projball2 = ub.Ball(radius=0.02,color='red')

sim = ub.Simulation.create_sim_grid([projball1, projball2])

t = 0

sum_time = 0
max_time = 0
cross_error_max = 0
N=100



for i in range(N):
    
    cont=True
        
    while cont:
        obj1 = generate_rand_object(color='red')
        obj2 = generate_rand_object(color='blue')
        
        type1 = ub.Utils.get_uaibot_type(obj1)
        type2 = ub.Utils.get_uaibot_type(obj2)
        cont =  type1 == 'uaibot.PointCloud' and type2 == 'uaibot.PointCloud'

        if not cont:
            start = time.time() 
            p1, p2, d, _ = ub.Utils.compute_dist(obj1, obj2)
            end = time.time() 
            cont = d < 0.2
            
            #Cross checking
            if not cont:
                error1 = np.linalg.norm(obj1.projection(p2)[0]-p1)
                error2 = np.linalg.norm(obj2.projection(p1)[0]-p2)
                cross_error_max = np.max([cross_error_max,error1, error2])
                
                # if error1 > 1e-3 or error2 > 1e-3:
                #     print("Warning, check "+str(i)+", "+type1+", "+type2+", error1 = "+str(round(error1,2))+", error2 = "+str(round(error2,2)))

    
    elapsed = end - start
    sum_time+=elapsed
    max_time=max(max_time,elapsed) 
    
    projball1.add_ani_frame(i,ub.Utils.trn(p1))
    projball2.add_ani_frame(i,ub.Utils.trn(p2))   

    sim.add(obj1)
    sim.add(obj2)
                         
    if type1 != 'uaibot.PointCloud':
        htm1 = np.matrix(obj1.htm)
        
    if type2 != 'uaibot.PointCloud':    
        htm2 = np.matrix(obj2.htm)
    
    if type1 == 'uaibot.PointCloud':
        obj1.add_ani_frame(0,0,0)
        obj1.add_ani_frame(i,0,np.shape(obj1.points)[1])
        obj1.add_ani_frame(i+1,0,0)   
    else:
        obj1.add_ani_frame(0,htm=ub.Utils.trn([1000,1000,1000]))
        obj1.add_ani_frame(i,htm=htm1)
        obj1.add_ani_frame(i+1,htm=ub.Utils.trn([1000,1000,1000]))        

    if type2 == 'uaibot.PointCloud':
        obj2.add_ani_frame(0,0,0)
        obj2.add_ani_frame(i,0,np.shape(obj2.points)[1])
        obj2.add_ani_frame(i+1,0,0) 
    else:
        obj2.add_ani_frame(0,htm=ub.Utils.trn([1000,1000,1000]))
        obj2.add_ani_frame(i,htm=htm2)
        obj2.add_ani_frame(i+1,htm=ub.Utils.trn([1000,1000,1000]))
        


    sys.stdout.write("\rCompleted: "+str(round(100*i/N))+"%")
    sys.stdout.flush()

             
print("\nAverage time took to compute distance: "+str(round(1000*sum_time/N,2))+" ms")
print("Max time took to compute distance: "+str(round(1000*max_time,2))+" ms")
print("Maximum cross error: "+str(round(1000*cross_error_max,3))+" mm")

print("Test was a success!")   

        

        

        
    
    

current_folder = os.path.dirname(os.path.abspath(__file__))           
sim.save(current_folder,"test_o2o")
