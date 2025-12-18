from utils_test import *
import uaibot as ub
import time
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')  
import os

print("Running Object to Object Smooth Distance Test")


projball1 = ub.Ball(radius=0.02,color='blue')
projball2 = ub.Ball(radius=0.02,color='red')

sim = ub.Simulation.create_sim_grid([projball1, projball2])



sum_time = 0
max_time = 0
cross_error_max = 0
N=100
eps=0.08, 
h=0.1
dmin0=0.2
dmin=0.05
eta=0.5
dt=0.01
Tmax=20
Nt = round(Tmax/dt)
No = 10
counter=0


hist_dist = []
hist_dist_0 = []
t = 0

for k in range(No):
    
    cont=True
    while cont:
        obj1 = generate_rand_object(color='red')
        obj2 = generate_rand_object(color='blue')
        type1 = ub.Utils.get_uaibot_type(obj1)
        type2 = ub.Utils.get_uaibot_type(obj2)
        
        if type1 == 'uaibot.PointCloud' and type2 == 'uaibot.PointCloud':
            cont=True
        else:
            cont = ub.Utils.compute_dist(obj1, obj2)[2]<dmin0

    sim.add(obj1)
    sim.add(obj2)
    
    if type1 != 'uaibot.PointCloud':
        htm1 = np.matrix(obj1.htm)
        obj1.add_ani_frame(0,htm=ub.Utils.trn([1000,1000,1000]))
        obj1.add_ani_frame(t,htm=htm1)
    else:
        obj1.add_ani_frame(0,0,0)
        obj1.add_ani_frame(t,0,np.shape(obj1.points)[1])
    
    if type2 != 'uaibot.PointCloud':
        htm2 = np.matrix(obj2.htm)
        obj2.add_ani_frame(0,htm=ub.Utils.trn([1000,1000,1000]))
        obj2.add_ani_frame(t,htm=htm2)    
    else:
        obj2.add_ani_frame(0,0,0)
        obj2.add_ani_frame(t,0,np.shape(obj2.points)[1])    

    for i in range(Nt):
        t += dt
        p1, p2, d, _ = ub.Utils.compute_dist(obj1, obj2, no_iter_max=2000, eps=0.02, h=0.1, tol=1e-5)
        p1_0, p2_0, d0, _ = ub.Utils.compute_dist(obj1, obj2)
        
        projball1.add_ani_frame(t,ub.Utils.trn(p1))
        projball2.add_ani_frame(t,ub.Utils.trn(p2))  
        
        
        w1 = np.matrix([np.cos(t), np.sin(t), np.cos(2*t)]).transpose()
        w2 = np.matrix([np.sin(2*t), np.sin(t), np.cos(0.5*t)]).transpose()
        
        alpha = max(d0-dmin,0)
        
        v1 = (1-alpha)*eta*(p1_0-p2_0)+alpha*np.matrix([np.sin(t), np.cos(t), 0]).transpose()
        v2 = (1-alpha)*eta*(p2_0-p1_0)+alpha*np.matrix([np.cos(t), np.cos(t), 0]).transpose()
        
        if type1 != 'uaibot.PointCloud':
            htm1_new = np.matrix(obj1.htm)
            htm1_new[0:3,-1] += v1*dt
            htm1_new = htm1_new * ub.Utils.rot(w1,np.linalg.norm(w1)*dt)
            obj1.add_ani_frame(t,htm=htm1_new) 
        
        if type2 != 'uaibot.PointCloud':
            htm2_new = np.matrix(obj2.htm)
            htm2_new[0:3,-1] += v2*dt   
            htm2_new = htm2_new * ub.Utils.rot(w2,np.linalg.norm(w2)*dt)
            obj2.add_ani_frame(t,htm=htm2_new) 

        sys.stdout.write("\rCompleted: "+str(round(100*counter/(Nt*No)))+"%")
        sys.stdout.flush()      
        
              
        hist_dist.append(d)
        hist_dist_0.append(d0)
        counter+=1



    if type1 != 'uaibot.PointCloud':
        obj1.add_ani_frame(t,htm=ub.Utils.trn([1000,1000,1000]))
    else:
        obj1.add_ani_frame(t,0,0)
            
    if type2 != 'uaibot.PointCloud':
        obj2.add_ani_frame(t,htm=ub.Utils.trn([1000,1000,1000]))
    else:
        obj2.add_ani_frame(t,0,0)

plt.plot(hist_dist)
plt.plot(hist_dist_0)
plt.show()


print("Test was a success!")   
    
    

current_folder = os.path.dirname(os.path.abspath(__file__))           
sim.save(current_folder,"test_o2o_smooth")
