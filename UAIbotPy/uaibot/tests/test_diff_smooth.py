from utils_test import *
import uaibot as ub
import time
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')  
import os


print("Running Object to Robot smooth distance test")


projball1 = ub.Ball(radius=0.02,color='blue')
projball2 = ub.Ball(radius=0.02,color='red')

robot = ub.Robot.create_kuka_lbr_iiwa()

sim = ub.Simulation.create_sim_grid([projball1, projball2, robot])



sum_time = 0
max_time = 0
cross_error_max = 0
N=100

eps=0.08
h=0.1



dmin0=0.2
dmin=0.05
eta=0.5
dt=0.01
Tmax=20
Nt = round(Tmax/dt)
No = 10
counter=0


der_ana_hist_all = []
der_num_hist_all = []
name_hist = []
t = 0

error_max = 0

for k in range(No):
    
    cont=True
    while cont:
        obj = generate_rand_object(color='red')
        cont = robot.compute_dist(q = robot.q, obj = obj).get_closest_item().distance<dmin0
        type = ub.Utils.get_uaibot_type(obj)    

    sim.add(obj)

    
    if type != 'uaibot.PointCloud':
        htm = np.matrix(obj.htm)
        obj.add_ani_frame(0,htm=ub.Utils.trn([1000,1000,1000]))
        obj.add_ani_frame(t,htm=htm)


    fun = lambda _q : robot.compute_dist(q = _q, obj = obj, eps=eps, h=h).get_item(5,0).distance()
    
    der_ana_hist = []
    der_num_hist = []
    dist_hist = []

    old_dr = None
    for i in range(Nt):
        t += dt
        
        qdot = np.matrix([1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0]).transpose()
        robot.add_ani_frame(t, robot.q+qdot*dt)
        
        dr = robot.compute_dist(q = robot.q, obj = obj, eps=eps, h=h, no_iter_max=10, old_dist_struct = old_dr)
        old_dr = dr
        
        der_ana_hist.append(dr.get_item(5,0).distance * (dr.get_item(5,0).jac_distance*qdot)[0,0])
        dist_hist.append(0.5*(dr.get_item(5,0).distance ** 2))
        
        sys.stdout.write("\rCompleted: "+str(round(100*counter/(Nt*No)))+"%")
        sys.stdout.flush()      
        
        counter+=1

    #Analyse
    for i in range(Nt-2):
        der_num = (dist_hist[i+2]-dist_hist[i])/(2*dt)
        der_num_hist.append(der_num)
        
        error = abs(der_num - der_ana_hist[i])
        error_max = max(error, error_max)
        
    der_ana_hist_all.append(der_ana_hist[1:-1])
    der_num_hist_all.append(der_num_hist)  
    name_hist.append("Collision check with "+type) 
    
    # rr=[]
    # for i in range(len(der_num_hist)):
    #     rr.append(der_num_hist[i]/der_ana_hist[i+1])
      
    # plt.plot(rr)    
    # plt.plot(der_num_hist)
    # plt.plot(der_ana_hist[1:-1])
    # plt.show()
    
    if type != 'uaibot.PointCloud':
        obj.add_ani_frame(t,htm=ub.Utils.trn([1000,1000,1000]))
    else:
        obj.add_ani_frame(t,0,0)
            

#Plot everything

n = len(name_hist)

cols = int(np.ceil(np.sqrt(n)))
rows = int(np.ceil(n / cols))


fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
axes = np.array(axes).flatten()  


for i in range(n):
    ax = axes[i]
    ax.plot(der_ana_hist_all[i], label='Analytical')
    ax.plot(der_num_hist_all[i], label='Numerical', linestyle='dashed')
    ax.set_title(name_hist[i])
    ax.legend()


for j in range(n, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


print("\nMaximum error was "+str(round(error_max,3)))
print("Test was a success!")   
    
    

current_folder = os.path.dirname(os.path.abspath(__file__))
sim.save(current_folder,"test_o2o_smooth")
