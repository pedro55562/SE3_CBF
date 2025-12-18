import uaibot as ub
import numpy as np
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import sys


def _constrained_control_demo_1():
    # Get the point cloud data
    url = "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/PointCloud/data_wall_with_hole.npy"
    wallpoints = np.load(BytesIO(requests.get(url).content))
    pc = ub.PointCloud(points = wallpoints, size=0.02, color='cyan')

    #Create the other objects and set up the simulation
    point_link = ub.Ball(color='red',radius=0.02)
    point_obj = ub.Ball(color='blue',radius=0.02)
    robot = ub.Robot.create_franka_emika_3()
    htm_tg = ub.Utils.trn([0.45,0.31,0.7])*ub.Utils.rotx(-np.pi/2)
    frame_tg = ub.Frame(htm_tg, size=0.1)
    sim = ub.Simulation.create_sim_grid([pc, robot, point_link, point_obj, frame_tg])
    sim.set_parameters(width=500,height=500,load_screen_color='#191919',background_color='#191919')

    #Define F function for control
    def funF(r):
        f = np.matrix(r)
        for j in range(np.shape(r)[0]):
            f[j,0] = np.sign(r[j,0])*np.sqrt(np.abs(r[j,0]))
            
        return f

    # for link in robot.links:
    #     for obj in link.col_objects:
    #         sim.add(obj[0])
            
    #Set some parameters
    dt = 0.02
    eta = 0.3
    Kp = 2.0
    Tmax = 35
    q_min = robot.joint_limit[:,0]
    q_max = robot.joint_limit[:,1]

    eps_to_obs = 0.003
    h_to_obs = 0.003
    eps_auto = 0.02
    h_auto = 0.05
    d_safe_obs = 0.02
    d_safe_auto = 0.002
    d_safe_jl = (np.pi/180)*5
    tolp=1.0 #1mm error tolerance
    tolo=1.0 #1deg error tolerance
    eps_reg = 0.01

    #Main loop
    t = 0

    robot.add_ani_frame(0,[0.0, 0.0, 0.0, -np.pi*4/180-2*d_safe_jl, 0.0, 2*d_safe_jl, 0.0])


    hist_data=[]
    cont = True

    while cont:
        
        #Get current configuration
        qr = robot.q
        
        #Get smooth distance structure from robot to point cloud
        dr_obj =robot.compute_dist(obj = pc, q = qr, eps=eps_to_obs, h=h_to_obs)
        
        A_obj = dr_obj.jac_dist_mat
        b_obj = dr_obj.dist_vect-d_safe_obs
        
        #Get smooth distance structure from robot to itself (autocollision)
        dr_auto =robot.compute_dist_auto(q = qr, eps=eps_auto, h=h_auto)    

        A_auto = dr_auto.jac_dist_mat
        b_auto = dr_auto.dist_vect-d_safe_auto
        
        #Assemble matrices for joint limit constraints
            
        A_joint = np.matrix(np.vstack(  (np.identity(7), -np.identity(7))  ))
        b_joint = np.matrix(np.vstack(  (qr-(q_min+d_safe_jl) , (q_max-d_safe_jl) - qr)  )) 
        
        #Get task function data
        
        r, Jr = robot.task_function(htm_tg, q=qr)
        
        #Create the optimization problem
        A =    np.matrix(np.vstack( (A_obj, A_auto, A_joint) ) )
        b =    -eta * np.matrix(np.vstack( (b_obj, b_auto, b_joint) ) )
        H = 2*(Jr.transpose() * Jr + eps_reg *np.identity(7))
        f = Jr.transpose() * Kp * funF(r)
        
        #Solve QP and obtain joint velocity
        
        qdot = ub.Utils.solve_qp(H, f, A, b)

        #Integrate joint velocity
        qr += qdot*dt
        
        #Compute some statics
        epx = abs(round(1000*r[0,0]))
        epy = abs(round(1000*r[1,0]))
        epz = abs(round(1000*r[2,0]))
        eox = round((180/np.pi)*np.arccos(1-r[3,0]),2)
        eoy = round((180/np.pi)*np.arccos(1-r[4,0]),2)
        eoz = round((180/np.pi)*np.arccos(1-r[5,0]),2)
        
        #Compute true distance to show
        dr_obj_0 =robot.compute_dist(obj = pc, q = qr)
        dr_auto_0 =robot.compute_dist_auto(q = qr)  
        
        dmin_obj = 1000*dr_obj_0.get_closest_item().distance
        dmin_auto = 1000*dr_auto_0.get_closest_item().distance
        
        str_msg = "\rt = "+str(round(t,2))+" s, epx = "+str(epx)+" mm, epy = "+str(epy)+" mm, epz = "+str(epz)+" mm, "
        str_msg += "eox = "+str(epx)+" deg, eoy = "+str(epy)+" deg, eoz = "+str(epz)+" deg     "
        
        prox_joints = (180/np.pi) * min((qr-q_min).min(), (q_max - qr).min())
        
        
        hist_data.append([t, epx, epy, epz, eox, eoy, eoz, dmin_obj, dmin_auto, prox_joints])

        
        sys.stdout.write(str_msg)
        sys.stdout.flush()
        
        #Update animation frames and time
        robot.add_ani_frame(t,qr)
        robot.update_col_object(t)
        point_link.add_ani_frame(t, htm = ub.Utils.trn(dr_obj.get_closest_item().point_link))
        point_obj.add_ani_frame(t, htm = ub.Utils.trn(dr_obj.get_closest_item().point_object ))            
        t+=dt
        
        #Check condition
        cont = t < Tmax and ((eox > tolo) or (eoy > tolo) or (eoz > tolo) or (epx > tolp) or (epy > tolp) or (epz > tolp))
        

    # Convert list to NumPy array for easier slicing
    data = np.array(hist_data)

    # Extract columns
    t = data[:, 0]  # Time
    epx, epy, epz = data[:, 1], data[:, 2], data[:, 3]
    eox, eoy, eoz = data[:, 4], data[:, 5], data[:, 6]
    dmin_obj, dmin_auto = data[:, 7], data[:, 8]
    prox_joints = data[:, 9]

    # Create a figure with subplots
    fig, axs = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

    # Plot epx, epy, epz vs. time
    axs[0].plot(t, epx, label='epx')
    axs[0].plot(t, epy, label='epy')
    axs[0].plot(t, epz, label='epz')
    axs[0].set_ylabel("Position error (mm)")
    axs[0].legend()
    axs[0].grid(True)

    # Plot eox, eoy, eoz vs. time
    axs[1].plot(t, eox, label='eox')
    axs[1].plot(t, eoy, label='eoy')
    axs[1].plot(t, eoz, label='eoz')
    axs[1].set_ylabel("Orientation error (mm)")
    axs[1].legend()
    axs[1].grid(True)

    # Plot dmin_obj, dmin_auto vs. time
    axs[2].plot(t, dmin_obj, label='to obj')
    axs[2].plot(t, dmin_auto, label='auto')
    axs[2].set_ylabel("Collision distance (mm)")
    axs[2].legend()
    axs[2].grid(True)

    # Plot prox_joints vs. time
    axs[3].plot(t, prox_joints, label='Joint limit margin')
    axs[3].set_xlabel("Time (t)")
    axs[3].set_ylabel("Joint limit margin (deg)")
    axs[3].grid(True)

    # Show the plots
    plt.tight_layout()
    plt.show()
    
    return sim

# sim.save("/home/vinicius/Desktop/uaibot_devel/uaibot_files/testing","intro_3")