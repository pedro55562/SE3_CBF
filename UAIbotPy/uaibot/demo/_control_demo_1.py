import uaibot as ub
import numpy as np
import matplotlib.pyplot as plt
import sys


def _control_demo_1():
    
    Tmax=20
    dt = 0.01
    K = 1.0
    v_speed = 0.3
    alpha = 1.0
    y_c = 0.5
    z_c = 0.6
    delta=0.3
    
    if ub.Utils.get_environment()=='Local':
        plt.figure()
        plt.xlim(-delta, delta)
        plt.ylim(z_c-delta, z_c+delta)
        plt.title("Right click to add points. Add at least 3. Left click to erase. Enter to finish")
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.grid(True)

        points = plt.ginput(n=0, timeout=0)
        
        if len(points)<3:
            raise Exception("At least 3 points are necessary!")
            
        plt.close()  
    else:
        print("In Google Colab/Jupyter Notebook is not possible to draw your own curve... ")
        print("Run locally for this feature. Type at least 3 points you want in the format ")
        print("x,z ")
        print("in which "+str(round(-delta,2))+" <= x <= "+str(round(delta,2))+" and "+str(round(z_c-delta,2))+" <= z <= "+str(round(z_c+delta,2)))
        print("Write 'done' when you want to be over.")
        points = []
        while True:
            user_input = input("Enter point (x, z): ").strip()
            
            if user_input.lower() == 'done':
                break

            try:
                # Evaluate the tuple safely
                point = eval(user_input, {"__builtins__": None}, {})
                
                if isinstance(point, tuple) and len(point) == 2:
                    x, z = point
                    x = float(x)
                    z = float(z)
                    
                    if x>=-delta and x<=delta and z<=z_c+delta and z>=z_c-delta:
                        points.append((x, z))
                    else:
                        print("Error: "+str(round(-delta,2))+" <= x <= "+str(round(delta,2))+" and "+str(round(z_c-delta,2))+" <= z <= "+str(round(z_c+delta,2)))
                        
                else:
                    print("Invalid format. Please enter a tuple like (1.0, 2.0).")
            
            except Exception as e:
                print(f"Invalid input: {e}")


    s = [i / 1000 for i in range(1000)]

    target_curve = ub.Utils.interpolate(points, is_closed = True)(s)
    target_curve_mod = []
    for p in target_curve:
        target_curve_mod.append(np.matrix([p[0,0], y_c, p[1,0]]).T)



    robot = ub.Robot.create_kuka_lbr_iiwa()
    point_cloud = ub.PointCloud(points=target_curve_mod,size=0.02,color='yellow')

    sim = ub.Simulation.create_sim_grid([robot, point_cloud])
    sim.set_parameters(width=500,height=500,load_screen_color='#191919',background_color='#191919')
    
    rot_tg = robot.fkm()*ub.Utils.rotx(-np.pi/2)

    hist_p = []
    for i in range(round(Tmax/dt)):
        t = i*dt
        htm = robot.fkm(robot.q)
        p = htm[0:3,-1]
        v, dist, _ = ub.Robot.vector_field(p,target_curve_mod,alpha=alpha)
        r, jac_r = robot.task_function(q = robot.q, htm_tg = rot_tg)
        r[0:3,-1] = v_speed*v
        r[3:6,-1] = -K*r[3:6,-1]
        qdot = ub.Utils.dp_inv_solve(jac_r,r)
        
        robot.add_ani_frame(t,robot.q+qdot*dt)

        str_msg = "\rt = "+str(round(t,2))+" s, dist = "+str(round(1000*dist,2))+" mm"
        
        hist_p.append(p)

        sys.stdout.write(str_msg)
        sys.stdout.flush()
   
    plt.figure()
    plt.plot([p[0, 0] for p in target_curve], [p[1, 0] for p in target_curve])
    plt.scatter([p[0] for p in points], [p[1] for p in points], color='red')
    plt.plot([p[0, 0] for p in hist_p], [p[2, 0] for p in hist_p],color='green')

    dm = 0.2
    
    plt.xlim(-delta-dm, delta+dm)
    plt.ylim(z_c-delta-dm, z_c+delta+dm)
    plt.title("Tracked curve")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True)
    plt.show()
                
    return sim





