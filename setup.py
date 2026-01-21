import os
import urllib.request
import uaibot as ub
import numpy as np
import matplotlib.pyplot as plt

def setup_motion_planning_simulation(problem_index, use_pc=False):

    filename = "fishbotics_mp_problems.npz"
    raw_url = "https://raw.githubusercontent.com/viniciusmgn/uaibot_content/master/contents/MotionPlanningProblems/" + filename

    if not os.path.isfile(filename):
        print(f"Baixando {filename}...")
        urllib.request.urlretrieve(raw_url, filename)
        print("Finalizado!")
    else:
        print("Arquivos já detectados")

    allproblems = np.load("fishbotics_mp_problems.npz", allow_pickle=True)
    allproblems = allproblems['arr_0'].item()
    name_problems = list(allproblems.keys())

    #Escolha um dos 2600 problemas pelo nome. Por exemplo, vamos pegar o primeiro
    name = name_problems[problem_index]
    prob = allproblems[name]

    #Extrai as informações
    all_obs = prob['all_obs']
    q0 = prob['q0']
    htm_tg = prob['htm_tg']
    htm_base = prob['htm_base']
    
    
    frame_tg = ub.Frame(htm=htm_tg, size=0.1)
    robot = ub.Robot.create_franka_emika_3(htm=htm_base)
    robot.add_ani_frame(time=0, q=q0)

    new_obs = []
    for obs in all_obs:
        cls_name = obs.__class__.__name__
        color = 'magenta'

        if cls_name == 'Box':
            width = obs.width
            depth = obs.depth
            height = obs.height
            htm = obs.htm

            new_box = ub.simobjects.Box(
                width=width,
                depth=depth,
                height=height,
                color=color,
                htm=htm
            )
            new_obs.append(new_box)

        elif cls_name == 'Cylinder':
            radius = obs.radius
            height = obs.height
            htm = obs.htm

            new_cyl = ub.simobjects.Cylinder(
                radius=radius,
                height=height,
                color=color,
                htm=htm
            )
            new_obs.append(new_cyl)

    all_obs = new_obs
    
    
    if use_pc:
        all_points = []
        for obs in all_obs:
            all_points+=[np.matrix(p).T for p in obs.to_point_cloud(disc=0.04).points.T]
            
        all_obs = [ub.PointCloud(points=all_points, color='cyan', size=0.02)]
    
    sim = ub.Simulation()
    sim.add(all_obs)
    sim.add(robot)
    sim.add(frame_tg)

    return robot, sim, all_obs, q0, htm_tg, htm_base

def set_configuration_speed(robot, q_dot, t, dt):
    q_next = robot.q + q_dot*dt
    robot.add_ani_frame(time = t+dt, q = q_next)

def draw_balls(pathhh_, robot, sim, color="cyan", radius = 0.01):
        sl = [ ]
        for q_c in pathhh_:
            fkm = robot.fkm(q = q_c)
            sl.append( fkm[ 0 : 3 , 3] )            
        balls = []
        for s in sl:
            balls.append( ub.Ball(htm = ub.Utils.trn(s), radius = radius, color = color))
        sim.add(balls)

def draw_pc(pathhh_, robot, sim, color="cyan", radius = 0.01):
    sl = [ ]
    for q_c in pathhh_:
        fkm = robot.fkm(q_c)
        sl.append( fkm[ 0 : 3 , 3] ) 
    pc = ub.PointCloud(size = radius, color = color, points = sl)
    sim.add(pc)


def vector_field(robot, curve ,alpha= 1, const_vel= 1.7, track_vel = 0.5):
    n = np.shape(robot.q)[0]

    index = -1
    dmin = float('inf')
    for i in range(len(curve)):
        d = np.linalg.norm(robot.q - curve[i])
        if d < dmin:
            dmin = d
            index = i

    f_g = 0.63 * np.arctan(alpha * dmin)
    f_h = track_vel * np.sqrt(max(1 - f_g**2, 0))

    if index < len(curve) - 1:
        diff = curve[index + 1] - curve[index]
    else: 
        diff = curve[index] - curve[index - 1]
    T = diff / (np.linalg.norm(diff) + 1e-8)
    
    N = (curve[index] - robot.q) / (np.linalg.norm(curve[index] - robot.q) + 1e-8)


    q_dot = abs(const_vel) * (f_g * N + f_h * T)

    return q_dot

def set_configuration_speed(robot, q_dot, t, dt):
    q_next = robot.q + q_dot*dt
    robot.add_ani_frame(time = t+dt, q = q_next)




def my_vector_field(robot, curve ,alpha= 1, const_vel= 1.7, track_vel = 0.5):
    n = np.shape(robot.q)[0]

    index = -1
    dmin = float('inf')
    for i in range(len(curve)):
        d = np.linalg.norm(robot.q - curve[i])
        if d < dmin:
            dmin = d
            index = i

    f_g = 0.63 * np.arctan(alpha * dmin)
    f_h = track_vel * np.sqrt(max(1 - f_g**2, 0))

    if index < len(curve) - 1:
        diff = curve[index + 1] - curve[index]
    else: 
        diff = curve[index] - curve[index - 1]
    T = diff / (np.linalg.norm(diff) + 1e-8)
    
    N = (curve[index] - robot.q) / (np.linalg.norm(curve[index] - robot.q) + 1e-8)


    q_dot = abs(const_vel) * (f_g * N + f_h * T)

    return q_dot