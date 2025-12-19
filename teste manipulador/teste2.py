import numpy as np
import uaibot as ub
import matplotlib.pyplot as plt

def draw_balls(pathhh_, robot, sim, color="cyan", radius = 0.01):
        sl = [ ]
        for q_c in pathhh_:
            fkm = robot.fkm(q = q_c)
            sl.append( fkm[ 0 : 3 , 3] )            
        balls = []
        for s in sl:
            balls.append( ub.Ball(htm = ub.Utils.trn(s), radius = radius, color = color))
        sim.add(balls)
        
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

t = 0
dt = 0.01
i = 0

robot = ub.Robot.create_kuka_lbr_iiwa(htm=ub.Utils.rotz(np.pi/2))
param_use_pc = False

htm_tg_0 = ub.Utils.trn([0.3, 0.0, 0.64])*ub.Utils.roty(np.pi)
htm_tg_1 = ub.Utils.trn([0.3, 0.0, 0.54])*ub.Utils.roty(np.pi)
htm_tg_2 = ub.Utils.trn([0.3, 0.0, 0.64])*ub.Utils.roty(np.pi)
htm_tg_3 = ub.Utils.trn([-0.65, 0.0, 0.63])*ub.Utils.roty(np.pi)
htm_tg_4 = ub.Utils.trn([-0.65, 0.0, 0.57])*ub.Utils.roty(np.pi)
htm_tg_5 = ub.Utils.trn([-0.65, 0.0, 0.63])*ub.Utils.roty(np.pi)
htm_tg_6 = ub.Utils.trn([-0.35, 0.35, 0.77])*ub.Utils.roty(-np.pi/2)
htm_tg_7 = ub.Utils.trn([0.30,  0.20, 0.64])*ub.Utils.roty(np.pi)


texture_steel = ub.Texture(
            url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/Textures/rough_metal.jpg',
            wrap_s='RepeatWrapping', wrap_t='RepeatWrapping', repeat=[4, 4])

texture_gold = ub.Texture(
            url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/Textures/gold_metal.png',
            wrap_s='RepeatWrapping', wrap_t='RepeatWrapping', repeat=[4, 4])


material_steel= ub.MeshMaterial(metalness=0.7, clearcoat=1, roughness=0.5, normal_scale=[0.5, 0.5], texture_map=texture_steel)
material_gold= ub.MeshMaterial(metalness=0.7, clearcoat=1, roughness=0.5, normal_scale=[0.5, 0.5], texture_map=texture_gold)

platform1 = ub.Box(htm=ub.Utils.trn([0.3, 0.0, 0.25]),width=0.2,depth=1.0,height=0.5,mesh_material=material_steel)
platform2 = ub.Box(htm=ub.Utils.trn([-0.7, 0.0, 0.25]),width=0.8,depth=1.0,height=0.5,mesh_material=material_steel)
platform3 = ub.Box(htm=ub.Utils.trn([-0.8, -0.35, 0.725]),width=0.6,depth=0.3,height=0.45,mesh_material=material_steel)
platform4 = ub.Box(htm=ub.Utils.trn([-0.8, 0.35, 0.725]),width=0.6,depth=0.3,height=0.45,mesh_material=material_steel)
platform5 = ub.Box(htm=ub.Utils.trn([-0.7, 0.0, 1.05]),width=0.8,depth=1.0,height=0.2,mesh_material=material_steel)
platform6 = ub.Box(htm=ub.Utils.trn([-0.4, 0.0, 0.55]),width=0.1,depth=0.4,height=0.1,mesh_material=material_steel)
platform7 = ub.Box(htm=ub.Utils.trn([-0.48,0.225,0.7]),width=0.1,depth=0.05,height=0.2,mesh_material=material_steel)
button = ub.Cylinder(htm=ub.Utils.trn([-0.6, 0.35, 0.77])*ub.Utils.roty(np.pi/2), radius=0.02, height =0.3, color='magenta')
disk = ub.Cylinder(htm=ub.Utils.trn([0.3,0,0.52]),height=0.04,radius=0.05,mesh_material=material_gold)



all_obs = [platform1, platform2, platform3, platform4, platform5, platform6, platform7]

#If point clouds are used, transform all objects in "all_obstalces" 
#into point clouds
if param_use_pc:
    
    all_points = []
    for obs in all_obs:
        all_points+=[np.matrix(p).T for p in obs.to_point_cloud(disc=0.04).points.T]
        
    all_obs = [ub.PointCloud(points=all_points, color='cyan', size=0.02)]
    
#Create simulation and add everything
sim = ub.Simulation.create_sim_factory([robot, button, disk])
sim.set_parameters(load_screen_color="#191919", background_color="#191919", width=500, height=500, show_world_frame=False, show_grid=False, camera_start_pose=[1.0,-1.0,1.0,-1.0,0.8,0,0.8])

sim.add(ub.Frame(htm_tg_0, size=0.1))
sim.add(ub.Frame(htm_tg_3, size=0.1))
sim.add(ub.Frame(htm_tg_6, size=0.1))
sim.add(ub.Frame(htm_tg_7, size=0.1))

for obs in all_obs:
    sim.add(obs)




success1, path1, iterations1, num_tries1, planning_time1 = robot.runRRT(q0=robot.q, htm_tg=htm_tg_0, obstacles=all_obs, usemultthread = True)
print("success1",success1)
print("planning_time1",planning_time1)
print("iterations1", iterations1)
print("num_tries1", num_tries1)

success2, path2, iterations2, num_tries2,planning_time2 = robot.runRRT(q0=path1[-1], htm_tg=htm_tg_3, obstacles=all_obs, usemultthread = True)
print("success2",success2)
print("planning_time2",planning_time2)
print("iterations2", iterations2)
print("num_tries2", num_tries2)

path = path1 + path2
q_goal = path[-1]


while np.linalg.norm(q_goal  - robot.q) > 0.01:
    t = i*dt
    q_dot = vector_field(robot=robot, curve=path, alpha=1, const_vel=1.7, track_vel=0.9)
    set_configuration_speed(robot=robot, q_dot=q_dot, t=t, dt=dt)
    
    i+=1
    



draw_balls(pathhh_=path2, robot=robot, sim=sim)

sim.save(address="/home/pedro/code_robot/SE3_CBF/",file_name="teste")


