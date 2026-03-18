from utils import *

from graphics.meshmaterial import *
from graphics.model3d import *

from simobjects.ball import *
from simobjects.box import *
from simobjects.cylinder import *

from .links import *


def _create_rigid_body_se3(htm, name, color, opacity):

    if not Utils.is_a_matrix(htm, 4, 4):
        raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

    if not (Utils.is_a_name(name)):
        raise Exception(
            "The parameter 'name' should be a string. Only characters 'a-z', 'A-Z', '0-9' and '_' are allowed. It should not begin with a number.")

    if not Utils.is_a_color(color):
        raise Exception("The parameter 'color' should be a HTML-compatible color.")

    if (not Utils.is_a_number(opacity)) or opacity < 0 or opacity > 1:
        raise Exception("The parameter 'opacity' should be a float between 0 and 1.")

    # 1 = Prismático, 0 = Revoluto
    
    link_info = [
       # Theta (Rot Z) [rad]
       [0.0, np.pi/2, 0.0, 0.0, 0.0, 0.0],
      
       # d (Trans Z) [m]  -> variáveis prismáticas
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],


       # Alpha (Rot X) [rad]
       [-np.pi/2, np.pi/2, 0.0, -np.pi/2, np.pi/2, 0.0],


       # a (Trans X) [m]
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],


       # (1 = Prismático, 0 = Revoluto)
       [1, 1, 1, 0, 0, 0]
   ]


    
    n = 6

    col_model = [[], [], [], [], [], []]
    
    col_model[5].append(Cylinder(htm=Utils.rotz(np.pi/2), 
                                 name=name + "_col", 
                                 radius=0.3, 
                                 height=0.20, 
                                 color="green", 
                                 opacity=0.55))




    base_3d_obj = [] 

    link_3d_obj = [[], [], [], [], [], []]

    link_3d_obj[5].append(
        Model3D(
        url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/CrazyFlie/crazyflie.obj',
        scale=3, 
        htm = Utils.trn([-0.0375 , 0, 0]) * Utils.rotx(-np.pi/2),
        mesh_material=MeshMaterial.create_rough_metal())
    ) 
    links = []
    for i in range(n):
        links.append(Link(i, link_info[0][i], link_info[1][i], link_info[2][i], link_info[3][i], link_info[4][i],
                          link_3d_obj[i]))

        for j in range(len(col_model[i])):
            links[i].attach_col_object(col_model[i][j], col_model[i][j].htm)

    q0 = [0, 0, 0, 0, np.pi/2, 0]

    large_val = 1000.0
    pi_val = np.pi
    
    joint_limits = np.matrix([
        [-large_val, large_val], 
        [-large_val, large_val], 
        [-large_val, large_val], 
        [-pi_val, pi_val], 
        [-pi_val, pi_val],
        [-pi_val, pi_val], 
       
    ])

    return base_3d_obj, links, np.identity(4), np.identity(4), q0, joint_limits