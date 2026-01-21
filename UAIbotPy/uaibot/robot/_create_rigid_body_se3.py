from utils import *

from graphics.meshmaterial import *
from graphics.model3d import *

from simobjects.ball import *
from simobjects.box import *
from simobjects.cylinder import *

from .links import *


def _create_rigid_body_se3(htm, name, color, opacity):
    # 1. Validações de entrada (Padrao do codigo fornecido)
    if not Utils.is_a_matrix(htm, 4, 4):
        raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

    if not (Utils.is_a_name(name)):
        raise Exception(
            "The parameter 'name' should be a string. Only characters 'a-z', 'A-Z', '0-9' and '_' are allowed. It should not begin with a number.")

    if not Utils.is_a_color(color):
        raise Exception("The parameter 'color' should be a HTML-compatible color.")

    if (not Utils.is_a_number(opacity)) or opacity < 0 or opacity > 1:
        raise Exception("The parameter 'opacity' should be a float between 0 and 1.")

    # 2. Definição da Tabela DH para 6-DoF (PPP + RRR)
    # Estrutura: [Theta, d, Alpha, a, Type]
    # Type: 1 = Prismático, 0 = Revoluto
    
    # Nota sobre a cadeia PPP:
    # Para mover em X, Y, Z usando juntas prismáticas (que normalmente atuam em Z),
    # rotacionamos os eixos entre as juntas.
    link_info = [
        # Theta (Rot Z) [rad]
        [ np.pi/2,     0.0,     np.pi/2,    0.0,    0.0,    0.0 ],

        # d (Trans Z) [m]
        [ 0.0,     0.0,     0.0,    0.0,    0.0,    0.0 ],

        # Alpha (Rot X) [rad]
        [ 0.0,    np.pi/2,  0.0,   -np.pi/2, 0.0,    0.0 ],

        # a (Trans X) [m]
        [ 0.0,     0.0,     0.0,    0.0,    0.0,    0.0 ],

        # Type (1 = Prismático, 0 = Revoluto)
        [ 1,       0,       1,      0,      1,      0 ]
    ]

    n = 6

    # 3. Modelo de Colisão
    # Cilindro simples, anexado apenas ao ultimo elo (o corpo do drone)
    col_model = [[], [], [], [], [], []]
    
    # Ajustamos o cilindro para englobar o hexacoptero.
    # Raio aprox 0.4m, Altura 0.15m. Rotacionado para alinhar com o plano do drone se necessario.
    # Assumindo Z up no modelo visual.
    col_model[5].append(Cylinder(htm=Utils.roty(np.pi/2), 
                                 name=name + "_col", 
                                 radius=0.3, 
                                 height=0.20, 
                                 color="red", 
                                 opacity=0.3))

    # 4. Objetos 3D Visuais
    # Os elos 0-4 são virtuais (sem malha), apenas o elo 5 (efetuador) tem o hexacoptero.
    
    # Base vazia (free floating body)
    base_3d_obj = [] 

    link_3d_obj = [[], [], [], [], [], []]

    link_3d_obj[5].append(
        Model3D(
        url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/CrazyFlie/crazyflie.obj',
        scale=3, 
        htm = Utils.trn([0 , 0.0375, 0]) * Utils.rotz(-np.pi/2),
        mesh_material=MeshMaterial.create_rough_metal())
    ) 
    # 5. Criação dos Links
    links = []
    for i in range(n):
        links.append(Link(i, link_info[0][i], link_info[1][i], link_info[2][i], link_info[3][i], link_info[4][i],
                          link_3d_obj[i]))

        for j in range(len(col_model[i])):
            links[i].attach_col_object(col_model[i][j], col_model[i][j].htm)

    # 6. Configuração Inicial e Limites
    q0 = [0, 000, 0, 000 , 0, 0]
# 1 3 5

    large_val = 1000.0
    pi_val = np.pi
    
    joint_limits = np.matrix([
        [-large_val, large_val], # X
        [-pi_val, pi_val], # Y
        [-large_val, large_val], # Z
        [-pi_val, pi_val],       # Yaw
        [-large_val, large_val],       # Pitch
        [-pi_val, pi_val]        # Roll
    ])

    return base_3d_obj, links, np.identity(4), np.identity(4), q0, joint_limits