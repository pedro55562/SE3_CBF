import numpy as np
import uaibot as ub
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull

def is_bounded(A, b):
    n_dim = A.shape[1]

    for i in range(n_dim):
        c = np.zeros(n_dim)
        c[i] = 1  

        res = linprog(c, A_ub=A, b_ub=b, method='highs', bounds=[(None, None)] * n_dim)

        if res.status == 3 or abs(res.x[i])>1e3: 
            return False  

        c[i] = -1  

        res = linprog(c, A_ub=A, b_ub=b, method='highs', bounds=[(None, None)] * n_dim)

        if res.status == 3 or abs(res.x[i])>1e3: 
            return False  
        
    return True  

def generate_bounded_polytope(n_halfspaces=10, size = 1.0):
    while True:
        A = np.random.uniform(-1, 1, (n_halfspaces, 3))  # Each row is a normal vector
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        b = size*np.abs(np.random.uniform(0.1, 0.3, (n_halfspaces,)))
        
        if is_bounded(A, b):
            return A, b

def generate_rand_object(objtype = -1, size = 1.0, spread = 1.0, color='red'):
    
    if objtype == -1:
        coin = np.random.randint(0,5)
    else:
        coin = objtype

    htm = ub.Utils.htm_rand([-spread*1.5,-spread*1.5,spread*0.5],[spread*1.5,spread*1.5,spread*1.5])
    
    if coin==0:
        r = size* np.random.uniform(0.3,1.0)
        obj = ub.Ball(htm=htm, radius=r, color=color)
        
        return obj
        
    if coin==1:
        w = size*np.random.uniform(0.3,0.8)
        d = size*np.random.uniform(0.3,0.8)
        h = size*np.random.uniform(0.3,0.8)
        obj = ub.Box(htm=htm, width=w, depth=d, height=h, color=color)
        
        return obj


    if coin==2:
        r = size*np.random.uniform(0.3,0.8)
        h = size*np.random.uniform(0.3,0.8)
        obj = ub.Cylinder(htm=htm, radius = r, height=h, color=color)
        
        return obj

        
    if coin==3:
        n = np.random.randint(500,1000)  
        m1 = np.matrix(np.random.uniform(low=0, high=1, size=(3, n))) 
        m2 = np.matrix(np.random.uniform(low=0, high=1, size=(3, n)))
        w = size*np.random.uniform(0.3,0.8)
        d = size*np.random.uniform(0.3,0.8)
        h = size*np.random.uniform(1.0,1.5)
        pc = np.matrix([w,d,h]).transpose()
        
        w = size*np.random.uniform(0.3,0.8)
        d = size*np.random.uniform(0.3,0.8)
        h = size*np.random.uniform(1.0,1.5)
        
        points = np.diag([w,d,h])*htm[0:3,0:3]*(m1+m2)+pc
        obj = ub.PointCloud(points = points, color=color)
        
        return obj
 
        
    if coin==4:
        A, b = generate_bounded_polytope(size = size)
        obj = ub.ConvexPolytope(A = A, b = b, color=color)
        obj.add_ani_frame(0,htm)
        
        return obj
            
        
