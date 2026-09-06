import numpy as np
import uaibot as ub
from scipy.linalg import expm

############################
# Basic functions
############################

def prop_htm(H,xi,dt):
    Hn = np.matrix(H)
    Hn[0:3,-1]+=xi[0:3,-1]*dt
    Hn[0:3,0:3] = expm(ub.Utils.S(xi[3:6,-1]*dt))*Hn[0:3,0:3]  
    return Hn  

def cmpt_lambda_terms(objA,objB,H,xi,h,eps):
    
    #Extract:
    sA = H[0:3,-1]
    vA = xi[0:3,-1]
    wA = xi[3:6,-1]
    
    #Put objA in the pose, temporarily
    htm_A = np.matrix(objA.htm)
    objA.set_ani_frame(H)
    
    #Compute dist
    a_star, b_star, lambda_AB, _ = objA.compute_dist(obj = objB, h = h, eps = eps, tol=1e-6, no_iter_max=6000)
    
    #Compute D_xi_lambda_AB
    D_xi_lambda_AB = np.matrix(np.zeros((1,6)))
    D_xi_lambda_AB[:,0:3] = (a_star-b_star).T/(1e-6+lambda_AB)
    D_xi_lambda_AB[:,3:6] = (ub.Utils.S(b_star-sA)*(a_star-b_star)).T/(1e-6+lambda_AB)
     
    #Compute (d/dt) D_xi_lambda_AB
    psi_A_a = vA + ub.Utils.S(wA)*(a_star-sA)
    psi_A_b = vA + ub.Utils.S(wA)*(b_star-sA)
    J_Pi_A = np.matrix(objA.cpp_obj.projection_jacobian(b_star,h,eps))
    J_Pi_B = np.matrix(objB.cpp_obj.projection_jacobian(a_star,h,eps))
    d_Pi_A_dt = psi_A_a - J_Pi_A*psi_A_b

        
    a_star_dot = np.linalg.inv(np.identity(3)-J_Pi_A*J_Pi_B)*d_Pi_A_dt
    b_star_dot = J_Pi_B*a_star_dot
    
    d_D_xi_Lambda_AB_dt = np.matrix(np.zeros((1,6)))
    d_D_xi_Lambda_AB_dt[:,0:3] = (a_star_dot-b_star_dot).T
    d_D_xi_Lambda_AB_dt[:,3:6] = (ub.Utils.S(b_star-sA)*(a_star_dot-b_star_dot)+ub.Utils.S(b_star_dot-vA)*(a_star-b_star)).T
    
    d_D_xi_lambda_AB_dt = (d_D_xi_Lambda_AB_dt - (D_xi_lambda_AB*xi)*D_xi_lambda_AB)/(1e-6+lambda_AB)

    #Put back the initial pose of the object
    objA.set_ani_frame(htm_A)
    
    return lambda_AB, D_xi_lambda_AB, d_D_xi_lambda_AB_dt
    
def cmpt_control(H,xi,obj_robot,list_obs,u_d,h=0.05,eps=0.01,eta=0.3,lambda_min=0.01,xi_lim=0.08):
    
    A = np.matrix(np.zeros((0,6)))
    b = np.matrix(np.zeros((0,1)))
    
    #Add the constraints for obstacle avoidance
    for obs in list_obs:
        lambda_RO, D_xi_lambda_RO, d_D_xi_lambda_RO_dt = cmpt_lambda_terms(obj_robot,obs,H,xi,h,eps)
        
        ff = -d_D_xi_lambda_RO_dt*xi
        d_lambda_RO_dt = D_xi_lambda_RO*xi
        b_temp = ff - 2*eta*d_lambda_RO_dt-eta*eta*(lambda_RO-lambda_min)
        
        A = np.vstack([A,D_xi_lambda_RO])
        b = np.vstack([b,b_temp])
        
        
    #Add limits for xi
    A = np.vstack([A,np.identity(6)])
    b = np.vstack([b,-xi_lim*np.ones((6,1))])
    A = np.vstack([A,-np.identity(6)])
    b = np.vstack([b,-xi_lim*np.ones((6,1))]) 
    
    #Solve the QP
    u = ub.Utils.solve_qp(2*np.identity(6),-2*u_d,A,b)   
    
    return u  
        
###########################################
#Test the derivatives, analytic vs numerical
############################################


max_error = 0
max_rel_error = 0
dt = 1e-3
h = 0.05
eps = 0.01  
max_test=5000

# count=0
# for i in range(max_test):
#     cont = True
    
#     #Sample two objects, ensure that they are not very close (distance is not differentiable 
#     # when distance is 0)
#     while cont:
#         if np.random.rand()<0.5:
#             objA = ub.Box(htm = ub.Utils.htm_rand([-5,-5,-5],[5,5,5]))
#         else:
#             objA = ub.Cylinder(htm = ub.Utils.htm_rand([-5,-5,-5],[5,5,5]))
            
#         if np.random.rand()<0.5:
#             objB = ub.Box(htm = ub.Utils.htm_rand([-5,-5,-5],[5,5,5]))
#         else:
#             objB = ub.Cylinder(htm = ub.Utils.htm_rand([-5,-5,-5],[5,5,5]))
            
#         cont = objA.compute_dist(objB)[2]<0.05        
        
        
#     H = np.matrix(objA.htm)
#     xi = 0.5*np.matrix(np.random.randn(6,1))
    
#     lambda_AB, D_xi_lambda_AB, d_D_xi_lambda_AB_dt = cmpt_lambda_terms(objA,objB,H,xi,h,eps)
    
#     #Compute super-careful numerical derivative
#     dtp = dt
    
#     cont = True
    
#     while cont:
    
#         _, D_p, _ = cmpt_lambda_terms(objA,objB,prop_htm(H,xi,dtp),xi,h,eps)
#         _, D_pp, _ = cmpt_lambda_terms(objA,objB,prop_htm(H,xi,2*dtp),xi,h,eps)
#         _, D_m, _ = cmpt_lambda_terms(objA,objB,prop_htm(H,xi,-dtp),xi,h,eps)
#         _, D_mm, _ = cmpt_lambda_terms(objA,objB,prop_htm(H,xi,-2*dtp),xi,h,eps)
        
#         d_D_xi_lambda_AB_dt_num = (-D_pp+8*D_p-8*D_m+D_mm)/(12*dtp)
#         error = np.linalg.norm(d_D_xi_lambda_AB_dt-d_D_xi_lambda_AB_dt_num)
        
#         cont = error>1e-1 and dtp>1e-4
#         dtp = 0.8*dtp
    

#     if error<1e-3:
#         count+=1
        
#     max_error = max(max_error, error)


# print("MAX ERROR:")
# print(max_error)
# print(str(round(100*count/max_test,2))+"% of the tests had error <1e-3")


###########################################
#Make a simple simulation
############################################

def cmpt_target_u_d(H, xi, H_d, kc=0.5):
#Spatial acceleration to reach a constant target pose Hd    
    
    def ext(M):
        return np.matrix(np.diag(M)).T
    
    s = H[0:3,-1]
    s_d = H_d[0:3,-1]
    Q = H[0:3,0:3]
    Q_d = H_d[0:3,0:3]
    
    v = xi[0:3,-1]
    w = xi[3:6,-1]
    Sx = ub.Utils.S([1,0,0])
    Sy = ub.Utils.S([0,1,0])
    Sz = ub.Utils.S([0,0,1])
    Sw = ub.Utils.S(w)
    
    r = np.matrix(np.zeros((6,1)))
    r[0:3,-1] = s-s_d
    r[3:6,-1] = ext(np.identity(3)-Q_d.T*Q)
    
    # #(d/dt) r = D_xi_r*xi
    D_xi_r = np.matrix(np.zeros((6,6)))
    D_xi_r[0:3,:] = np.hstack([np.identity(3), np.zeros((3,3))])
    ax = ext(-Q_d.T*Sx*Q)
    ay = ext(-Q_d.T*Sy*Q)
    az = ext(-Q_d.T*Sz*Q)
    D_xi_r[3:6,:] = np.hstack([np.zeros((3,3)), ax, ay, az]) 
    
    #d_D_xi_r_dt = (d/dt) D_xi_r
    d_D_xi_r_dt = np.matrix(np.zeros((6,6)))
    dax = ext(-Q_d.T*Sx*Sw*Q)
    day = ext(-Q_d.T*Sy*Sw*Q)
    daz = ext(-Q_d.T*Sz*Sw*Q)
    d_D_xi_r_dt[3:6,:] = np.hstack([np.zeros((3,3)), dax, day, daz]) 
    
    #Compute u_d
    u_d = ub.Utils.dp_inv(D_xi_r)*(-d_D_xi_r_dt*xi-2*kc*D_xi_r*xi-kc*kc*r) 
  
    return u_d, r


dt = 0.003
t_max = 36
H = ub.Utils.trn([0,0,1])
xi= np.matrix(np.zeros((6,1)))
H_d = ub.Utils.trn([4,4,2])*ub.Utils.rotx(0.3)*ub.Utils.roty(0.4)

robot = ub.Cylinder(htm = H,radius=0.3,height=0.2,opacity=0.5,color='yellow')
#CREATE A COPY OF THE ROBOT OBJECT JUST TO USE IN THE FUNCTION. NO NEED TO UPDATE ITS POSE
#WE ONLY NEED THE GEOMETRY
robot_copy = robot.copy()

robot_frame = ub.Frame(size=0.4)
target_frame = ub.Frame(size=0.4,htm=H_d)



list_obs=[]
list_obs.append(ub.Box(htm=ub.Utils.trn([1,1,2]),width=0.2,depth=0.2,height=4,color='magenta'))
list_obs.append(ub.Box(htm=ub.Utils.trn([3,1.6,2]),width=0.2,depth=2.0,height=4,color='magenta'))


sim = ub.Simulation.create_sim_mountain([robot, robot_frame, target_frame])
sim.add(list_obs)


hist_r = []
hist_xi = []
hist_u = []
hist_t = []


for i in range(round(t_max/dt)):
    
    t = i*dt
    u_d, r =  cmpt_target_u_d(H, xi, H_d)
    print(np.shape(H))
    print(np.shape(xi))
    print(np.shape(u_d))
    u = cmpt_control(H,xi,robot_copy,list_obs,u_d)
    break
    xi +=u*dt
    H = prop_htm(H,xi,dt)
    
    hist_t.append(t)
    hist_r.append(np.linalg.norm(r))
    hist_xi.append(np.matrix(xi))
    hist_u.append(u)

    
    
    robot.add_ani_frame(t,H)
    robot_frame.add_ani_frame(t,H)
    

import matplotlib.pyplot as plt



for i in range(6):
    plt.plot(hist_t, [ua[i,0] for ua in hist_u])
    
plt.show()
    


for i in range(6):
    plt.plot(hist_t, [xia[i,0] for xia in hist_xi])
    
plt.show()

# sim.run()






