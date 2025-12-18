import uaibot as ub
import numpy as np
from utils_test import *
# from uaibot.utils import Utils



input1 = [1,2,3]
input2 = [[1],[2],[3]]
input3 = np.matrix([1,2,3])
input4 = np.array([1,2,3])
input5 = np.matrix([1,2,3]).T


print(ub.Utils.S(input1))
print(ub.Utils.S(input2))
print(ub.Utils.S(input3))
print(ub.Utils.S(input4))
print(ub.Utils.S(input5))

print(ub.Utils.rot(input1,3))
print(ub.Utils.rot(input2,3))
print(ub.Utils.rot(input3,3))
print(ub.Utils.rot(input4,3))
print(ub.Utils.rot(input5,3))


print(ub.Utils.trn(input2))
print(ub.Utils.trn(input3))
print(ub.Utils.trn(input4))
print(ub.Utils.trn(input5))

print(ub.Utils.inv_htm(ub.Utils.rot(input1,3)*ub.Utils.trn([0,1,2])))
print(ub.Utils.inv_htm(ub.Utils.rot(input2,3)*ub.Utils.trn([0,1,2])))
print(ub.Utils.inv_htm(ub.Utils.rot(input3,3)*ub.Utils.trn([0,1,2])))
print(ub.Utils.inv_htm(ub.Utils.rot(input4,3)*ub.Utils.trn([0,1,2])))
print(ub.Utils.inv_htm(ub.Utils.rot(input5,3)*ub.Utils.trn([0,1,2])))

a, b = ub.Utils.axis_angle(ub.Utils.inv_htm(ub.Utils.rot(input1,3)*ub.Utils.trn([0,1,2])))

print(a)
print(b)

a, b, c = ub.Utils.euler_angles(ub.Utils.inv_htm(ub.Utils.rot(input1,3)*ub.Utils.trn([0,1,2])))

print([a,b,c])

A1 = [[-1,-2,3],[4,5,6]]
A2 = np.matrix([[-1,-2,3],[4,5,6]])
A3 = np.array([[-1,-2,3],[4,5,6]])

b1 = [1,0]
b2 = np.matrix([1,0])
b3 = np.matrix([1,0]).T

print(ub.Utils.dp_inv(A1))
print(ub.Utils.dp_inv(A2))
print(ub.Utils.dp_inv(A3))

print(ub.Utils.dp_inv_solve(A1, b1))
print(ub.Utils.dp_inv_solve(A2, b2))
print(ub.Utils.dp_inv_solve(A3, b3))
print(ub.Utils.dp_inv_solve(A1, b1, mode='python'))
print(ub.Utils.dp_inv_solve(A2, b2, mode='python'))
print(ub.Utils.dp_inv_solve(A3, b3, mode='python'))

H1 = [[2,1],[1,3]]
H2 = np.array(H1)
H3 = np.matrix(H1)
H4 = np.matrix(H1)

f1 = [1,2]
f2 = np.matrix([1,2])
f3 = np.matrix([1,2]).T
f4 = np.array([1,2])

A1 = [[1,2],[3,4],[4,5],[5,6]]
A2 = np.matrix([[1,2],[3,4],[4,5],[5,6]])
A3 = np.array([[1,2],[3,4],[4,5],[5,6]])
A4 = np.array([[1,2],[3,4],[4,5],[5,6]])

b1 = [10,20,30,40]
b2 = np.matrix([10,20,30,40])
b3 = np.matrix([10,20,30,40]).T
b4 = np.array([10,20,30,40])

print("QP Solver")

print(ub.Utils.solve_qp(H1, f1, A1, b1))
print(ub.Utils.solve_qp(H2, f2, A2, b2))
print(ub.Utils.solve_qp(H3, f3, A3, b3))
print(ub.Utils.solve_qp(H4, f4, A4, b4))

robot = ub.Robot.create_franka_emika_3()

NO_TRY = 20

print("-------------------------------------------------------------")
print("Testing FK: ")

max_error = 0
for i in range(NO_TRY):
    q_input = [np.random.randn() for j in range(7)]
    
    fk1 = robot.fkm(q=q_input,mode='c++')
    fk2 = robot.fkm(q=np.matrix(q_input),mode='c++')
    fk3 = robot.fkm(q=q_input,mode='python')
    fk4 = robot.fkm(q=np.matrix(q_input),mode='python')  
    
    max_error = max(max_error, np.linalg.norm(fk1-fk2))
    max_error = max(max_error, np.linalg.norm(fk3-fk4))  
    max_error = max(max_error, np.linalg.norm(fk1-fk3))
    max_error = max(max_error, np.linalg.norm(fk2-fk4))      

print("Max error FK  = "+str(max_error))

print("-------------------------------------------------------------")
print("Testing Jacobian: ")

max_error = 0
for i in range(NO_TRY):
    q_input = [np.random.randn() for j in range(7)]
    
    jg1, fk1 = robot.jac_geo(q=q_input,mode='c++')
    jg2, fk2 = robot.jac_geo(q=np.matrix(q_input),mode='c++')
    jg3, fk3 = robot.jac_geo(q=q_input,mode='python')
    jg4, fk4 = robot.jac_geo(q=np.matrix(q_input),mode='python')  
    
    max_error = max(max_error, np.linalg.norm(fk1-fk2))
    max_error = max(max_error, np.linalg.norm(fk3-fk4))  
    max_error = max(max_error, np.linalg.norm(fk1-fk3))
    max_error = max(max_error, np.linalg.norm(fk2-fk4))      
    max_error = max(max_error, np.linalg.norm(jg1-jg2))
    max_error = max(max_error, np.linalg.norm(jg3-jg4))  
    max_error = max(max_error, np.linalg.norm(jg1-jg3))
    max_error = max(max_error, np.linalg.norm(jg2-jg4))  
    
print("Max error Jacobian  = "+str(max_error))

print("-------------------------------------------------------------")
print("Testing IK  ")

max_error = 0
no_fail = 0

obs = ub.Ball(htm = ub.Utils.trn([0.5,0,0.6]),radius=0.25)
for i in range(NO_TRY):
    q_input = np.matrix(robot.q)
    
    cont = True
    
    while cont:
        for i in range(7):
            q_input[i,0] = robot.joint_limit[i,0]+np.random.rand()*(robot.joint_limit[i,1]-robot.joint_limit[i,0])
            
        ok, _, _ = robot.check_free_config(q=q_input, obstacles=[obs])
        cont = not ok

    
    htm_rand = robot.fkm(q_input)
    
    try:
        q_inv_cpp = robot.ikm(htm_tg = htm_rand, mode='c++', obstacles=[obs])
        q_inv_py = robot.ikm(htm_tg = htm_rand, mode='python', obstacles=[obs])
        
        error_cpp = np.linalg.norm(htm_rand-robot.fkm(q_inv_cpp))
        error_py = np.linalg.norm(htm_rand-robot.fkm(q_inv_py))
        
        max_error = max(max_error, error_cpp)
        max_error = max(max_error, error_py)
    except:
        no_fail = no_fail+1
        
print("No IK fail: "+str(no_fail))
print("Max error IK = "+str(max_error))

print("-------------------------------------------------------------")
print("Testing Task Function: ")

max_error = 0
for i in range(NO_TRY):
    q_input = [np.random.randn() for j in range(7)]
    htm_rand = robot.fkm(q_input)
    q_input = [np.random.randn() for j in range(7)]
    
    r1, jac_r1 = robot.task_function(q=q_input,mode='c++', htm_tg = htm_rand)
    r2, jac_r2 = robot.task_function(q=np.matrix(q_input),mode='c++', htm_tg = htm_rand)
    r3, jac_r3 = robot.task_function(q=q_input,mode='python', htm_tg = htm_rand)
    r4, jac_r4 = robot.task_function(q=np.matrix(q_input),mode='python', htm_tg = htm_rand)  
    
    max_error = max(max_error, np.linalg.norm(r1-r2))
    max_error = max(max_error, np.linalg.norm(r3-r4))  
    max_error = max(max_error, np.linalg.norm(r1-r3))
    max_error = max(max_error, np.linalg.norm(r2-r4))      
    max_error = max(max_error, np.linalg.norm(jac_r1-jac_r2))
    max_error = max(max_error, np.linalg.norm(jac_r3-jac_r4))  
    max_error = max(max_error, np.linalg.norm(jac_r1-jac_r3))
    max_error = max(max_error, np.linalg.norm(jac_r2-jac_r4))
    
print("Max error Task Function = "+str(max_error))

print("-------------------------------------------------------------")

print("Testing distance computation: ")

max_error = 0
for i in range(NO_TRY):
    
    coin1 = np.random.randint(0,5)
    coin2 = np.random.randint(0,5)
    
    obj1 = generate_rand_object(objtype = coin1 if coin1!=3 else 0)
    obj2 = generate_rand_object(objtype = coin2 if coin2!=3 else 0)
    
    _,_, d_cpp, _ = obj1.compute_dist(obj2,  p_init=[np.random.randn() for j in range(3)], tol=0.0001, no_iter_max=2000, mode='c++')
    _,_, d_py, _  = obj1.compute_dist(obj2,  p_init=[np.random.randn() for j in range(3)], tol=0.0001, no_iter_max=2000, mode='python')
    
    max_error = max(max_error,abs(d_cpp-d_py))
    
print("Max error distance computation: "+str(max_error))

print("-------------------------------------------------------------")
print("Testing check_free_config: ")

no_comp_failed = 0
for i in range(NO_TRY):
    
    obj = generate_rand_object()

    
    q_input = [np.random.randn() for j in range(7)]

    ok_cpp, _, _ = robot.check_free_config(q=q_input, obstacles=[obj], mode='c++')
    ok_py, _, _  = robot.check_free_config(q=q_input, obstacles=[obj], mode='python',  no_iter_max=5000)
    
    if not (ok_cpp==ok_py):
        no_comp_failed = no_comp_failed+1
        
print("Failed in "+str(no_comp_failed))




    