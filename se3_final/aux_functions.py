import numpy as np
import uaibot as ub
import heapq
import math
from dataclasses import dataclass



@dataclass(order=True)
class _Interval:
    # heap order key first
    lower_bound: float
    a: float
    b: float
    fa: float
    fb: float
    x_star: float   # point where the piecewise-linear lower bound attains its minimum on [a,b]
    lb_star: float  # that minimum value (equals lower_bound)


def _interval_lower_envelope(a, b, fa, fb, L):
    """
    For Lipschitz f with constant L, on [a,b] we have:
      f(x) >= max( fa - L|x-a|, fb - L|b-x| )
    The minimum of this lower envelope occurs at:
      x* = (a + b)/2 + (fa - fb)/(2L)
    clamped to [a,b].
    The lower bound value there is:
      lb* = (fa + fb)/2 - L(b-a)/2   (if x* inside)
    If clamped, we evaluate at the clamp point.
    """
    if L <= 0:
        raise ValueError("L must be > 0")

    # candidate minimizer of the intersection point
    x_star = 0.5 * (a + b) + (fa - fb) / (2.0 * L)
    if x_star < a:
        x_star = a
    elif x_star > b:
        x_star = b

    # lower envelope at x_star
    lb = max(fa - L * abs(x_star - a), fb - L * abs(b - x_star))
    return x_star, lb


def lipschitz_argmin(f, L, delta=1e-3, a0=0.0, b0=1.0, max_evals=20000):
    """
    Global minimization of f on [0,1] given Lipschitz constant L.

    Returns: (s_hat, f(s_hat), info_dict)

    Guarantee (informal but standard):
      The algorithm maintains intervals with valid lower bounds.
      It keeps refining the interval with the smallest lower bound.
      When the best interval length <= 2*delta, returning its midpoint
      gives a point within delta of *some* global minimizer.

    Parameters
    ----------
    f : callable
        Function f(x) for x in [0,1].
    L : float
        Known Lipschitz constant of f on [0,1].
    delta : float
        Target error in the minimizer location (x). Must be > 0.
    max_evals : int
        Safety cap on number of function evaluations.

    Notes
    -----
    - Works for nonconvex, nonsmooth functions as long as Lipschitz holds.
    - If L is an overestimate, it still works (just slower).
    - If L is an underestimate, guarantees break.
    """
    if delta <= 0:
        raise ValueError("delta must be > 0")
    if L <= 0:
        raise ValueError("L must be > 0")

    # initial samples at endpoints
    fa0, fb0 = float(f(a0)), float(f(b0))
    evals = 2

    best_x = a0 if fa0 <= fb0 else b0
    best_fx = min(fa0, fb0)

    x_star, lb = _interval_lower_envelope(a0, b0, fa0, fb0, L)
    heap = []
    heapq.heappush(heap, _Interval(lb, a0, b0, fa0, fb0, x_star, lb))

    # main loop: refine most "promising" interval (smallest lower bound)
    while heap:
        itv = heapq.heappop(heap)
        a, b, fa, fb = itv.a, itv.b, itv.fa, itv.fb

        # stopping criterion in terms of location accuracy
        if (b - a) <= delta:
            s_hat = 0.5 * (a + b)
            return s_hat, float(f(s_hat)), {
                "status": "ok",
                "evals": evals + 1,
                "best_seen_x": best_x,
                "best_seen_fx": best_fx,
                "final_interval": (a, b),
                "final_interval_length": (b - a),
            }

        # evaluate at the point that minimizes the lower envelope on this interval
        x_new = itv.x_star
        fx_new = float(f(x_new))
        evals += 1
        if evals > max_evals:
            return best_x, best_fx, {
                "status": "max_evals_reached",
                "evals": evals,
                "best_seen_x": best_x,
                "best_seen_fx": best_fx,
                "last_interval": (a, b),
            }

        # update incumbent
        if fx_new < best_fx:
            best_fx = fx_new
            best_x = x_new

        # split interval at x_new (if x_new equals endpoint, split at midpoint instead)
        if x_new <= a + 1e-15 or x_new >= b - 1e-15:
            x_new = 0.5 * (a + b)
            fx_new = float(f(x_new))
            evals += 1
            if fx_new < best_fx:
                best_fx = fx_new
                best_x = x_new
            if evals > max_evals:
                return best_x, best_fx, {
                    "status": "max_evals_reached",
                    "evals": evals,
                    "best_seen_x": best_x,
                    "best_seen_fx": best_fx,
                    "last_interval": (a, b),
                }

        # left interval [a, x_new]
        xL = x_new
        fL = fx_new
        if xL > a + 1e-15:
            x_star_L, lb_L = _interval_lower_envelope(a, xL, fa, fL, L)
            heapq.heappush(heap, _Interval(lb_L, a, xL, fa, fL, x_star_L, lb_L))

        # right interval [x_new, b]
        xR = x_new
        fR = fx_new
        if b > xR + 1e-15:
            x_star_R, lb_R = _interval_lower_envelope(xR, b, fR, fb, L)
            heapq.heappush(heap, _Interval(lb_R, xR, b, fR, fb, x_star_R, lb_R))

    # Shouldn't get here
    return best_x, best_fx, {"status": "empty_heap", "evals": evals}

# ------------------------------------------------------------
# Basic SO(3) / SE(3) utilities (hat/vee, Jacobians, adjoint)
# ------------------------------------------------------------

_EPS = 1e-12


def skew(w):
    """w: (3,) -> 3x3 skew"""
    wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
    return np.array([[0.0, -wz,  wy],
                     [wz,  0.0, -wx],
                     [-wy, wx,  0.0]], dtype=float)


def inv_SE3(H):
    return np.vstack((np.hstack( (H[0:3,0:3].transpose(), -H[0:3,0:3].transpose()*H[0:3,-1])),(0,0,0,1)))

def vee_so3(W):
    """W: 3x3 skew -> (3,)"""
    return np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=float)


def exp_SO3(phi):
    """Rodrigues: exp([phi]^)"""
    phi = np.asarray(phi, dtype=float).reshape(3)
    th = np.linalg.norm(phi)
    W = skew(phi)

    if th < 1e-8:
        # series: I + W + 1/2 W^2
        return np.eye(3) + W + 0.5 * (W @ W)

    A = np.sin(th) / th
    B = (1.0 - np.cos(th)) / (th * th)
    return np.eye(3) + A * W + B * (W @ W)


def log_SO3(R):
    """
    Log in SO(3), returning phi with angle in [-pi, pi].
    R: 3x3 rotation
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    # numerical safety
    c = (np.trace(R) - 1.0) * 0.5
    c = float(np.clip(c, -1.0, 1.0))
    th = np.arccos(c)

    if th < 1e-8:
        # For small angles, log(R) ≈ 0.5*(R - R^T)
        W = 0.5 * (R - R.T)
        return vee_so3(W)

    if np.pi - th < 1e-6:
        # Near pi: use robust axis extraction
        # Compute axis from diagonal elements
        A = (R + np.eye(3)) * 0.5
        axis = np.empty(3, dtype=float)
        axis[0] = np.sqrt(max(A[0, 0], 0.0))
        axis[1] = np.sqrt(max(A[1, 1], 0.0))
        axis[2] = np.sqrt(max(A[2, 2], 0.0))

        # Fix signs using off-diagonals
        if R[2, 1] - R[1, 2] < 0: axis[0] = -axis[0]
        if R[0, 2] - R[2, 0] < 0: axis[1] = -axis[1]
        if R[1, 0] - R[0, 1] < 0: axis[2] = -axis[2]

        n = np.linalg.norm(axis)
        if n < _EPS:
            # fallback
            W = (R - R.T) / (2.0 * np.sin(th))
            axis = vee_so3(W)
            n = np.linalg.norm(axis)
            if n < _EPS:
                axis = np.array([1.0, 0.0, 0.0])
            else:
                axis = axis / n
        else:
            axis = axis / n

        # angle in [0, pi]; make it within [-pi, pi] => keep +pi
        return axis * th

    # general case
    W = (R - R.T) / (2.0 * np.sin(th))
    axis = vee_so3(W)
    return axis * th

def jac_left_SO3(phi):
    """Left Jacobian of SO(3): J(phi) such that dexp_phi maps to left-trivialized tangent."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    th = np.linalg.norm(phi)
    W = skew(phi)

    if th < 1e-8:
        # J ≈ I + 1/2 W + 1/6 W^2
        return np.eye(3) + 0.5 * W + (1.0 / 6.0) * (W @ W)

    th2 = th * th
    A = (1.0 - np.cos(th)) / th2
    B = (th - np.sin(th)) / (th2 * th)
    return np.eye(3) + A * W + B * (W @ W)

def inv_jac_left_SO3(phi):
    """Inverse of left Jacobian of SO(3)."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    th = np.linalg.norm(phi)
    W = skew(phi)

    if th < 1e-8:
        # J^{-1} ≈ I - 1/2 W + 1/12 W^2
        return np.eye(3) - 0.5 * W + (1.0 / 12.0) * (W @ W)

    half = 0.5 * th
    cot_half = np.cos(half) / np.sin(half)  # cot(th/2)
    th2 = th * th
    return (np.eye(3)
            - 0.5 * W
            + (1.0 / th2) * (1.0 - th * 0.5 * cot_half) * (W @ W))


def exp_SE3(A):
    """
    Compute exp in SE(3).
    A: 4x4 np.matrix in se(3) (top-left skew, last row zeros).
    Returns: 4x4 np.matrix in SE(3).
    """
    A = np.asarray(A, dtype=float).reshape(4, 4)
    w_hat = A[:3, :3]
    v = A[:3, 3]
    phi = vee_so3(w_hat)

    R = exp_SO3(phi)
    J = jac_left_SO3(phi)
    p = J @ v

    H = np.eye(4, dtype=float)
    H[:3, :3] = R
    H[:3, 3] = p
    return np.matrix(H)


def log_SE3(H):
    """
    Compute log in se(3) from H in SE(3).
    The rotation angle is returned within [-pi, pi] (via SO(3) log convention).
    H: 4x4 np.matrix in SE(3).
    Returns: 4x4 np.matrix in se(3).
    """
    H = np.asarray(H, dtype=float).reshape(4, 4)
    R = H[:3, :3]
    p = H[:3, 3]

    phi = log_SO3(R)                # angle in [-pi, pi] convention
    Jinv = inv_jac_left_SO3(phi)
    v = Jinv @ p
    w_hat = skew(phi)

    A = np.zeros((4, 4), dtype=float)
    A[:3, :3] = w_hat
    A[:3, 3] = v
    return np.matrix(A)


def vec(A):
    """
    A: 4x4 np.matrix in se(3).
    Returns: 6x1 np.matrix [a_x a_y a_z alpha_x alpha_y alpha_z]^T
             where 'a' is translational component, 'alpha' is angular.
    """
    A = np.asarray(A, dtype=float).reshape(4, 4)
    w = vee_so3(A[:3, :3])
    v = A[:3, 3].reshape(3)
    xi = np.zeros((6, 1), dtype=float)
    xi[0:3, 0] = v
    xi[3:6, 0] = w
    return np.matrix(xi)


def vec_inv(a):
    """
    Inverse of vec.
    a: 6x1 (or length-6) -> 4x4 np.matrix in se(3)
    """
    a = np.asarray(a, dtype=float).reshape(6)
    v = a[0:3]
    w = a[3:6]
    A = np.zeros((4, 4), dtype=float)
    A[:3, :3] = skew(w)
    A[:3, 3] = v
    return np.matrix(A)

def Dexp_num(A,E):
    #Implement analytical later!
    
    ds=0.001
    return (exp_SE3(A+E*ds)-exp_SE3(A-E*ds))/(2*ds)


        
def left_jac(A):
    phi = np.sqrt(A[0,1]**2+A[0,2]**2+A[1,2]**2)+1e-5
    phi2 = phi*phi
    phi3 = phi2*phi
    phi4 = phi3*phi
    phi5 = phi4*phi
    
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    phicphi = phi*cphi
    phisphi = phi*sphi
    
    vs = skew(A[0:3,-1])
    ws = A[0:3,0:3]
    
    A_hat = np.vstack((np.hstack((ws,vs)),np.hstack((np.zeros((3,3)),ws))))
    
    A0 = np.identity(6)
    A1 = A_hat
    A2 = A1*A_hat
    A3 = A2*A_hat
    A4 = A3*A_hat
    
    c1 = (4-phisphi-4*cphi)/(2*phi2)
    c2 = (4*phi-5*sphi+phicphi)/(2*phi3)
    c3 = (2-phisphi-2*cphi)/(2*phi4)
    c4 = (2*phi-3*sphi+phicphi)/(2*phi5)
    
    
    return A0+c1*A1+c2*A2+c3*A3+c4*A4
    

def Dexp(A,E):
    return vec_inv(left_jac(A)*vec(E))*exp_SE3(A)

def I_SE3(A):
    
    J = left_jac(A)
    # Hinv = exp_SE3(A)
    # H = inv_SE3(Hinv)
    
    H = exp_SE3(-A)
    
    Q = H[0:3,0:3]
    d = H[0:3,-1]
    
    a0 = Q*J[0:3,0]
    a1 = Q*J[0:3,1]
    a2 = Q*J[0:3,2]

    a3 = -ub.Utils.S(Q*J[3:6,3])*d+Q*J[0:3,3]
    a4 = -ub.Utils.S(Q*J[3:6,4])*d+Q*J[0:3,4]
    a5 = -ub.Utils.S(Q*J[3:6,5])*d+Q*J[0:3,5]
        
    
    # I = np.matrix(np.zeros((6,6)))

    # for i in range(6):
    #     I[:,i] = vec(H*vec_inv(J[:,i])*Hinv)
      
    zv = np.zeros((3,1))
    I = np.vstack((np.hstack((a0,a1,a2,a3,a4,a5)),np.hstack((zv,zv,zv,a0,a1,a2))))
      
    
    return I

def I_SE3_num(A):
    
    I = np.matrix(np.zeros((6,6)))
    for i in range(6):
        e = np.zeros((6,1))
        e[i]=1
        I[:,i] = vec(exp_SE3(-A)*Dexp_num(A,vec_inv(e)))

    return I

def L_SE3(A,E,mat=None):
    
    if not mat:
        I = I_SE3(-A)
        I1_inv = np.linalg.inv(I[0:3,0:3])
        I2 = I[0:3,3:6]
        B_Inv = I1_inv*I2*I1_inv
    else:
        I1_inv = mat[0]
        B_Inv = mat[1]
        
    xi = vec(E)
    
    return vec_inv(np.vstack((I1_inv*xi[0:3]-B_Inv*xi[3:6], I1_inv*xi[3:6]))), [I1_inv, B_Inv]
    
    
def min_seg_approx(H,C,E1,E2):
    
    A = log_SE3(inv_SE3(C)*H)
    U1, mat = L_SE3(A,E1)
    U2, _ = L_SE3(A,E2, mat)
    b0 = np.einsum('ij,ji->', A.T, A)
    b1 = -2*np.einsum('ij,ji->', A.T, U1)
    b2 = -2*np.einsum('ij,ji->', A.T, U2) + np.einsum('ij,ji->', U1.T, U1)
    b3 = 2*np.einsum('ij,ji->', U1.T, U2)
    b4 = np.einsum('ij,ji->', U2.T, U2)
    
    def eval_pol(s):
        return (((b4*s+b3)*s+b2)*s+b1)*s+b0
    
    c = np.array([4*b4, 3*b3, 2*b2, b1], dtype=np.complex128)
    roots = np.roots(c)
    roots = roots[np.abs(roots.imag) <= 1e-12].real
    roots = roots[roots<=1]
    roots = roots[roots>=0]
    
    pairs = [[0,eval_pol(0)]]+[[1,eval_pol(1)]]+[ [r,eval_pol(r)] for r in roots]
    
    return min(pairs, key=lambda x: x[1])
    


def create_SE3_pol(target_curve, K):


  C = [target_curve(k/K) for k in range(K+1)]
  A = [log_SE3(np.linalg.inv(C[k])*C[k+1]) for k in range(K) ]

  mat_left = np.matrix(np.zeros((6*K,6*K)))
  mat_right = np.matrix(np.zeros((6*K,1)))

  
  
  for k in range(K-1):
    mat_left[6*k:6*(k+1),6*k:6*(k+1)] = I_SE3(A[k])
    mat_left[6*k:6*(k+1),6*(k+1):6*(k+2)] = np.identity(6)
    mat_right[6*k:6*(k+1)] = 2*np.matrix(vec(A[k]))

  mat_left[6*(K-1):6*K,6*(K-1):6*K] = I_SE3(A[K-1])
  mat_left[6*(K-1):6*K,0:6] = np.identity(6)
  mat_right[6*(K-1):6*K] = 2*np.matrix(vec(A[K-1]))

  #############
  M = np.matrix(np.identity(6))
  for k in range(K):
      M = I_SE3(A[k])*M
  #############

  coef = np.linalg.inv(mat_left)*mat_right

  E1 = [vec_inv(coef[6*k:6*(k+1)]) for k in range(K)]
  E2 = [A[k]-E1[k] for k in range(K)]

  def _aux_fun(s):
    ind = int(min(np.floor(K*(s%1)),K-1))
    snorm = K*(s%1)-ind
    return C[ind]*exp_SE3(E1[ind]*snorm+E2[ind]*(snorm**2))


  return lambda s: _aux_fun(s), C, E1, E2

def rand_se3():
    cont = True
    while cont:
        a = np.matrix(np.random.randn(6, 1))
        cont = np.linalg.norm(a)>=np.pi
        
    return vec_inv(a)

def rand_SE3():
    return exp_SE3(rand_se3())


def estimate_lipschitz(path_fun):
    
    ds = 0.01
    L = 0
    for i in range(15000):
        H = rand_SE3()
        s = np.random.rand()
        
        Hp = H*path_fun(s+ds)
        Hn = H*path_fun(s-ds)
        
        use = np.trace(Hp)>0.05 and np.trace(Hn)>0.05
        
        if use:
            Dp = np.linalg.norm(log_SE3(Hp), 'fro')
            Dn = np.linalg.norm(log_SE3(Hn), 'fro')
            
            L = max(L, abs((Dp-Dn)/(2*ds)))
        
    return 1.25*L
        

def compute_arc_length(original_fun, s0, s1, ds=0.001):

    L = 0
    s=s0
    while s<s1:
        H = original_fun(s)
        H_inv=inv_SE3(H)
        dH = original_fun(s+ds)-H
        L+= np.linalg.norm(H_inv*dH, 'fro')
        s+=ds 
        
    return L 

def process_fun(original_fun, K):
    
    grouppol_fun , C, E1, E2 = create_SE3_pol(original_fun, K)
    lip_cons = estimate_lipschitz(grouppol_fun)
    length = [compute_arc_length(original_fun,i/K,(i+1)/K) for i in range(K)]
    
    return [grouppol_fun , C, E1, E2, lip_cons, length]
    
    
def fun_D_ori(H, data_fun, delta=1e-4):

    grouppol_fun = data_fun[0]
    lip_cons = data_fun[4] 
    fun_H = lambda s: np.linalg.norm(log_SE3(inv_SE3(H)*grouppol_fun(s)), 'fro')
    
    result = lipschitz_argmin(fun_H, lip_cons, delta)
    
    return result[0], fun_H(result[0])
    
    
def fun_D_our(H, data_fun, delta=1e-4, delta0 = None, mult=None):

    grouppol_fun = data_fun[0]
    C = data_fun[1]
    E1 = data_fun[2] 
    E2 = data_fun[3] 
    lip_cons = data_fun[4]
    K = len(E1)
    delta0p = 1/K if not delta0 else delta0
 
    fun_H = lambda s: np.linalg.norm(log_SE3(inv_SE3(H)*grouppol_fun(s)), 'fro')
           
    if delta0p<=1-1e-6:
        result = lipschitz_argmin(fun_H, lip_cons, delta0p)
        
        s_min = result[2]["final_interval"][0]
        s_max = result[2]["final_interval"][1]
        
        i_check = []
        
        for i in range(K):
            if (i+1)/K >= s_min and i/K <=s_max:
                i_check.append(i)
    else:
        i_check = [i for i in range(K)]
           
    s_best = -1
    f_best = 1000
    
    for i in i_check:
        pair = min_seg_approx(H,C[i],E1[i],E2[i])
        if pair[1]<f_best:
            f_best= pair[1]
            s_best = (i+pair[0])/K
           
    # mult_aux = 900/K if not mult else mult
    # s_min = s_best-mult_aux*delta
    # s_max = s_best+mult_aux*delta
    
    # result =  lipschitz_argmin(fun_H, lip_cons, delta, s_min, s_max)
    
    cont=True
    dfun_H = (fun_H(s_best+0.001)-fun_H(s_best-0.001))
    ds_base = 1e-2/lip_cons
    ds = -ds_base if dfun_H > 0 else ds_base
    
    i=0
    f_cur = fun_H(s_best)
    while cont:
        f_new  = fun_H(s_best+ds)
        if f_new < f_cur:
            s_best+=ds
            f_cur = f_new
        else:
            cont=False
            
    # return result[0] % 1, fun_H(result[0])
    return s_best % 1, f_cur
    
    
def vecfield_SE3(H, data_fun, KN=1, KT=1, alpha=1, beta=0.8):
    
    K = len(data_fun[2])
    
    s_star, D, = fun_D_our(H, data_fun, delta0=1, delta=1e-4)
    
    # s_ori, D_ori = fun_D_ori(H, data_fun, delta = 1e-5)
    
    # s_star = s_ori
    
    i_star = int(math.floor(K*s_star))
    H_star = data_fun[0](s_star)
    
    
    H_inv = inv_SE3(H)
    H_star_inv = inv_SE3(H_star)
    
    U = log_SE3(H_inv*H_star)
    I_U_inv = np.linalg.inv(I_SE3(U))
    
    N_vec = np.matrix(np.zeros((6,1)))
    
    for i in range(6):
        ei = np.matrix(np.zeros((6,1)))
        ei[i] = 1
        Ei = vec_inv(ei)
        dUi = vec_inv(I_U_inv*vec(H_star_inv*Ei*H_star))
        
        N_vec[i] = np.einsum('ij,ji->', U.T, dUi)
        
    ##Implement N_vec_num
    # N_vec_num = np.matrix(np.zeros((6,1)))
    
    # eps = 0.001
    # for i in range(6):
    #     ei = np.matrix(np.zeros((6,1)))
    #     ei[i] = 1
    #     Ei = vec_inv(ei)
    #     Dp = np.linalg.norm(log_SE3(H_inv*exp_SE3(-Ei*eps)*H_star),'fro')
    #     Dn = np.linalg.norm(log_SE3(H_inv*exp_SE3(Ei*eps)*H_star),'fro')
    #     N_vec_num[i] = (Dp-Dn)/(2*eps)
        
    # N_vec_num = N_vec_num/(1e-6+np.linalg.norm(N_vec_num))  
        
    ####################
        
    s_mod = K*s_star-i_star
    E1 = data_fun[2][i_star]
    E2 = data_fun[3][i_star]
    E = (E1+E2*s_mod)*s_mod
    dE = E1+2*E2*s_mod
    T_vec = vec(H_star*vec_inv(I_SE3(E)*vec(dE))*H_star_inv)
    
    # T_vec_num = vec( ((data_fun[0](s_star+0.001)-data_fun[0](s_star-0.001))/(2*0.001))*H_star_inv )
    # T_vec_num = T_vec_num/(np.linalg.norm(T_vec_num)+1e-6)
    
    N_vec_norm = N_vec/(1e-6+np.linalg.norm(N_vec))
    T_vec_norm = T_vec/(1e-6+np.linalg.norm(T_vec))
    
    cos_angle = N_vec_norm.T*T_vec_norm
    
    # if abs(cos_angle)>0.1:
    #     kku=0
    
    # f = lambda s: np.linalg.norm(log_SE3(H_inv*data_fun[0](s)),'fro')
    # dD = (f(s_ori+0.001)-f(s_ori-0.001))/(2*0.001)
    
    g = (2/np.pi)*np.arctan(alpha*(D**beta))
    cN = KN*g
    cT = KT*np.sqrt(1+1e-6-g*g)
    
    return cN*N_vec_norm+cT*T_vec_norm, D, cos_angle
    
    
    
        
    
    
    
    