import numpy as np
import PullData as ed
from scipy import linalg
from scipy.linalg import lstsq

def get_polymax(H, freq,order_model,delta_t) :
    M = np.zeros((len(H[0]) * (order_model + 1), len(H[0]) * (order_model +1)),dtype=object)
    for l in range(len(H)) :
        X_l = np.zeros((len(freq), order_model+1), dtype=complex)
        Y_l = []
        for n in range(len(freq)) :
            H_line_freq = H[l , : ,n]
            for m in range(order_model + 1) :
                weigt_fct = 1
                X_l[n][m] = weigt_fct * (np.exp(1j * 2 * np.pi * freq[n] * m * delta_t))
            Y_l.append(np.kron( - X_l[n], H_line_freq))
        Y_l  = np.array(Y_l)
        Xl_H = np.conjugate(X_l).T
        Yl_H = np.conjugate(Y_l).T
        R_l  = np.real(Xl_H @ X_l)
        S_l  = np.real(Xl_H @ Y_l)
        T_l  = np.real(Yl_H @ Y_l)
        M   += (T_l - S_l.T @ np.linalg.pinv(R_l) @ S_l) 
    
    M *= 2
    M_11, M_12    = decomposition_M(M, H, order_model)
    alpha         = np.linalg.solve(M_11, -M_12)
    C             = get_C(alpha, order_model)
    eigenvals, x  = linalg.eig(C)
    eigenvals     = np.log(eigenvals) / delta_t

    w_i           = np.sqrt(np.real(eigenvals)**2 + np.imag(eigenvals)**2)
    damping_i     =  - np.real(eigenvals) / w_i
    arg_sorted    = np.argsort(w_i)
    eigenvals     = eigenvals[arg_sorted]
    w_i           = w_i[arg_sorted]
    damping_i     = damping_i[arg_sorted]
    return w_i, damping_i, eigenvals

def decomposition_M(M, H, p) :
    m = len(H[0])

    M_11 = M[:m*p, :m*p]  
    M_12 = M[:m*p, m*p:]  
    return np.array(M_11, dtype=float), np.array(M_12, dtype=float)

def get_C(alpha, order_model) :
    m = len(alpha[0]) 
    zero_upper    = np.zeros(((order_model - 1) * m, m))
    identity_uper = np.eye((order_model - 1) * m)
    upper_part    = np.hstack([zero_upper, identity_uper])
    lower_part    = np.zeros((m , m * order_model))
    for p in range(order_model) :
        alpha_p = alpha[m * p: m * (p + 1),:]
        lower_part[:, p * m:(p + 1) * m] = - alpha_p.T
    C = np.vstack([upper_part, lower_part])
    return C


def get_stabilisation(dic_order):
    # Convertir les clés en une liste triée pour garantir l'ordre
    sorted_keys = sorted(dic_order.keys())
    for i in range(len(sorted_keys) - 1): 
        key = sorted_keys[i]
        next_key = sorted_keys[i + 1]
        
        dic_n = dic_order[key]
        dic_n_next = dic_order[next_key]
        
        for j, w in enumerate(dic_n_next["wn"]):
            w_next   = np.array(dic_n["wn"])
            tol_low  = (1 - 0.01) * w
            tol_high = (1 + 0.01) * w
            idx_w    = np.where((w_next >= tol_low) & (w_next <= tol_high))[0] 
            if len(idx_w) == 0:
                continue
            else:
                damping_n    = np.array(dic_n["zeta"])[idx_w]
                damping_next = dic_n_next["zeta"][j]
                tol_low  = (1 - 0.05) * damping_next
                tol_high = (1 + 0.05) * damping_next
                idx_damp = np.where((damping_n >= tol_low) & (damping_n <= tol_high))[0]
                if len(idx_damp) == 0:
                    dic_n_next["stable"][j] = "v"
                else :
                    dic_n_next["stable"][j] = "d"

        dic_order[next_key] = dic_n_next  
    
    return dic_order

                
def compute_lsfd(lambdak, f, H):
    f[0] = 10**-6    # like sart to 0, ... 
    ni = H.shape[0]  
    no = H.shape[1]  
    n  = H.shape[2]   
    nmodes = lambdak.shape[0]  
    omega = 2 * np.pi * f 

    # Extract like p_k and q_k
    b = 1 / np.subtract.outer(1j * omega, lambdak).T
    c = 1 / np.subtract.outer(1j * omega, np.conj(lambdak)).T

    hr = H.real
    hi = H.imag
    br = b.real
    bi = b.imag
    cr = c.real
    ci = c.imag

    hri = np.dstack((hr, hi))
    bri = np.hstack((br+cr, bi+ci))
    cri = np.hstack((-bi+ci, br-cr))

    Kr = np.ones(n)
    ur_zeros      = np.zeros(n)
    Mr = -1/(omega**2)

    Krr = np.hstack((Kr, ur_zeros))
    Kri = np.hstack((ur_zeros, Kr))
    Mrr = np.hstack((Mr, ur_zeros))
    Mri = np.hstack((ur_zeros, Mr))

    A = np.vstack((bri, cri, Krr, Kri, Mrr, Mri))

    hri = hri.reshape(ni*no, 2*n)

    uv = lstsq(A.T, hri.T)[0]
    
    uv = uv.T.reshape(ni, no, 2*nmodes+4)

    u   = uv[:, :, :nmodes]
    v   = uv[:, :, nmodes:-4]

    a  = u + 1j*v       

    return a

def extract_eigenmode(A) :
    """
    Eingenmode is extract using the residue method
    """
    n, m ,q = A.shape # m represente the accelrometer n the number of shock and q the mode
    mode    = np.zeros((q, n), dtype=complex)
    for i in range(q):
        mode_s    = np.sqrt(A[0,0,i])
        mode[i, 0] = mode_s
        for j in range(1,n):
            mode[i, j] = A[j,0,i]/mode_s

    return mode
