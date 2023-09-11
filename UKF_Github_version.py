# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 23:09:48 2023

@author: tarun
"""

# Github Version
# Takes Initial Conditions as Posterior

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import copy
plt.rcParams['figure.dpi'] = 300

time_step = .1                                                                   # Model time step
time_start = 0                                                                  # Model start time, inclusive
time_end = 5                                                                   # Model end time, inclusive
n = int((time_end - time_start) / time_step) + 1
t = np.linspace(time_start, time_end, n)                                        # Vector of model time steps

const = dict(g = 32.2, rho_0 = 3.4e-3, k_rho = 22e3, alpha = 0.5, UKF_beta = 2, sig_bounds = 3, h = time_step)
x_0 = np.random.normal(10e4, np.sqrt(500))
x_dot_0 = np.random.normal(-6000, np.sqrt(2e4))
beta_0 = np.random.normal(2000, np.sqrt(2.5e5))

Q = np.diag(np.array([0, 2, 0])) * (1/time_step)
w = (0, Q)
R = np.diag(np.array([200]))
v = (0, R)
G = time_step * np.identity(3)
P_0 = np.diag((np.array([500, 2e4, 2.5e5])))

x = np.zeros((3, t.size))
x[:, 0] = x_0, x_dot_0, beta_0
rng = np.random.default_rng(seed=2)

############# Msmt Functions #############

def zCalc(x, v, const):
    return (x[0]**2) + (x[1]**2) + np.random.multivariate_normal([np.array(v[0])], v[1])

def zPartial(x):
    return np.array([[2*x[0]], [2*x[1]], [0]])

############# Equations of Motion ############

def xDot(t, x, w, const, axis):
    if axis == 0:
        return x[1]
    elif axis == 1:
        rho = const["rho_0"] * np.exp(-x[0] / const["k_rho"])
        x2 = x[2]
        
        if x[2] == 0.0:
            x2 = 10e-15
        
        d = 0.5 * rho * (x[1] ** 2) / x2
        x2_dot = d - const['g']
        return x2_dot
    elif axis == 2:
        return 0
    
def xDotPartial(x, w, const, axis):
    if axis == 0:
        return np.array([1, const["h"], 0])
    elif axis == 1:
        rho_0 = const["rho_0"]
        k_rho = const["k_rho"]
        h = const["h"]
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        rho = rho_0 * np.exp(-x1 / k_rho)

        
        if x3 == 0.0:
            x3 = 10e-15
        wrt_x1 = - 0.5 * (x2**2) * rho * h / (x3*k_rho)
        wrt_x2 = 1 + x2 * rho * h / x3
        wrt_x3 = - 0.5 * (x2**2) * h * rho / (x3**2)
        return np.array([wrt_x1, wrt_x2, wrt_x3])
    elif axis == 2:
        return np.array([0, 0, 1])

############# RK4 Function ##############

def RK4(t, x, w, v, G, const, rng, est):
    # Necessary const parameters
    n = t.size
    t_step = t[1] - t[0]
    num_x = x.shape[0] # Num of state variables
    
    # Matrices for RK4 calc
    K = np.zeros((num_x, 4))
    
    # Measurement matrices
    z = np.zeros(n)
    # Add noise to initial timestep measurements
    z[0] = zCalc(x[:, 0], v, const)
    K = np.zeros((num_x, 4))
    for i in range(1, n):
        for k in range(num_x):
            K[k, 0] = t_step * xDot(t[i-1], x[:, i-1], w, const, k)
            
            K[k, 1] = t_step * xDot(t[i-1] + t_step/2, x[:, i-1] + K[:, 0]/2, w, const, k)
            
            K[k, 2] = t_step * xDot(t[i-1] + t_step/2, x[:, i-1] + K[:, 1]/2, w, const, k)
            
            K[k, 3] = t_step * xDot(t[i-1] + t_step, x[:, i-1] + K[:, 2], w, const, k)
                
            #x[k][i] = x[k][i - 1] + (L[0] + 2 * (L[1] + L[2]) + L[3]) / 6
            x[k, i] = x[k, i - 1] + (K[k, 0] + 2 * (K[k, 1] + K[k, 2]) + K[k, 3]) / 6
        rand = np.random.multivariate_normal(np.zeros((3,)), w[1])
        x[:, i] += G @ rand

        z[i] = zCalc(x[:, i], v, const)
        K.fill(0)
        
    return x, z

##############

def KFError(KF_cov, sig_bounds):
    sig_cov = sig_bounds * np.sqrt(KF_cov.diagonal()).T
    return sig_cov

############## UKF ##############

def UKF(t, x_k, z_RK4, P_k, w, v, G, const, rng):
    # No need to use augmented state method since correlations b/w state error,
    # process noise and measurement noise are assumed to be 0
    L = int(len(x_k)) 
    a = const["alpha"]
    b = const["UKF_beta"]
    sig_bounds = const["sig_bounds"]
    I = np.identity(L)
    # If kap is (-) may lead to non-positive semi-definite predicted covariance
    kap = 3 - L
    lam = (a**2) * (L + kap) - (L)
    gam = np.sqrt(L + lam)
    xi_k = np.zeros((L, 2*L + 1))
    xi_k_next = np.zeros((L, 2*L + 1))
    temp_xi = np.zeros((L, 2))
    
    # Sigma Point Weight Calc, Only need to calc 1 time for entire t
    
    W_mean_diag = np.zeros(2*L + 1)
    W_mean_diag[0] = lam / (L + lam)
    W_mean_diag[1:] = 0.5 / (L + lam)
    W_cov_diag = np.zeros(2*L + 1)
    W_cov_diag[0] = 1 - a**2 + b + (lam / (L + lam))
    W_cov_diag[1:] = 0.5 / (L + lam)

    h_k_next = np.zeros(2*L + 1)
    e_k = np.zeros(1)
    
    err_bar = np.zeros((L, t.size))
    err_bar[:, 0] = KFError(P_k, sig_bounds)
    
    for k in range(0, t.size-1):
        y_k_next = 0
        xi_k.fill(0)
        xi_k_next.fill(0)
        # P_k_yy = P_k_eyey, added for theory's sake
        P_k_yy = 0
        P_k_eyey = 0
        P_k_exey = 0
        e_k.fill(0)
        sig = la.cholesky(P_k) * gam
        
        xi_k[:, 0] = x_k[:, k]
        for i in range(1, L + 1):
            xi_k[:, i] = x_k[:, k] + sig[:,i-1]
            xi_k[:, L+i] = x_k[:, k] - sig[:,i-1]

        # Propagation Step
        for i in range(0, 2*L + 1):
            temp_xi.fill(0)
            temp_xi[:, 0] = xi_k[:, i]
            temp_xi, _ = RK4(t[k:k+2], temp_xi, (0.0, np.zeros(w[1].shape)), (0.0, np.zeros(v[1].shape)), G, const, rng, True)
            xi_k_next[:, i] = temp_xi[:, -1]
        for i in range(0, 2*L + 1):
            x_k[:, k + 1] += W_mean_diag[i] * xi_k_next[:, i]
        
        for i in range(0, 2*L + 1):
            h_k_next[i] = zCalc(xi_k_next[:, i], (0.0, np.zeros(v[1].shape)), const)
            #y_k += W_mean[i, i] * h_k[i]
            y_k_next += W_mean_diag[i] * h_k_next[i]

        P_k.fill(0)
        # Calc Covariances_k, Correlations_k, P_k-
        for i in range(0, 2*L + 1):
            P_k += W_cov_diag[i] * ((xi_k_next[:, i] - x_k[:, k+1]).reshape((L,1)) @ (xi_k_next[:, i] - x_k[:, k + 1]).reshape((1, L)))
            P_k_yy += W_cov_diag[i] * ((h_k_next[i] - y_k_next) * (h_k_next[i] - y_k_next))
            P_k_exey += W_cov_diag[i] * ((xi_k_next[:, i] - x_k[:, k + 1]) * (h_k_next[i] - y_k_next))
            
        P_k += ((G @ w[1]) @ G.T)
        P_k_yy += v[1]
        P_k_eyey = P_k_yy
        # Calc Gain_k
        K_k = (P_k_exey / P_k_eyey).reshape((L,1))
        e_k[:] = z_RK4[k+1] - y_k_next
    
        
        # Update Step
        x_k[:, k+1] += (K_k * e_k).reshape((L,))
        P_k -= (K_k * P_k_eyey) @ K_k.T
        err_bar[:, k+1] = KFError(P_k, sig_bounds)

    return x_k, P_k, err_bar

############## Main ###############

x_RK4 = copy.deepcopy(x)
x_RK4, z_RK4 = RK4(t, x_RK4, w, v, G, const, rng, False)

x_UKF = copy.deepcopy(x)
P_UKF = copy.deepcopy(P_0)
x_UKF, P_UKF, err_UKF = UKF(t, x_UKF, z_RK4, P_UKF, w, v, G, const, rng)
############# Plotting ###############

fig, x_graphs = plt.subplots(1, 3, sharex=True, figsize=(15,9))
for i in range(x_graphs.shape[0]):
    x_graphs[i].plot(t, x_RK4[i], c='r', label='Actual')
    x_graphs[i].plot(t, x_UKF[i], c='b', linestyle='--', label='Estimate')
    x_graphs[i].set_title("x%i" % i)
    x_graphs[i].legend(loc='best')
fig.suptitle("UKF Estimation vs. Time")
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('Time (s)')
plt.show()

zero = np.zeros((3, t.size))
fig, x_graphs = plt.subplots(1, 3, sharex=True, figsize=(15,9))
for i in range(x_graphs.shape[0]):
    x_graphs[i].plot(t, x_UKF[i] - x_RK4[i], c='r', label='UKF Error')
    x_graphs[i].errorbar(t, zero[i], yerr=err_UKF[i], fmt=' ')
    x_graphs[i].set_title("x%i" % i)
fig.suptitle('UKF Error with %.3f\u03C3 Error Bars vs. Time' % const["sig_bounds"])
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('Time (s)')
plt.show()