# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 23:14:22 2023

@author: tarun
"""

# Github version
# Takes Initial conditions as Prior

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import copy
import timeit as time
plt.rcParams['figure.dpi'] = 300


time_step = .1                                                                   # Model time step
time_start = 0                                                                  # Model start time, inclusive
time_end = 5                                                                   # Model end time, inclusive
t = np.arange(time_start, time_end + time_step, time_step)                    # Vector of model time steps

const = dict(g = 32.2, rho_0 = 3.4e-3, k_rho = 22e3, num_ensemble = 100, Upsilon = 1, sig_bounds = 3, h = time_step)
x_0 = np.random.normal(10e4, np.sqrt(500))
x_dot_0 = np.random.normal(-6000, np.sqrt(2e4))
beta_0 = np.random.normal(2000, np.sqrt(2.5e5))

Q = np.diag(np.array([0, 2, 0])) * (1/time_step)
w = (0, Q)
R = np.diag(np.array([200])) #* (1/time_step)
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

############# RK4 Function ##############

def RK4(t, x, w, v, const, rng, est):
    # Necessary const parameters
    n = t.size
    t_step = t[1] - t[0]
    num_x = int(x[:, 0].size) # Num of state variables
    
    # Matrices for RK4 calc
    K = np.zeros((num_x, 4))
    
    # Measurement matrices
    z = np.zeros(n)
    
    # Add noise to initial timestep measurements
    z[0] = zCalc(x[:, 0], v, const)
    
    for i in range(1, n):
        for k in range(num_x):
            K[k, 0] = t_step * xDot(t[i-1], x[:, i-1], w, const, k)

            K[k, 1] = t_step * xDot(t[i-1] + t_step/2, x[:, i-1] + K[:, 0]/2, w, const, k)
            
            K[k, 2] = t_step * xDot(t[i-1] + t_step/2, x[:, i-1] + K[:, 1]/2, w, const, k)
            
            K[k, 3] = t_step * xDot(t[i-1] + t_step, x[:, i-1] + K[:, 2], w, const, k)

            x[k][i] = x[k][i - 1] + (K[k, 0] + 2 * (K[k, 1] + K[k, 2]) + K[k, 3]) / 6

        rand = np.random.multivariate_normal(np.zeros((num_x,)), w[1])
        x[:, i] += G @ rand

        z[i] = zCalc(x[:, i], v, const)
        K.fill(0)
    return x, z

################################

def KFError(KF_cov, sig_bounds):
    sig_cov = sig_bounds * np.sqrt(KF_cov.diagonal()).T
    return sig_cov

############## EnKF ###############

def EnKF(t, x_k, y_RK4, P_k, w, v, G, const, rng):
    # Initialize
    # Covariances
    # Gain
    # Update
    # Propagation
    Upsilon = const["Upsilon"]
    N = const["num_ensemble"]
    sig_bounds = const["sig_bounds"]
    dim_state = len(w[1])
    dim_msmt = len(v[1])
    y_k = np.zeros(x_k.shape[1])
    x_j = np.zeros((dim_state, N))
    y_j = np.zeros(N)
    temp_x = np.zeros((dim_state, 2))
    w_j = (w[0], np.zeros((dim_state, N)))
    v_j = (v[0], np.zeros((dim_msmt, N)))
    err_bar = np.zeros((3, t.size))
    for j in range(0, N):
        x_j[:, j] = np.random.multivariate_normal(x_k[:, 0], P_k)
    for k in range(0, t.size-1):
        P_k_exey = 0
        P_k_eyey = 0
        P_k.fill(0)
        K_k = 0
        for j in range(0, N):
            y_j[j] = zCalc(x_j[:, j], (0.0, np.zeros(v[1].shape)), const)
            w_j[1][:, j] = np.random.multivariate_normal(np.zeros(dim_state), w[1])
            v_j[1][:, j] = np.random.multivariate_normal(np.zeros(dim_msmt), v[1])
        x_k[:, k] = np.sum(x_j, axis=1) / N
        y_k[k] = np.sum(y_j) / N
        for j in range(0, N):
            P_k_exey += (x_j[:, j] - x_k[:, k]).reshape(dim_state, 1) @ (y_j[j] - y_k[k]).reshape(1, dim_msmt)
            P_k_eyey += (y_j[j] - y_k[k]).reshape(dim_msmt, 1) @ (y_j[j] - y_k[k]).reshape(1, dim_msmt)
            P_k += (x_j[:, j] - x_k[:, k]).reshape(dim_state, 1) @ (x_j[:, j] - x_k[:, k]).reshape(1, dim_state)
        P_k_exey /= (N - 1)
        P_k_eyey /= (N - 1)
        P_k /=  (N - 1)
        err_bar[:, k] = KFError(P_k, sig_bounds)
        K_k = P_k_exey / P_k_eyey
        x_j += K_k @ (y_RK4[k] + v_j[1][:] - y_j)
        for j in range(0, N):            
            temp_x.fill(0)
            temp_x[:, 0] = x_j[:, j]
            temp_x, _ = RK4(t[k:k+2], temp_x, (0.0, np.zeros(w[1].shape)), (0.0, np.zeros(v[1].shape)), const, rng, False)#(v_j[0], v_j[1][j]), (w_j[0], w[1][j]), const)

            x_j[:, j] = temp_x[:, -1] + Upsilon * w_j[1][:, j]
    P_k_exey = 0
    P_k_eyey = 0
    P_k.fill(0)
    K_k = 0
    for j in range(0, N):
        y_j[j] = zCalc(x_j[:, j], (0.0, np.zeros(v[1].shape)), const)
        w_j[1][:, j] = np.random.multivariate_normal(np.zeros(dim_state), w[1])
        v_j[1][:, j] = np.random.multivariate_normal(np.zeros(dim_msmt), v[1])
    x_k[:, -1] = np.sum(x_j, axis=1) / N
    y_k[-1] = np.sum(y_j) / N
    for j in range(0, N):
        P_k_exey += (x_j[:, j] - x_k[:, -1]).reshape(dim_state, 1) @ (y_j[j] - y_k[-1]).reshape(1, dim_msmt)
        P_k_eyey += (y_j[j] - y_k[-1]).reshape(dim_msmt, 1) @ (y_j[j] - y_k[-1]).reshape(1, dim_msmt)
        P_k += (x_j[:, j] - x_k[:, -1]).reshape(dim_state, 1) @ (x_j[:, j] - x_k[:, -1]).reshape(1, dim_state)
    P_k_exey /= (N - 1)
    P_k_eyey /= (N - 1)
    P_k /=  (N-1)
    err_bar[:, -1] = KFError(P_k, sig_bounds)  
    K_k = P_k_exey / P_k_eyey
    x_j += K_k @ (y_RK4[k] + v_j[1][:] - y_j)
    return x_k, P_k, err_bar

############## Main ###############

x_RK4 = copy.deepcopy(x)
x_RK4, z_RK4 = RK4(t, x_RK4, w, v, const, rng, False)


x_EnKF = copy.deepcopy(x)
P_EnKF = copy.deepcopy(P_0)
x_EnKF, P_Eknf, err_EnKF = EnKF(t, x_EnKF, z_RK4, P_EnKF, w, v, G, const, rng)
############# Plotting ###############

fig, x_graphs = plt.subplots(1, 3, sharex=True, figsize=(15,9))
for i in range(x_graphs.shape[0]):
    x_graphs[i].plot(t, x_RK4[i], c='r', label='Actual')
    x_graphs[i].plot(t, x_EnKF[i], c='b', linestyle='--', label = 'Estimate')
    x_graphs[i].set_title("x%i" % i)
    x_graphs[i].legend(loc='best')
fig.suptitle("EnKF Estimation vs. Time")
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('Time (s)')
plt.show()

zero = np.zeros((3, t.size))
fig, x_graphs = plt.subplots(1, 3, sharex=True, figsize=(15,9))
for i in range(x_graphs.shape[0]):
    x_graphs[i].plot(t, x_EnKF[i] - x_RK4[i], c='r', label='EnKF Error')
    x_graphs[i].errorbar(t, zero[i], yerr=err_EnKF[i], fmt=' ')
    x_graphs[i].set_title("x%i" % i)
fig.suptitle('EKF/UKF Error with %.3f\u03C3 Error Bars vs. Time' % const["sig_bounds"])
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('Time (s)')
plt.show()
