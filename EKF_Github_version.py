# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 23:04:07 2023

@author: tarun
"""

# Github Verison
# Takes Initial Conditions as prior

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
R = np.diag(np.array([100]))
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

############## EKF ##############

def CD_EKF(t, x_k, z_RK4, P_k, w, v, G, const, x_RK4, rng):
    m = w[1].shape[0]
    n = v[1].shape[0]
    F = np.zeros((m, m))
    I = np.identity(m)
    temp_x = np.zeros((m, 2))
    err_bar = np.zeros((m, t.size))
    sig_bounds = const["sig_bounds"]
    last = 0
    for k in range(t.size - 1):
        # 1. Update
        H = zPartial(x_k[:, k]).reshape(n, m)
        K = ((P_k @ H.T) @ la.inv((H @ P_k) @ H.T + v[1])).reshape(m, 1)

        x_k[:, k] += (K @ (z_RK4[k] - zCalc(x_k[:, k], (0.0, np.zeros(v[1].shape)), const)))

        P_k = (I - (K @ H)) @ P_k
        err_bar[:, k] = KFError(P_k, sig_bounds)
        
        # 2. Propagation
        temp_x[:, 0] = x_k[:, k]
        temp_x, temp_z = RK4(t[k:k+2], temp_x, (0.0, np.zeros(w[1].shape)), (0.0, np.zeros(v[1].shape)), G, const, rng, True)
        x_k[:, k+1] = temp_x[:, -1]
        temp_x.fill(0)
        temp_x[:, 0] = x_k[:, k]
        for i in range(m):
            F[i, :] = xDotPartial(x_k[:, k], w, const, i)
        P_k = np.matmul(np.matmul(F,P_k),F.T) + ((G @ w[1]) @ G.T)
    
    # Final Update
    H = zPartial(x_k[:, -1]).reshape(n, m)
    K = ((P_k @ H.T) @ la.inv(H @ P_k @ H.T + R)).reshape(m, 1)
    x_k[:, -1] += (K @ (z_RK4[-1] - zCalc(x_k[:, -1], (0.0, np.zeros(v[1].shape)), const)))

    P_k = (I - (K @ H)) @ P_k
    err_bar[:, -1] = KFError(P_k, sig_bounds)
    return x_k, P_k, err_bar

############## Main ###############

x_RK4 = copy.deepcopy(x)
x_RK4, z_RK4 = RK4(t, x_RK4, w, v, G, const, rng, False)

x_EKF = copy.deepcopy(x)
P_EKF = copy.deepcopy(P_0)
x_EKF, P_EKF, err_EKF = CD_EKF(t, x_EKF, z_RK4, P_EKF, w, v, G, const, x_RK4, rng)
############# Plotting ###############

fig, x_graphs = plt.subplots(1, 3, sharex=True, figsize=(15,9))
for i in range(x_graphs.shape[0]):
    x_graphs[i].plot(t, x_RK4[i], c='r', label = 'Actual')
    x_graphs[i].plot(t, x_EKF[i], c='b', linestyle='--', label='Estimate')
    x_graphs[i].set_title("x%i" % i)
    x_graphs[i].legend(loc='best')
fig.suptitle("EKF Estimation vs. Time")
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('Time (s)')
plt.show()


zero = np.zeros((3, t.size))
fig, x_graphs = plt.subplots(1, 3, sharex=True, figsize=(15,9))
for i in range(x_graphs.shape[0]):
    x_graphs[i].plot(t, x_EKF[i] - x_RK4[i], c='r', label='EKF Error')
    x_graphs[i].errorbar(t, zero[i], yerr=err_EKF[i], fmt=' ')
    x_graphs[i].set_title("x%i" % i)
fig.suptitle('EKF Error with %.3f\u03C3 Error Bars vs. Time' % const["sig_bounds"])
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('Time (s)')
plt.show()