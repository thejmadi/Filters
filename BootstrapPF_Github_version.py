# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:46:11 2023

@author: tarun
"""

# Github version
# Takes initial conditions as posterior
# Has trouble with nonlinear model due to severe particle depletion

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

const = dict(g = 32.2, rho_0 = 3.4e-3, k_rho = 22e3, Upsilon = 1, sig_bounds = 3, h = time_step, num_PF = 100)
x_0 = 100000
x_dot_0 = 0
beta_0 = 2000

Q = np.diag(np.array([0, 2, 0])) * (1/time_step)
w = (0, Q)
R = np.diag(np.array([200]))
v = (0, R)
G = time_step * np.identity(3)
T = 1*np.identity((3))
P_0 = np.diag((np.array([500, 2e4, 2.5e5])))

x = np.zeros((3, t.size))
x[:, 0] = x_0, x_dot_0, beta_0
rng = np.random.default_rng(seed=2)

############# Msmt Functions #############

def zCalc(x, v, const):
    return x[0] + np.random.multivariate_normal([np.array(v[0])], v[1])
    #return (x[0]**2) + (x[1]**2) + np.random.multivariate_normal([np.array(v[0])], v[1])

def zPartial(x):
    return np.array([[1], [0], [0]])
    #return np.array([[2*x[0]], [2*x[1]], [0]])

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

############## Bootstrap Particle Filter ##############

def BPF(t, x_k, x_RK4, y_RK4, P_k, w, v, T, const, rng):
    N = const["num_PF"]
    sig_bounds = const["sig_bounds"]
    dim_state = len(w[1])
    dim_msmt = len(v[1])
    y_k = np.zeros(N)
    x_j = np.zeros((dim_state, N))
    y_j = np.zeros(N)
    W = np.ones(N) / N
    N_eff = np.zeros(t.size)
    N_eff[0] = N
    temp_x = np.zeros((dim_state, 2))
    x_err = np.zeros(x_j.shape)
    u_resampling = np.zeros(N)
    z_resampling = np.zeros(N)
    J = np.zeros((dim_state, dim_state))
    E_n = np.zeros((dim_state, dim_state))
    err_bar = np.zeros((3, t.size))
    err_bar[:, 0] = KFError(P_k, sig_bounds)
    ax = plt.figure().add_subplot(projection='3d')
    #ax2 = plt.figure().add_subplot(projection='3d')
    colors = ['r', 'g', 'b', 'k'] * int(np.ceil(t.size/4))
    
    for j in range(0, N):
        x_j[:, j] = np.random.multivariate_normal(x_k[:, 0], P_k)
        #u_resampling[j] = ((j + 1) - 1 + nu) / N
        
    
    for k in range(t.size - 1):
        # Propagation
        x_err.fill(0)
        z_resampling.fill(0)
        for j in range(N):            
            temp_x.fill(0)
            temp_x[:, 0] = x_j[:, j]
            temp_x, _ = RK4(t[k:k+2], temp_x, w, v, const, rng, True)
            x_j[:, j] = temp_x[:, -1]
            y_j[j] = zCalc(x_j[:, j], (v[0], np.zeros(v[1].shape)), const)

        # Update
        for j in range(N):
            W[j] = np.exp(-0.5 * (y_RK4[k+1] - y_j[j])**2 / v[1])
        W /= np.sum(W)
        N_eff[k+1] = 1 / np.sum(W **2)
        
        ax.scatter(x_j[0, :], t[k+1], W, zdir='z', c=colors[k], s=1)
        
        for j in range(N):
            x_k[:, k+1] += x_j[:, j] * W[j]
            y_k[k+1] += y_j[j] * W[j]
        P_k.fill(0)
        for j in range(N):
            x_err[:, j] = x_j[:, j] - x_k[:, k+1]
            P_k += W[j] * (x_err[:, j].reshape((dim_state, 1)) @ x_err[:, j].reshape((1, dim_state)))

        err_bar[:, k+1] = KFError(P_k, sig_bounds)
        
        # Resampling
        '''
        for j in range(N):
            z_resampling[j] = sum(W[:j+1])
            u_resampling[j] = ((j + 1) - 1 + rng.random()) / N
        i = 0
        for j in range(N):
            if u_resampling[j] < z_resampling[i]:
                x_j[:, j] = x_j[:, i]
                j +=1
            else:
                i +=1
        
        # Roughening  
        
        
        J.fill(0)
        E_n.fill(0)
        for n in range(dim_state):
            E_n[n, n] = abs(max(x_j[n, :]) - min(x_j[n, :]))
        for n in range(dim_state):
            J[n, n] = ((T[n,n] * E_n[n, n]) / (N**(1/3)))**2
        for j in range(N):
            x_j[:, j] += np.random.multivariate_normal(np.zeros(3), J)
        
        ax2.scatter(x_j[0, :], t[k], W, zdir='z', c=colors[k], s=1)
        
        W = np.ones(N) / N
        '''
        
    ax.set_title("Particle Weight Distribution for x0")
    #ax2.set_title("After Roughening")
    ax.set_xlabel('x0 (ft)')
    ax.set_ylabel('Time (s)')
    ax.set_zlabel('Weight')
    ax.view_init(elev=15., azim=60)
    #ax2.view_init(elev=15., azim=60)
    plt.show
    
    return x_k, P_k, err_bar, N_eff

############## Main ###############

x_RK4 = copy.deepcopy(x)
x_RK4, z_RK4 = RK4(t, x_RK4, w, v, const, rng, False)

x_BPF = copy.deepcopy(x)
P_BPF = copy.deepcopy(P_0)
t_start = time.default_timer()
x_BPF, P_BPf, err_BPF, N_eff = BPF(t, x_BPF, x_RK4, z_RK4, P_BPF, w, v, T, const, rng)
t_end = time.default_timer()
print((t_end - t_start) , ' seconds')

############# Plotting ###############


fig, x_graphs = plt.subplots(1, 3, sharex=True, figsize=(15,9))
for i in range(x_graphs.shape[0]):
    x_graphs[i].plot(t, x_RK4[i], c='r', label='Actual')
    x_graphs[i].plot(t, x_BPF[i], c='b', linestyle='--', label='Estimate')
    x_graphs[i].set_title("x%i" % i)
    x_graphs[i].legend(loc='best')
fig.suptitle("PF Estimation vs. Time")
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('Time (s)')
plt.show()

zero = np.zeros((3, t.size))
fig, x_graphs = plt.subplots(1, 3, sharex=True, figsize=(15,9))
for i in range(x_graphs.shape[0]):
    x_graphs[i].plot(t, x_BPF[i] - x_RK4[i], c='r', label='UKF Error')
    x_graphs[i].errorbar(t, zero[i], yerr=err_BPF[i], fmt=' ')
    x_graphs[i].set_title("x%i" % i)
fig.suptitle('PF Error with %.3f\u03C3 Error Bars vs. Time' % const["sig_bounds"])
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('Time (s)')
plt.show()



plt.plot(t, N_eff)
plt.title("Effective Sample Size vs. Time")
plt.xlabel("Time (s)")
plt.ylabel("Effective Sample Size")
plt.show()