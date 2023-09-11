# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 14:18:34 2023

@author: tarun
"""

# Github Version
# Projectile Problem


import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import timeit as time
plt.rcParams['figure.dpi'] = 300

############ Param Inputs ############

time_step = 1                                                                   # Model time step
time_start = 0                                                                  # Model start time, inclusive
time_end = 65                                                                   # Model end time, inclusive
x_0_input = np.array([[-100], [500], [100], [10], [10], [350]])                 # Pos, Vel Initial Value

msmt_stddev = np.array([np.radians(0.5), np.radians(0.5)])                      # Standard Deviation of Noise
num_of_msmt_param = msmt_stddev.size                                            # Azimuth and Elevation
tol = 1e-20                                                                     # Tolerance required to consider NLS as converged
sig_bounds_EKF = 3
x_initial_est = np.array([[-110], [450], [200], [9], [4.5], [290]])             # Initial guess of state used for NLS

############ Param Calcs #############

t = np.arange(time_start, time_end + time_step, time_step)                      # Vector of model time steps
x_next = np.zeros((x_0_input.size, t.size))
x_next[:, 0] = x_0_input.T                                                      # Create List of Position values to be filled
msmt_var = msmt_stddev ** 2                                                     # Variance of Noise, assumes stddev for each msmt param are same
R = np.identity(num_of_msmt_param)
for i in range(0, num_of_msmt_param):
    R[i] *= msmt_var[i]                                                         # Measurement Covariance R
    
############ Azimuth, Elevation Functions ############

def Az(x):                                                                      # Calc azimuth given position
    azimuth = np.arctan2(x[1], x[0])
    if np.sign(azimuth) == -1: azimuth += 2*np.pi                               # Converts azimuth from (-pi,pi) to (0,2pi)
    return azimuth

def El(x):                                                                      # Calc elevation given position
    r_mag = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    return np.arcsin(x[2] / r_mag)

def PartialAz(x):                                                               # Calc partial derivatives of Az with respect to r_x, r_y, r_z, v_x, v_y, v_z
    r_x = x[0]
    r_y = x[1]
    # Derivatives wrt to v are all 0
    if r_x == 0.0:
        r_x = 10E-6
    return np.array([(-r_y/r_x**2) / (1 + (r_y/r_x)**2), (1/r_x) / (1 + (r_y/r_x)**2), 0.0, 0.0, 0.0, 0.0])


def PartialEl(x):                                                               # Calc partial derivatives of El with respect to r_x, r_y, r_z, v_x, v_y, v_z
    r_x = x[0]
    r_y = x[1]
    r_z = x[2]
    r_mag = np.sqrt(r_x**2 + r_y**2 + r_z**2)
    # Derivatives wrt v are all 0
    return np.array([-r_x*r_z / ((r_mag**2) * np.sqrt(r_x**2 + r_y**2)), -r_y*r_z / ((r_mag**2) * np.sqrt(r_x**2 + r_y**2)), np.sqrt(r_x**2 + r_y**2) / (r_mag**2), 0.0, 0.0, 0.0])

############ Noise Functions ############

def Noise(x, var):
    az = Az(x)                                                                  # Azimuth of given pos data
    el = El(x)                                                                  # Elevation of given pos data
    h = np.array([az, el])                                                      # Nominal Azimuth, Elevation in radians
    # True Az, El in radians
    y = np.array([sum(i) for i in zip(h, list(np.random.normal(0, np.sqrt(var[i]), len(var))))])
    return h, y    

############ RK4 Eqs ############

def RDot(t, r, v, axis):
    # Returns vel of same axis
    return v[axis]

def VDot(t, r, v, axis):
    if axis == 0:
        return 0
    if axis == 1:
        return 0
    if axis == 2:
        return -9.81
    
def RDotPartial(t, r, v, axis, h):
    if axis == 0:
        return np.array([1, 0, 0, h, 0, 0])
    if axis == 1:
        return np.array([0, 1, 0, 0, h, 0])
    if axis == 2:
        return np.array([0, 0, 1, 0, 0, h])

def VDotPartial(t, r, v, axis, h):
    if axis == 0:
        return np.array([0, 0, 0, 1, 0, 0])
    if axis == 1:
        return np.array([0, 0, 0, 0, 1, 0])
    if axis == 2:
        return np.array([0, 0, 0, 0, 0, 1])

def RK4(t, x, var):
    # Necessary const parameters
    n = t.size
    dim_msmt = var.size
    t_step = t[1] - t[0]
    dim_x = int(x[:, 0].size / 2) # Dim of axes. Usually 3
    
    # Matrices for RK4 calc
    K = np.zeros(4)
    L = np.zeros(4)
    
    # Measurement matrices
    h = np.zeros((dim_msmt, n))
    y = np.zeros((dim_msmt, n))
    
    # Add noise to initial timestep measurements
    h_temp, y_temp = Noise(x[:, 0], var) # Add noise to initial position msmts
    h[:, 0] = h_temp
    y[:, 0] = y_temp
    
    for i in range(1, n):
        for k in range(dim_x):
            K[0] = t_step * VDot(t[i-1], x[:dim_x, i-1], x[dim_x:, i-1], k)
            L[0] = t_step * RDot(t[i-1], x[:dim_x, i-1], x[dim_x:, i-1], k)

            K[1] = t_step * VDot(t[i-1] + t_step/2, x[:dim_x, i-1] + L[0]/2, x[dim_x:, i-1] + K[0]/2, k)
            L[1] = t_step * RDot(t[i-1] + t_step/2, x[:dim_x, i-1] + L[0]/2, x[dim_x:, i-1] + K[0]/2, k)
            
            K[2] = t_step * VDot(t[i-1] + t_step/2, x[:dim_x, i-1] + L[1]/2, x[dim_x:, i-1] + K[1]/2, k)
            L[2] = t_step * RDot(t[i-1] + t_step/2, x[:dim_x, i-1] + L[1]/2, x[dim_x:, i-1] + K[1]/2, k)

            K[3] = t_step * VDot(t[i-1] + t_step, x[:dim_x, i-1] + L[2], x[dim_x:, i-1] + K[2], k)
            L[3] = t_step * RDot(t[i-1] + t_step, x[:dim_x, i-1] + L[2], x[dim_x:, i-1] + K[2], k)
            
            x[k][i] = x[k][i - 1] + (L[0] + 2 * (L[1] + L[2]) + L[3]) / 6
            x[k + 3][i] = x[k + 3][i - 1] + (K[0] + 2 * (K[1] + K[2]) + K[3]) / 6
        
        # Uses Noise() to change position vals to measurement vals, adds noise to y_temp
        h_temp, y_temp = Noise(x[:, i], var)
            
        h[:, i] = h_temp
        y[:, i] = y_temp
        K.fill(0)
        L.fill(0)
    return x, h, y

############ Nonlinear Least Squares Functions ############

def NLS(t, y, x_initial, var, tol):
    dim_msmt = var.size # Num of measurement angles
    t_step = t[1] - t[0]
    n = t.size # Num of timesteps
    n2 = n * dim_msmt # Total number of measurements taken
    m = int(x_initial[:, 0].size / 2) # Dim of pos and vel vectors. Usually 3
    m2 = 2*m # Dim of state vector
    I = np.identity(m) # Identity Matrix (mxm)
    I2 = np.identity(m2) # Identity Matrix (m2xm2)
    A = np.array([[np.zeros((m,m)), I], [np.zeros((m,m)), np.zeros((m,m))]])
    A = np.concatenate(np.concatenate(A,axis=1),axis=1) # gives 6x6 matrix
    H = np.zeros((dim_msmt, m2)) # H(2nx2m) = partial(Az, El) / partial(x(t))
    J = np.zeros((n2, m2)) # J = partial(H) / partial(x(0))
    W = np.identity(n2)
    for i in range(0, n2):
        W[i] *= var[i % 2]
    W = la.pinv(W)
    diff1 = tol * 2
    diff2 = tol * 2
    x_RK4 = np.zeros((m2, n))
    x_RK4[:, 0] = x_initial.T
    i = 1 # Num of iterations taken for NLS to converge
    while(diff1 >= tol and diff2 >= tol):
        # f(x_RK4) calculation using RK4 into Azimuth and Elevation functions, keeps positions and true measurements

        x_i, h_i, _ = RK4(t, x_RK4, var)
        
        # Remove bias from the true measurement values
        y_i = (y - h_i).reshape((n2,1), order='F')
        
        for k in range(0, n2, dim_msmt):
            k2 = int(k/dim_msmt) # k/2
            # Next 3 lines calc partial(Az, El) / partial(x(t))
            # H = partial(Az, El) / partial(x(t))
            H[0] = PartialAz(x_i[:, k2])
            H[1] = PartialEl(x_i[:, k2])
            # Calc partial(Az, El) / partial(x(0)) = H * phi_k
            # phi_k = partial(x(k)) / partial(x(0))
            phi_k = la.matrix_power(I2+A*t_step, k2)
            J[k:k+2] = H @ phi_k
        
        del_x = (la.pinv(J.T @ J) @ J.T) @ y_i
        diff1 = la.norm(x_initial[:3])
        diff2 = la.norm(x_initial[3:])
        
        # Calc x_k+1
        x_initial += del_x
        
        # Copy x into temporary variable
        x_RK4 = np.zeros((m2, n))
        x_RK4[:, 0] = x_initial.T
        
        # Calc value to exit while loop
        diff1 = abs(diff1 - la.norm(x_initial[:3]))
        diff2 = abs(diff2 - la.norm(x_initial[3:]))
        i += 1
    P_i1 = la.pinv(J.T @ W @ J)
    return x_initial, P_i1

############ Ext. Kalman Filter ############

def CD_EKF_Error(EKF_cov, sig_bounds):
    sig_cov = sig_bounds * np.sqrt(EKF_cov.diagonal()).T
    return sig_cov


def CD_EKF(t, y_RK4, NLS_x, NLS_cov, R, err, x_RK4, sig_bounds):
    n = t.size # Num timesteps
    h = t[1] - t[0]
    m = int(len(NLS_x) / 2) # Dim of each state var eg. 3 pos and 3 vel
    m2 = 2*m                # Total number of state var eg. 6 pos and vel
    dim_msmt = y_RK4.shape[0]
    # 1. Initialization Step
    x_k = np.zeros((m2, n))
    x_k[:, 0] = NLS_x.T
    P_k = NLS_cov
    h_k = np.zeros((2,))
    H = np.zeros((2, m2))
    I = np.identity(m2)
    temp_x = np.zeros((m2, 2))
    F = np.zeros((m2, m2))
    err_bar = np.zeros((m2, n))
    
    err_bar[:, 0] = CD_EKF_Error(P_k, sig_bounds)
    
    for k in range(0, t.size-1):
        # 2. Propagation
        temp_x[:, 0] = x_k[:, k]
        # Calling RK4
        temp_x, temp_h, _ = RK4(t[k:k+2], temp_x, err)
        # Setting k+1 state prediction
        x_k[:, k+1] = np.array(temp_x)[:, -1]
        # Calc x_k+ = x_k- +  K_k(y_k - h_k-)
        h_k = temp_h[:, -1]
        temp_x.fill(0)
        for i in range(m):
            F[i, :] = RDotPartial(t, x_k[:3, k], x_k[3:, k], i, h)
            F[i + m] = VDotPartial(t, x_k[:3, k], x_k[3:, k], i, h) 
        P_k = np.matmul(np.matmul(F,P_k),F.T)
        H[0] = PartialAz(x_k[:, k])
        H[1] = PartialEl(x_k[:, k])
        
        #K_k = P_k @ H.T @ la.pinv((H @ P_k) @ H.T + (R**2))
        K_k = P_k @ H.T @ la.pinv((H @ P_k) @ H.T + (R))
        # 3. Update
        # Calc x_k-
        # Resetting variables for RK4
        
        x_k[:, k+1] += (K_k @ (y_RK4[:, k+1] - h_k)).reshape((m2,))
        
        # Calc P_k+ = [I - K_k @ H_k]P_k-
        P_k = (I - (K_k @ H)) @ P_k
        err_bar[:, k+1] = CD_EKF_Error(P_k, sig_bounds)
        h_k.fill(0)
    return x_k, P_k, err_bar

############ Main ############
x_nom_RK4, h_RK4, y_RK4 = RK4(t, x_next, msmt_var)

msmt_err_RK4 = h_RK4 - y_RK4

x_initial_NLS, cov_NLS = NLS(t, y_RK4, x_initial_est, msmt_var, tol)
print("The estimation of the initial state is: ")
print(x_initial_NLS)
print()
print("The NLS covariance matrix is positive definite: ", all( i > 0 for i in la.eigvals(cov_NLS)))
print()

x_true_EKF, cov_EKF, err_EKF = CD_EKF(t, y_RK4, x_initial_NLS, cov_NLS, R, msmt_var, x_nom_RK4, sig_bounds_EKF)
x_EKF_err = x_true_EKF - x_nom_RK4
cov_eigvals_EKF = la.eigvals(cov_EKF)
print("EKF Covariance Matrix: ")
print(cov_EKF)
print()
print("EKF Covariance Eigenvalues: ")
print(cov_eigvals_EKF)
print()
print("The EKF covariance matrix is positive definite: ", all( i > 0 for i in cov_eigvals_EKF))

########### Plotting ###########

fig, RK4_EKF_graphs = plt.subplots(1, 2, sharex=True, figsize=(15,9))
RK4_EKF_graphs[0].plot(t, x_nom_RK4[0], c = 'b', label='b1')
RK4_EKF_graphs[0].plot(t, x_true_EKF[0], c = 'r', label='b1', linestyle='--')
RK4_EKF_graphs[0].plot(t, x_nom_RK4[1], c = 'g', label='b2')
RK4_EKF_graphs[0].plot(t, x_true_EKF[1], c = 'r', label='b2', linestyle='--')
RK4_EKF_graphs[0].plot(t, x_nom_RK4[2], c = 'k', label='b3')
RK4_EKF_graphs[0].plot(t, x_true_EKF[2], c = 'r', label='b3', linestyle='--')
RK4_EKF_graphs[1].plot(t, x_nom_RK4[3], c = 'b', label='b1')
RK4_EKF_graphs[1].plot(t, x_true_EKF[3], c = 'r', label='b1', linestyle='--')
RK4_EKF_graphs[1].plot(t, x_nom_RK4[4], c = 'g', label='b2')
RK4_EKF_graphs[1].plot(t, x_true_EKF[4], c = 'r', label='b2', linestyle='--')
RK4_EKF_graphs[1].plot(t, x_nom_RK4[5], c = 'k', label='b3')
RK4_EKF_graphs[1].plot(t, x_true_EKF[5], c = 'r', label='b3', linestyle='--')
RK4_EKF_graphs[0].set_title("Position (m)")
RK4_EKF_graphs[1].set_title("Velocity (m/s)")
RK4_EKF_graphs[0].legend(loc='best')
RK4_EKF_graphs[1].legend(loc='best')
plt.suptitle("EKF Estimation vs Time")
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('Time (s)')
plt.show()

plt.plot(t, np.degrees(h_RK4[0]), c = 'r', label='Nominal Azimuth')
plt.plot(t, np.degrees(h_RK4[1]), c = 'r', label='Nominal Elevation')
plt.plot(t, np.degrees(y_RK4[0]), c = 'b', label='True Azimuth', linestyle='--')
plt.plot(t, np.degrees(y_RK4[1]), c = 'g', label='True Elevation', linestyle='--')
plt.title('Nominal and True Azimuth, Elevation Angles vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('Angles (deg)')
plt.legend(loc='best')
plt.show()

fig, EKF_err_graphs = plt.subplots(2, 3, sharex=True, figsize=(15,9))
x_EKF_err = x_EKF_err.reshape(EKF_err_graphs.shape[0], EKF_err_graphs.shape[1], t.size)
err_EKF = err_EKF.reshape(EKF_err_graphs.shape[0], EKF_err_graphs.shape[1], t.size)
c = 0
for k in range(EKF_err_graphs.shape[0]):
    for i in range(EKF_err_graphs.shape[1]):
        EKF_err_graphs[k][i].plot(t, x_nom_RK4[c], c = 'b', label='Actual')
        EKF_err_graphs[k][i].plot(t, x_true_EKF[c], c = 'r', linestyle='--', label='Estimate')
        EKF_err_graphs[k][i].legend(loc='best')
        c += 1
EKF_err_graphs[0][0].set_title('Position b1')
EKF_err_graphs[0][1].set_title('Position b2')
EKF_err_graphs[0][2].set_title('Position b3')
EKF_err_graphs[1][0].set_title('Velocity b1')
EKF_err_graphs[1][1].set_title('Velocity b2')
EKF_err_graphs[1][2].set_title('Velocity b3')
fig.suptitle('EKF Error vs. Time')
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('Time (s)')
plt.ylabel('EKF Error')
plt.show()

fig, EKF_err_graphs = plt.subplots(2, 3, sharex=True, figsize=(15,9))
zero = np.zeros((2, 3, t.size))
for k in range(EKF_err_graphs.shape[0]):
    for i in range(EKF_err_graphs.shape[1]):
        EKF_err_graphs[k][i].plot(t, x_EKF_err[k][i], c = 'r')
        EKF_err_graphs[k][i].errorbar(t, zero[k][i], yerr=err_EKF[k][i], fmt=' ')
EKF_err_graphs[0][0].set_title('Position b1')
EKF_err_graphs[0][1].set_title('Position b2')
EKF_err_graphs[0][2].set_title('Position b3')
EKF_err_graphs[1][0].set_title('Velocity b1')
EKF_err_graphs[1][1].set_title('Velocity b2')
EKF_err_graphs[1][2].set_title('Velocity b3')
fig.suptitle('EKF Error with %.3f\u03C3 Error Bars vs. Time' % sig_bounds_EKF)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('Time (s)')
plt.ylabel('EKF Error')
plt.show()
