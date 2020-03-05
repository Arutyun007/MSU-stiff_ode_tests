# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.integrate import ode
from matplotlib import pyplot as plt
from ode_sdirk import ode_sdirk as sdirk


y0, t0 = [-2, 0], 0
#y0, t0 = 1.0, 0

def f(t, y, arg1):
#    return [arg1*y[0]]
    return [y[1],arg1*(1-y[0]**2)*y[1]-y[0]]
#    return [arg1*(y[1]-y[0]**3/3.0 + y[0]),-y[0]]
    
def J(t, y, arg1):
#    return [arg1]
    return [[0.0, 1.0],[arg1*(-2.0*y[0])*y[1], arg1*(1.0-y[0]**2)]]
#    return [[arg1*(-y[0]**2+1), arg1],[-1.0, 0.0]]

    
param_value = 8.0
t1 = 100.0
dt = 0.05

""" SDIRK Butcher tables 
    *2 - 2-d order A and L-stable
    *3 - 4-th order A-stable
    *4 - 4-th order A and L-stable
"""
# 2
A2 = [[1.0/4.0, 0],[1.0/2.0, 1.0/4.0]]
b2 = [1.0/2.0, 1.0/2.0]
c2 = [1.0/4.0, 3.0/4.0]
# 4
A4 = [[1/4, 0, 0, 0,0],[1/2,1/4,0,0,0],[17/50, -1/25, 1/4, 0, 0],[371/1360, -137/2720, 15/544, 1/4, 0],[25/24, -49/48, 125/16, -85/12, 1/4]]
b4 = [25/24, -49/48, 125/16, -85/12, 1/4]
c4 = [1/4, 3/4, 11/20, 1/2, 1]
# 3
gamma = 1/np.sqrt(3)*np.cos(np.pi/18)+1/2
delta = 1/(6*(2*gamma-1)**2)
A3 = [[gamma,0,0],[1/2-gamma,gamma,0],[2*gamma,1-4*gamma,gamma]]
b3 = [delta, 1-2*delta, delta]
c3 = [gamma, 1/2, 1-gamma]


r_sdirk2 = sdirk(f = f, jac = J, A = A2, c = c2, b = b2, b_hat = None, use_full_newton = False)
r_sdirk3 = sdirk(f = f, jac = J, A = A3, c = c3, b = b3, b_hat = None, use_full_newton = False)
r_sdirk4 = sdirk(f = f, jac = J, A = A4, c = c4, b = b4, b_hat = None, use_full_newton = False)



#r = ode(f).set_integrator('dop853')
#r = ode(f).set_integrator('dopri5')
#r = ode(f, J).set_integrator('vode', method = 'adams', with_jacobian = True)
r = ode(f, J).set_integrator('vode', method = 'bdf', with_jacobian = True, min_step = dt, max_step = dt, first_step = dt, atol = 100.0, rtol = 100.0, order = 2)

r.set_initial_value(y0, t0).set_f_params(param_value).set_jac_params(param_value)
r_sdirk2.set_initial_value(y0, t0).set_f_params(param_value).set_jac_params(param_value)
r_sdirk3.set_initial_value(y0, t0).set_f_params(param_value).set_jac_params(param_value)
r_sdirk4.set_initial_value(y0, t0).set_f_params(param_value).set_jac_params(param_value)


all_t = []
all_y = []
all_t_sdirk2 = []
all_y_sdirk2 = []
all_t_sdirk3 = []
all_y_sdirk3 = []
all_t_sdirk4 = []
all_y_sdirk4 = []

print('SDIRK2\n')
while r_sdirk2.t < t1:
    r_sdirk2.integrate(r_sdirk2.t+dt)
    all_t_sdirk2.append(r_sdirk2.t)
    all_y_sdirk2.append(r_sdirk2.y)


print('SDIRK3\n')
while r_sdirk3.t < t1:
    r_sdirk3.integrate(r_sdirk3.t+dt)
    all_t_sdirk3.append(r_sdirk3.t)
    all_y_sdirk3.append(r_sdirk3.y)

print('SDIRK4\n')
while r_sdirk4.t < t1:
    r_sdirk4.integrate(r_sdirk4.t+dt)
    all_t_sdirk4.append(r_sdirk4.t)
    all_y_sdirk4.append(r_sdirk4.y)

print('BDF\n')
while r.successful() and r.t < t1:
    r.integrate(r.t+dt)
    all_t.append(r.t)
    all_y.append(r.y)

#plt.figure()
#plt.plot(all_t_sdirk,np.transpose(np.transpose(all_y_sdirk)[0]),'*')
#plt.plot(all_t,np.transpose(np.transpose(all_y)[0]))
#plt.show()

plt.figure()
plt.plot(np.transpose(np.transpose(all_y)[1]),np.transpose(np.transpose(all_y)[0]),'.')
plt.plot(np.transpose(np.transpose(all_y_sdirk2)[1]),np.transpose(np.transpose(all_y_sdirk2)[0]))
plt.plot(np.transpose(np.transpose(all_y_sdirk3)[1]),np.transpose(np.transpose(all_y_sdirk3)[0]))
plt.plot(np.transpose(np.transpose(all_y_sdirk4)[1]),np.transpose(np.transpose(all_y_sdirk4)[0]))
plt.legend(['python native','sdirk2','sdirk3','sdirk4'])
plt.show()