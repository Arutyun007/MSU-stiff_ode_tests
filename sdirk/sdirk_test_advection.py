# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.integrate import ode
from matplotlib import pyplot as plt
from ode_sdirk import ode_sdirk as sdirk



class advection_first_order(object):
    def __init__(self, N):
        super(advection_first_order, self).__init__()
        self.N = N
        self.dh = 2.0/N;
        self.A = self.__build_matrix();



    def __build_matrix(self):
        A_impl = np.zeros((self.N, self.N))  # Iinitialize A_impl
        A_impl[np.diag_indices(self.N)] = 1./self.dh;
        A_impl[0, -1] = -1./self.dh
        A_impl[self.N - 1, 0] = 0

        k = -1

        rows, cols = np.indices(A_impl.shape)
        row_values = np.diag(rows, k=k)
        col_values = np.diag(cols, k=k)
        A_impl[row_values, col_values] = -1./self.dh
        return -A_impl

    def f(self, t, y, args):
        return np.dot(self.A, y);

    def jac(self, t, y, args):
        return self.A


def initial_condition(c, x):
    u0 = np.exp(-c * pow(x, 2))
    #plt.plot(x, u0)
    #plt.show()
    return u0

def exact_advection(c, x, Tf):
    # exact solution
    U_exact = np.zeros((len(x)))
    #print(x)
    x = x+1;
    x_ch = x - c * Tf
    #print(x_ch)
    # for i in range(len(x_ch)):
    #    if (x_ch[i] < 0):
    #       x_mod[i] = newMod(x_ch[i], 2 * math.pi)
    #      print(x_mod[i])
    # else:
    #    x_mod[i] = x_ch[i] % (2 * math.pi)

    x_mod = np.mod(x_ch, 2)
    # print(x_mod)
    for i in range(len(U_exact)):
        U_exact[i] = np.exp(-c*(x_mod[i]-1)**2)
    # plt.plot(x, U_exact)
    # plt.show()
    #myfunc = initial_condition(c, x_mod)
    return U_exact


t0 = 0
t1 = 5.0
dt = 0.1
N = 100
a = -1
b = 1
c_exp = 5.0


AD1 = advection_first_order(N)
x = np.linspace(a, b, N)  # x span

y0 = initial_condition(c_exp, x)
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


r_sdirk2 = sdirk(f = AD1.f, jac = AD1.jac, A = A2, c = c2, b = b2, b_hat = None, use_full_newton = 'sdirk')
r_sdirk3 = sdirk(f = AD1.f, jac = AD1.jac, A = A3, c = c3, b = b3, b_hat = None, use_full_newton = 'sdirk')
r_sdirk4 = sdirk(f = AD1.f, jac = AD1.jac, A = A4, c = c4, b = b4, b_hat = None, use_full_newton = 'sdirk')



r = ode(AD1.f).set_integrator('dop853', rtol = 1.0e-11, atol = 1.0e-12)
# r = ode(f).set_integrator('dopri5')
#r = ode(f, J).set_integrator('vode', method = 'bdf', with_jacobian = True)
#r = ode(f, J).set_integrator('vode', method = 'bdf', with_jacobian = True, min_step = dt, max_step = dt, first_step = dt, atol = 100.0, rtol = 100.0, order = 5)

param_value = 0;

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
while r_sdirk2.t <= t1:
    r_sdirk2.integrate(r_sdirk2.t+dt)
    all_t_sdirk2.append(r_sdirk2.t)
    all_y_sdirk2.append(r_sdirk2.y)

print('SDIRK3\n')
while r_sdirk3.t <= t1:
    r_sdirk3.integrate(r_sdirk3.t+dt)
    all_t_sdirk3.append(r_sdirk3.t)
    all_y_sdirk3.append(r_sdirk3.y)

print('SDIRK4\n')
while r_sdirk4.t <= t1:
    r_sdirk4.integrate(r_sdirk4.t+dt)
    all_t_sdirk4.append(r_sdirk4.t)
    all_y_sdirk4.append(r_sdirk4.y)

print('BDF\n')
while r.successful() and r.t <= t1:
    r.integrate(r.t+dt)
    all_t.append(r.t)
    all_y.append(r.y)

#plt.figure()
#plt.plot(all_t_sdirk,np.transpose(np.transpose(all_y_sdirk)[0]),'*')
#plt.plot(all_t,np.transpose(np.transpose(all_y)[0]))
#plt.show()

plt.figure()
plt.plot(x, exact_advection(c_exp, x, t1) )
plt.plot(x, r_sdirk2.y, '*')
plt.plot(x, r_sdirk3.y, '*')
plt.plot(x, r_sdirk4.y, '*')
plt.plot(x, r.y,'.')
plt.legend(['exact','sdirk2','sdirk3','sdirk4','python bdf'])
plt.show()


plt.figure()
plt.plot(np.transpose(np.transpose(all_y)[1]),np.transpose(np.transpose(all_y)[0]),'.')
plt.plot(np.transpose(np.transpose(all_y_sdirk2)[1]),np.transpose(np.transpose(all_y_sdirk2)[0]))
# plt.plot(np.transpose(np.transpose(all_y_sdirk3)[1]),np.transpose(np.transpose(all_y_sdirk3)[0]))
# plt.plot(np.transpose(np.transpose(all_y_sdirk4)[1]),np.transpose(np.transpose(all_y_sdirk4)[0]))
# plt.legend(['python native','sdirk2','sdirk3','sdirk4'])
plt.show()

# err_sdirk2 = np.array(all_y_sdirk2) - np.array(all_y)
# err_sdirk3 = np.array(all_y_sdirk3) - np.array(all_y)
# err_sdirk4 = np.array(all_y_sdirk4) - np.array(all_y)
# err_sdirk2_norm = np.sqrt(np.transpose(err_sdirk2)[0]**2+np.transpose(err_sdirk2)[1]**2)
# err_sdirk3_norm = np.sqrt(np.transpose(err_sdirk3)[0]**2+np.transpose(err_sdirk3)[1]**2)
# err_sdirk4_norm = np.sqrt(np.transpose(err_sdirk4)[0]**2+np.transpose(err_sdirk4)[1]**2)
# plt.figure()
# plt.plot(np.log(err_sdirk2_norm))
# plt.plot(np.log(err_sdirk3_norm))
# plt.plot(np.log(err_sdirk4_norm))
# plt.legend(['err sdirk2','err sdirk3','err sdirk4'])
# plt.show()
