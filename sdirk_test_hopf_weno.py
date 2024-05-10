# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.integrate import ode
from matplotlib import pyplot as plt
from ode_sdirk import ode_sdirk as sdirk
from ode_irk import ode_irk as irk
from spatial_discretization import weno_interpolation
from riemann_solvers import riemann_solver
import B_stability_check as stabcheck
import time

class advection(object):
    def __init__(self, N, solver, order):
        super(advection, self).__init__()
        self.N = N
        self.dh = 2.0/N;
        self.a = 1.0;
        self.flux = lambda u: self.a*u
        self.flux_derivative = lambda u: self.a
        self.weno = weno_interpolation(order)
        self.flux = riemann_solver(solver = solver, flux = self.flux, flux_derivative = self.flux_derivative)
        self.splitting = False

    def set_order(self, order):
        self.weno = weno_interpolation(order)
    
    def set_flux_solver(self, solver, flux, flux_derivative):
        self.flux = riemann_solver(solver = solver, flux = flux, flux_derivative = flux_derivative)

    def set_flux_splitting(self, splitting):
        self.splitting = splitting

    def f(self, t, y, args):
        if self.splitting:
            FL, FR = self.flux.apply_split(y)
            fl, fr = self.weno.apply_periodic_split(FL, FR)

        else:
            Lw, Rw, Le, Re = self.weno.apply_periodic(y)
            fl, fr = self.flux.apply_side(Lw, Rw, Le, Re)

        ret = -(fr-fl)/self.dh
        return ret


    def jac(self, t, y, args):
        local_eps = 1.0e-7
        N = len(y)
        M = np.zeros((N,N))
        for j in range(0,N):
            e = np.zeros(len(y))
            e[j] = local_eps
            yp = y+e
            ym = y-e
            M[j] = 0.5*(self.f(t, yp, args) - self.f(t, ym, args) )/local_eps 
        
        return np.transpose(M)



class exact_solution(object):
    def __init__(self, solution_type):
        super(exact_solution, self).__init__()
        self.solution_type = solution_type
        self.canonic = self.canonic()

    class canonic:
        """implements a canonical advection test problem on -1<=x<=1"""
        def __init__(self):
            self.__a=0.5
            self.__z=-0.7
            self.__delta=0.005
            self.__alpha=10
            self.__beta=np.log(2.0)/(36.0*self.__delta**2)
            self.__G = lambda x,b,r: np.exp(-b*(x-r)**2)
            self.__F = lambda x,d,r: np.sqrt(np.maximum(1.0-d**2.0*(x-r)**2,0))

        def initial_conditions(self, x):
            x1 = x*((x>=-0.8) & (x<=-0.6))
            x2 = x*((x>=-0.4) & (x<=-0.2))
            x3 = x*((x>= 0.0) & (x<= 0.2))
            x4 = x*((x>= 0.4) & (x<= 0.6))
            a = self.__a
            z = self.__z
            delta = self.__delta
            alpha = self.__alpha
            beta = self.__beta

            u1 = 1/6.0*( self.__G(x1,beta,z-delta) + self.__G(x1,beta,z+delta) + 4*self.__G(x1,beta,z)) 
            u2 =  1.0*(x2 != 0)
            u3 = 1.0-np.abs(10.0*(x3-0.1))
            u4 = 1/6.0*(self.__F(x4,alpha,a-delta) + self.__F(x4,alpha,a+delta) + 4.0*self.__F(x4,alpha,a))
            u0 = u1 + u2 + u3 + u4
            return u0
        
        def advect_solution(self, x, Tf):
            x = x+1;
            x_ch = x - Tf
            x_mod = np.mod(x_ch, 2)
            return self.initial_conditions(x_mod-1)


    def __exact_advection_exp(self, c, x, Tf):
        # exact solution exponent
        x = x+1;
        x_ch = x - Tf
        x_mod = np.mod(x_ch, 2)
        U_exact = np.exp(-c*(x_mod-1)**2)

        return U_exact
    def __exact_advection_step(self,c, x, Tf):
        # exact solution step
        U_exact = np.zeros((len(x)))
        x = x+1;
        x_ch = x - Tf
        x_mod = np.mod(x_ch, 2)
        for i in range(len(U_exact)):
            if x_mod[i]-1 >0:
                U_exact[i] = 1
        return U_exact

    def __initial_condition_exp(self, c, x):
        u0 = np.exp(-c * pow(x, 2))
        return u0


    def __initial_condition_step(self, c, x):
        mask = x>0.0;
        u0 = mask*1.0;
        return u0;

    def advect(self, c, x, final_time):
        if self.solution_type == "exponent":
            return self.__exact_advection_exp(c, x, final_time)
        elif self.solution_type == "step":
            return self.__exact_advection_step(c, x, final_time)
        elif self.solution_type == "canonic":
            return self.canonic.advect_solution(x, final_time)
        else:
            return x*0

    def init(self, c, x):
        if self.solution_type == "exponent":
            return self.__initial_condition_exp(c, x)
        elif self.solution_type == "step":
            return self.__initial_condition_step(c, x)
        elif self.solution_type == "canonic":
            return self.canonic.initial_conditions(x)            
        else:
            return x*0        

class hopf(object):
    def __init__(self, N, solver, order):
        super(hopf, self).__init__()
        self.N = N
        #self.dh = 2.0 / N;
        self.dh = 2.0 * np.pi / N;
        self.a = 1.0;
        #self.flux = lambda u: self.a * u
        #self.flux_derivative = lambda u: self.a  #??????????????????????
        self.flux = lambda u: (u * u * 0.5)
        self.flux_derivative = lambda u: u  # ??????????????????????
        self.weno = weno_interpolation(order)
        self.flux = riemann_solver(solver=solver, flux=self.flux, flux_derivative=self.flux_derivative)
        self.splitting = False

    def set_order(self, order):
        self.weno = weno_interpolation(order)

    def set_flux_solver(self, solver, flux, flux_derivative):
        self.flux = riemann_solver(solver=solver, flux=flux, flux_derivative=flux_derivative)

    def set_flux_splitting(self, splitting):
        self.splitting = splitting

    def f(self, t, y, args):
        if self.splitting:
            FL, FR = self.flux.apply_split(y)
            fl, fr = self.weno.apply_periodic_split(FL, FR)

        else:
            Lw, Rw, Le, Re = self.weno.apply_periodic(y)
            fl, fr = self.flux.apply_side(Lw, Rw, Le, Re)

        ret = -(fr - fl) / self.dh
        return ret

    def jac(self, t, y, args):
        local_eps = 1.0e-7
        N = len(y)
        M = np.zeros((N, N))
        for j in range(0, N):
            e = np.zeros(len(y0))
            e[j] = local_eps
            yp = y0 + e
            ym = y0 - e
            M[j] = 0.5 * (AD1.f(t, yp, args) - AD1.f(t, ym, args)) / local_eps
            # M[j] = (AD1.f(t, yp, args) - AD1.f(t, y0, args) )/local_eps
            # M[j] = (AD1.f(t, y0, args) - AD1.f(t, ym, args) )/local_eps

        return np.transpose(M)


class exact_solution_hopf(object):
    def __init__(self, solution_type):
        #super(exact_solution, self).__init__()
        self.solution_type = solution_type
        #self.canonic = self.canonic()

    def __initial_condition_sin(self, c, x):
        u0 = (np.sin(x)) ** 9
        return u0

    def hopf(self, c, x, final_time):
        if self.solution_type == "sinus":
            return self.__exact_hopf_sinus(c, x, final_time)
        #if self.solution_type == "exponent":
        #    return self.__exact_advection_exp(c, x, final_time)

        #elif self.solution_type == "step":
        #    return self.__exact_advection_step(c, x, final_time)
        #elif self.solution_type == "canonic":
        #    return self.canonic.advect_solution(x, final_time)
        #else:
        #    return x * 0

    def init(self, c, x):
        if self.solution_type == "sinus":
            return self.__initial_condition_sin(c, x)
        """
        if self.solution_type == "exponent":
            return self.__initial_condition_exp(c, x)
        elif self.solution_type == "step":
            return self.__initial_condition_step(c, x)
        elif self.solution_type == "canonic":
            return self.canonic.initial_conditions(x)
        else:
            return x * 0
        """

    def __exact_hopf_sinus(self, c, x, Tf):  # need to change!!!!!!!!!
        u_exact = np.zeros(len(x))
        x_t = np.zeros(len(x))
        x_t = x + self.__initial_condition_sin(1, x) * Tf
        u_exact = self.__initial_condition_sin(1, x)

        return u_exact

def trapz_method(u, h, Nx):
    res = 0
    for i in range(1, Nx):
        res = res + ((u[i - 1] + u[i]) / 2) * h
    # print(res)
    return res

def L2_error(u0, u, Nx, h):
    res = 0
    # print(u0)
    u0_u_dif = np.power(u0 - u, 2)
    res = np.sqrt(trapz_method(u0_u_dif, h, Nx))
    return res

def L2_error_relative(u0, u, Nx, h):
    res = 0
    # print(u0)
    u0_u_dif = np.power(u0 - u, 2)
    res = np.sqrt(trapz_method(u0_u_dif, h, Nx)) / np.sqrt(trapz_method(np.power(u0, 2), h, Nx))
    return res

def read_from_file(path, files, axis):
    evstigneevnnm_data = []
    for i in range(len(files)):
        evstigneevnnm_data.append([])
    for i in range(len(files)):
        f = open(path + '\\' + files[i], 'r')
        for line in f:
            columns = line.split()
            if axis == "x":
                evstigneevnnm_data[i].append(float(columns[0]))
            elif axis == "y":
                evstigneevnnm_data[i].append(float(columns[1]))
    return evstigneevnnm_data



path_to_evstigneevnm_data = "C:\\Users\\amartirosyan\\Documents\\msu\\Diplom\\Evstigneev data"
#files = ["U_weno_7_4_0p534.dat", "U_weno_7_4_1p0.dat", "U_weno_7_4_1p5.dat", "U_weno_7_4_2p0.dat",
#             "U_weno_7_4_2p5.dat", "U_weno_7_4_3p0.dat", "U_weno_7_4_3p5.dat", "U_weno_7_4_4p0.dat"]
#files = ["U_weno_7_4_1p5.dat"]
files = ["U_weno_7_4_3p5.dat"]
#             , "U_weno_7_4_3p0.dat", "U_weno_7_4_3p5.dat", "U_weno_7_4_4p0.dat"]



evstigneevnnm_data_x = read_from_file(path_to_evstigneevnm_data, files, "x")
evstigneevnnm_data_u = read_from_file(path_to_evstigneevnm_data, files, "y")
t0 = 0.0
#t1_arr = [2187/4096, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
#t1_arr = [1.5]
t1_arr = [3.5]
start_time = 0.0
step = 0
CFL_arr = [0.5, 1.0, 4.0, 6.0, 8.0, 10.0]
#CFL_arr = [0.5]
#CFL_arr = [4.0, 6.0]
N = 200
dh = 2.0 * np.pi/N
#dt = CFL * dh
a = 0
b = 2*np.pi
c_exp = 1.0
orders = [1,3,5]
implicit_method_arr = ["IE", "IM", "CN", "L2C", "L3A", "G2P4", "G3P6", "L3B", "L3AS4", "SDIRK1BDF", "SDIRK4", "SDIRK3", "SDIRK2", "L3C", "R2A", "R2AS"]
#implicit_method_arr = ["G2P4", "G3P6", "L3B", "L3AS4", "SDIRK1BDF", "SDIRK4", "SDIRK3", "SDIRK2", "L3C", "R2A", "R2AS"]


#implicit_method_arr = ["SDIRK4"]
#implicit_method_arr = ["SDIRK3", "SDIRK2", "L3C", "R2A", "R2AS"]#"IE", "IM", "CN", "L2C", "L3A", "G2P4", "G3P6", "L3B", "L3AS4", "SDIRK1BDF", "SDIRK4", "SDIRK3", "SDIRK2", "L3C", "R2A", "R2AS"
newton_jacobian = "freeze" #"full"  "sdirk" "freeze"
riemann_solver_type = "LF" #"upwind"# "LF"
exact_solution_type = "sinus" #"exponent"
equation = "hopf"  # "advection", "Hopf(inviscid Burgers)"
print("equation = ", equation)

counter = 0
for t1 in t1_arr:

    for implicit_method in implicit_method_arr:
        print("using implicit_method = ", implicit_method)
        for CFL in CFL_arr:
            dt = CFL * dh
            print("using CFL = ", CFL, " dt = ", dt)
            if equation == "advection":
                AD1 = advection(N, riemann_solver_type, 5)
                x = np.linspace(a, b, N)  # x span
                ES = exact_solution(exact_solution_type)
                y0 = ES.init(c_exp, x)
            elif equation == "hopf":
                AD1 = hopf(N, riemann_solver_type, 5)
                x = np.linspace(a, b, N)  # x span
                ES = exact_solution_hopf(exact_solution_type)
                y0 = ES.init(c_exp, x)
            """ SDIRK Butcher tables 
                *1 - 1-s order Implicit Euler A and L-stable
                *2 - 2-d order A and L-stable
                *3 - 4-th order A-stable
                *4 - 4-th order A and L-stable
            """
            # 1 +
            A1 = [[1.0]]
            b1 = [1.0]
            c1 = [1.0]
            #IM +
            A1_IM = [[1/2]]
            b1_IM = [1.0]
            c1_IM = [1/2]
            # CN +
            A2_CN = [[0, 0],[1.0/2.0, 1.0/2.0]]
            b2_CN = [1.0/2.0, 1.0/2.0]
            c2_CN = [0, 1.0]

            # G2P4 +
            A2_G2P4 = [[1/4, (3-2*np.sqrt(3))/12],[(3+2*np.sqrt(3))/12, 1.0/4.0]]
            b2_G2P4 = [1.0/2.0, 1.0/2.0]
            c2_G2P4 = [(3-np.sqrt(3))/6, (3+np.sqrt(3))/6]

            # 2 Lobatto IIIC B-stable method +
            A2_L2C = [[1/2, -1/2],[1.0/2.0, 1.0/2.0]]
            b2_L2C = [1.0/2.0, 1.0/2.0]
            c2_L2C = [0, 1.0]
            # 2+
            A2 = [[1.0/4.0, 0],[1.0/2.0, 1.0/4.0]]
            b2 = [1.0/2.0, 1.0/2.0]
            c2 = [1.0/4.0, 3.0/4.0]
            # 3+
            gamma = 1.0/np.sqrt(3)*np.cos(np.pi/18.0)+1/2.0
            delta = 1.0/(6.0*(2.0*gamma-1.0)**2)
            A3 = [[gamma,0,0],[1/2-gamma,gamma,0],[2*gamma,1-4*gamma,gamma]]
            b3 = [delta, 1-2*delta, delta]
            c3 = [gamma, 1/2, 1-gamma]
            # 4
            A4 = [[1/4, 0, 0, 0, 0],[1/2,1/4,0,0,0],[17/50, -1/25, 1/4, 0, 0],[371/1360, -137/2720, 15/544, 1/4, 0],[25/24, -49/48, 125/16, -85/12, 1/4]]
            b4 = [25/24, -49/48, 125/16, -85/12, 1/4]
            c4 = [1/4, 3/4, 11/20, 1/2, 1]

            #SDIRK1BDF
            A3_SDIRK1BDF = [[1, 0, 0],[1, 1, 0],[1/2, -1/2, 1]]
            b3_SDIRK1BDF = [1/2, -1/2, 1]
            c3_SDIRK1BDF = [1, 2, 1]

            # China paper Lobatto IIIA Ehle Method: s = 3, p = 4,
            A4_L3A = [[0, 0, 0],[5/24, 1/3, -1/24],[1/6, 2/3, 1/6]]
            b4_L3A = [1/6, 2/3, 1/6]
            c4_L3A = [0, 1/2, 1]

            #Gauss: s = 3, p = 6,
            A3_G3P6 = [[5/36, 2/9-np.sqrt(15.0)/15, 5/36-np.sqrt(15.0)/30],[5/36+np.sqrt(15.0)/24, 2/9, 5/36-np.sqrt(15.0)/24],[5/36+np.sqrt(15.0)/30, 2/9+np.sqrt(15.0)/15, 5/36]]
            b3_G3P6 = [5/18, 4/9, 5/18]
            c3_G3P6 = [1/2 - np.sqrt(15.0)/10, 1/2, 1/2+np.sqrt(15.0)/10]

            # Lobatto IIIB Ehle Method: s = 3, p = 4,
            A4_L3B = [[1/6, -1/6, 0],[1/6, 1/3, 0],[1/6, 5/6, 0]]
            b4_L3B = [1/6, 2/3, 1/6]
            c4_L3B = [0, 1/2, 1]

            # 4 Lobatto IIIC B-stable method
            A4_L3C = [[1/6, -1/3, 1/6],[1/6, 5/12, -1/12],[1/6, 2/3, 1/6]]
            b4_L3C = [1/6, 2/3, 1/6]
            c4_L3C = [0, 1/2, 1]
            # 3 Radau IIA methods, algebraically stable.
            A3_R2A = [[5/12, -1/12],[3/4, 1/4]]
            b3_R2A = [3/4, 1/4]
            c3_R2A = [1/3, 1]
            # 3 Radau IIA methods, algebraically and B-stable.
            A3_R2AS = [[1/4, -1/4],[1/4, 5/12]]
            b3_R2AS = [1/4, 3/4]
            c3_R2AS = [0, 2/3]
            # 4 Lobatto IIIA s = 4, p = 6
            A4_L3AS4 = [[0, 0, 0, 0],[(11+np.sqrt(5))/120, (25-np.sqrt(5))/120, (25-13*np.sqrt(5))/120, (-1+np.sqrt(5))/120],[(11-np.sqrt(5))/120, (25+13*np.sqrt(5))/120, (25+np.sqrt(5))/120, (-1-np.sqrt(5))/120],[1/12, 5/12, 5/12, 1/12]]
            b4_L3AS4 = [1/12, 5/12, 5/12, 1/12]
            c4_L3AS4 = [0, (5-np.sqrt(5))/10, (5+np.sqrt(5))/10, 1]

            y_orders = []
            yi_orders = []
            duration_calc = []
            step = 0
            steps = []
            # steps = []

            for order in orders:
                AD1.set_order(order)
                # step = 0
                print(implicit_method, ', order = ', order)
                param_value = 0
                if implicit_method == "IE":
                    start_time = time.time()
                    stabcheck.check(A1, b1)
                    r_ie = irk(f=AD1.f, jac=AD1.jac, A=A1, c=c1, b=b1, b_hat=None, use_full_newton=newton_jacobian)
                    r_ie.set_initial_value(y0, t0).set_f_params(param_value).set_jac_params(param_value)
                    while r_ie.t <= t1:
                        dt1 = dt
                        # step += 1
                        if (r_ie.t + dt) > t1:
                            dt1 = t1 - r_ie.t
                            if (dt1 < 1.0e-12):
                                break
                        r_ie.integrate(r_ie.t + dt1)
                        print("t = ", r_ie.t)
                    end_time = time.time() - start_time
                    yi_orders.append(r_ie.y)
                    duration_calc.append(end_time)
                    # steps.append(step)

                elif implicit_method == "IM":
                    start_time = time.time()
                    stabcheck.check(A1_IM, b1_IM)
                    l3a2 = irk(f=AD1.f, jac=AD1.jac, A=A1_IM, c=c1_IM, b=b1_IM, b_hat=None, use_full_newton=newton_jacobian)
                    l3a2.set_initial_value(y0, t0).set_f_params(param_value).set_jac_params(param_value)
                    while l3a2.t <= t1:
                        dt1 = dt
                        # step += 1
                        if (l3a2.t + dt) > t1:
                            dt1 = t1 - l3a2.t
                        if (dt1 < 1.0e-12):
                            break
                        l3a2.integrate(l3a2.t + dt1)
                        print("t = ", l3a2.t)
                    end_time = time.time() - start_time
                    yi_orders.append(l3a2.y)
                    duration_calc.append(end_time)
                    # steps.append(step)

                elif implicit_method == "CN":
                    start_time = time.time()
                    stabcheck.check(A2_CN, b2_CN)
                    l3a2 = irk(f=AD1.f, jac=AD1.jac, A=A2_CN, c=c2_CN, b=b2_CN, b_hat=None, use_full_newton=newton_jacobian)
                    l3a2.set_initial_value(y0, t0).set_f_params(param_value).set_jac_params(param_value)
                    while l3a2.t <= t1:
                        dt1 = dt
                        # step += 1
                        if (l3a2.t + dt) > t1:
                            dt1 = t1 - l3a2.t
                        if (dt1 < 1.0e-12):
                            break
                        l3a2.integrate(l3a2.t + dt1)
                        print("t = ", l3a2.t)
                    end_time = time.time() - start_time
                    yi_orders.append(l3a2.y)
                    duration_calc.append(end_time)
                    # steps.append(step)

                elif implicit_method == "G2P4":
                    start_time = time.time()
                    stabcheck.check(A2_G2P4, b2_G2P4)
                    l3a2 = irk(f=AD1.f, jac=AD1.jac, A=A2_G2P4, c=c2_G2P4, b=b2_G2P4, b_hat=None,
                               use_full_newton=newton_jacobian)
                    l3a2.set_initial_value(y0, t0).set_f_params(param_value).set_jac_params(param_value)
                    while l3a2.t <= t1:
                        dt1 = dt
                        # step += 1
                        if (l3a2.t + dt) > t1:
                            dt1 = t1 - l3a2.t
                        if (dt1 < 1.0e-12):
                            break
                        l3a2.integrate(l3a2.t + dt1)
                        print("t = ", l3a2.t)
                    end_time = time.time() - start_time
                    yi_orders.append(l3a2.y)
                    duration_calc.append(end_time)
                    # steps.append(step)




                elif implicit_method == "L2C":
                    start_time = time.time()
                    stabcheck.check(A2_L2C, b2_L2C)
                    l3a2 = irk(f=AD1.f, jac=AD1.jac, A=A2_L2C, c=c2_L2C, b=b2_L2C, b_hat=None,
                               use_full_newton=newton_jacobian)
                    l3a2.set_initial_value(y0, t0).set_f_params(param_value).set_jac_params(param_value)
                    while l3a2.t <= t1:
                        dt1 = dt
                        # step += 1
                        if (l3a2.t + dt) > t1:
                            dt1 = t1 - l3a2.t
                        if (dt1 < 1.0e-12):
                            break
                        l3a2.integrate(l3a2.t + dt1)
                        print("t = ", l3a2.t)
                    end_time = time.time() - start_time
                    yi_orders.append(l3a2.y)
                    duration_calc.append(end_time)
                    # steps.append(step)

                elif implicit_method == "L3A":
                    start_time = time.time()
                    stabcheck.check(A4_L3A, b4_L3A)
                    l3a2 = irk(f=AD1.f, jac=AD1.jac, A=A4_L3A, c=c4_L3A, b=b4_L3A, b_hat=None,
                               use_full_newton=newton_jacobian)
                    l3a2.set_initial_value(y0, t0).set_f_params(param_value).set_jac_params(param_value)
                    while l3a2.t <= t1:
                        dt1 = dt
                        # step += 1
                        if (l3a2.t + dt) > t1:
                            dt1 = t1 - l3a2.t
                        if (dt1 < 1.0e-12):
                            break
                        l3a2.integrate(l3a2.t + dt1)
                        print("t = ", l3a2.t)
                    end_time = time.time() - start_time
                    yi_orders.append(l3a2.y)
                    duration_calc.append(end_time)
                    # steps.append(step)

                elif implicit_method == "L3B":
                    start_time = time.time()
                    stabcheck.check(A4_L3B, b4_L3B)
                    l3a2 = irk(f=AD1.f, jac=AD1.jac, A=A4_L3B, c=c4_L3B, b=b4_L3B, b_hat=None,
                               use_full_newton=newton_jacobian)
                    l3a2.set_initial_value(y0, t0).set_f_params(param_value).set_jac_params(param_value)
                    while l3a2.t <= t1:
                        dt1 = dt
                        # step += 1
                        if (l3a2.t + dt) > t1:
                            dt1 = t1 - l3a2.t
                        if (dt1 < 1.0e-12):
                            break
                        l3a2.integrate(l3a2.t + dt1)
                        print("t = ", l3a2.t)
                    end_time = time.time() - start_time
                    yi_orders.append(l3a2.y)
                    duration_calc.append(end_time)
                    # steps.append(step)


                elif implicit_method == "G3P6":
                    start_time = time.time()
                    stabcheck.check(A3_G3P6, b3_G3P6)
                    l3a2 = irk(f=AD1.f, jac=AD1.jac, A=A3_G3P6, c=c3_G3P6, b=b3_G3P6, b_hat=None,
                               use_full_newton=newton_jacobian)
                    l3a2.set_initial_value(y0, t0).set_f_params(param_value).set_jac_params(param_value)
                    while l3a2.t <= t1:
                        dt1 = dt
                        # step += 1
                        if (l3a2.t + dt) > t1:
                            dt1 = t1 - l3a2.t
                        if (dt1 < 1.0e-12):
                            break
                        l3a2.integrate(l3a2.t + dt1)
                        print("t = ", l3a2.t)
                    end_time = time.time() - start_time
                    yi_orders.append(l3a2.y)
                    duration_calc.append(end_time)
                    # steps.append(step)

                elif implicit_method == "L3AS4":
                    start_time = time.time()
                    stabcheck.check(A4_L3AS4, b4_L3AS4)
                    l3a2 = irk(f=AD1.f, jac=AD1.jac, A=A4_L3AS4, c=c4_L3AS4, b=b4_L3AS4, b_hat=None,
                               use_full_newton=newton_jacobian)
                    l3a2.set_initial_value(y0, t0).set_f_params(param_value).set_jac_params(param_value)
                    while l3a2.t <= t1:
                        dt1 = dt
                        # step += 1
                        if (l3a2.t + dt) > t1:
                            dt1 = t1 - l3a2.t
                        if (dt1 < 1.0e-12):
                            break
                        l3a2.integrate(l3a2.t + dt1)
                        print("t = ", l3a2.t)
                    end_time = time.time() - start_time
                    yi_orders.append(l3a2.y)
                    duration_calc.append(end_time)
                    # steps.append(step)


                elif implicit_method == "R2A":
                    start_time = time.time()
                    stabcheck.check(A3_R2A, b3_R2A)
                    l3a2 = irk(f=AD1.f, jac=AD1.jac, A=A3_R2A, c=c3_R2A, b=b3_R2A, b_hat=None,
                               use_full_newton=newton_jacobian)
                    l3a2.set_initial_value(y0, t0).set_f_params(param_value).set_jac_params(param_value)
                    while l3a2.t <= t1:
                        dt1 = dt
                        # step += 1
                        if (l3a2.t + dt) > t1:
                            dt1 = t1 - l3a2.t
                        if (dt1 < 1.0e-12):
                            break
                        l3a2.integrate(l3a2.t + dt1)
                        print("t = ", l3a2.t)
                    end_time = time.time() - start_time
                    yi_orders.append(l3a2.y)
                    duration_calc.append(end_time)
                    # steps.append(step)

                elif implicit_method == "R2AS":
                    start_time = time.time()
                    stabcheck.check(A4_L3A, b4_L3A)
                    l3a2 = irk(f=AD1.f, jac=AD1.jac, A=A3_R2AS, c=c3_R2AS, b=b3_R2AS, b_hat=None,
                               use_full_newton=newton_jacobian)
                    l3a2.set_initial_value(y0, t0).set_f_params(param_value).set_jac_params(param_value)
                    while l3a2.t <= t1:
                        dt1 = dt
                        # step += 1
                        if (l3a2.t + dt) > t1:
                            dt1 = t1 - l3a2.t
                        if (dt1 < 1.0e-12):
                            break
                        l3a2.integrate(l3a2.t + dt1)
                        print("t = ", l3a2.t)
                    end_time = time.time() - start_time
                    yi_orders.append(l3a2.y)
                    duration_calc.append(end_time)
                    # steps.append(step)

                elif implicit_method == "L3C":
                    start_time = time.time()
                    stabcheck.check(A4_L3A, b4_L3A)
                    l3a2 = irk(f=AD1.f, jac=AD1.jac, A=A4_L3C, c=c4_L3C, b=b4_L3C, b_hat=None,
                               use_full_newton=newton_jacobian)
                    l3a2.set_initial_value(y0, t0).set_f_params(param_value).set_jac_params(param_value)
                    while l3a2.t <= t1:
                        dt1 = dt
                        # step += 1
                        if (l3a2.t + dt) > t1:
                            dt1 = t1 - l3a2.t
                        if (dt1 < 1.0e-12):
                            break
                        l3a2.integrate(l3a2.t + dt1)
                        print("t = ", l3a2.t)
                    end_time = time.time() - start_time
                    yi_orders.append(l3a2.y)
                    duration_calc.append(end_time)
                    # steps.append(step)

                elif implicit_method == "SDIRK2":
                    start_time = time.time()
                    stabcheck.check(A2, b2)
                    r_sdirk2 = sdirk(f=AD1.f, jac=AD1.jac, A=A2, c=c2, b=b2, b_hat=None, use_full_newton=newton_jacobian)
                    r_sdirk2.set_initial_value(y0, t0).set_f_params(param_value).set_jac_params(param_value)
                    while r_sdirk2.t <= t1:
                        dt1 = dt
                        # step += 1
                        if (r_sdirk2.t + dt) > t1:
                            dt1 = t1 - r_sdirk2.t
                        if (dt1 < 1.0e-12):
                            break
                        r_sdirk2.integrate(r_sdirk2.t + dt1)
                        print("t = ", r_sdirk2.t)
                    end_time = time.time() - start_time
                    yi_orders.append(r_sdirk2.y)
                    duration_calc.append(end_time)
                    # steps.append(step)

                elif implicit_method == "SDIRK1BDF":
                    start_time = time.time()
                    stabcheck.check(A3_SDIRK1BDF, b3_SDIRK1BDF)
                    r_sdirk3 = sdirk(f=AD1.f, jac=AD1.jac, A=A3_SDIRK1BDF, c=c3_SDIRK1BDF, b=b3_SDIRK1BDF, b_hat=None,
                                     use_full_newton=newton_jacobian)
                    r_sdirk3.set_initial_value(y0, t0).set_f_params(param_value).set_jac_params(param_value)
                    while r_sdirk3.t <= t1:
                        dt1 = dt
                        # step += 1
                        if (r_sdirk3.t + dt) > t1:
                            dt1 = t1 - r_sdirk3.t
                        if (dt1 < 1.0e-12):
                            break
                        r_sdirk3.integrate(r_sdirk3.t + dt1)
                        print("t = ", r_sdirk3.t)
                    end_time = time.time() - start_time
                    yi_orders.append(r_sdirk3.y)
                    duration_calc.append(end_time)
                    # steps.append(step)

                elif implicit_method == "SDIRK3":
                    start_time = time.time()
                    stabcheck.check(A3, b3)
                    r_sdirk3 = sdirk(f=AD1.f, jac=AD1.jac, A=A3, c=c3, b=b3, b_hat=None, use_full_newton=newton_jacobian)
                    r_sdirk3.set_initial_value(y0, t0).set_f_params(param_value).set_jac_params(param_value)
                    while r_sdirk3.t <= t1:
                        dt1 = dt
                        # step += 1
                        if (r_sdirk3.t + dt) > t1:
                            dt1 = t1 - r_sdirk3.t
                        if (dt1 < 1.0e-12):
                            break
                        r_sdirk3.integrate(r_sdirk3.t + dt1)
                        print("t = ", r_sdirk3.t)
                    end_time = time.time() - start_time
                    yi_orders.append(r_sdirk3.y)
                    duration_calc.append(end_time)
                    # steps.append(step)

                elif implicit_method == "SDIRK4":
                    start_time = time.time()
                    stabcheck.check(A4, b4)
                    r_sdirk4 = sdirk(f=AD1.f, jac=AD1.jac, A=A4, c=c4, b=b4, b_hat=None, use_full_newton=newton_jacobian)
                    r_sdirk4.set_initial_value(y0, t0).set_f_params(param_value).set_jac_params(param_value)
                    while r_sdirk4.t <= t1:
                        dt1 = dt
                        # step += 1
                        if (r_sdirk4.t + dt) > t1:
                            dt1 = t1 - r_sdirk4.t
                        if dt1 < 1.0e-12:
                            break
                        r_sdirk4.integrate(r_sdirk4.t + dt1)
                        print("t = ", r_sdirk4.t)
                    end_time = time.time() - start_time
                    yi_orders.append(r_sdirk4.y)
                    duration_calc.append(end_time)
                    # steps.append(step)

                step = 0

                # refernce solution
                reference_method = 'dop853'
                start_time = time.time()
                r = ode(AD1.f).set_integrator(reference_method, rtol=1.0e-6, atol=1.0e-11)
                r.set_initial_value(y0, t0).set_f_params(param_value).set_jac_params(param_value)
                print(reference_method, ', order = ', order)
                while r.successful() and r.t <= t1:
                    dt1 = dt
                    step += 1
                    print("step = ", step)
                    if (r.t + dt) > t1:
                        dt1 = t1 - r.t
                    if (dt1 < 1.0e-12):
                        break
                    r.integrate(r.t + dt1)
                end_time = time.time() - start_time
                y_orders.append(r.y)
                duration_calc.append(end_time)
                # steps.append(step)

            plt.figure()

            legend_array = []
            if t1 <= 2187/4096:
                plt.plot(x,  np.interp(x, (x + ES.init(c_exp, x) * t1), ES.hopf(c_exp, x, t1)))
                legend_array.append("exact solution, t = {Tf}".format(Tf=round(t1,4)))

            # plt.legend(legend_array)
            # plt.title("CFL = {cfl}, Riemann solver = {RS}".format(cfl=CFL, RS=riemann_solver_type))
            # plt.show()

            for y, yi, o in zip(y_orders, yi_orders, orders):
                plt.plot(x, yi, '*')
                plt.plot(x, y, '.')
                print("CFL = ", CFL)
                print("steps = ", step)

                if o == 1:
                    legend_array.append("1st order, {}".format(implicit_method))
                    legend_array.append("1st order, ref:{}".format(reference_method))
                    # print("L2_error 1st order{} = " .format(implicit_method), L2_error(np.interp(x, (x+ES.init(c_exp, x) * t1),  ES.hopf(c_exp, x, t1)), yi, N, dh))
                    # print("L2_error 1st order{} = ".format(reference_method), L2_error(np.interp(x, (x+ES.init(c_exp, x) * t1),  ES.hopf(c_exp, x, t1)), y, N, dh))

                    # print("L2_error relative 1st order{} = ".format(implicit_method),
                    #      L2_error_relative(np.interp(x, (x + ES.init(c_exp, x) * t1), ES.hopf(c_exp, x, t1)), yi, N, dh))
                    # print("L2_error relative 1st order{} = ".format(reference_method),
                    #      L2_error_relative(np.interp(x, (x + ES.init(c_exp, x) * t1), ES.hopf(c_exp, x, t1)), y, N, dh))
                    f = open('{eq},{exact},t={tf},IM={im}.txt'.format(eq=equation, exact=exact_solution_type, tf=t1,
                                                                      im=implicit_method), 'a')
                    g = open(
                        '{eq},{exact},t={tf},IM={im}-time.txt'.format(eq=equation, exact=exact_solution_type, tf=t1,
                                                                      im=implicit_method),
                        'a')
                    f.write(str(L2_error_relative(np.interp(x, (x + ES.init(c_exp, x) * t1), ES.hopf(c_exp, x, t1)), yi, N, dh)) + ' ')
                    f.write(str(L2_error_relative(np.interp(x, (x + ES.init(c_exp, x) * t1), ES.hopf(c_exp, x, t1)), y, N, dh)) + ' ')
                    g.write(str(duration_calc[o - 1]) + ' ')
                    g.write(str(duration_calc[o]) + ' ')
                    # print("1st order, {} steps=".format(implicit_method), steps[o-1])
                    # print("1st order, {} steps=".format(reference_method), steps[o])

                    # f.write(str(L2_error_relative(np.interp(x, (x + ES.init(c_exp, x) * t1), ES.hopf(c_exp, x, t1)), yi, N,
                    #                              dh)) + ' ')
                    # f.write(str(L2_error_relative(np.interp(x, (x + ES.init(c_exp, x) * t1), ES.hopf(c_exp, x, t1)), y, N,
                    #                              dh)) + ' ')
                    # f.write("hello")

                else:
                    legend_array.append("WENO{order}, {method}".format(order=o, method=implicit_method))
                    legend_array.append("WENO{order}, ref:{method}".format(order=o, method=reference_method))
                    # print("L2_error WENO{order}, {method}".format(order = o, method = implicit_method), L2_error(np.interp(x, (x+ES.init(c_exp, x) * t1),  ES.hopf(c_exp, x, t1)), yi, N, dh))
                    # print("L2_error WENO{order}, ref:{method}".format(order = o, method = reference_method), L2_error(np.interp(x, (x+ES.init(c_exp, x) * t1),  ES.hopf(c_exp, x, t1)), y, N, dh))
                    f.write(str(L2_error_relative(np.interp(x, (x + ES.init(c_exp, x) * t1), ES.hopf(c_exp, x, t1)), yi, N,
                                                  dh)) + ' ')
                    f.write(str(L2_error_relative(np.interp(x, (x + ES.init(c_exp, x) * t1), ES.hopf(c_exp, x, t1)), y, N,
                                                  dh)) + ' ')
                    g.write(str(duration_calc[o - 1]) + ' ')
                    g.write(str(duration_calc[o]) + ' ')
                    # print("WENO{order}, {method} steps=".format(order=o, method=implicit_method), steps[o-1])
                    # print("WENO{order}, ref:{method} steps=".format(order=o, method=reference_method), steps[o])

                    # print("L2_error relative WENO{order}, {method}".format(order=o, method=implicit_method),
                    #      L2_error_relative(np.interp(x, (x + ES.init(c_exp, x) * t1), ES.hopf(c_exp, x, t1)), yi, N, dh))
                    # print("L2_error relative WENO{order}, ref:{method}".format(order=o, method=reference_method),
                    #      L2_error_relative(np.interp(x, (x + ES.init(c_exp, x) * t1), ES.hopf(c_exp, x, t1)), y, N, dh))
                    # f.write(str(L2_error_relative(np.interp(x, (x + ES.init(c_exp, x) * t1), ES.hopf(c_exp, x, t1)), yi, N,
                    #                              dh)) + ' ')
                    # f.write(str(L2_error_relative(np.interp(x, (x + ES.init(c_exp, x) * t1), ES.hopf(c_exp, x, t1)), y, N,
                    #                             dh)) + ' ')

                    # f.close()
            plt.plot(evstigneevnnm_data_x[counter], evstigneevnnm_data_u[counter])

            if t1 <= 2187 / 4096:
                legend_array.append("article (WENO7, RK4, t={tf})".format(tf=round(t1,4)))
            else:
                legend_array.append("article (WENO7, RK4, t={tf})".format(tf=round(t1, 1)))

            plt.legend(legend_array, loc='upper right')
            plt.title(
                "equation = {eq}, CFL = {cfl}, Riemann solver = {RS}".format(eq=equation, cfl=CFL, RS=riemann_solver_type))
            plt.savefig('{eq},{exact},t={tf},IM={im},CFL={cfl}.png'.format(eq=equation, exact=exact_solution_type, tf=t1,
                                                                           im=implicit_method, cfl=CFL))
            # plt.show()
            f.write('\n')
            g.write('\n')
            g.write(str(step))
            # f.write('\n')
            g.write('\n')
        f.close()
        g.close()
    counter += 1
        #plt.legend(legend_array)
        #plt.title("CFL = {cfl}, Riemann solver = {RS}".format(cfl=CFL, RS=riemann_solver_type))
        #plt.show()
    #f.close()

"""
        #plt.figure()
        
        #plt.plot(x, y0 )
        #u = np.linspace(-1, 1, N)  # x span
        
        
        
        fig, ax = plt.subplots(nrows=1, ncols=3)
        
        #plt.plot((x+ES.init(c_exp, x) * t1), ES.hopf(c_exp, x, t1))
        #legend_array = ["exact solution, t = {Tf}".format(Tf=t1)]
        
        ax[0].plot((x+ES.init(c_exp, x) * t1), ES.hopf(c_exp, x, t1))
        
        #plt.legend(legend_array)
        #plt.title("CFL = {cfl}, Riemann solver = {RS}".format(cfl = CFL, RS = riemann_solver_type))
        #plt.show()
        print("argmax exact = ", np.argmax(ES.hopf(c_exp, x, t1)))
        for y,yi,o in zip(y_orders, yi_orders, orders):
            #plt.plot(x, ES.hopf(c_exp, x, t1) - yi, '*')
            #plt.plot(x, ES.hopf(c_exp, x, t1) - y, '.')
            ax[1].plot(x, yi)
            ax[1].plot(x, y)
        
            ax[2].plot(x, ES.hopf(c_exp, x, t1) - yi)
            ax[2].plot(x, ES.hopf(c_exp, x, t1) - y)
            if o==1:
                #legend_array.append("1st order{}, ".format(implicit_method) )
                #legend_array.append("1st order, ref:{}".format(reference_method) )
                #print("L2_error 1st order{} = " .format(implicit_method), L2_error(ES.hopf(c_exp, x, t1), yi, N, dh))
                #print("L2_error 1st order{} = ".format(reference_method), L2_error(ES.hopf(c_exp, x, t1), y, N, dh))
                print("argmax 1st order{} = ".format(implicit_method), np.argmax(yi))
                print("argmax 1st order{} = ".format(reference_method), np.argmax(y))
            else:
                #legend_array.append("WENO{order}, {method}".format(order = o, method = implicit_method))
                #legend_array.append("WENO{order}, ref:{method}".format(order = o, method = reference_method))
                #print("L2_error WENO{order}, {method}".format(order = o, method = implicit_method), L2_error(ES.hopf(c_exp, x, t1), yi, N, dh))
                #print("L2_error WENO{order}, ref:{method}".format(order = o, method = reference_method), L2_error(ES.hopf(c_exp, x, t1), y, N, dh))
                print("argmax WENO{order}, {method}".format(order=o, method=implicit_method), np.argmax(yi))
                print("argmax WENO{order}, ref:{method}".format(order=o, method=reference_method), np.argmax(y))
        #plt.legend(legend_array)
        plt.title("CFL = {cfl}, Riemann solver = {RS}".format(cfl = CFL, RS = riemann_solver_type))
        plt.show()
        """

"""
        
        """

        # plt.figure()
        # plt.plot(np.transpose(np.transpose(all_y)[1]),np.transpose(np.transpose(all_y)[0]),'.')
        # plt.plot(np.transpose(np.transpose(all_y_sdirk2)[1]),np.transpose(np.transpose(all_y_sdirk2)[0]))
        # plt.plot(np.transpose(np.transpose(all_y_sdirk3)[1]),np.transpose(np.transpose(all_y_sdirk3)[0]))
        # plt.plot(np.transpose(np.transpose(all_y_sdirk4)[1]),np.transpose(np.transpose(all_y_sdirk4)[0]))
        # plt.legend(['python native','sdirk2','sdirk3','sdirk4'])
        # plt.show()

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

