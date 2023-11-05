# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.integrate import ode
from matplotlib import pyplot as plt
from ode_sdirk import ode_sdirk as sdirk


class weno_interpolation(object):
    def __init__(self, scheme):
        super(weno_interpolation, self).__init__()
        self.order = scheme
        self.stride = int((self.order+1)/2)
        self.weno_epsilon = 1.0e-16
        self.__use_vectorization = True

    def __shift_indexes(self, N):
        idx = []
        for j in range(0,self.stride*2+1):
            idx.append( list( range(j,N+j) ) )
        return idx

    def __interpolate_7(self, f1, f2, f3, f4, f5, f6, f7):
        eps = self.weno_epsilon
        q0=-1.0/4.0*f1+13.0/12.0*f2-23.0/12.0*f3+25.0/12.0*f4;
        q1=1.0/12.0*f2-5.0/12.0*f3+13.0/12.0*f4+1.0/4.0*f5;
        q2=-1.0/12.0*f3+7.0/12.0*f4+7.0/12.0*f5-1.0/12.0*f6;
        q3=1.0/4.0*f4+13.0/12.0*f5-5.0/12.0*f6+1.0/12.0*f7;

        beta0=(547.0*f1*f1 + 7043.0*f2*f2 + 11003.0*f3*f3 - 9402.0*f3*f4 +  2107.0*f4*f4 - 2.0*f1 *(1941.0*f2 - 2321.0*f3 + 927.0*f4) +  f2*(-17246.0*f3 + 7042.0*f4));
        beta1=(267.0*f2*f2 + 2843.0*f3*f3 + 3443.0*f4*f4 - 2522.0*f4*f5 +  547.0*f5*f5 - 2.0*f2*(821.0*f3 - 801.0*f4 + 247.0*f5) +  f3*(-5966.0*f4 + 1922.0*f5));
        beta2=(547.0*f3*f3 + 3443.0*f4*f4 + 2843.0*f5*f5 - 1642.0*f5*f6 + 267.0*f6*f6 - 2.0*f3*(1261.0*f4 - 961.0*f5 + 247.0*f6) + f4*(-5966.0*f5 + 1602.0*f6));
        beta3=(2107.0*f4*f4 + 11003.0*f5*f5 + 7043.0*f6*f6 - 3882.0*f6*f7 + 547.0*f7*f7 - 2.0*f4*(4701.0*f5 - 3521.0*f6 + 927.0*f7) + f5*(-17246.0*f6 + 4642.0*f7));

        d0=1.0/35.0
        d1=12.0/35.0
        d2=18.0/35.0
        d3=4.0/35.0; 

        alpha0=d0/((eps+beta0)*(eps+beta0));
        alpha1=d1/((eps+beta1)*(eps+beta1));
        alpha2=d2/((eps+beta2)*(eps+beta2));
        alpha3=d3/((eps+beta3)*(eps+beta3));
        alphaum=alpha0+alpha1+alpha2+alpha3;
        w0=alpha0/alphaum;
        w1=alpha1/alphaum;
        w2=alpha2/alphaum;
        w3=alpha3/alphaum;

        ret = w0*q0+w1*q1+w2*q2+w3*q3


        return ret


    def __interpolate_5(self, f1, f2, f3, f4, f5):
        eps = self.weno_epsilon
        q0=1.0/3.0*f1-7.0/6.0*f2+11.0/6.0*f3;
        q1=-1.0/6.0*f2+5.0/6.0*f3+1.0/3.0*f4;
        q2=1.0/3.0*f3+5.0/6.0*f4-1.0/6.0*f5;
        beta0=13.0/12.0*(f1-2.0*f2+f3)*(f1-2.0*f2+f3)+0.25*(f1-4.0*f2+3.0*f3)*(f1-4.0*f2+3.0*f3);
        beta1=13.0/12.0*(f2-2.0*f3+f4)*(f2-2.0*f3+f4)+0.25*(f2-f4)*(f2-f4);
        beta2=13.0/12.0*(f3-2.0*f4+f5)*(f3-2.0*f4+f5)+0.25*(3.0*f3-4.0*f4+f5)*(3.0*f3-4.0*f4+f5);
        d0=0.1;
        d1=0.6;
        d2=0.3; 
        alpha0=d0/((eps+beta0)*(eps+beta0));
        alpha1=d1/((eps+beta1)*(eps+beta1));
        alpha2=d2/((eps+beta2)*(eps+beta2));
        alpha_sum=alpha0+alpha1+alpha2;
        w0=alpha0/alpha_sum;
        w1=alpha1/alpha_sum;
        w2=alpha2/alpha_sum;
        res=w0*q0+w1*q1+w2*q2;
        return res

    def __interpolate_3(self, f1, f2, f3):
        eps = self.weno_epsilon
        q0=-0.5*f1+1.5*f2;
        q1=0.5*f2+0.5*f3;
        beta0=(f2-f1)*(f2-f1);
        beta1=(f3-f2)*(f3-f2);
        d0=1.0/3.0;
        d1=2.0/3.0; 
        alpha0=d0/((eps+beta0)*(eps+beta0));
        alpha1=d1/((eps+beta1)*(eps+beta1));
        alpha_sum=alpha0+alpha1;
        w0=alpha0/alpha_sum;
        w1=alpha1/alpha_sum;
        res = w0*q0+w1*q1
        return res

    def __interpolate_1(self, f1):
        return f1


    def __form_stencil_values(self, vec_ext, center):
        actial_position_in_extended_vec = center + self.stride
        return vec_ext[actial_position_in_extended_vec-self.stride:actial_position_in_extended_vec+self.stride+1]

    def __interpolate_w_left_local(self, stencil_vals):
        if(self.order == 7):
            f1 = stencil_vals[0]
            f2 = stencil_vals[1]
            f3 = stencil_vals[2]
            f4 = stencil_vals[3]
            f5 = stencil_vals[4]
            f6 = stencil_vals[5]            
            f7 = stencil_vals[6]            
            val = self.__interpolate_7(f1, f2, f3, f4, f5, f6, f7)        
        elif(self.order == 5):
            f1 = stencil_vals[0]
            f2 = stencil_vals[1]
            f3 = stencil_vals[2]
            f4 = stencil_vals[3]
            f5 = stencil_vals[4]
            val = self.__interpolate_5(f1, f2, f3, f4, f5)
        elif(self.order == 3):
            f1 = stencil_vals[0]
            f2 = stencil_vals[1]
            f3 = stencil_vals[2]
            val = self.__interpolate_3(f1, f2, f3)
        else:
            f1 = stencil_vals[0]
            val = self.__interpolate_1(f1)
        return val
    def __interpolate_w_right_local(self, stencil_vals):
        if(self.order == 7):
            f1 = stencil_vals[1]
            f2 = stencil_vals[2]
            f3 = stencil_vals[3]
            f4 = stencil_vals[4]
            f5 = stencil_vals[5]
            f6 = stencil_vals[6]
            f7 = stencil_vals[7]
            val = self.__interpolate_7(f7, f6, f5, f4, f3, f2, f1)
        elif(self.order == 5):
            f1 = stencil_vals[1]
            f2 = stencil_vals[2]
            f3 = stencil_vals[3]
            f4 = stencil_vals[4]
            f5 = stencil_vals[5]
            val = self.__interpolate_5(f5, f4, f3, f2, f1)
        elif(self.order == 3):
            f1 = stencil_vals[1]
            f2 = stencil_vals[2]
            f3 = stencil_vals[3]
            val = self.__interpolate_3(f3, f2, f1)
        else:
            f1 = stencil_vals[1]
            val = self.__interpolate_1(f1)
        return val

    def __interpolate_e_left_local(self, stencil_vals):
        if(self.order == 7):
            f1 = stencil_vals[1]
            f2 = stencil_vals[2]
            f3 = stencil_vals[3]
            f4 = stencil_vals[4]
            f5 = stencil_vals[5]
            f6 = stencil_vals[6]
            f7 = stencil_vals[7]            
            val = self.__interpolate_7(f1, f2, f3, f4, f5, f6, f7)        
        elif(self.order == 5):
            f1 = stencil_vals[1]
            f2 = stencil_vals[2]
            f3 = stencil_vals[3]
            f4 = stencil_vals[4]
            f5 = stencil_vals[5]
            val = self.__interpolate_5(f1, f2, f3, f4, f5)
        elif(self.order == 3):
            f1 = stencil_vals[1]
            f2 = stencil_vals[2]
            f3 = stencil_vals[3]
            val = self.__interpolate_3(f1, f2, f3)
        else:
            f1 = stencil_vals[1]
            val = self.__interpolate_1(f1)
        return val
    def __interpolate_e_right_local(self, stencil_vals):
        if(self.order == 7):
            f1 = stencil_vals[2]
            f2 = stencil_vals[3]
            f3 = stencil_vals[4]
            f4 = stencil_vals[5]
            f5 = stencil_vals[6]
            f6 = stencil_vals[7]
            f7 = stencil_vals[8]            
            val = self.__interpolate_7(f7, f6, f5, f4, f3, f2, f1) 
        elif(self.order == 5):
            f1 = stencil_vals[2]
            f2 = stencil_vals[3]
            f3 = stencil_vals[4]
            f4 = stencil_vals[5]
            f5 = stencil_vals[6]
            val = self.__interpolate_5(f5, f4, f3, f2, f1)
        elif(self.order == 3):
            f1 = stencil_vals[2]
            f2 = stencil_vals[3]
            f3 = stencil_vals[4]
            val = self.__interpolate_3(f3, f2, f1)
        else:
            f1 = stencil_vals[2]
            val = self.__interpolate_1(f1)
        return val

    def __interpolate_l_w(self, vec, idxs):
        if(self.order == 7):
            f1 = vec[idxs[0]]
            f2 = vec[idxs[1]]
            f3 = vec[idxs[2]]
            f4 = vec[idxs[3]]
            f5 = vec[idxs[4]]
            f6 = vec[idxs[5]]
            f7 = vec[idxs[6]]
            val = self.__interpolate_7(f1, f2, f3, f4, f5, f6, f7) 
        elif(self.order == 5):
            f1 = vec[idxs[0]]
            f2 = vec[idxs[1]]
            f3 = vec[idxs[2]]
            f4 = vec[idxs[3]]
            f5 = vec[idxs[4]]
            val = self.__interpolate_5(f1, f2, f3, f4, f5)
        elif(self.order == 3):
            f1 = vec[idxs[0]]
            f2 = vec[idxs[1]]
            f3 = vec[idxs[2]]
            val = self.__interpolate_3(f1, f2, f3)
        else:
            f1 = vec[idxs[0]]
            val = self.__interpolate_1(f1)  

        return val      
    def __interpolate_r_w(self, vec, idxs):
        if(self.order == 7):
            f1 = vec[idxs[1]]
            f2 = vec[idxs[2]]
            f3 = vec[idxs[3]]
            f4 = vec[idxs[4]]
            f5 = vec[idxs[5]]
            f6 = vec[idxs[6]]
            f7 = vec[idxs[7]]
            val = self.__interpolate_7(f7, f6, f5, f4, f3, f2, f1) 
        elif(self.order == 5):
            f1 = vec[idxs[1]]
            f2 = vec[idxs[2]]
            f3 = vec[idxs[3]]
            f4 = vec[idxs[4]]
            f5 = vec[idxs[5]]
            val = self.__interpolate_5(f5, f4, f3, f2, f1)
        elif(self.order == 3):
            f1 = vec[idxs[1]]
            f2 = vec[idxs[2]]
            f3 = vec[idxs[3]]
            val = self.__interpolate_3(f3, f2, f1)
        else:
            f1 = vec[idxs[1]]
            val = self.__interpolate_1(f1)   
        return val    
    def __interpolate_l_e(self, vec, idxs):
        if(self.order == 7):
            f1 = vec[idxs[1]]
            f2 = vec[idxs[2]]
            f3 = vec[idxs[3]]
            f4 = vec[idxs[4]]
            f5 = vec[idxs[5]]
            f6 = vec[idxs[6]]
            f7 = vec[idxs[7]]
            val = self.__interpolate_7(f1, f2, f3, f4, f5, f6, f7) 
        elif(self.order == 5):
            f1 = vec[idxs[1]]
            f2 = vec[idxs[2]]
            f3 = vec[idxs[3]]
            f4 = vec[idxs[4]]
            f5 = vec[idxs[5]]
            val = self.__interpolate_5(f1, f2, f3, f4, f5)
        elif(self.order == 3):
            f1 = vec[idxs[1]]
            f2 = vec[idxs[2]]
            f3 = vec[idxs[3]]
            val = self.__interpolate_3(f1, f2, f3)
        else:
            f1 = vec[idxs[1]]
            val = self.__interpolate_1(f1) 
        return val                   
    def __interpolate_r_e(self, vec, idxs):
        if(self.order == 7):
            f1 = vec[idxs[2]]
            f2 = vec[idxs[3]]
            f3 = vec[idxs[4]]
            f4 = vec[idxs[5]]
            f5 = vec[idxs[6]]
            f6 = vec[idxs[7]]
            f7 = vec[idxs[8]]
            val = self.__interpolate_7(f7, f6, f5, f4, f3, f2, f1) 
        elif(self.order == 5):
            f1 = vec[idxs[2]]
            f2 = vec[idxs[3]]
            f3 = vec[idxs[4]]
            f4 = vec[idxs[5]]
            f5 = vec[idxs[6]]
            val = self.__interpolate_5(f5, f4, f3, f2, f1)
        elif(self.order == 3):
            f1 = vec[idxs[2]]
            f2 = vec[idxs[3]]
            f3 = vec[idxs[4]]
            val = self.__interpolate_3(f3, f2, f1)
        else:
            f1 = vec[idxs[2]]
            val = self.__interpolate_1(f1)   
        return val
        
    def apply_periodic(self, vec):
        # extend vector by boundary conditions of width stride
        vec_extended = np.hstack((vec[-self.stride:], vec, vec[:self.stride]))
        vec_size = len(vec)
        if self.__use_vectorization:
            idxs = self.__shift_indexes(vec_size)
            Lw = self.__interpolate_l_w(vec_extended, idxs)
            Rw = self.__interpolate_r_w(vec_extended, idxs)
            Le = self.__interpolate_l_e(vec_extended, idxs)
            Re = self.__interpolate_r_e(vec_extended, idxs)  

        else:
            Lw = np.zeros(vec_size)
            Rw = np.zeros(vec_size)
            Le = np.zeros(vec_size)
            Re = np.zeros(vec_size)
            for j in range(0, vec_size):
                # --X------X--
                #  LwR    LeR
                # LwR: west L/R values
                # LeR: east L/R values
                stencil = self.__form_stencil_values(vec_extended, j)
                Lw[j] = self.__interpolate_w_left_local(stencil)
                Rw[j] = self.__interpolate_w_right_local(stencil)
                Le[j] = self.__interpolate_e_left_local(stencil)
                Re[j] = self.__interpolate_e_right_local(stencil)

        return Lw, Rw, Le, Re

    def apply_periodic_split(self, FL, FR):
        FL_extended = np.hstack((FL[-self.stride:], FL, FL[:self.stride]))
        FR_extended = np.hstack((FR[-self.stride:], FR, FR[:self.stride]))

        vec_size = len(FL)

        if self.__use_vectorization:
            idxs = self.__shift_indexes(vec_size)
            Lw = self.__interpolate_l_w(vec_extended, idxs)
            Rw = self.__interpolate_r_w(vec_extended, idxs)
            Le = self.__interpolate_l_e(vec_extended, idxs)
            Re = self.__interpolate_r_e(vec_extended, idxs)  

        return Lw, Rw, Le, Re        


class riemann_solver(object):
    def __init__(self, solver, flux, flux_derivative):
        super(riemann_solver, self).__init__()
        self.__solver = solver
        self.__flux = flux
        self.__flux_derivative = flux_derivative
     
    def set_solver(self, solver):
        self.__solver = solver

    def set_flux_data(self, flux, flux_derivative):
        self.__flux = flux
        self.__flux_derivative = flux_derivative

    def __upwind(self, l, r):
        c = self.__flux_derivative(l)
        if c<0.0:
            return c*r;
        else:
            return c*l;

    # side flux
    def __calc_flux_side(self, l, r): 
        if self.__solver == "upwind":
            return self.__upwind(l,r)
        else:
            return 0
    # flux splitting
    def __calc_flux_split(self, u):
        if self.__solver == "LLF":
            return self.__rusanov(u)
        else:
            return 0

    def __rusanov(self, u):
        F = self.__flux(u)
        a = self.__flux_derivative(u) 
        Fp=0.5*(F + a*u); 
        Fm=0.5*(F - a*u); 
        return Fp, Fm

    def apply_side(self, Lw, Rw, Le, Re):
        # fl = Lw*0
        # fr = Lw*0
        # for j in range(0,len(Lw)):
        #     fl[j] =  self.__calc_flux_side(Lw[j], Rw[j]);
        #     fr[j] =  self.__calc_flux_side(Le[j], Re[j]);
        fl = self.__calc_flux_side(Lw, Rw);
        fr = self.__calc_flux_side(Le, Re);
        return fl, fr
    
    def apply_split(self, u):
        fl, fr = self.__calc_flux_split(u);
        return fl, fr



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


    # def jac(self, t, y, args):
        # return 0




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


t0 = 0
t1 = np.sqrt(17.0)
dt = 0.1
N = 200
a = -1
b = 1
c_exp = 20.0


AD1 = advection(N, "upwind", 5)
x = np.linspace(a, b, N)  # x span
ES = exact_solution("canonic");

y0 = ES.init(c_exp, x)
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

orders = [1,3,5,7]
y_orders = []

for order in orders:
#TODO: implicit methods are to be implemented

    # r_sdirk2 = sdirk(f = AD1.f, jac = AD1.jac, A = A2, c = c2, b = b2, b_hat = None, use_full_newton = 'sdirk')
    # r_sdirk3 = sdirk(f = AD1.f, jac = AD1.jac, A = A3, c = c3, b = b3, b_hat = None, use_full_newton = 'sdirk')
    # r_sdirk4 = sdirk(f = AD1.f, jac = AD1.jac, A = A4, c = c4, b = b4, b_hat = None, use_full_newton = 'sdirk')

    AD1.set_order(order)

    r = ode(AD1.f).set_integrator('dop853', rtol = 1.0e-6, atol = 1.0e-11)
    # r = ode(f).set_integrator('dopri5')
    #r = ode(f, J).set_integrator('vode', method = 'bdf', with_jacobian = True)
    #r = ode(f, J).set_integrator('vode', method = 'bdf', with_jacobian = True, min_step = dt, max_step = dt, first_step = dt, atol = 100.0, rtol = 100.0, order = 5)

    param_value = 0;

    r.set_initial_value(y0, t0).set_f_params(param_value)#.set_jac_params(param_value)
    # r_sdirk2.set_initial_value(y0, t0).set_f_params(param_value).set_jac_params(param_value)
    # r_sdirk3.set_initial_value(y0, t0).set_f_params(param_value).set_jac_params(param_value)
    # r_sdirk4.set_initial_value(y0, t0).set_f_params(param_value).set_jac_params(param_value)


    all_t = []
    all_y = []
    # all_t_sdirk2 = []
    # all_y_sdirk2 = []
    # all_t_sdirk3 = []
    # all_y_sdirk3 = []
    # all_t_sdirk4 = []
    # all_y_sdirk4 = []

    # print('SDIRK2\n')
    # while r_sdirk2.t <= t1:
    #     r_sdirk2.integrate(r_sdirk2.t+dt)
    #     all_t_sdirk2.append(r_sdirk2.t)
    #     all_y_sdirk2.append(r_sdirk2.y)

    # print('SDIRK3\n')
    # while r_sdirk3.t <= t1:
    #     r_sdirk3.integrate(r_sdirk3.t+dt)
    #     all_t_sdirk3.append(r_sdirk3.t)
    #     all_y_sdirk3.append(r_sdirk3.y)

    # print('SDIRK4\n')
    # while r_sdirk4.t <= t1:
    #     r_sdirk4.integrate(r_sdirk4.t+dt)
    #     all_t_sdirk4.append(r_sdirk4.t)
    #     all_y_sdirk4.append(r_sdirk4.y)

    print('BDF, order = ', order)
    while r.successful() and r.t <= t1:
        if (r.t+dt)>t1:
            dt1 = t1-r.t
        else:
            dt1 = dt
        r.integrate(r.t+dt1)
        all_t.append(r.t)
        all_y.append(r.y)

    #plt.figure()
    #plt.plot(all_t_sdirk,np.transpose(np.transpose(all_y_sdirk)[0]),'*')
    #plt.plot(all_t,np.transpose(np.transpose(all_y)[0]))
    #plt.show()

    y_orders.append(r.y)


plt.figure()
plt.plot(x, ES.advect(c_exp, x, t1) )
# plt.plot(x, r_sdirk2.y, '*')
# plt.plot(x, r_sdirk3.y, '*')
# plt.plot(x, r_sdirk4.y, '*')
legend_array = ["exact"]
for y,o in zip(y_orders, orders):
    plt.plot(x, y, '.')
    if o==1:
        legend_array.append("1st order")
    else:
        legend_array.append("WENO{order}".format(order = o))
# plt.legend(['exact','sdirk2','sdirk3','sdirk4','python bdf'])
plt.legend(legend_array)
plt.show()


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
