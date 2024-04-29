#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:19:50 2020

@author: noctum
"""
#import re
#import warnings

from numpy import asarray, zeros, isscalar, linalg, identity, multiply, dot, hstack, vstack, split, sum, abs
import scipy.sparse as sparse

class ode_irk(object):
    def __init__(self, f, jac, A, c, b, b_hat = None, use_full_newton = 'full', skip_nonconverged_step = False, eps_newton = 1.0e-9, iters_newton = 1000, newton_rebuild_jacobian_frequency = 7):
        self.stiff = 0
        self.f = f
        self.jac = jac
        self.f_params = ()
        self.jac_params = ()
        self._u = []
        self.use_full_newton = use_full_newton
        if self.use_full_newton!='freeze' and self.use_full_newton!='full':
            print('Incorrect newton method scheme.')
            print('Only "full" or "freeze" approaches are possible.')
            raise          
        self.A = A
        self.c = c
        self.b = b
        self.s = len(b) # number of stages
#        self.y_stages = zeros(self.s)
        
#        TODO: Implement and test adoptation process
        if b_hat != None:
            self.b_hat = b_hat
            self.use_adoptation = True
        else:
            self.b_hat = []
            self.use_adoptation = False
        self._u1 = []
        self._dt = 0.0
        self.eps_newton = eps_newton
        self.iters_newton = iters_newton
        self.__newtin_initial_iteration = True
        self.__newton_latest_converged_norm = -1
        self.__newton_rebuild_jacobian_frequency = newton_rebuild_jacobian_frequency
        self.skip_nonconverged_step = skip_nonconverged_step
    @property
    def y(self):
        return self._u
    def y1(self):
        return self._u1
    
    def set_f_params(self, *args):
        """Set extra parameters for user-supplied function f."""
        self.f_params = args[0]
        return self

    def set_jac_params(self, *args):
        """Set extra parameters for user-supplied function jac."""
        self.jac_params = args[0]
        return self
    
    def set_initial_value(self, y, t=0.0):
        if isscalar(y):
            y = [y]
        self._u = asarray(y)
        self.t = t
        n_vec = len(self._u)
        self.n_vec = n_vec
        self.E = identity(n_vec*self.s) # '''sets ident matrix of the extended system'''
        return self
    

    def __solve_linear_system(self, b):
        if self.use_full_newton == 'full':
            x = linalg.solve(self.op, b)
            # x = sparse.linalg.spsolve(self.op, b)
        else:
            x = dot(self.op, b)
        return(x)

    def __form_operator(self, y_v):
        y_a = self.__cut(y_v)
        a_row = []
        for y_k,  c_l in zip(y_a, self.c ):
            a_row.append(  self.jac(self.t+c_l*self._dt, y_k, self.jac_params) )
        a = []
        for j in range(0, self.s):
            a_l = []
            for a_k, a_jk in zip( a_row, self.A[j] ):
                a_l.append(self._dt*a_jk*a_k)
            aa_l = hstack(a_l)
            if len(a)>0:
                a = vstack((a,aa_l))
            else:
                a = aa_l
        a = self.E - a
        return a
        
    def __set_operator(self, y_v):
        if self.use_full_newton != 'full' and self.__newtin_initial_iteration:
            self.__newtin_initial_iteration = False # we are forming the operator. Next time it will be triggered only if it is True
            a = self.__form_operator(y_v)
            self.op = linalg.inv(a)
        elif self.use_full_newton == 'full':
            # op_sp = sparse.csc_matrix(self.__form_operator(y_v))
            # self.op = op_sp
            self.op = self.__form_operator(y_v)
            
    def __cut(self, y_v):
        return split(y_v, self.s)
    def __glue(self, y_m):
        return hstack(y_m)

    # using IRK in the following form:
    # u_{n+1} = u_{n} + \tau*sum_{j=1}^{s} b_j f(t+\tau*c_j, y_j)
    # y_j = u_{n}+\tau*\sum_{k=1}^{s} a_{jk} f(t+\tau*c_k, y_k)
    # here we use the Newton's method to find the solution simultaniously for a general IRK metod

    # self._u1 has an initial y_n solution
    # accepts a single long vector y_v
    def __form_rhs(self, y_v):
        rhs = []
        y_a = self.__cut(y_v)
        for j in range(0,self.s): #running for all stages
            c_l = self.c[j]
            k = 0
            res = self._u #y_n
            for y_k in y_a: #running for all K-s for each stage
                res = res + self._dt*self.A[j][k]*self.f(self.t+c_l*self._dt, y_k, self.f_params)
                k = k + 1
            rhs.append(res - y_a[j]) 
        rhs_v = self.__glue(rhs)
        return(rhs_v)
    
    def __newton(self, u_in):
        y_a = [] 
        for j in range(0, self.s):
            c_l = self.c[j]
            y_approx = u_in + c_l*self._dt*self.f(self.t+c_l*self._dt, u_in, self.f_params)
            y_a.append(y_approx)
        y_v = self.__glue(y_a)
        b_v = self.__form_rhs(y_v)
        self.__newtin_initial_iteration = True #set status for freeze operator action
        norm_newton = linalg.norm(b_v)
        iter_newton = 0
        while norm_newton>self.eps_newton and iter_newton<self.iters_newton:
            iter_newton = iter_newton + 1
            wight = 1.0
            self.__set_operator(y_v)
            dy_v = self.__solve_linear_system(b_v)
            y_v_1 = y_v + wight*dy_v
            b_v = self.__form_rhs(y_v_1)
            norm_local = linalg.norm(b_v)
            if iter_newton%self.__newton_rebuild_jacobian_frequency == 0:
                self.__newtin_initial_iteration = True #rebuild jacobian
                
            while norm_local>norm_newton and wight>1.0e-10:
                wight = wight*0.5
                y_v_1 = y_v + wight*dy_v
                b_v = self.__form_rhs(y_v_1)
                norm_local = linalg.norm(b_v)
            if wight<=1.0e-10:
                status = False
                break
            norm_newton = norm_local
            y_v = y_v_1

        if  norm_newton>self.eps_newton:
            status = False
        else:
            status = True
        self.__newton_latest_converged_norm = norm_newton
        return(y_v, status)

    # accepts a single long vector y_v
    def __form_solutoin_b(self, y_v):
        update = zeros(self.n_vec)
        y_a = self.__cut(y_v)
        for j in range(0,self.s):
            c_l = self.c[j]
            update = update + self._dt*multiply(self.b[j], self.f(self.t+c_l*self._dt, y_a[j], self.f_params))
        y_l = self._u + update
        return(y_l)
    
    def integrate(self, t, step=False, relax=False):
        self._dt =  t - self.t
        self._u1 = self._u
        u1_v, status = self.__newton(self._u1)
        if status or not self.skip_nonconverged_step:
            self._u = self.__form_solutoin_b(u1_v)
            self.t = t
            if not status:
                print('Step failed to converge with Newton norm:', self.__newton_latest_converged_norm)
        else:
            print('Step failed to converge. Try to decrease a timestep, no solutoin is produced')
        return(status)