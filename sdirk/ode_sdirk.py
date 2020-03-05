#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:19:50 2020

@author: noctum
"""
#import re
#import warnings

from numpy import asarray, zeros, isscalar, linalg, identity, multiply, dot


class ode_sdirk(object):
    def __init__(self, f, jac, A, c, b, b_hat = None, use_full_newton = False):
        self.stiff = 0
        self.f = f
        self.jac = jac
        self.f_params = ()
        self.jac_params = ()
        self._y = []
        self.use_full_newton = use_full_newton
        
        """ Constants for the SDIRK method 
            It is assumed that a_j^j = gamma
        """
        self.gamma = A[0][0]
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
        self._y1 = []
        self._dt = 0.0
        self.eps_newton = 1.0e-10
        

    @property
    def y(self):
        return self._y
    def y1(self):
        return self._y1
    
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
        self._y = asarray(y)
        self.t = t
        n_vec = len(self._y)
        self.n_vec = n_vec
        self.E = identity(n_vec) # '''sets ident matrix'''
        return self
    
#    def form_rhs_method(self, stage):
    
#    def __ttt(self):
#        j_matrix = self.jac(self.t, self.y, self.jac_params);
#        a = multiply(1.0, j_matrix)
#        return(a)
        
    def __solve_linear_system(self,y_n, b, stage_number):
        c_l = self.c[stage_number]
        a1 = self._dt*multiply(self.gamma, self.jac(self.t+c_l*self._dt, y_n, self.jac_params))
        a = self.E - a1
        x = linalg.solve(a, b)
        return(x)
    def __newton(self, y_n, rhs, stage_number):
        c_l = self.c[stage_number]
        
        du = zeros(self.n_vec);
        norm_newton = 1
        iterations = 0
        while(norm_newton>self.eps_newton):
            b1 = self._dt*multiply(self.gamma, self.f(self.t+c_l*self._dt, y_n, self.f_params)) - y_n
            b = rhs + b1
            du = self.__solve_linear_system(y_n, b, stage_number)
            y_n = y_n + du
            norm_newton = linalg.norm(b)
            iterations = iterations + 1
            if iterations > 1000:
                print('Newton filed to converge with norm = ', norm_newton, '\n')
                break
        return(y_n)
        
    def __solve_linear_system_matrix(self, y_n, c_l):
        a1 = self._dt*multiply(self.gamma, self.jac(self.t+c_l*self._dt, y_n, self.jac_params))
        a = self.E - a1
        iJ = linalg.inv(a)
        return(iJ)
        
    def __apply_system_imatrix(self, iJ, b):
        x = dot(iJ, b)
        return(x)
        
    def __newton_matrix(self, y_n, rhs, stage_number):
        c_l = self.c[stage_number]        
        if stage_number == 0:
            self.iJ = self.__solve_linear_system_matrix(y_n, c_l)
        du = zeros(self.n_vec);
        norm_newton = 1    
        iterations = 0
        while(norm_newton>self.eps_newton):
            b1 = self._dt*multiply(self.gamma, self.f(self.t+c_l*self._dt, y_n, self.f_params)) - y_n
            b = rhs + b1        
            du = self.__apply_system_imatrix(self.iJ, b)
            y_n = y_n + du
            norm_newton = linalg.norm(b)
            iterations = iterations + 1
            if iterations > 1000:
                print('Newton filed to converge with norm = ', norm_newton, '\n')
                break
        return(y_n)
        
    def __form_rhs(self, stage_number):
        rhs = zeros(self.n_vec)
        for k in range(0,stage_number):
            c_l = self.c[k]
            f_l = self.f(self.t+c_l*self._dt, self.y_stages[k], self.f_params)
            rhs = rhs + self._dt*multiply(self.A[stage_number][k],f_l)
        
        return(rhs)
        
    def __form_solutoin_b(self):
        
        update = zeros(self.n_vec)
        for j in range(0,self.s):
            c_l = self.c[j]
            update = update + multiply(self.b[j], self.f(self.t+c_l*self._dt, self.y_stages[j], self.f_params))
            
        y_l = self._y + self._dt*update
        return(y_l)
    
    def integrate(self, t, step=False, relax=False):
        self._dt =  t - self.t
        self._y1 = self._y
        self.y_stages = []
        for j in range(0,self.s):
            rhs = self._y + self.__form_rhs(j)
            
            if self.use_full_newton:
                self._y1 = self.__newton(self._y1, rhs, j) # solves linsys on each iter
            else:
                self._y1 = self.__newton_matrix(self._y1, rhs, j) # applies initial inverse

            self.y_stages.append(self._y1)
            
        self._y = self.__form_solutoin_b()
        
        self.t = t
        return(self)