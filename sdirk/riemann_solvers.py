import numpy as np


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

    def __lf(self, l, r):
        c = np.abs(self.__flux_derivative(l))
        return 0.5*(self.__flux(l) + self.__flux(r) ) - c*(r-l)


    # side flux
    def __calc_flux_side(self, l, r): 
        if self.__solver == "upwind":
            return self.__upwind(l,r)
        if self.__solver == "LF":
            return self.__lf(l,r)
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
