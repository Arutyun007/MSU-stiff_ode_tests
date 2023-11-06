import numpy as np


class weno_interpolation(object):
    def __init__(self, scheme):
        super(weno_interpolation, self).__init__()
        self.order = scheme
        self.stride = int((self.order+1)/2)
        self.weno_epsilon = 1.0e-5
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
