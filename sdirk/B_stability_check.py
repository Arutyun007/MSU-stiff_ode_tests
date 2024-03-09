import numpy as np

def check(A, b):
    try:
        iA = np.linalg.inv(A)
        B = np.diag(b)
        AT = np.transpose(A)
        iAT = np.linalg.inv(AT)
        BB = np.outer(b,b)
        M = B.dot(A)+AT.dot(B) - BB
        Q = B.dot(iA)+iAT.dot(B)-iAT.dot(BB).dot(iA)
        eB = np.linalg.eigvals(B)
        eM = np.linalg.eigvals(M)
        eQ = np.linalg.eigvals(Q)
        min_eB = np.min(np.real(eB))
        min_eM = np.min(np.real(eM))
        min_eQ = np.min(np.real(eQ))
        at_lease_one = False
        if min_eB >= 0 and min_eM >= 0:
            print("the method is algebraically stable.")
            at_lease_one = True
        if min_eB >= 0 and min_eQ >= 0:
            print("the method is B-stable.")
            at_lease_one = True
        if not at_lease_one:
            raise
    except:
        print("the method is NOT algebraically or B-stable.")