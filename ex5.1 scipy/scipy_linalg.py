# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 20:32:02 2022

@author: zhao
"""
import numpy as np 
from scipy import linalg

#a. define matrx
A = np.linspace(1,9,num = 9).reshape(3,3)

#b. define a vector

b = np.array([1,2,3])

#c. caluclate x in Ax = b
x1 = linalg.pinv(A) @ b

#d. check the result
print(np.allclose(b, A@x1))

#e .repeat witha matrix
B = np.random.rand(3,3)
x2 = linalg.pinv(A) @ B
#dont understand why but results colse
print(np.allclose(B, A@x2))

#f. eigon value and eigon vector
eig_V,eig_w = linalg.eig(A)

#g.calculate inverse and determinant, det is 0 not invertible
A_inv = linalg.pinv(A)
A_det = linalg.det(A)

#h. calculate the norm in different orders
A_norm_None = linalg.norm(A,ord = None)
A_norm_fro = linalg.norm(A,ord = 'fro')
A_norm_nuc = linalg.norm(A,ord = 'nuc')
A_norm_inf = linalg.norm(A,ord = np.inf)
A_norm_ninf = linalg.norm(A,ord = -np.inf)
A_norm_1 = linalg.norm(A,ord = 1)
A_norm_2 = linalg.norm(A,ord = 2)
A_norm_N1 = linalg.norm(A,ord = -1)
