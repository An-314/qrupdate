import numpy as np
from scipy import linalg
import dch1dn

a=np.mat([[1,1,1,1],[1,2,3,4],[1,3,6,10],[1,4,10,21]])
R=linalg.cholesky(a)

u=np.mat([[0],[0],[0],[1]])
print(a - np.dot(u,u.T))

R, info =dch1dn.dch1dn(4,R,u)
print(np.dot(R.T,R))

print(linalg.cholesky(a - np.dot(u,u.T)))
print(R)

a=np.mat([[10,8,6],[8,17,17],[6,17,44]])
R=linalg.cholesky(a)

u=np.mat([[1],[2],[3]])
print(a - np.dot(u,u.T))

R, info=dch1dn.dch1dn(3,R,u)
print(np.dot(R.T,R))

print(linalg.cholesky(a - np.dot(u,u.T)))
print(R)

