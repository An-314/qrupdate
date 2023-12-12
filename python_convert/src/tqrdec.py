import numpy as np
import dqrdec

a=np.mat([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
print(a)

q,r=np.linalg.qr(a)
print("factorizing")
print(q)
print(r)
print(np.dot(q,r))

q,r=dqrdec.dqrdec(4,3,3,q,r,2)
print("updating")
print(q)
print(r)
print(np.dot(q,r))