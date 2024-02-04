import numpy as np
import dqrder

a=np.mat([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
print(a)

q,r=np.linalg.qr(a, mode = "complete")

q,r=dqrder.dqrder(4,3,q,r,2)
print("updating")
print(q)
print(r)
print(np.dot(q,r))