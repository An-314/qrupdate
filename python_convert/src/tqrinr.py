import numpy as np
import dqrinr

print("完全分解")
a=np.mat([[1,2],[4,5],[7,8],[10,11]])
print(a)

q,r=np.linalg.qr(a,mode='complete')

x = np.ones([1,2])
q,r=dqrinr.dqrinr(4,2,q,r,2,x)
print("updating")
print(q)
print(r)
print(np.dot(q,r))