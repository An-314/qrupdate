import numpy as np
import dqrinc

print("不完全分解")
a=np.mat([[1,2],[5,6],[9,10],[13,14]])
print(a)

q,r=np.linalg.qr(a)

x = np.ones(4)
q,r=dqrinc.dqrinc(4,2,q,r,2,x)
print("updating")
print(q)
print(r)
print(np.dot(q,r))

print("完全分解")
a=np.mat([[1,2],[4,5],[7,8],[10,11]])
print(a)

q,r=np.linalg.qr(a,mode='complete')

x = np.ones(4)
q,r=dqrinc.dqrinc(4,2,q,r,2,x)
print("updating")
print(q)
print(r)
print(np.dot(q,r))
