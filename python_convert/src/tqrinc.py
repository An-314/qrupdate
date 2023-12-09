import numpy as np
import dqrinc

print("example1")
a=np.mat([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
print(a)

q,r=np.linalg.qr(a)
print("factorizing")
print(q)
print(r)
print(np.dot(q,r))

x = np.ones(4)
q,r=dqrinc.dqrinc(4,4,4,q,r,3,x)
print("updating")
print(q)
print(r)
print(np.dot(q,r))

print("example2")
a=np.mat([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
print(a)

q,r=np.linalg.qr(a)
print("factorizing")
print(q)
print(r)
print(np.dot(q,r))

x = np.ones(4)
q,r=dqrinc.dqrinc(4,3,3,q,r,3,x)
print("updating")
print(q)
print(r)
print(np.dot(q,r))