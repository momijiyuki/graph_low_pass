import numpy as np
l = [[ 3, -1, -1,  0, -1],
     [-1,  3, -1,  0, -1],
     [-1, -1,  3, -1,  0],
     [ 0,  0, -1,  2, -1],
     [-1, -1,  0, -1,  3]]

val, vec = np.linalg.eig(l)

vec= vec[:, np.argsort(val)]
val.sort()

print(val)
for i in vec:
    print(i)

print(vec @ np.diag(val) @ vec.T)
