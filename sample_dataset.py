import random 
import numpy as np
l1=[1,1,-1,-1]
l2=[1,-1,1,-1]
r=[0,1,2,3]

def generate(samples):
    X=[]
    Y=[]
    for i in range(samples):
        for j,k,l in zip(l1,l2,r):
            x=np.random.randint(1,10000)*j
            y=np.random.randint(1,10000)*k
            X.append([x,y])
            Y.append(l)
    return X,Y

# X,Y=generate(10)
# print(X)
# print(Y)
