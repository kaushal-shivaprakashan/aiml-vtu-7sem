import numpy as np
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
X = X/np.amax(X,axis=0)
y = y/100

def sigmoid (x):
    return 1/(1 + np.exp(-x))


e,lr,iln,hln,oln=7000,0.1,2,3,1

wh=np.random.uniform(size=(iln,hln))
bh=np.random.uniform(size=(1,hln))
wout=np.random.uniform(size=(hln,oln))
bout=np.random.uniform(size=(1,oln))

for i in range(e):
    hla = sigmoid(np.dot(X,wh)+bh)
    op = sigmoid(np.dot(hla,wout)+bout)
    
    
print("Input: \n",str(X))
print("Actual Output: \n",str(y))
print("Predicted Output: \n" ,op)

#######op
Input: 
 [[0.66666667 1.        ]
 [0.33333333 0.55555556]
 [1.         0.66666667]]
Actual Output: 
 [[0.92]
 [0.86]
 [0.89]]
Predicted Output: 
 [[0.82276992]
 [0.79714673]
 [0.82248118]]
