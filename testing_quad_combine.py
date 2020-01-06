"""Want to combine multiple quadratic approximations of a function
   What is the best way to do it?"""

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.exp(x)

def f1(x):
    return np.exp(x)

def f2(x):
    return np.exp(x)

def f3(x):
    return np.exp(x)

def quad(x,a):
    return f(a) + f1(a)*(x-a) + f2(a)*(x-a)**2/2.

def w(x,a,b):
    w1 = 1./(x-a)**2
    w2 = 1./(x-b)**2
    w = w1+w2
    return w1/w, w2/w

def double_quad(x,a,b):
    w1,w2 = w(x,a,b)
    print(w1)
    print(w2)
    return w1*quad(x,a) + w2*quad(x,b)

#def error(x):
#    """Error bound on second-order Taylor approx"""

x = np.linspace(-5,5,100)
y = np.exp(x)
ya1 = quad(x,-2)
ya2 = quad(x,2)
yac = double_quad(x,0,2)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,y,label='true')
ax.plot(x,ya1,label='quad approx 1')
ax.plot(x,ya2,label='quad approx 2')
ax.plot(x,yac,label='comb. quad approx')
ax.legend()
plt.show()


