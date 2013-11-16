
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform,exponential


def g((x,y)):
    """
    The function over which we want to integrate.
    
    This function must take a single argument, which can
    be a list or a tuple or a numpy array. The argument
    defines a point.
    """
    r = np.sqrt(x**2 + y**2)
    if r >= 2. and r <= 3. and x >= 1. and y >= -2.:
        # (x,y) in correct volume
        return True
    else:
        return False

def distribution(size):
    xs = uniform(size=size,low=1.0,high=3.0)
    ys = exponential(size=size,scale=1.0)-2.
    return np.array((xs,ys)).T

npoints = 200
points = distribution(npoints)

x_in,y_in = [],[]
x_out,y_out = [],[]

for (x,y) in points:
    if g((x,y)):
        x_in.append(x)
        y_in.append(y)
    else:
        x_out.append(x)
        y_out.append(y)

r1 = 2.
r2 = 3.

thetas = np.linspace(0.,2.*np.pi,2000)

fig = plt.figure(figsize=(6,6))

# inner circle
plt.plot(r1*np.cos(thetas),r1*np.sin(thetas),"k--")

# outer circle
plt.plot(r2*np.cos(thetas),r2*np.sin(thetas),"k--")

# box
plt.hlines((-2.,3.),1.,3.,lw=2,color="blue")
plt.vlines((1.,3.),-2.,3.,lw=2,color="blue")

plt.scatter(x_in,y_in,color="red")
plt.scatter(x_out,y_out,color="black")

plt.xlim(-3.5,3.5)
plt.ylim(-3.5,3.5)

plt.xlabel("$x$",fontsize=20)
plt.ylabel("$y$",fontsize=20)

#plt.show()
plt.savefig("import_plot.pdf",bbox_inches="tight")
plt.savefig("import_plot.png",bbox_inches="tight")

