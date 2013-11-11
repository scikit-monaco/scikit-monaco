
import numpy as np
import matplotlib.pyplot as plt


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

npoints = 200

x_in,y_in = [],[]
x_out,y_out = [],[]

for i in range(npoints):
    x,y = np.random.ranf(2)
    x = 1.+2*x
    y = -2.+5*y
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

plt.savefig("scatter.pdf",bbox_inches="tight")
plt.savefig("scatter.png",bbox_inches="tight")

