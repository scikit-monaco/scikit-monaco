
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def g(x,y):
    """
    The function over which we want to integrate.
    
    This function must take a single argument, which can
    be a list or a tuple or a numpy array. The argument
    defines a point.
    """
    r = np.sqrt(x**2 + y**2)
    return np.where((r > 2.) & (r < 3.) & (x>1.) & (y>-2.),
            np.exp(-1.0*(y+2)),0.)

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

xs = np.linspace(1.,3.,200)
ys = np.linspace(-2.,3.,200)
xs, ys = np.meshgrid(xs,ys)
zs = g(xs,ys)

#fig = plt.figure()
#ax = fig.add_subplot(111,projection="3d")

#ax.set_xlabel("x")
#ax.set_ylabel("y")

#ax.plot_surface(xs,ys,zs,rstride=2,cstride=2,cmap=cm.coolwarm,linewidth=0)
#ax.plot_wireframe(xs,ys,zs,rstride=3,cstride=3)

plt.contourf(xs,ys,zs,30,cmap=cm.Reds)

plt.xlabel("$x$",fontsize=20)
plt.ylabel("$y$",fontsize=20)

plt.savefig("import_integrand.pdf",bbox_inches="tight")
plt.savefig("import_integrand.png",bbox_inches="tight")
#plt.show()
