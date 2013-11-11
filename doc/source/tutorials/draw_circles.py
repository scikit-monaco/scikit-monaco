
import matplotlib.pyplot as plt
import numpy as np

r1 = 2.
r2 = 3.

thetas = np.linspace(0.,2.*np.pi,2000)
xs = np.linspace(-3.,3.,2000)

fig = plt.figure(figsize=(6,6))

# inner circle
plt.plot(r1*np.cos(thetas),r1*np.sin(thetas),"k--")

# outer circle
plt.plot(r2*np.cos(thetas),r2*np.sin(thetas),"k--")
ys_below = -np.sqrt(9.-xs**2)

# box
plt.hlines(-2.,1.,5.,linestyle="--")
plt.vlines(1.,-2.,5.,linestyle="--")

xmask = xs > 1.
plt.fill_between(xs,np.sqrt(9.-xs**2),np.where(xs<2.,np.sqrt(4.-xs**2),0.),where=xmask,
        edgecolor="red",facecolor="red")
plt.fill_between(xs,np.where(ys_below>-2.,ys_below,-2.),np.where(xs<2.,-np.sqrt(4.-xs**2),0.),where=np.logical_and(xmask,1.),
        edgecolor="red",facecolor="red")

plt.xlabel("$x$",fontsize=20)
plt.ylabel("$y$",fontsize=20)

plt.xlim(-3.5,3.5)
plt.ylim(-3.5,3.5)

plt.annotate("$\Omega$",xy=(2.4,-0.1),fontsize=20)

plt.savefig("rings.png",bbox_inches="tight")
plt.savefig("rings.pdf",bbox_inches="tight")
