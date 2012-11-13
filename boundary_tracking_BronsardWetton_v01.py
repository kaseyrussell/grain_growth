from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

"""Try to copy fig 4 from Bronsard & Wetton
    "A Numerical Method for Tracking Curve Networks Moving with Curvature Motion"
    Journal of Computational Physics vol. 120, pp. 66 (1995)
    """

""" This script just plots the "before" picture. Later versions
    try to actually simulate boundary motion. """


sigma = np.linspace(0,1,20)
boundary1 = np.array([1-sigma, np.sin(np.pi*sigma)**2/4])
boundary2 = np.array([-1/2*(1-sigma), np.sqrt(3)/2*(1-sigma)])
boundary3 = np.array([-1/2*(1-sigma), -np.sqrt(3)/2*(1-sigma)])

s = np.linspace(0,2*np.pi, 100)
domain_boundary = np.array([np.cos(s), np.sin(s)])

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.cla()
ax.plot( boundary1[0], boundary1[1], '-' )
ax.plot( boundary2[0], boundary2[1], '-' )
ax.plot( boundary3[0], boundary3[1], '-' )
ax.plot( domain_boundary[0], domain_boundary[1], '-k' )
ax.set_aspect('equal')
ax.set_xlim(-1.1,1.1)
ax.set_ylim(-1.1,1.1)
plt.show()

