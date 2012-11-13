from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys
import cPickle
import grain_growth
reload(grain_growth)

"""Try to copy fig 4 from Bronsard & Wetton
    "A Numerical Method for Tracking Curve Networks Moving with Curvature Motion"
    Journal of Computational Physics vol. 120, pp. 66 (1995)
    
    v04 separates model into module grain_growth.py
    """


#sigma = np.linspace(0.05,0.9,16)
sigma = np.linspace(0.01,0.99,16)
boundary1 = np.array([1-sigma, np.sin(np.pi*sigma)**2/4])
boundary2 = np.array([-1/2*(1-sigma), np.sqrt(3)/2*(1-sigma)])
boundary3 = np.array([-1/2*(1-sigma), -np.sqrt(3)/2*(1-sigma)])
boundaries = [boundary1, boundary2, boundary3]

s = np.linspace(0,2*np.pi, 100)
domain_boundary = np.array([np.cos(s), np.sin(s)])

""" Plot the initial conditions: """
fig = plt.figure(1)
ax1 = fig.add_subplot(121)
ax1.cla()
ax1.plot( boundary1[0], boundary1[1], '-' )
ax1.plot( boundary2[0], boundary2[1], '-' )
ax1.plot( boundary3[0], boundary3[1], '-' )
ax1.plot( domain_boundary[0], domain_boundary[1], '-k' )
ax1.set_aspect('equal')
ax1.set_xlim(-1.1,1.1)
ax1.set_ylim(-1.1,1.1)
ax1.set_xticks([])
ax1.set_yticks([])


t = 0.0
k = 0.0003 # time-step
C = [0,0]  # initial value only
h = 1/16   # grid spacing parameter

if 'load' in sys.argv:
    fname = sys.argv[sys.argv.index('load')+1]
    with open(fname, 'rb') as fobj:
        data = cPickle.load(fobj)
        
    t = data['t']
    boundary_objects = data['boundaries']
    triple_points = data['triple_points']
    film = grain_growth.ThinFilm( boundary_objects, triple_points )
else:
    boundary_objects = [grain_growth.Boundary(b) for b in boundaries]
    triple_points = [grain_growth.TriplePoint( *[dict(b=b, end='tail') for b in boundary_objects] )]
    for b in boundary_objects:
        b.set_termination( end='head', termination=grain_growth.DomainBoundary('head') )
    film = grain_growth.ThinFilm( boundary_objects, triple_points )

film.set_timestep(k)
film.set_resolution(h)
for b in boundary_objects:
    b.set_resolution(h)

if 'time' in sys.argv:
    time_end = float(sys.argv[sys.argv.index('time') + 1])
else:
    time_end = 1.35

while t <= time_end: #1.0:
    film.update_triple_points()
    film.update_grain_boundaries()
    if not film.regrid():
        break
    t += film.get_timestep()


if 'save' in sys.argv:
    data = dict( boundaries=film.boundaries, triple_points=film.triple_points, t=t )
    fname = "state_time_{0:.3f}.pkl".format(t) 
    with open(fname, 'wb') as fobj:
        cPickle.dump(data, fobj)

""" Plot the result at t=1.0: """
ax2 = fig.add_subplot(122)
ax2.cla()
markersize=4.0
markers = ['-o','-d','-x']
for b,marker in zip(boundary_objects, markers):
    print len(b.b)
    b.convert_to_columns()
    d = b.b
    ax2.plot( d[0], d[1], marker, ms=markersize )
ax2.plot( domain_boundary[0], domain_boundary[1], '-k' )
ax2.set_aspect('equal')
ax2.set_xlim(-1.1,1.1)
ax2.set_ylim(-1.1,1.1)
ax2.set_xticks([])
ax2.set_yticks([])

# take it back to point-by-point format in case we want to poke around in the data:
for b in boundary_objects:
    b.convert_to_points()


plt.show()
fig.canvas.draw()
