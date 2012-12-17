from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys
import cPickle
from scipy import interpolate
import pyximport
pyximport.install(setup_args = {'options' :
                                {'build_ext' :
                                 {'libraries' : 'lapack',
                                  'include_dirs' : np.get_include(),
                                  }}})

import grain_growth_cython as grain_growth
#import grain_growth
#reload(grain_growth)


"""Try to copy fig 4 from Bronsard & Wetton
    "A Numerical Method for Tracking Curve Networks Moving with Curvature Motion"
    Journal of Computational Physics vol. 120, pp. 66 (1995)
    
    v05 tries for the six-node network example (Fig. 9)
    """

""" Generate the initial conditions using a spline fit to a few discrete points
    (four per boundary). B&W don't say exactly what points they use, so don't
    expect exact agreement with their simulation.
    The 'head' of each boundary will lie closer to the origin, and again we'll
    be using a unit circle DomainBoundary with Neumann boundary conditions.
    """
n = 16 # number of interior points per boundary in initial graph
def make_boundary(b):
    """ interpolate boundary b and return 
        a Boundary instance
        the interpolation is done over an array of
        length n+2, which includes node points;
        these are stripped before this is 
        passed to create a Boundary instance """
    f = interpolate.UnivariateSpline(b[0], b[1], s=0)
    xf = np.linspace(b[0][0], b[0][-1], n+2)
    yf = f(xf)
    return grain_growth.Boundary(np.array([xf[1:-1],yf[1:-1]]))
    
def make_special_boundary(b):
    """ boundaries 1-2 and 1-6 do not lie on a circle of radius 0.5
        (they squish out the x-axis to hit 0.6), so we will just
        linearly interpolate them. """
    f = interpolate.interp1d(b[0], b[1])
    xf = np.linspace(b[0][0], b[0][1], n+2)
    #print b
    #print xf
    yf = f(xf)
    return grain_growth.Boundary(np.array([xf[1:-1],yf[1:-1]]))
    

r = np.array([0.5,0.6,0.8,1.0])
rrev = r[::-1] # reversed! (need b/c interpolation must have increasing x)
delta = np.pi/12 # perturbation
dx = np.array([0,-0.02,-0.03,0]) # perturbation
a2 = np.pi/3
a3 = 2*np.pi/3
a4 = np.pi
a5 = -2*np.pi/3
a6 = -np.pi/3
b1 = np.array([[0.6,0.65,0.7,np.cos(delta)],[0,0,0,np.sin(delta)]])
b2 = np.array([r*np.cos(a2), r*np.sin(a2)])
b3 = np.array([rrev*np.cos(a3), rrev*np.sin(a3)])
b4 = np.array([rrev*np.cos(a4), rrev*np.sin(a4)])
b5 = np.array([rrev*np.cos(a5), rrev*np.sin(a5)])
b6 = np.array([r*np.cos(a6)+dx, r*np.sin(a6)])

""" Most connector boundaries will lie on a circle of radius 0.5 """
theta = np.array([0.0, np.pi/36, np.pi/18, np.pi/9, np.pi/6, 2*np.pi/9, 5*np.pi/18, 11*np.pi/36, np.pi/3])
thetarev = theta[::-1]
rc  = 0.5
b32 = rc*np.array([np.cos(thetarev+a2), np.sin(thetarev+a2)])
b43 = rc*np.array([np.cos(thetarev+a3), np.sin(thetarev+a3)])
b45 = rc*np.array([np.cos(theta+a4), np.sin(theta+a4)])
b56 = rc*np.array([np.cos(theta+a5), np.sin(theta+a5)])

""" boundaries 1-2 and 1-6 do not lie on a circle of radius 0.5
    (they squish out the x-axis to hit 0.6), so we will just
    linearly interpolate them. They both run toward b1 because
    x must be increasing for interp1d. """
b21 = np.array([[b2[0][0],b1[0][0]], [b2[1][0],b1[1][0]]])
b61 = np.array([[b6[0][0],b1[0][0]], [b6[1][0],b1[1][0]]])


""" Convert arrays to Boundary instances: """
B1, B2, B3, B4, B5, B6 = [make_boundary(b) for b in [b1,b2,b3,b4,b5,b6]]
B32, B43, B45, B56     = [make_boundary(b) for b in [b32,b43,b45,b56]]
B21, B61               = [make_special_boundary(b) for b in [b21, b61]]

""" Manual override to force points closer together at triple point w/ B4: """
B43.manual_override(1,1,0.01)
B45.manual_override(1,1,-0.01)


nodes = []

""" Set appropriate ends to DomainBoundary:"""
for B in [B1,B2,B6]:
    nodes.append(grain_growth.Node(node_type=grain_growth.type_domain_boundary()))
    nodes[-1].add_single_boundary( dict(b=B, end=grain_growth.get_tail_end()) )

for B in [B3,B4,B5]:
    nodes.append(grain_growth.Node(node_type=grain_growth.type_domain_boundary()))
    nodes[-1].add_single_boundary( dict(b=B, end=grain_growth.get_head_end()) )


""" Make triple-points """
nodes.append(grain_growth.Node(node_type=grain_growth.type_triple_point()))
nodes[-1].add_boundaries(
    dict(b=B1, end=grain_growth.get_head_end()),
    dict(b=B21, end=grain_growth.get_tail_end()),
    dict(b=B61, end=grain_growth.get_tail_end()))

nodes.append(grain_growth.Node(node_type=grain_growth.type_triple_point()))
nodes[-1].add_boundaries(
    dict(b=B2, end=grain_growth.get_head_end()),
    dict(b=B32, end=grain_growth.get_tail_end()),
    dict(b=B21, end=grain_growth.get_head_end()))

nodes.append(grain_growth.Node(node_type=grain_growth.type_triple_point()))
nodes[-1].add_boundaries(
    dict(b=B3, end=grain_growth.get_tail_end()),
    dict(b=B43, end=grain_growth.get_tail_end()),
    dict(b=B32, end=grain_growth.get_head_end()))

nodes.append(grain_growth.Node(node_type=grain_growth.type_triple_point()))
nodes[-1].add_boundaries(
    dict(b=B4, end=grain_growth.get_tail_end()),
    dict(b=B45, end=grain_growth.get_head_end()),
    dict(b=B43, end=grain_growth.get_head_end()))

nodes.append(grain_growth.Node(node_type=grain_growth.type_triple_point()))
nodes[-1].add_boundaries(
    dict(b=B5, end=grain_growth.get_tail_end()),
    dict(b=B56, end=grain_growth.get_head_end()),
    dict(b=B45, end=grain_growth.get_tail_end()))

nodes.append(grain_growth.Node(node_type=grain_growth.type_triple_point()))
nodes[-1].add_boundaries(
    dict(b=B6, end=grain_growth.get_head_end()),
    dict(b=B61, end=grain_growth.get_head_end()),
    dict(b=B56, end=grain_growth.get_tail_end()))


boundary_objects = [B1,B2,B3,B4,B5,B6,B21,B32,B43,B45,B56,B61]
   

t = 0.0
k = 0.0003 # time-step
C = [0.0,0.0]  # initial value only
h = 1.0/n   # grid spacing parameter

film = grain_growth.ThinFilm( boundary_objects, nodes )

film.set_timestep(k)
film.set_resolution(h)
for b in boundary_objects:
    b.set_resolution(h)

if 'time' in sys.argv:
    time_end = float(sys.argv[sys.argv.index('time') + 1])
else:
    time_end = 1.5 #0.5

phi = np.linspace(0,2*np.pi, 100)
domain_boundary = np.array([np.cos(phi), np.sin(phi)])
if 'noprofile' in sys.argv:
    fig = plt.figure(1)
    ax2 = fig.add_subplot(111)
    def printplot(i):
        ax2.cla()
        markersize=4.0
        for b in boundary_objects:
            b.convert_to_columns()
            d = b.get_boundary()
            ax2.plot( d[0], d[1], '-k' )
            b.convert_to_points()
        ax2.plot( domain_boundary[0], domain_boundary[1], '-k' )
        ax2.set_aspect('equal')
        ax2.set_xlim(-1.1,1.1)
        ax2.set_ylim(-1.1,1.1)
        ax2.set_xticks([])
        ax2.set_yticks([])
        fig.canvas.draw()
        if 'movie' in sys.argv:
            fname = 'tmp/sim_{0:0>4}.png'.format(i)
            plt.savefig(fname, dpi=100)

for b in boundary_objects:
    b.extrapolate()

dt_movie = 0.005  # increment 
t_movie  = 0.0   # acquire frame if t>t_movie
i        = 0     # frame number
if 'noprofile' in sys.argv:
    while t <= time_end: #1.0:
        film.update_nodes()
        film.update_grain_boundaries()
        if 'movie' in sys.argv:
            if t > t_movie:
                printplot(i)
                i += 1
                t_movie += dt_movie
        if not film.regrid():
            break
        if film.get_timestep() < 1.9e-7:
            print "Time:", t
        t += film.get_timestep()
else:
    def runloop(t,film):
        while t <= time_end: #1.0:
            film.update_nodes()
            film.update_grain_boundaries()
            if not film.regrid():
                break
            t += film.get_timestep()


    def run():
        t = 0.0
        runloop(t,film)


if 'save' in sys.argv:
    data = dict( boundaries=film.boundaries, nodes=film.nodes, t=t )
    fname = "state_time_{0:.3f}.pkl".format(t) 
    with open(fname, 'wb') as fobj:
        cPickle.dump(data, fobj)

def f(j):
    for i in [3,4]:
        n = boundary_objects[j].get_termination(i)
        print n, n.get_type()

def make_movie():
    command = ('mencoder', 'mf://tmp/*.png', '-mf', 'fps=25', '-o', 'output.avi', '-ovc', 'lavc', '-lavcopts', 'vcodec=mpeg4')
    import subprocess
    subprocess.check_call(command)

if 'movie' in sys.argv:
    make_movie()

if 'noprofile' in sys.argv:
    plt.show()
    fig.canvas.draw()


