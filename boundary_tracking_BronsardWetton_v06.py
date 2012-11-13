from __future__ import division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import cPickle
from scipy import interpolate
import grain_growth_cython as grain_growth
import voronoi


n = 8 # number of interior points per boundary in initial graph
t = 0.0
k = 0.0003 # time-step
C = [0.0,0.0]  # initial value only
h = 1.0/n   # grid spacing parameter

film = grain_growth.ThinFilm()
film.set_timestep(k)
film.set_resolution(h)

""" Generate the initial conditions using a linear fit to the points of 
    a Voronoi diagram.
    """
successful = False
while not successful:
    try:
        film.initialize_graph(100)
        successful = True
    except ValueError:
        """ Triple point angles were too far off to calculate. """
        pass

if 'time' in sys.argv:
    time_end = float(sys.argv[sys.argv.index('time') + 1])
else:
    time_end = 0.5

fig = plt.figure(1)
ax = fig.add_subplot(111)
def printplot(i):
    ax.cla()
    markersize=4.0
    
    segments = film.get_plot_lines()
    lines = mpl.collections.LineCollection(segments, color='0.5')
    ax.add_collection(lines)
    ax.set_aspect('equal')
    ax.set_xlim(-1.1,1.1)
    ax.set_ylim(-1.1,1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.canvas.draw()
    if 'movie' in sys.argv:
        fname = 'tmp/sim_{0:0>4}.png'.format(i)
        plt.savefig(fname, dpi=100)

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

def make_movie():
    command = ('mencoder', 'mf://tmp/*.png', '-mf', 'fps=25', '-o', 'output.avi', '-ovc', 'lavc', '-lavcopts', 'vcodec=mpeg4')
    import subprocess
    subprocess.check_call(command)

if 'movie' in sys.argv:
    make_movie()

plt.show()
fig.canvas.draw()


