from __future__ import division
import numpy as np
import matplotlib as mpl
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
import voronoi_cython as voronoi


n = 8 # number of interior points per boundary in initial graph
t = 0.0
k = 1.0e-2 #0.00003 # time-step
C = [0.0,0.0]  # initial value only
h = 1.0/n   # grid spacing parameter

film = grain_growth.ThinFilm()
film.set_timestep(k)
film.set_resolution(h)

""" Generate the initial conditions using a linear fit to the points of 
    a Voronoi diagram.
    """
film.initialize_graph(30, load_voronoi='init_voronoi_30.pkl')
#film.initialize_graph(100, load_voronoi='init_voronoi_100.pkl')
print "YES!!!"

if 'time' in sys.argv:
    time_end = float(sys.argv[sys.argv.index('time') + 1])
else:
    time_end = 0.5

fig = plt.figure(1)
ax = fig.add_subplot(111)
def printplot(i, film):
    ax.cla()
    markersize=4.0
    
    segments = film.get_plot_lines()
    if True:
        shift        = film.get_domain_width()
        for dx in [-1.0,0.0,1.0]:
            for dy in [-1.0,0.0,1.0]:
                shiftedlines = []
                for s in segments:
                    shiftedlines.append(s+np.array((dx*shift, 0.0))+np.array((0.0,dy*shift)))
                ax.add_collection(mpl.collections.LineCollection(shiftedlines, color='0.8'))

    lines = mpl.collections.LineCollection(segments, color='0.4')
    ax.add_collection(lines)
    ax.set_aspect('equal') #, 'datalim')
    #ax.autoscale_view(True,True,True)
    ax.set_xlim(-7,7)
    ax.set_ylim(-7,7)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
           
    fig.canvas.draw()
    if 'movie' in sys.argv:
        fname = 'tmp/sim_{0:0>4}.png'.format(i)
        plt.savefig(fname, dpi=100)

dt_movie = 0.005  # increment 
t_movie  = 0.0   # acquire frame if t>t_movie
i        = 0     # frame number
if True:
    if 'noprofile' in sys.argv:
        film.update_nodes()
        while t <= time_end: #1.0:
            #film.update_nodes()
            #film.update_grain_boundaries()
            film.junction_iteration()
            if 'movie' in sys.argv:
                if t > t_movie:
                    printplot(i, film)
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
                #film.update_nodes()
                #film.update_grain_boundaries()
                film.junction_iteration()
                if not film.regrid():
                    break
                t += film.get_timestep()


        def run():
            t = 0.0
            runloop(t,film)

def s():
    film.junction_iteration()
    film.regrid()
    printplot(0,film)
    
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


