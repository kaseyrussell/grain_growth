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
def runtest():
    film.initialize_graph(50)

if __name__ == '__main__':
    film.initialize_graph(50)


