
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyximport
pyximport.install(setup_args = {'options' :
                                {'build_ext' :
                                 {'libraries' : 'lapack',
                                  'include_dirs' : np.get_include(),
                                  }}})

import voronoi_cython as voronoi

if False:
    """ Plot periodic Voronoi diagram made using tiles method
    and overlay the center tile """
    segments    = voronoi.voronoi(100, periodic=True)
    center_tile = voronoi.periodic_voronoi(segments=segments)
    
    """ Plot all segments: """
    fig = plt.figure(1)
    ax  = fig.add_subplot(111)
    ax.cla()
    lines = mpl.collections.LineCollection(segments, color='0.75')
    ax.add_collection(lines)

    """ Overlay those that cross into the unit square centered on (0,0): """
    goodlines = mpl.collections.LineCollection(center_tile, color='0.15')
    ax.add_collection(goodlines)

    """ Add a box from (0,0) to (1,1) to show boundaries """
    l,u = -0.5, 0.5
    box = [[(l,l),(l,u)], [(l,l),(u,l)], [(u,u),(l,u)], [(u,u),(u,l)]]
    lines2 = mpl.collections.LineCollection(box, color='0.25')
    ax.add_collection(lines2)

    #ax.axis([0,1,0,1])
    ax.axis([-1.6,1.6,-1.6,1.6])
    ax.set_aspect('equal')

    plt.show()
    fig.canvas.draw()
else:
    """ Same, but do it with node ID turned on and label nodes on
    the center tile. """
    segments    = voronoi.voronoi(100, periodic=True, include_nodes=True)
    center_tile = voronoi.periodic_voronoi(segments=segments, include_nodes=True)
    
    """ Plot all segments: """
    fig = plt.figure(1)
    ax  = fig.add_subplot(111)
    ax.cla()
    line_segments = [s['points'] for s in segments]
    lines = mpl.collections.LineCollection(line_segments, color='0.75')
    ax.add_collection(lines)

    """ Overlay those that cross into the unit square centered on (0,0): """
    center_segments = [s['points'] for s in center_tile]
    goodlines = mpl.collections.LineCollection(center_segments, color='0.15')
    ax.add_collection(goodlines)

    nodelist      = []
    for seg in center_tile:
        if seg['head'] not in nodelist:
            nodelist.append(seg['head'])
            x,y = seg['points'][0]
            ax.text(x,y,str(seg['head']))
        if seg['tail'] not in nodelist:
            nodelist.append(seg['tail'])
            x,y = seg['points'][1]
            ax.text(x,y,str(seg['tail']))

    """ Add a box from (0,0) to (1,1) to show boundaries """
    l,u = -0.5, 0.5
    box = [[(l,l),(l,u)], [(l,l),(u,l)], [(u,u),(l,u)], [(u,u),(u,l)]]
    lines2 = mpl.collections.LineCollection(box, color='0.25')
    ax.add_collection(lines2)

    #ax.axis([0,1,0,1])
    ax.axis([-1.6,1.6,-1.6,1.6])
    ax.set_aspect('equal')

    plt.show()
    fig.canvas.draw()

def inspect(n):
    for s in center_tile:
        if s['head'] == n or s['tail'] == n:
            print s


