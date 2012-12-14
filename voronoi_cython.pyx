""" Generate a 2D Voronoi diagram from a random series of points.
    Can make periodic (quasi) using a Robert Kern idea of tiling
    the group of points by hand on a 3x3 matrix, creating the diagram, and 
    then taking the center tile

Based on code from stack overflow at 
http://stackoverflow.com/questions/5596317/getting-the-circumcentres-from-a-delaunay-triangulation-generated-using-matplotl
and
http://stackoverflow.com/questions/10650645/python-calculate-voronoi-tesselation-from-scipys-delaunay-triangulation-in-3d
and Robert Kern's idea at:
http://matplotlib.1069221.n5.nabble.com/matplotlib-delauney-with-periodic-boundary-conditions-td26196.html

   Copyright 2012 Kasey Russell ( email: krussell _at_ post.harvard.edu )
   Distributed under the GNU General Public License
"""
from __future__ import division

import numpy as np
cimport numpy as np
import cython
from cpython cimport bool
import matplotlib as mpl
import matplotlib.pyplot as plt

def voronoi(Py_ssize_t n, bool periodic=False, bool include_nodes=False):
    ''' Return a list of line segments describing the 
    voronoi diagram of n random points.
    If periodic==True, then the diagram will be of 
    a tiled 3x3 array of points centered
    on (0,0), otherwise the points run [0,1) in both x-y. 
    If include_nodes==True, then the returned list 
    will be of dictionaries:
    [dict(points=(x,y), head=node, tail=node), ...]
    to identify the nodes to which the segment is connected. '''
    
    cdef:
        Py_ssize_t i, j, k, ncenters
        np.ndarray[dtype=np.float64_t, ndim=2] P
        np.ndarray[dtype=np.float64_t, ndim=2] C
        np.ndarray[dtype=np.float64_t, ndim=2] tiled_P
        np.ndarray[dtype=np.float64_t, ndim=1] X
        np.ndarray[dtype=np.float64_t, ndim=1] Y
        np.ndarray[dtype=int, ndim=1] neighbors
        double shiftx, shifty
        
    P = np.random.random((n,2))
    if periodic:
        tiled_P = np.zeros((n*9,2))
        for i,shiftx in enumerate([-1.5, -0.5, 0.5]):
            for j,shifty in enumerate([-1.5, -0.5, 0.5]):
                tiled_P[(i+j*3)*n:(i+j*3+1)*n,0] = P[:,0] + shiftx
                tiled_P[(i+j*3)*n:(i+j*3+1)*n,1] = P[:,1] + shifty
        P = tiled_P
        
    D   = mpl.delaunay.triangulate.Triangulation(P[:,0],P[:,1])
    C   = D.circumcenters
    X,Y = C[:,0], C[:,1]
    ncenters  = len(C)
    
    segments     = []
    indx_counted = []
    node         = 0
    print "number of centers:", ncenters
    for i in xrange(ncenters):
        neighbors = D.triangle_neighbors[i]
        for k in neighbors:
            """ The triangle neighbors are the triangles that share edges
            with a given triangle. The segments of the Voronoi diagram 
            connect the centers of circumscribed circles in the neighboring
            triangles. """
            if k != -1:
                if include_nodes:
                    """ Avoid double-counting """
                    if False:
                        already_counted = False
                        for s in segments:
                            if s['points'] == [(X[k],Y[k]), (X[i],Y[i])]:
                                s['tail'] = node
                                already_counted = True
                                break

                        if not already_counted:
                            segments.append(dict(points=[(X[i],Y[i]), (X[k],Y[k])], head=node))
                    else:
                        try:
                            n = indx_counted.index((k,i))
                            """ The segment is already in the list; make the 
                            current node the tail node. """
                            segments[n]['tail'] = node
                        except ValueError:
                            """ The segment is not already in the list, so add it and 
                            make the current node the head node. """
                            segments.append(dict(points=[(X[i],Y[i]), (X[k],Y[k])], head=node))
                            indx_counted.append((i,k))
                else:
                    segments.append([(X[i],Y[i]), (X[k],Y[k])])
        node += 1
    return segments

def periodic_voronoi(n=100, include_nodes=False, segments=None):
    """ Returns a list of line segments corresponding to
    a voronoi diagram of n random points 
    with periodic boundary conditions. If segments is None,
    a new set of segments are generated, otherwise n is ignored
    and the segments are used.
    If include_nodes==True, and segments are not supplied,
    then the returned list will be of dictionaries:
    [dict(points=(x,y), node=node), ...]
    to identify the node to which the segment is connected. 
    """

    """ voronoi(n, True) will return a list of segments corresponding to
    the tiled array of random points """
    if segments is None:
        segments = voronoi(n, True, include_nodes)

    """ Select the segments from the center tile (which should be as 
    if we had calculated the diagram on just the center tile but
    using periodic boundary conditions """
    goodsegments = []
    doubles      = []
    wrappers     = []
    for s in segments:
        extends_into_region = False
        is_double = False
        points    = s['points'] if include_nodes else s
        for i,pt in enumerate(points):
            if (abs(pt[0])<=0.5 and abs(pt[1])<=0.5):
                extends_into_region = True

        for i,pt in enumerate(points):
            if extends_into_region and (pt[0]<=-0.5 or pt[1]<=-0.5):
                """ Extends across boundary in the negative direction 
                (After wrapping the periodic B.C., there will be two identical 
                boundaries, one across in the positive direction, and one across 
                in the negative. Here we flag the one going negative to reject it."""
                is_double = True
                if include_nodes:
                    """ keep track of which node the OTHER end of this segment
                    (i.e. not the end extending out of the central region)
                    was attached to b/c this is the node that we will 
                    connect to the boundary that wraps around the simulation
                    region."""
                    node = s['tail'] if i==0 else s['head']
                    indx = 1 if i==0 else 0
                    doubles.append(dict(segment=s, node_id=node, point_out=pt, point_in=points[indx]))
                    
            if extends_into_region and (pt[0]>0.5 or pt[1]>0.5) and include_nodes:
                """ This node will get replaced with one on the 
                other side of the simulation b/c of the periodic 
                boundary conditions. """
                end = 'head' if i==0 else 'tail'
                phantom = s[end]
                # indx is the index of the segment point inside the region
                indx = 1 if i==0 else 0
                wrappers.append(dict(segment=s, node_id=phantom,
                    end=end, point_in=points[indx], point_out=pt))

        if extends_into_region and not is_double:
            goodsegments.append(s)
                
    
    if include_nodes:
        """ Fix the nodes of the segments that wrap around the
        simulation region """
        tol = 1.0e-3
        for d in doubles:
            """ find segment in wrappers that is the duplicate of d
            once periodic b.c. are imposed. """
            for w in wrappers:
                """ First check if w shares a vertex with d """
                if (abs(d['point_out'][0]+1.0 - w['point_in'][0])<tol or
                    abs(d['point_out'][1]+1.0 - w['point_in'][1])<tol):

                    """ Does it share both verticies? """
                    if (abs(d['point_in'][0]+1.0 - w['point_out'][0])<tol or
                        abs(d['point_in'][1]+1.0 - w['point_out'][1])<tol):
                    
                        #print 'replaced {0} with {1}'.format(w['node_id'], d['node_id'])
                        phantom_end = w['end']
                        w['segment'][phantom_end] = d['node_id']

    return goodsegments


def periodic_diagram_with_nodes(n):
    """ Return a periodic Voronoi tesselation of n random
    points, a list of the nodes, and a list of lists containing
    the indices of the boundaries connected to each node. """
    diagram  = periodic_voronoi(n, include_nodes=True)
    nodelist = []
    node_segments = []
    for i,seg in enumerate(diagram):
        if seg['head'] not in nodelist:
            nodelist.append(seg['head'])
            node_segments.append([i])
        else:
            indx = nodelist.index(seg['head'])
            node_segments[indx].append(i)

        if seg['tail'] not in nodelist:
            nodelist.append(seg['tail'])
            node_segments.append([i])
        else:
            indx = nodelist.index(seg['tail'])
            node_segments[indx].append(i)
        
    print "number of nodes:", len(node_segments)
    for i,n in enumerate(node_segments):
        if len(n) < 3:
            raise ValueError("too few ({0}) segments for node {1} of {2}".format(len(n), i, len(node_segments)))

    return diagram, nodelist, node_segments
    
def runtest():
    """ test of periodic tesselation for profiler """
    segments, nodelist, node_segments = periodic_diagram_with_nodes(1000)

def plot_test():
    """ Plot periodic voronoi diagram with node ID turned on and label nodes on
    the center tile. """
    segments    = voronoi(100, periodic=True, include_nodes=True)
    center_tile = periodic_voronoi(segments=segments, include_nodes=True)
    
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
        if seg['points'][0] not in nodelist:
            nodelist.append(seg['points'][0])
            x,y = seg['points'][0]
            ax.text(x,y,str(seg['head']))
        if seg['points'][1] not in nodelist:
            nodelist.append(seg['points'][1])
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


if __name__ == '__main__':

    if False:
        """ Plot periodic Voronoi diagram made using tiles method
        and overlay the center tile """
        segments    = voronoi(100, periodic=True)
        center_tile = periodic_voronoi(segments=segments)
        
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
        segments    = voronoi(100, periodic=True, include_nodes=True)
        center_tile = periodic_voronoi(segments=segments, include_nodes=True)
        
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
            if seg['points'][0] not in nodelist:
                nodelist.append(seg['points'][0])
                x,y = seg['points'][0]
                ax.text(x,y,str(seg['head']))
            if seg['points'][1] not in nodelist:
                nodelist.append(seg['points'][1])
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

