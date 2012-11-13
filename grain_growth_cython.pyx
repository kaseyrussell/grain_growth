# encoding: utf-8
# cython: profile=True
from __future__ import division

import numpy as np
cimport numpy as np
import cython
from cpython cimport bool
import uuid
import voronoi
from scipy import interpolate

""" Implementation of boundary tracking method from Bronsard & Wetton
    "A Numerical Method for Tracking Curve Networks Moving with Curvature Motion"
    Journal of Computational Physics vol. 120, pp. 66 (1995)

    Author: Kasey J. Russell
    Harvard University SEAS
    Laboratory of Evelyn Hu
    Copyright (C) 2012 Kasey J. Russell, all rights reserved
    Released as open-source software under
    the GNU public license (GPL). This code
    is not guaranteed to work, not guaranteed
    against anything, use at your own risk, etc.
    """

cdef:
    int triple_point    = 0
    int domain_boundary = 1
    int empty           = 2
    int head_end        = 3
    int tail_end        = 4
    bool verbose        = False

cpdef int type_empty():
    return empty

cpdef int type_triple_point():
    return triple_point

cpdef int type_domain_boundary():
    return domain_boundary

cpdef int get_head_end():
    return head_end

cpdef int get_tail_end():
    return tail_end

cdef double pi = 3.14159265359

cdef extern from "math.h":
    double fabs(double)
    double sin(double)
    double cos(double)
    double asin(double)
    double acos(double)
    double atan(double)
    double sqrt(double)
    long round(double)

cdef double sign(double x):
    if x > 0.0:
        return 1.0
    else:
        return -1.0


@cython.profile(False)
cdef inline double norm(np.ndarray[np.float64_t, ndim=1] n):
    """ n is assumed to be a two-element array or list. """
    return sqrt(n[0]*n[0] + n[1]*n[1])


cdef class Boundary(object):
    """ Pass in a 2-d array of x-y values of the initial boundary points. By default,
        this array is assumed to be a two-column array, with column0 for x and
        column1 for y. If as_points=True, then the format is already
        converted into a 1-D array of (x,y) values, with one (x,y) value for each point.
        phase_left and phase_right are the phases of grain adjacent to the grain boundary,
        with left and right referring to the perspective looking from head to tail along
        the boundary."""
    cdef:
        np.ndarray b
        np.ndarray v
        bool _as_points, _to_be_deleted
        char* _phase_left
        char* _phase_right
        double _h
        Node head, tail
        str _id

    def __init__(self, np.ndarray[np.float64_t, ndim=2] b_initial,
            char* phase_left='None',
            char* phase_right='None',
            bool as_points=False):
        #assert type(b_initial) == np.ndarray
        self._as_points = as_points
        self._id = str(uuid.uuid1())
        if not as_points:
            """ Tack on elements to beginning and end of array as place-holders for
                the extrapolated points. """
            self.b = np.array([ [0]+list(b_initial[0])+[0], [0]+list(b_initial[1])+[0] ], dtype=np.float64)
            self.convert_to_points()
        else:
            """ assume it already has elements at beginning and end for extrapolated points. """
            self.b = b_initial
            
        self.v = np.zeros([len(self.b)-2,2])
        self._phase_left = phase_left
        self._phase_right = phase_right
        self._to_be_deleted = False
        self._h = 1.0/16.0
        
    def __str__(self):
        return str(self._id)

    def __richcmp__(Boundary self, Boundary other, int op):
        """ Establish basis for comparison between Boundary instances (check _id values)"""
        if op == 2:
            return self._id == other._id
        elif op == 3:
            return self._id != other._id
        
    def manual_override(self, i1, i2, value):
        """ argh. just need to tweak starting parameters on two boundaries.
        """
        self.b[i1][i2] = value

    def convert_to_points(self):
        """ It is easier to do point-by-point vector-style work with the
            x and y points associated with each other, so you
            have just one column of data, with each data point
            being an (x,y) pair. This function converts
            into this point-by-point format."""
        assert not self._as_points
        self.b = np.array(zip(list(self.b[0]), list(self.b[1])))
        self._as_points = True


    def convert_to_columns(self):
        """ Undoes function 'convert_to_points' for easier plotting."""
        assert self._as_points
        self.b = np.array(zip(*self.b))
        self._as_points = False


    def set_resolution(self, h):
        """ set resolution parameter h """
        self._h = h


    def extrapolate(self):
        """ Extrapolate the grain boundary across the terminations at each end of the boundary.
            The boundary needs to be in 'point-by-point' format. """
        #assert self._as_points
        if self.head._type == empty:
            """ The grain boundary loops on itself. """
            self.b[0] = self.b[-2]
            self.b[-1] = self.b[1]
            return True
            
        cdef:
            np.ndarray[np.float64_t, ndim=1] head
            np.ndarray[np.float64_t, ndim=1] tail
        head = self.head.position
        tail = self.tail.position

        #!!! TODO!!!: CAN I clean this up?
        self.b[0,0], self.b[0,1] = 2*head[0]-self.b[1,0], 2*head[1]-self.b[1,1]
        self.b[-1,0], self.b[-1,1] = 2*tail[0]-self.b[-2,0], 2*tail[1]-self.b[-2,1]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double get_velocity(self) except *:
        """ Calculate the instantaneous velocity vector for
            each point on the grain boundary.
            b needs to be in 'point-by-point' format.
            This will return an array in point-by-point format
            with n-2 rows, where n is the number of rows
            in the boundary array (which has the 2
            extrapolated extra points.)
            Returns the min velocity (used for 4RK time-stepping adjustment by B&W)."""
        #assert self._as_points
        cdef:
            np.ndarray[np.float64_t, ndim=2] v
            np.ndarray[np.float64_t, ndim=2] b
            Py_ssize_t i, stop, lenb
            double min_velocity = 0.0
            double D1norm2, D1x, D1y, D2x, D2y
    
        lenb = len(self.b)
        b = self.b
        
        stop = lenb-1
        v = np.zeros([lenb-2, 2], dtype=np.float)
        for i in xrange(1,stop):
            D2x = b[i+1,0] - 2.0*b[i,0] + b[i-1,0]    # numerical second derivative (w/o normalization) 
            D2y = b[i+1,1] - 2.0*b[i,1] + b[i-1,1]    # numerical second derivative (w/o normalization) 

            D1x = (b[i+1,0] - b[i-1,0])/2.0         # numerical first derivative (w/o normalization)
            D1y = (b[i+1,1] - b[i-1,1])/2.0         # numerical first derivative (w/o normalization)

            D1norm2 = D1x*D1x + D1y*D1y
            v[i-1,0] = D2x/D1norm2             # normalizations cancel since we take ratio
            v[i-1,1] = D2y/D1norm2             # normalizations cancel since we take ratio
            if i == 1:
                min_velocity = D1norm2/4.0
            elif D1norm2/4.0 < min_velocity:
                min_velocity = D1norm2/4.0

        self.v = v
        return min_velocity
        
    cpdef np.ndarray get_boundary(self):
        """ Returns a numpy array of the boundary """
        return self.b

    def get_scheduled_for_deletion(self):
        """ returns whether the gb is slated for deletion. """
        return self._to_be_deleted


    def get_average_velocity(self):
        """ calculate average velocity of points on the gb. """
        print np.mean(self.v)


    cdef update_position(self, double k):
        """ Using the velocity vector, update the positions of the interior
            points (i.e. not including the extrapolated points).
            k is the time increment (delta-t), using Bronsard-Wetton notation. """

        self.b[1:-1] = self.b[1:-1] + self.v*k


    def densify(self):
        """ Double the number of interior points for higher resolution
        and redistribute the points evenly along the length of the path."""
        if verbose: print "densify: current length:", len(self.b)
        cdef Py_ssize_t new_length
        new_length   = round((len(self.b)-2)*2 + 2)
        self.interpolate(new_length)

    def interpolate(self, Py_ssize_t new_length):
        """ linearly interpolate and evenly redistribute points along the
        boundary after coarsening or densifying the boundary. """
        cdef:
            np.ndarray[np.float64_t, ndim=2] old_boundary
            np.ndarray[np.float64_t, ndim=2] delta_vectors
            np.ndarray[np.float64_t, ndim=1] delta_scalars
            np.ndarray[np.float64_t, ndim=1] t
            np.ndarray[np.float64_t, ndim=1] s
            double boundary_length, alpha_n
            Py_ssize_t i, n
    
        """Temporarily use the 'extrapolation' spaces in the arrays
        to hold the locations of the triple point or domain boundary. """
        if self.head._type == empty:
            # gb loops on itself, so put the head mid-way between start and end
            self.b[0]  = (self.b[-2] - self.b[1])/2
            self.b[-1] = self.b[0]
        else:
            self.b[0]  = self.head.position
            self.b[-1] = self.tail.position

        old_boundary    = self.b[:]
        self.b          = np.zeros([new_length,2])
        delta_vectors   = np.diff(old_boundary, axis=0)
        delta_scalars   = np.array([0]+[norm(d) for d in delta_vectors])
        boundary_length = sum(delta_scalars)
        
        """ We parameterize the original boundary with the parameter s describing
        the fractional distance along the boundary from head to tail. """
        s = np.cumsum(delta_scalars)/boundary_length
        
        """ the array t will hold parameters describing fractional distance along the
        boundary for the new set of interior points. """
        t = np.arange(1,new_length-1, dtype=np.float64)/(new_length-1)

        """ The new points are to be evenly spaced along the boundary """
        for i,ti in enumerate(t):
            n = np.where(s<ti)[0][-1] # ti should always be greater than zero, and s[0]=0
            alpha_n = (ti - s[n])/(s[n+1]-s[n])
            self.b[i+1] = old_boundary[n]*(1-alpha_n) + old_boundary[n+1]*alpha_n

        self.extrapolate()
        self.v = np.zeros([len(self.b)-2,2])
        
        
    def coarsen(self):
        """ halve the number of interior points, but keep the two interior end points and
            the two exterior extrapolated points. """
        #TODO: should interpolate and make this smarter and keep num interior points
        # to be a multiple of two
        cdef Py_ssize_t new_length
        new_length = round((len(self.b)-2)/2 + 2)
        self.interpolate(new_length)
        
        #old_boundary = self.b[:]
        #print new_length
        #self.b = np.zeros([new_length,2])
        #self.b[-2] = old_boundary[-2]
        #print 'half way.', np.shape(self.b[1:-2]), np.shape(old_boundary[1:-2:2])
        #self.b[1:-2] = old_boundary[1:-2:2]
        #print 'survived!'
        #self.extrapolate()
        #self.v = np.zeros([len(self.b)-2,2])
        

    cdef check_spacing(self):
        """ we need to refine our mesh when it gets too coarse, coarsen it when it gets dense,
            and cut it when it gets to 2 interior points. """

        cdef:
            np.ndarray[np.float64_t, ndim=1] normd1x2
            np.ndarray[np.float64_t, ndim=2] b
            Py_ssize_t i, stop, lenb
            double min_velocity = 0.0
            double h, D1x, D1y, nu, mu, norm2
        
        b = self.b
        h = self._h
        lenb = len(b)
        normd1x2 = np.zeros(lenb-2, dtype=np.float64)
        nu = 0.0    # just pick something smaller than anything you would encounter
        mu = 10.0*h # just pick something bigger than anything you would encounter
        for i in xrange(1,lenb-1):
            D1x = (b[i+1,0]-b[i-1,0])/2.0/h
            D1y = (b[i+1,1]-b[i-1,1])/2.0/h
            norm2 = D1x*D1x + D1y*D1y
            normd1x2[i-1] = norm2
            if norm2 > nu:
                nu = norm2
            if norm2 < mu:
                mu = norm2
        #normd1x2 = [norm( (self.b[i+1] - self.b[i-1])/2.0/self._h )**2 for i in range(1,len(self.b)-1)]
        #nu = max(normd1x2)
        #mu = min(normd1x2)
        
        if lenb < 5 and mu < h*h/10.0:
            """ we're down to two interior points and need to kill the boundary """
            self._to_be_deleted = True
            return False

        if nu > 4.0:
            self.densify()
        if nu < 1.0/4.0 and lenb > 4.0:
            self.coarsen()


    def set_termination(self, int end, Node termination):
        """ Set the termination object for the specified end of the grain boundary.
            The value of 'end' should be either head_end or tail_end.
            The value of termination should be a Node object.
            """
        if end == head_end:
            self.head = termination
            if termination._type == empty:
                self.tail = termination 
        else:
            self.tail = termination


    def get_termination(self, int end):
        """ Get the specified triple point (if there is one)
            The value of 'end' should be either 'head' or 'tail'.
            The returned value will be a TriplePoint object,
            DomainBoundary object if the grain boundary terminates at
            the domain boundary, or None if the boundary loops
            onto itself and so has no termination.
            """
        if end == head_end:
            return self.head
        else:
            return self.tail


    cdef bool will_die_alone(self):
        """ Are there any to_be_deleted grain boundaries attached
            to the same triple points as this grain boundary? """
        cdef int num_also_dying
        num_also_dying = 0
        
        if self.head._type == triple_point:
            num_also_dying += self.head.get_number_dying_boundaries() - 1

        if self.tail._type == triple_point:
            num_also_dying += self.tail.get_number_dying_boundaries() - 1

        return True if num_also_dying == 0 else False


cdef class Node(object):
    """ A class for managing the Node points at the ends of grain boundaries.
        If three grain boundaries meet at this node, it is a triple point.
        If the grain boundary terminates at the domain boundary instead
        (which is typically used only for testing, as in B&W's paper), then
        this is a Neumann boundary condition on a single grain boundary.
        
        Constructor requires a single argument:
            char* type: 
                a string describing the type of Node this is:
                    either 'triple point' or 'domain boundary'
        
        
        where Boundary is a Boundary object and 'end' can be 'head' or 'tail' to
        specify which end of the boundary is connected to the triple point. """
    cdef np.ndarray position
    cdef int _type
    cdef str _id
    cdef Boundary b
    cdef list boundaries, nodes, ends, endindx
    cdef Py_ssize_t bendindx
    cdef int bend
    def __init__(self, int node_type):
        assert node_type in [empty, triple_point, domain_boundary]
        self._type = node_type
        self._id = str(uuid.uuid1())
        self.position = np.array([0.0, 0.0], dtype=np.float64)
        self.boundaries = []
        self.nodes = []
        self.ends = []
        self.endindx = []
        
    def __str__(self):
        return str(self._id)

    def __richcmp__(Node self, Node other, int op):
        """ Establish basis for comparison between Node instances (check _id values)"""
        if isinstance(self, Node) and isinstance(other, Node):
            if op == 2:
                return self._id == other._id
            elif op == 3:
                return self._id != other._id
        else:
            return NotImplemented

    def get_id(self):
        """ Generate a unique ID for this node using uuid module. """
        self._id = str(uuid.uuid1())

    def get_type(self):
        """ return integer describing node type """
        return self._type

    def get_number_dying_boundaries(self):
        """ # boundaries to be deleted from this node. """
        num_dying = 0
        for gb in self.boundaries:
            if gb.get_scheduled_for_deletion():
                num_dying += 1
        return num_dying
        
    def get_boundaries(self):
        """ return the list of attached boundaries """
        return self.boundaries
        
    def get_sole_survivor(self):
        """ For a node that we know has two gb to be
            deleted, return the one that won't be deleted. """
        cdef Boundary gb
        for gb in self.boundaries:
            if not gb._to_be_deleted:
                return gb, self.ends[self.boundaries.index(gb)]
                

    def add_single_boundary(self, dict b):
        """ For a DomainBoundary Node, there's only one boundary, so add this boundary only.
        """
        assert self._type == domain_boundary
        self.b, self.bendindx, self.bend = self.parse_boundary(b)
        self.update_position()


    def add_boundaries(self, b1, b2, b3):
        """ 
        For triple point, pass in an ordered sequence of three dictionaries containing information
        about the boundaries (in anti-clockwise order around the junction),
        each formatted like so:
        b1 = dict( b=Boundary(), end='head' )
        
        split the dictionaries and save similar info for each of 3 boundaries:
        b1: boundary
        b1pt: the index of the point in the grain boundary array
            to use when calculating triple point location (i.e. either 1 or -2)
        b1end: a string (either 'head' or 'tail') specifying the end of the grain boundary
            that is connected to this triple point.
        """
        assert self._type == triple_point
        for b in [b1,b2,b3]:
            boundary, indx, end = self.parse_boundary(b)
            self.boundaries.append(boundary)
            self.endindx.append(indx)
            self.ends.append(end)
        self.update_position()
        
    def replace_boundary(self, Boundary a, Boundary b, int bend):
        """ Replace boundary a with b (for example after a concatenation). """
        cdef int indx
        indx = self.boundaries.index(a)
        self.boundaries[indx] = b
        self.ends[indx] = bend
        self.endindx[indx] = 1 if bend == head_end else -2
        


    def parse_boundary(self, bdict):
        """ Parse dictionary, saving boundary and info on proper end point
        and set_termination on the boundary so that it points to this triple-point. """
        assert bdict['end'] in [head_end, tail_end]
        array_position = 1 if bdict['end'] == head_end else -2
        bdict['b'].set_termination( bdict['end'], self )
        return bdict['b'], array_position, bdict['end']


    cpdef int update_position( self, bool verbose=False, bool check_angles=False ) except *:
        """ Send in the list of boundaries, each of which ends at the center-point,
            and use them to calculate the new center-point. """
        cdef:
            Boundary gb
            double x,y,theta,theta1,theta2,theta3
            double l1, l2, l3
            double alpha, delta, beta, dilation
            np.ndarray[np.float64_t, ndim=2] b
            np.ndarray[np.float64_t, ndim=1] b1N
            np.ndarray[np.float64_t, ndim=1] b2N
            np.ndarray[np.float64_t, ndim=1] b3N
            np.ndarray[np.float64_t, ndim=1] v13
            np.ndarray[np.float64_t, ndim=1] v21
            np.ndarray[np.float64_t, ndim=1] v23
            np.ndarray[np.float64_t, ndim=1] C
            #np.ndarray[np.float64_t, ndim=2] rotation
            Py_ssize_t i
            
        if self._type == domain_boundary:
            """ Find where the grain boundary 'gb' would intersect the domain boundary if
            it were to arrive normal to the domain boundary.
            self.b needs to be in 'point-by-point' format. """
            gb = self.b
            b = gb.b
            x, y = b[self.bendindx,0], b[self.bendindx,1]
            theta = atan(fabs(y/x))
            self.position[0] = sign(x)*cos(theta)
            self.position[1] = sign(y)*sin(theta)
        else:
            theta1 = theta2 = theta3 = 2.0*pi/3.0
            gb = self.boundaries[0]
            b1N = gb.b[self.endindx[0]]
            gb = self.boundaries[1]
            b2N = gb.b[self.endindx[1]]
            gb = self.boundaries[2]
            b3N = gb.b[self.endindx[2]]
            #b1N, b2N, b3N = [gb.b[endpt] for gb, endpt in zip(self.boundaries, self.endindx)]
            v13 = b3N - b1N # vector running from pt1 to pt3
            v21 = b1N - b2N # vector running from pt2 to pt1
            v23 = b3N - b2N # vector running from pt2 to pt3
            l1 = norm( v23 )
            l2 = norm( v13 )
            l3 = norm( v21 )
            alpha = acos(-(v21[0]*v13[0]+v21[1]*v13[1])/l2/l3) # np.arccos( dot(-v21,v13)/l2/l3) )
            delta = theta2 - alpha
            beta = atan( sin(delta)/(l3*sin(theta3)/(l2*sin(theta1))+cos(delta)) )
            dilation = sin(pi - beta - theta1)/sin(theta1)
            
            if v23[0]*v21[1] > v21[0]*v23[1]: # same as: if np.cross(v23,v21) > 0:
                # rotate clockwise
                beta *= -1
            #rotation = np.matrix([[cos(beta), -sin(beta)],[sin(beta), cos(beta)]])
            #C = b2N + dilation*np.array(np.dot(rotation, v21))[0]
            # Hand-code the rotation-dilation matrix multiplication:
            C = np.arange(2, dtype=np.float64)
            C[0] = b2N[0] + dilation*(v21[0]*cos(beta) - v21[1]*sin(beta))
            C[1] = b2N[1] + dilation*(v21[0]*sin(beta) + v21[1]*cos(beta))

            if check_angles == True:
                vc1, vc2, vc3 = C-b1N, C-b2N, C-b3N
                t1 = acos((vc2[0]*vc1[0] + vc2[1]*vc1[1])/norm(vc2)/norm(vc1))*180.0/pi # np.dot(vc2,vc1) is (vc2[0]*vc1[0]+vc2[1]*vc1[1])
                t2 = acos((vc3[0]*vc2[0] + vc3[1]*vc2[1])/norm(vc3)/norm(vc2))*180.0/pi
                t3 = acos((vc3[0]*vc1[0] + vc3[1]*vc1[1])/norm(vc3)/norm(vc1))*180.0/pi
                if (round(t1) != round(theta1*180.0/pi)
                    or round(t2) != round(theta2*180.0/pi)
                    or round(t3) != round(theta3*180.0/pi)):
                    print "There is some problem with triple point angles."
                    print "  t1, t2, t3:", t1, t2, t3
                    print "  locations:", b1N, b2N, b3N

                if verbose:
                    print 'points:', b1N, b2N, b3N
                    print 'angles are in degrees:'
                    print 'alpha:', alpha*180.0/pi
                    print 'delta:', delta*180.0/pi
                    print 'beta:', beta*180.0/pi
                    print 'l1, l2, l3:', l1, l2, l3
                    print 'dilation:', dilation
                    print 'new triple point location:', C
                    print "theta1, theta2, theta3:", t1, t2, t3

            if delta < 0:
                print "  C is:", C
                print "  b1N, b2N, b3N:", b1N, b2N, b3N
                raise ValueError("Bring points together. Your triangle of endpoints is too open to contain the triple point.")

            self.position = C
        

cdef class ThinFilm(object):
    """ A class for simulating thin-film grain growth using the boundary
        front-tracking method. The physics follows what came out of Carl
        Thompson's lab, but the detailed implementation is an attempt to
        follow Bronsard & Wetton.
    """
    cdef:
        double _k, _h
        list boundaries, nodes
        
    def __init__(self, boundaries=None, nodes=None):
        if boundaries is not None and nodes is not None:
            self.add_boundaries(boundaries)
            self.add_nodes(nodes)
        self._k = 0.0003
        self._h = 1.0/16.0
    
    def initialize_graph(self, num_grains=2500, num_interior_points_per_boundary=8):
        """ Generate an initial set of Boundary and Node objects
        using a Voronoi diagram from a set of num_grains randomly spaced
        points on a domain with periodic boundary conditions. 
        The width and height of the simulation domain will both be set 
        equal to the square root of n (so that one spatial unit of 
        the domain is approximately equal to the average grain 
        diameter in the initial graph)."""
        width = height = np.sqrt(num_grains)
        segments, nodelist, node_segments = voronoi.periodic_diagram_with_nodes(num_grains)
        boundaries = []
        nodes = []
        for s in segments:
            """ 
            each segment is a dictionary containing:
                points: list of floats, [(x0,y0), (x1,y1)]
                head: int, node id number
                tail: int, node id number """
            
            # magnification factor from Voronoi diagram to our simulation region:
            mag = np.sqrt(num_interior_points_per_boundary)
            p0 = np.array(s['points'][0], dtype=np.float64)*mag
            p1 = np.array(s['points'][1], dtype=np.float64)*mag
            if norm(p1-p0) < self._h:
                n  = num_interior_points_per_boundary
            else:
                n  = 2
            
            b  = np.zeros((n+2, 2), dtype=np.float64)
            for i in range(n):
                b[i+1,:] = p0 + (i+1.0)/(n+1.0)*(p1-p0)

            #if norm(p1-p0)/n > self._h/5.0:
            #    b[1,:] = p0 + self._h/5.0*(p1-p0)/norm(p1-p0)
            #    b[-2,:] = p1 - self._h/5.0*(p1-p0)/norm(p1-p0)

            # These two spaces in b will be over-written by extrapolated points, 
            # but for now I'll just store the node locations
            b[0,:] = p0
            b[-1,:] = p1
            boundaries.append( Boundary(b, as_points=True) )
            
        for node_id, [j, k, l] in zip(nodelist, node_segments):
            b1 = boundaries[j]
            b2 = boundaries[k]
            b3 = boundaries[l]
            
            b1end = head_end if node_id == segments[j]['head'] else tail_end
            b2end = head_end if node_id == segments[k]['head'] else tail_end
            b3end = head_end if node_id == segments[l]['head'] else tail_end

            b1b = b1.get_boundary()
            b2b = b2.get_boundary()
            b3b = b3.get_boundary()
            v1 = np.array(b1b[-2]) - np.array(b1b[1])
            v2 = np.array(b2b[-2]) - np.array(b2b[1])
            v3 = np.array(b3b[-2]) - np.array(b3b[1])
            if b1end == head_end:
                v1 = -1*v1 
            if b2end == head_end:
                v2 = -1*v2 

            if v1[0]*v2[1] < v2[0]*v1[1]: # same as np.cross(v1,v2) < 0:
                b_temp = b2
                b2 = b3
                b3 = b_temp
            
            node = Node(node_type=triple_point)
            node.add_boundaries(
                dict(b=b1, end=b1end),
                dict(b=b2, end=b2end),
                dict(b=b3, end=b3end))

            # check update_position to see if delta<0 ?
            try:
                node.update_position()
            except ValueError:
                # I think this is most likely to fail when you have a segment intersect
                # an otherwise straight segment. Let's try to identify the intersecting segment
                # and bring the other two points in closer.
                print 'hi'
                
            nodes.append( node )

        self.add_boundaries(boundaries)
        self.add_nodes(nodes)

    def add_boundaries(self, boundaries):
        """ Add a list of Boundary objects to the ThinFilm object."""
        self.boundaries = boundaries


    def add_nodes(self, nodes):
        """ Add a list of Node objects to the ThinFilm object."""
        self.nodes = nodes
        

    cdef bool delete_single_boundary(self, Boundary gb):
        """ Delete an isolated boundary. There are four different scenarios
        here, and each one must be handled separately.
        
        In the simplest case, the gb loops onto itself and so will be deleted
        without affecting any triple points.
        
        In the case that the gb connects two triple points, we have a 4* situation
        (in B&W classification): we need to delete the two triple points, delete the
        current grain boundary, generate two new triple points connecting different
        boundaries, and connect the two new triple points with a new boundary.
        
        In the case that there is a domain boundary in the simulation (depends on your
        boundary conditions) and one end of the gb terminates
        at it with a triple point at the other end, the triple point will be deleted
        and the two other gb from the triple point will now be terminated at the
        domain boundary.

        In the case that the gb just connects two different points on the domain
        boundary, we treat it like a gb that loops onto itself and just delete it.
        
        """
        cdef:
            Boundary gb2, new_gb
            Boundary head_prev_boundary, head_subs_boundary
            Boundary tail_prev_boundary, tail_subs_boundary
            Boundary a, b, b1,b2,b3
            int end, aend, bend
            Node TP1, TP2, new_node, n

        if gb.head._type == empty:
            """ gb loops onto itself, so it can just be removed. """
            self.boundaries.remove(gb)
            return True
        
        if gb.head._type == domain_boundary:
            if verbose: print "  Number nodes before single gb deletion:", len(self.nodes)

            if gb.tail._type == domain_boundary:
                """ gb connects two points on domain boundary & can just be removed. """
                if verbose: print "  Deleting a single gb that connects two points on the domain boundary."
                self.nodes.remove(gb.tail)
                self.nodes.remove(gb.head)
                self.boundaries.remove(gb)
            else:
                """ tail connects to a triple point. connect the two other gb
                from that triple point to the domain boundary"""
                if verbose: print "  Deleting single boundary that connects DB to TP."
                self.nodes.remove(gb.tail)
                self.nodes.remove(gb.head)
                self.boundaries.remove(gb)

                if verbose:
                    b1,b2,b3 = gb.tail.boundaries
                    e1,e2,e3 = gb.tail.ends
                    print "  gb.tail.boundaries, gb.tail.ends, (gb.tail = {0}):".format(gb.tail)
                    print "    {0}, {1}".format(b1,e1)
                    print "    {0}, {1}".format(b2,e2)
                    print "    {0}, {1}".format(b3,e3)
                
                for gb2, end in zip(gb.tail.boundaries, gb.tail.ends):
                    if gb2 != gb:
                        self.nodes.append( Node(node_type=domain_boundary) )
                        self.nodes[-1].add_single_boundary(dict(b=gb2, end=end))
            return True
        else:
            if gb.tail._type == triple_point:
                """ gb connects two triple points.
                There are two possibilities:
                    1) 4* situation: there are 5 unique grain 
                    boundaries involved. Make a new grain boundary roughly
                    orthogonal to the one we are going to be deleting,
                    then connect it to the 4 living grain boundaries.
                
                    2) A bubble trapped on a grain boundary is shrinking
                    into non-existence (like Fig. 14 of B&W). In this case
                    there are only 4 unique grain boundaries. One of the 
                    survivors connects the same two Nodes as the gb that
                    is being deleted. This survivor should be broken off 
                    from the main gb and will form a small node-less bubble
                    that will then die alone (I'll kill it immediately if
                    it only has 2 interior points).
                """
                
                """ Grab the two living boundaries off of each end
                (these are referenced in order--previous and subsequent--
                relative to the gb that is going to be deleted) """
                head_index = gb.head.boundaries.index(gb)
                head_prev_boundary = gb.head.boundaries[head_index-1]
                head_prev_endindx = gb.head.endindx[head_index-1]
                head_prev_end = gb.head.ends[head_index-1]
                
                subs_indx = head_index+1 if head_index+1 < len(gb.head.boundaries) else 0
                head_subs_boundary = gb.head.boundaries[subs_indx]
                head_subs_endindx = gb.head.endindx[subs_indx]
                head_subs_end = gb.head.ends[subs_indx]
                
                tail_index = gb.tail.boundaries.index(gb)
                tail_prev_boundary = gb.tail.boundaries[tail_index-1]
                tail_prev_endindx = gb.tail.endindx[tail_index-1]
                tail_prev_end = gb.tail.ends[tail_index-1]

                subs_indx = tail_index+1 if tail_index+1 < len(gb.tail.boundaries) else 0
                tail_subs_boundary = gb.tail.boundaries[subs_indx]
                tail_subs_endindx = gb.tail.endindx[subs_indx]
                tail_subs_end = gb.tail.ends[subs_indx]

                if verbose: print "  deleting a single gb that connects two triple points."
                """ First check for possibility #2:
                """
                if ((head_prev_boundary == tail_prev_boundary) or 
                    (head_prev_boundary == tail_subs_boundary) or
                    (head_subs_boundary == tail_prev_boundary) or 
                    (head_subs_boundary == tail_subs_boundary)):
                    if verbose: print "  making a loop."

                    self.nodes.remove(gb.head)
                    self.nodes.remove(gb.tail)
                    self.boundaries.remove(gb)

                    if ((head_prev_boundary == tail_prev_boundary) or 
                        (head_prev_boundary == tail_subs_boundary)):
                        """ head previous will become the loop. """
                        a, aend = head_subs_boundary, head_subs_end
                        if head_prev_boundary == tail_prev_boundary:
                            b, bend = tail_subs_boundary, tail_subs_end
                        else:
                            b, bend = tail_prev_boundary, tail_prev_end

                        if len(head_prev_boundary.b) == 4:
                            """ just kill it now. """
                            self.boundaries.remove(head_prev_boundary)
                        else:
                            head_prev_boundary.set_termination( head_end, Node(node_type=empty) )
                        #head_prev_boundary.set_termination( head_end, Node(node_type=empty) )
                        
                    elif ((head_subs_boundary == tail_prev_boundary) or 
                        (head_subs_boundary == tail_subs_boundary)):
                        """ head subsequent will become the loop. """
                        a, aend = head_prev_boundary, head_prev_end
                        if head_subs_boundary == tail_prev_boundary:
                            b, bend = tail_subs_boundary, tail_subs_end
                        else:
                            b, bend = tail_prev_boundary, tail_prev_end

                        if len(head_subs_boundary.b) == 4:
                            """ just kill it now. """
                            self.boundaries.remove(head_subs_boundary)
                        else:
                            head_subs_boundary.set_termination( head_end, Node(node_type=empty) )
                        #head_subs_boundary.set_termination( head_end, Node(node_type=empty) )

                    self.concatenate_boundaries(a, aend, b, bend)
                    return True

                if verbose: print "  4* situation, should conserve # nodes and # boundaries."
                
                """ Then it must be possibility #1.
                Our new triple points will connect the following
                grain boundaries in the following order:
                1) tail_prev
                2) head_subs
                3) new_head
                and
                1) head_prev
                2) tail_subs
                3) new_tail
                
                Where the new gb runs from new_head to new_tail.

                First we need to make the new grain boundary:
                """
                p1 = (head_subs_boundary.b[head_subs_endindx] + tail_prev_boundary.b[tail_prev_endindx])/2
                p2 = (head_prev_boundary.b[head_prev_endindx] + tail_subs_boundary.b[tail_subs_endindx])/2
                new_head = p1+(p2-p1)/3
                new_tail = p2+(p1-p2)/3
                new_gb = Boundary(np.array([[0,0],new_head,new_tail,[0,0]]), as_points=True)

                """ Now delete the old grain boundary from the list and insert the new one,
                and delete the two old triple points, replacing them with two new ones. """
                self.nodes.remove(gb.head)
                self.nodes.remove(gb.tail)
                self.boundaries.remove(gb)
                
                self.boundaries.append(new_gb)
                TP1 = Node(node_type=triple_point)
                TP1.add_boundaries(
                    dict(b=tail_prev_boundary, end=tail_prev_end),
                    dict(b=head_subs_boundary, end=head_subs_end),
                    dict(b=new_gb, end=head_end))
                TP2 = Node(node_type=triple_point)
                TP2.add_boundaries(
                    dict(b=head_prev_boundary, end=head_prev_end),
                    dict(b=tail_subs_boundary, end=tail_subs_end),
                    dict(b=new_gb, end=tail_end))
                self.nodes.append(TP1)
                self.nodes.append(TP2)
            else:
                """ head connects to a triple point. connect the two other gb
                from that triple point to the domain boundary. """
                if verbose:
                    print "  Deleting gb that connects TP to DB (head to tail)."
                    print "  Deleting TP {0} and DB {1}".format(gb.head, gb.tail)
                self.nodes.remove(gb.head)
                self.nodes.remove(gb.tail)
                self.boundaries.remove(gb)
                for gb2, end in zip(gb.head.boundaries, gb.head.ends):
                    if gb2 != gb:
                        if verbose: print "  Appending DB node to gb {0}".format(gb2)
                        self.nodes.append( Node(node_type=domain_boundary) )
                        self.nodes[-1].add_single_boundary(dict(b=gb2, end=end))
                    else:
                        if verbose: print "  gb2 == gb for gb2={0}".format(gb2)
            return True

    cdef concatenate_boundaries(self, Boundary a, int aend, Boundary b, int bend):
        """ Stitch boundaries a and b together, connecting aend to bend and
        equalizing the number of points of each boundary so that the number 
        of interior points of the combined boundary will still be a power of 2
        for easier densifying/coarsening later on. """
        cdef:
            np.ndarray[np.float64_t, ndim=2] new_array
            Boundary new_boundary, gb
            Node a_surviving_node, b_surviving_node
            int a_surviving_end, b_surviving_end, indx
        
        while len(a.b) > len(b.b):
            b.densify()
        while len(a.b) < len(b.b):
            a.densify()
        
        if aend == tail_end:
            a_surviving_end = head_end
            if bend == head_end:
                b_surviving_end = tail_end
                new_array = np.concatenate((a.b[:-1], b.b[1:]), axis=0)
            else:
                b_surviving_end = head_end
                new_array = np.concatenate((a.b[:-1], b.b[-1::-1]), axis=0)
        else:
            a_surviving_end = tail_end
            if bend == head_end:
                b_surviving_end = tail_end
                new_array = np.concatenate((a.b[-1::-1], b.b[1:]), axis=0)
            else:                    
                b_surviving_end = head_end
                new_array = np.concatenate((a.b[-1::-1], b.b[-1::-1]), axis=0)

        a_surviving_node = a.get_termination(a_surviving_end)
        b_surviving_node = b.get_termination(b_surviving_end)
        new_boundary = Boundary( new_array, as_points=True )
        new_boundary.set_termination(head_end, a_surviving_node)
        new_boundary.set_termination(tail_end, b_surviving_node)
        
        """ and don't forget to update the surviving node to be attached to 
        the new_boundary rather than a and b! """
        a_surviving_node.replace_boundary(a, new_boundary, head_end)
        b_surviving_node.replace_boundary(b, new_boundary, tail_end)
                
        self.boundaries.append( new_boundary )
        self.boundaries.remove(a)
        self.boundaries.remove(b)


    def boundaries_share_two_nodes(self, Boundary b1, Boundary b2):
        """ Really? You need more of a description? """
        if ((b1.head == b2.head or b1.head == b2.tail) and
            (b1.tail == b2.head or b1.tail == b2.tail)):
            return True
        else:
            return False


    def delete_two_boundaries(self, to_be_deleted):
        """ Remove two boundaries that share at least one triple point. """
        assert len(to_be_deleted) == 2
        cdef:
            Node tp, tp_new
            Boundary survivor, a, b, d1, d2
            int aend, bend
        d1, d2 = to_be_deleted
        if self.boundaries_share_two_nodes(*to_be_deleted):
            """ Connect the two living grain boundaries and turn them into
            one longer grain boundary. If the number of points per boundary
            is not the same, B&W increase the density of the less-dense one
            to match the more-dense one to make it easier to keep track of
            the density of points. """
            a, aend = d1.head.get_sole_survivor()
            b, bend = d1.tail.get_sole_survivor()
            self.concatenate_boundaries(a, aend, b, bend)
            self.nodes.remove(d1.head)
            self.nodes.remove(d1.tail)
            if verbose: print "  Two boundaries share two nodes, done concatenating."
        else:
            """ The two are connected to the domain boundary at the other ends, so
            just connect the surviving member of the TriplePoint to the DomainBoundary. """
            
            if verbose: print "  Deleting two boundaries that are both connected to DomainBoundary"
            #assert (d1.head._type == domain_boundary) or (d1.tail._type == domain_boundary)
            if d1.head._type == triple_point:
                tp = d1.head
                self.nodes.remove(d1.tail)
            else:
                tp = d1.tail
                self.nodes.remove(d1.head)

            if d2.head._type == domain_boundary:
                self.nodes.remove(d2.head)
            else:
                self.nodes.remove(d2.tail)

            survivor, survivor_end = tp.get_sole_survivor()
            self.nodes.remove(tp)
            tp_new = Node(node_type=domain_boundary)
            tp_new.add_single_boundary(dict(b=survivor, end=survivor_end))
            self.nodes.append( tp_new )

        self.boundaries.remove(d1)
        self.boundaries.remove(d2)

    def delete_boundaries(self, list to_be_deleted):
        """ Remove boundaries through surgery. """
        cdef Boundary gb
        for gb in to_be_deleted:
            if gb.will_die_alone():
                if verbose: print "  killing single boundary."
                self.delete_single_boundary(gb)
                to_be_deleted.remove(gb)
                
        if len(to_be_deleted) > 2:
            """Shit. Delete three boundaries at once? BUT I AM LAZY!!!"""
            raise ValueError("AHHHHHHHHHHHH. Not implemented.")
        elif len(to_be_deleted) == 0:
            return True
        else:
            if verbose: print "  deleting two boundaries at once. dt:", self._k
            if verbose: print "  Length of boundary list before deletion:", len(self.boundaries)
            self.delete_two_boundaries(to_be_deleted)
            if verbose: print "  Length of boundary list after deletion:", len(self.boundaries)
            
                           
    def get_plot_lines(self):
        """ return a list of line segments for use in a matplotlib line collection. """
        segments = []
        for b in self.boundaries:
            segments.append( b.get_boundary() )
        
        return segments
            

    def regrid(self):
        """ Check each boundary to ensure that its density of points is
            not too high and not too low. If it only has two interior points
            and is too dense, then it is deleted and the remaining boundaries
            are surgically repaired across the void. """
            
        to_be_deleted = []
        cdef Boundary b
        for b in self.boundaries:
            if b.check_spacing() == False:
                to_be_deleted.append(b)
        
        if len(to_be_deleted) > 0:
            if verbose: 
                print "Pre-deletion number of Nodes: {0} and list:".format(len(self.nodes))
                for n in self.nodes:
                    if n.get_type() == triple_point:
                        b1,b2,b3 = n.get_boundaries()
                        print " {0}, {1} has boundaries:".format(n,n.get_type())
                        print "    {0}".format(b1)
                        print "    {0}".format(b2)
                        print "    {0}".format(b3)
                    else:
                        print " {0}, {1}".format(n, n.get_type())
            if verbose:
                print " "
                print "  And as listed from self.boundaries:"
                for i,b in enumerate(self.boundaries):
                    for j in [3,4]:
                        n = b.get_termination(j)
                        end = "head" if j ==3 else "tail"
                        xtra = "  NOT IN self.nodes." if n not in self.nodes else ""
                        print "  boundary {0} {1} ({5}): {2}, {3} {4}".format(i,end,n,n.get_type(),xtra,b)

            self.delete_boundaries(to_be_deleted)
            if verbose:
                print "    Post-deletion number of Nodes: {0} and list:".format(len(self.nodes))
                for n in self.nodes:
                    if n.get_type() == triple_point:
                        b1,b2,b3 = n.get_boundaries()
                        print "   {0}, {1} has boundaries:".format(n,n.get_type())
                        print "      {0}".format(b1)
                        print "      {0}".format(b2)
                        print "      {0}".format(b3)
                    else:
                        print "   {0}, {1}".format(n, n.get_type())

                print " "
                print "     And as listed from self.boundaries:"
                for i,b in enumerate(self.boundaries):
                    for j in [3,4]:
                        n = b.get_termination(j)
                        end = "head" if j ==3 else "tail"
                        xtra = "  NOT IN self.nodes." if n not in self.nodes else ""
                        print "  boundary {0} {1} ({5}): {2}, {3} {4}".format(i,end,n,n.get_type(),xtra,b)
        
        return True


    def calculate_timestep(self, min_velocity):
        """ alter time step according to the overly-restrictive 4RK condition of B&W
            Eq. 15 """
        self._k = min_velocity
        

    cpdef double get_timestep(self):
        """ Return the most recent value of time step. """
        return self._k


    def set_timestep(self, k):
        """ Set the time increment to be used for the subsequent iteration. """
        self._k = k


    def set_resolution(self, h):
        """ set resolution parameter h """
        self._h = h


    cpdef update_grain_boundaries(self):
        """ Cycle through the list of grain boundaries and update
            their locations. """
        cdef:
            double min_velocity, v
            Py_ssize_t i
            Boundary b
            
        for i,b in enumerate(self.boundaries):
            b.extrapolate()
            v = b.get_velocity()
            if i == 0:
                min_velocity = v
            elif v < min_velocity:
                min_velocity = v
                
        self.calculate_timestep(min_velocity)
        for b in self.boundaries:
            b.update_position(self.get_timestep())


    cpdef update_nodes(self):
        """ Cycle through the list of triple points and update their
            locations using the most recent interior points of the
            grain boundaries. """
        cdef Node tp
        for tp in self.nodes:
            tp.update_position()
    


