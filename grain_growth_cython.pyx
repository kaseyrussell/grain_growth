# encoding: utf-8
# cython: profile=True
from __future__ import division

import numpy as np
cimport numpy as np
import cython
from cpython cimport bool
import uuid
import voronoi_cython as voronoi
from scipy import interpolate
import pyximport
pyximport.install(setup_args = {'options' :
                                {'build_ext' :
                                 {'libraries' : 'lapack',
                                  'include_dirs' : np.get_include(),
                                  }}})
from matrix_solvers import dgtsvx, dgesv, tridiagonal_solve

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
    
    I build this using a setup.py file along with the following command:

        python setup.py build_ext --inplace
    
    And if you want an html file with color-coded 'python-ness' of each line:

        cython -a thismodule.pyx
    
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

from libc.math cimport sqrt

cdef extern from "math.h":
    double fabs(double)
    double sin(double)
    double cos(double)
    double asin(double)
    double acos(double)
    double atan(double)
    #double sqrt(double)
    long round(double)

cdef double sign(double x):
    if x > 0.0:
        return 1.0
    else:
        return -1.0


@cython.profile(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double norm2(np.float64_t[:] n):
    """ n is assumed to be a two-element array or list. """
    return n[0]*n[0] + n[1]*n[1]

@cython.profile(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double norm(np.ndarray[np.float64_t, ndim=1] n):
#cdef inline double norm(np.float64_t[:] n):
    """ n is assumed to be a two-element array or list. """
    #return sqrt(norm2(n))
    return sqrt(n[0]*n[0] + n[1]*n[1])


cdef class Boundary(object):
    """ Pass in a 2-d array of x-y values of the initial boundary points. By default,
        this array is assumed to be a two-column array, with column0 for x and
        column1 for y. If as_points=True, then the format is already
        converted into a 1-D array of (x,y) values, with one (x,y) value for each point.
        phase_left and phase_right are the phases of grain adjacent to the grain boundary,
        with left and right referring to the perspective looking from head to tail along
        the boundary.
        
        period is the width of the periodic region (if periodic b.c.).
        """
    cdef:
        np.ndarray b
        #np.float64_t[:,::1] b # doesn't even compile...
        np.ndarray b_prev_timestep
        np.ndarray v
        bool _as_points, _to_be_deleted
        char* _phase_left
        char* _phase_right
        double _h, _iteration_change
        double _length
        double _prev_length
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
            
        self.b_prev_timestep = self.b.copy()
        self.v               = np.zeros([len(self.b)-2,2], dtype=np.float64)
        self._phase_left     = phase_left
        self._phase_right    = phase_right
        self._to_be_deleted  = False
        self._h              = 1.0/16.0
        self._iteration_change = 0.0
        self._length      = 0.0
        self._prev_length = 0.0
        
    def __str__(self):
        """ What should be printed when 'print Boundary' is called? The Boundary._id value."""
        return str(self._id)

    def __richcmp__(Boundary self, Boundary other, int op):
        """ Establish basis for comparison between Boundary instances (check _id values)"""
        if op == 2:
            return self._id == other._id
        elif op == 3:
            return self._id != other._id
        
    cdef int initialize_iteration_buffer(self) except *:
        """ Reset self.b_prev_timestep to the current value. """
        self.b_prev_timestep = self.b.copy()


    cdef double get_iteration_change(self) except *:
        """ returns the maximum change in boundary location between this iteration and the last."""
        return self._iteration_change


    #@cython.boundscheck(False)
    #@cython.wraparound(False)
    cpdef int implicit_step_junction_iteration(self, double timestep) except *:
        """         # implicit time-step boundaries
            # extrapolate across nodes from previous iteration
            # Find 1/|D1Xi|^2 and other elements of matrix
            # Solve for Xn+1
        """
        cdef:
            Py_ssize_t lenb = len(self.b)
            Py_ssize_t i
            np.ndarray[dtype=np.float64_t, ndim=2] b_prev_iteration = np.empty((lenb,2), dtype=np.float64)
            #np.ndarray[dtype=np.float64_t, ndim=2] b_prev_timestep
            #np.ndarray[dtype=np.float64_t, ndim=2] b_prev_timestep_y
            np.ndarray[dtype=np.float64_t, ndim=2] b_solve = np.empty((lenb,2), dtype=np.float64)
            #np.ndarray[dtype=np.float64_t, ndim=2] b_solve_x
            #np.ndarray[dtype=np.float64_t, ndim=2] b_solve_y
            np.ndarray[dtype=np.float64_t, ndim=1] offdiagdn = np.empty((lenb-1), dtype=np.float64)
            np.ndarray[dtype=np.float64_t, ndim=1] diag = np.empty((lenb), dtype=np.float64)
            np.ndarray[dtype=np.float64_t, ndim=1] offdiagup = np.empty((lenb-1), dtype=np.float64)
            np.ndarray[dtype=np.float64_t, ndim=2] M = np.empty((lenb,lenb), dtype=np.float64)
            np.ndarray[dtype=np.float64_t, ndim=1] inv_d1xsq = np.empty((lenb), dtype=np.float64)
            np.ndarray[dtype=np.float64_t, ndim=1] node_location = np.empty((2), dtype=np.float64)
            double delta, delta_max, d1x, normx, normy, ptemp
            double lx, ly

        lenb = len(self.b)
        b_prev_iteration  = self.b[:]
        inv_d1xsq         = np.zeros(lenb, dtype=np.float64)
        for i in range(lenb-2):
            #d1x = norm( b_prev_iteration[i+2]-b_prev_iteration[i] )
            normx = b_prev_iteration[i+2,0]-b_prev_iteration[i,0]
            normy = b_prev_iteration[i+2,1]-b_prev_iteration[i,1]
            d1x = sqrt( normx*normx + normy*normy )
            inv_d1xsq[i+1] = 4.0/d1x/d1x
        
        """ build matrix
        First and last row are the Dirichlet boundary conditions on the 
        extrapolated points.
        """
        diag      = np.ones(lenb) + 2.0*timestep*inv_d1xsq
        offdiagup = -timestep*inv_d1xsq[:lenb-1]*np.ones(lenb-1, dtype=np.float64)
        offdiagdn = -timestep*inv_d1xsq[1:] *np.ones(lenb-1, dtype=np.float64)
        offdiagup[0]      = 1.0
        offdiagdn[lenb-2] = 1.0

        self.get_head_location(node_location)
        self.b_prev_timestep[0]  = 2.0*node_location
        
        self.get_tail_location(node_location)
        self.b_prev_timestep[lenb-1] = 2.0*node_location
        
        #self.b = dgtsvx( offdiagdn, diag, offdiagup, self.b_prev_timestep.transpose() )
        #self.b = tridiagonal_solve( offdiagdn, diag, offdiagup, self.b_prev_timestep.transpose() )
        
        M      = np.diag(diag) + np.diag(offdiagup, k=1) + np.diag(offdiagdn, k=-1)
        b_solve = np.linalg.solve( M, self.b_prev_timestep )
        self.b = b_solve
        #b_prev_timestep = self.b_prev_timestep[:]
        #self.b = dgesv( M, b_prev_timestep )

        #self.b = b_solve.transpose()
        
        delta_max = 0.0
        for i in range(lenb):
            normx = self.b[i,0]-b_prev_iteration[i,0]
            normy = self.b[i,1]-b_prev_iteration[i,1]
            delta = sqrt( normx*normx + normy*normy )
            if delta > delta_max:
                delta_max = delta
        self._iteration_change = delta_max
        
    def get_previous_boundary(self, Node node):
        """ Get the grain boundary previous to (i.e. clockwise from) 
        the current boundary at the given node. """
        cdef:
            Boundary prev_boundary
            Py_ssize_t indx, prev_endindx
            int prev_end
        indx = node.boundaries.index(self)
        prev_boundary = node.boundaries[indx-1]
        prev_endindx = node.endindx[indx-1]
        prev_end = node.ends[indx-1]
        return prev_boundary, prev_endindx, prev_end
        

    def get_subsequent_boundary(self, Node node):
        """ Get the grain boundary subsequent to (i.e. anti-clockwise from) 
        the current boundary at the given node. """
        cdef:
            Boundary subs_boundary
            Py_ssize_t indx, subs_endindx, subs_indx
            int subs_end
        indx = node.boundaries.index(self)
        subs_indx = indx+1 if indx+1 < len(node.boundaries) else 0
        subs_boundary = node.boundaries[subs_indx]
        subs_endindx = node.endindx[subs_indx]
        subs_end = node.ends[subs_indx]
        return subs_boundary, subs_endindx, subs_end
        
    def has_node(self, Node node):
        """ Is the given node attached to this boundary? """
        return True if (node == self.head or node == self.tail) else False

    def retract_from_node(self, Node node):
        """ The closest point of this boundary is too close to the Node 
        and we violate `fits_inside_triangle`. Retract the point by the 
        distance to the previous point and interpolate to redistribute."""

        if len(self.b) < 5:
            # probably we should just delete?
            return False

        if node == self.head:
            self.b[0] = self.b[1]
            self.b[-1] = self.tail.position
        else:
            self.b[-1] = self.b[-2]
            self.b[0] = self.head.position

        self.interpolate(len(self.b), True)
        self.extrapolate()


    def curve_thy_neighbors(self, Node node):
        """ Transitioning from the Voronoi diagram is a pain because the
        intersections are not at the proper angles. Here we rotate the
        two other boundaries that share this 'node'. The way this method is 
        intended to be used, the actual location of the node has not yet
        been determined (which is the whole problem). Instead, the
        Voronoi intersection is stored in the spaces of the boundary 
        array reserved for the extrapolated points.
        """
        cdef:
            Boundary prev_boundary, subs_boundary
            Py_ssize_t indx, prev_endindx, subs_endindx, subs_indx, lprevb, lsubsb
            int prev_end, subs_end
        prev_boundary, prev_endindx, prev_end = self.get_previous_boundary(node)
        subs_boundary, subs_endindx, subs_end = self.get_subsequent_boundary(node)

        b0_nodeindx, b0_indx = 0, 1
        if self.tail == node:
            # check if Node is connected to tail rather than head
            b0_nodeindx, b0_indx = len(self.b)-1, len(self.b)-2
        v0 = self.b[b0_indx] - self.b[b0_nodeindx]

        prev_nodeindx, prev_indx, prev_nn = 0, 1, 2
        if prev_boundary.tail == node:
            # check if Node is connected to tail rather than head
            lprevb = len(prev_boundary.b)
            prev_nodeindx, prev_indx, prev_nn = lprevb-1, lprevb-2, lprevb-3
        vprev = prev_boundary.b[prev_indx] - prev_boundary.b[prev_nodeindx]

        subs_nodeindx, subs_indx, subs_nn = 0, 1, 2
        if subs_boundary.tail == node:
            # check if Node is connected to tail rather than head
            lsubsb = len(subs_boundary.b)
            subs_nodeindx, subs_indx, subs_nn = lsubsb-1, lsubsb-2, lsubsb-3
        vsubs = subs_boundary.b[subs_indx] - subs_boundary.b[subs_nodeindx]

        #beta_prev = np.arccos( np.dot(vprev,v0)/norm(vprev)/norm(v0) )
        #beta_subs = np.arccos( np.dot(vsubs,v0)/norm(vsubs)/norm(v0) )
        ## implemented without array math:
        beta_prev = np.arccos( (vprev[0]*v0[0] + vprev[1]*v0[1])/norm(vprev)/norm(v0) )
        beta_subs = np.arccos( (vsubs[0]*v0[0] + vsubs[1]*v0[1])/norm(vsubs)/norm(v0) )

        del_prev  = -(2*np.pi/3 - beta_prev) # cw
        del_subs  =   2*np.pi/3 - beta_subs  # ccw rotation

        #prev_rotation = np.matrix([[np.cos(del_prev), np.sin(-del_prev)],[np.sin(del_prev), np.cos(del_prev)]])
        #subs_rotation = np.matrix([[np.cos(del_subs), np.sin(-del_subs)],[np.sin(del_subs), np.cos(del_subs)]])
        #prev_rotated = np.array(np.dot(prev_rotation, vprev))[0]
        #subs_rotated = np.array(np.dot(subs_rotation, vsubs))[0]
        ## implemented without array math:
        prev_rotated = np.array([
            np.cos(del_prev)*vprev[0] + np.sin(-del_prev)*vprev[1],
            np.sin(del_prev)*vprev[0] + np.cos(del_prev)*vprev[1]])
        subs_rotated = np.array([
            np.cos(del_subs)*vsubs[0] + np.sin(-del_subs)*vsubs[1],
            np.sin(del_subs)*vsubs[0] + np.cos(del_subs)*vsubs[1]])

        prev_change = prev_rotated - vprev
        subs_change = subs_rotated - vsubs

        prev_boundary.b[prev_indx] = prev_boundary.b[prev_indx] + prev_change
        subs_boundary.b[subs_indx] = subs_boundary.b[subs_indx] + subs_change

        #prev_nn_change = prev_change - np.dot(prev_change, vprev)*vprev/norm(vprev)**2
        #subs_nn_change = subs_change - np.dot(subs_change, vsubs)*vsubs/norm(vsubs)**2
        ## implemented without array math:
        prev_nn_change = prev_change - (prev_change[0]*vprev[0] + prev_change[1]*vprev[1])*vprev/norm(vprev)**2
        subs_nn_change = subs_change - (subs_change[0]*vsubs[0] + subs_change[1]*vsubs[1])*vsubs/norm(vsubs)**2

        prev_boundary.b[prev_nn] = prev_boundary.b[prev_nn] + prev_nn_change
        subs_boundary.b[subs_nn] = subs_boundary.b[subs_nn] + subs_nn_change


    def manual_override(self, i1, i2, value):
        """ argh. just need to tweak starting parameters...
        """
        self.b[i1][i2] = value


    def convert_to_points(self):
        """ It is easier to do point-by-point vector-style work with the
            x and y points associated with each other, so you
            have just one column of data, with each data point
            being an (x,y) pair. This function converts
            into this point-by-point format."""
        assert not self._as_points
        # Couldn't I just do self.b.transpose() ?...
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


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef get_head_location(self, np.ndarray[dtype=np.float64_t, ndim=1] loc):
        """ Return the location of the head node and shift it by node._domain_width 
        if it exceeds the periodic b.c. """
        cdef:
            double x,y,x0,y0
            Py_ssize_t indx
            double period = self.head._domain_width
        indx = 1
        x,y = self.b[indx]
        x0 = self.head.position[0]
        y0 = self.head.position[1]
        if fabs(x-x0) > period/2.0:
            x0 = x0 + period if (x-x0) > 0.0 else x0 - period
        if fabs(y-y0) > period/2.0:
            y0 = y0 + period if (y-y0) > 0.0 else y0 - period
        loc[0] = x0
        loc[1] = y0


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef get_tail_location(self, np.ndarray[dtype=np.float64_t, ndim=1] loc):
        """ Return the location of the tail node and shift it by node._domain_width 
        if it exceeds the periodic b.c. """
        cdef:
            double x,y,x0,y0
            double period = self.tail._domain_width
            Py_ssize_t indx
        indx = len(self.b)-2
        x,y = self.b[indx]
        x0 = self.tail.position[0]
        y0 = self.tail.position[1]
        if fabs(x-x0) > period/2.0:
            x0 = x0 + period if (x-x0) > 0.0 else x0 - period
        if fabs(y-y0) > period/2.0:
            y0 = y0 + period if (y-y0) > 0.0 else y0 - period
        loc[0] = x0
        loc[1] = y0


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef get_node_location(self, Node node, np.ndarray[dtype=np.float64_t, ndim=1] xy):
        """ Return the location of the node and shift it by node._domain_width 
        if it exceeds the periodic b.c. """
        cdef:
            Boundary gb
            double x,y,period,x0,y0
            Py_ssize_t indx
        if node == self.head:
            indx = 1
        else:
            indx = len(self.b)-2
        period = node._domain_width
        x,y = self.b[indx]
        x0 = node.position[0]
        y0 = node.position[1]
        #x0, y0 = node.position
        if fabs(x-x0) > period/2.0:
            x0 = x0 + period if (x-x0) > 0.0 else x0 - period
        if fabs(y-y0) > period/2.0:
            y0 = y0 + period if (y-y0) > 0.0 else y0 - period
        xy[0] = x0
        xy[1] = y0
        #return np.array([x0,y0], dtype=np.float64)


    cpdef bool extrapolate(self):
        """ Extrapolate the grain boundary across the terminations at each end of the boundary.
            The boundary needs to be in 'point-by-point' format. """
        #assert self._as_points
        if self.head._type == empty:
            """ The grain boundary loops on itself. """
            self.b[0] = self.b[len(self.b)-2]
            self.b[len(self.b)-1] = self.b[1]
            return True
            
        cdef:
            np.ndarray[np.float64_t, ndim=1] head = np.empty((2), dtype=np.float64)
            np.ndarray[np.float64_t, ndim=1] tail = np.empty((2), dtype=np.float64)

        if self.head._type == domain_boundary:
            head = self.head.position
            tail = self.tail.position
        else:
            self.get_head_location(head) # self.head.position
            self.get_tail_location(tail) # self.tail.position

        #!!! TODO!!!: CAN I clean this up?
        self.b[0,0], self.b[0,1] = 2*head[0]-self.b[1,0], 2*head[1]-self.b[1,1]
        self.b[len(self.b)-1,0], self.b[len(self.b)-1,1] = 2*tail[0]-self.b[len(self.b)-2,0], 2*tail[1]-self.b[len(self.b)-2,1]

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
            np.ndarray[np.float64_t, ndim=2] b = np.empty((len(self.b),2), dtype=np.float64)
            np.ndarray[np.float64_t, ndim=2] v = np.zeros((len(self.b)-2,2), dtype=np.float64)
            Py_ssize_t i, stop, lenb
            double min_velocity = 0.0
            double D1norm2, D1x, D1y, D2x, D2y
    
        lenb = len(self.b)
        b = self.b
        
        stop = lenb-1
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

    def get_length(self):
        """ Returns the straight-line distance from first interior
        point to last interior point """
        return self._length

    def get_previous_length(self):
        """ Returns the straight-line distance from first interior
        point to last interior point before current iteration"""
        return self._prev_length

    cpdef get_scheduled_for_deletion(self):
        """ returns whether the gb is slated for deletion. """
        return self._to_be_deleted


    def get_average_velocity(self):
        """ calculate average velocity of points on the gb. """
        print np.mean(self.v)


    cdef void update_position(self, double k) except *:
        """ Using the velocity vector, update the positions of the interior
            points (i.e. not including the extrapolated points).
            k is the time increment (delta-t), using Bronsard-Wetton notation. """
        cdef double lx,ly
        cdef Py_ssize_t lenb = len(self.b)
        self.b[1:lenb-1] = self.b[1:lenb-1] + self.v*k
        if False:
            self._prev_length = self._length
            lx = self.b[lenb-2,0] - self.b[1,0]
            ly = self.b[lenb-2,1] - self.b[1,1]
            self._length = sqrt(lx*lx + ly*ly)

    cpdef densify(self, bool nodeless=False):
        """ Double the number of interior points for higher resolution
        and redistribute the points evenly along the length of the path.
        If nodeless==True, then use the values already stored in self.b[0] and
        self.b[-1] as the extrapolation end points rather than accessing
        the Node objects stored in self.head and self.tail """
        if verbose:
            print "densify: current length:", len(self.b)
        cdef Py_ssize_t new_length
        new_length   = round((len(self.b)-2)*2 + 2)
        self.interpolate(new_length, nodeless)

    cpdef interpolate(self, Py_ssize_t new_length, bool nodeless):
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
            self.b[0]  = (self.b[len(self.b)-2] - self.b[1])/2
            self.b[len(self.b)-1] = self.b[0]
        elif nodeless:
            pass
        elif self.head._type == triple_point:
            self.get_head_location(self.b[0]) # self.head.position
            self.get_tail_location(self.b[len(self.b)-1]) # self.tail.position
        else:
            self.b[0]  = self.head.position
            self.b[len(self.b)-1] = self.tail.position

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
            s_lessthan_ti = np.where(s<ti)[0]
            n = s_lessthan_ti[len(s_lessthan_ti)-1] # ti should always be greater than zero, and s[0]=0
            alpha_n = (ti - s[n])/(s[n+1]-s[n])
            self.b[i+1] = old_boundary[n]*(1-alpha_n) + old_boundary[n+1]*alpha_n

        if not nodeless:
            self.extrapolate()
        self.v = np.zeros([len(self.b)-2,2])
        
        
    cdef void coarsen(self, bool nodeless=False) except *:
        """ halve the number of interior points, but keep the two interior end points and
            the two exterior extrapolated points.
            If nodeless==True, then use the values already stored in self.b[0] and
            self.b[-1] as the extrapolation end points rather than accessing
            the Node objects stored in self.head and self.tail (which won't yet exist if 
            you are still building the graph, for example)."""

        cdef Py_ssize_t new_length
        new_length = round((len(self.b)-2)/2 + 2)
        if len(self.b) < 5:
            raise ValueError("Trying to coarsen boundary to fewer than 2 interior points.")
        self.interpolate(new_length, nodeless)
        

    cpdef check_spacing(self, bool nodeless=False):
        """ we need to refine our mesh when it gets too coarse, coarsen it when it gets dense,
            and cut it when it gets to 2 interior points. """

        cdef:
            Py_ssize_t lenb = len(self.b)
            np.ndarray[np.float64_t, ndim=2] b
            np.ndarray normd1x2 = np.zeros(lenb-2, dtype=np.float64)
            Py_ssize_t i, stop
            double min_velocity = 0.0
            double h, D1x, D1y, nu, mu, norm2
        
        b = self.b
        h = self._h
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
        
        if lenb < 5 and ((mu < h*h/10.0)
                          or  ((self._prev_length > self._length)
                                and  (self._length < h/10.0))):
            """ we're down to two interior points and need to kill the boundary """
            #print "delete"
            self._to_be_deleted = True
            return False

        if nu > 4.0:
            self.densify(nodeless)
        if nu < 1.0/4.0 and lenb > 4.0:
            self.coarsen(nodeless)


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


    cpdef will_die_alone(self):
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
    cdef double _domain_width
    def __init__(self, int node_type, double period=1.0e15):
        assert node_type in [empty, triple_point, domain_boundary]
        self._type = node_type
        self._id = str(uuid.uuid1())
        self.position = np.array([0.0, 0.0], dtype=np.float64)
        self.boundaries = []
        self.nodes = []
        self.ends = []
        self.endindx = []
        self._domain_width = period
        
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

    cpdef get_id(self):
        """ Return the id of this Node object. """
        return self._id

    cpdef get_type(self):
        """ return integer describing node type """
        return self._type

    cpdef get_number_dying_boundaries(self):
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
        cdef Boundary bndry
        for b in [b1,b2,b3]:
            bndry, indx, end = self.parse_boundary(b)
            #bndry.check_spacing(True)
            self.boundaries.append(bndry)
            self.endindx.append(indx)
            self.ends.append(end)

        if not self.fits_inside_triangle():
            print 'adjusting boundaries for node {0} at {1}'.format(self.get_id().split('-')[0], self.position)
            #bndry = self.get_shorty_boundary()
            bndry = self.get_obtuse_boundary()
            bndry.curve_thy_neighbors(self)
            #bndry.retract_from_node(self)
            
            #self.shift_closest_points()
            #for i in range(2):
            #    if not self.fits_inside_triangle():
            #        bndry = self.get_shorty_boundary()
            #        bndry.coarsen(True)

            #if not self.fits_inside_triangle():
            #    self.shift_closest_points()
            
        self.update_position()


    def shift_closest_points(self):
        """ We're having trouble making a Node that works, so shift all points
        to be the same distance from self.position and this should
        guarantee that a node solution is possible. """
        cdef:
            Boundary b
            Py_ssize_t indx, i
            np.ndarray[dtype=np.float64_t, ndim=1] point = np.empty((2))

        dist_to_node = []
        for i in range(3):
            self.get_closest_interior_point(i, point)
            dist_to_node.append( norm(point - self.position) )
        
        closest      = np.argmin(dist_to_node)
        closest_dist = min(dist_to_node)
        for i,b,indx in zip(range(3),self.boundaries,self.endindx):
            if i == closest:
                continue
            
            self.get_closest_interior_point(i, point)
            v = self.position - point
            shift_direction = v/norm(v)
            shift_distance  = norm(v) - closest_dist
            b.b[indx] = b.b[indx] + shift_direction*shift_distance
            
        
    def replace_boundary(self, Boundary a, Boundary b, int bend):
        """ Replace boundary a with b (for example after a concatenation). """
        cdef int indx
        indx = self.boundaries.index(a)
        self.boundaries[indx] = b
        self.ends[indx] = bend
        self.endindx[indx] = 1 if bend == head_end else len(b.b)-2
        


    cpdef parse_boundary(self, bdict):
        """ Parse dictionary, saving boundary and info on proper end point
        and set_termination on the boundary so that it points to this triple-point. """
        assert bdict['end'] in [head_end, tail_end]
        cdef Boundary bndry
        bndry = bdict['b']
        array_position = 1 if bdict['end'] == head_end else len(bndry.b)-2
        bndry.set_termination( bdict['end'], self )
        return bndry, array_position, bdict['end']


    cpdef set_position(self, np.ndarray[dtype=np.float64_t, ndim=1] position):
        """ Manually set the position of the Node (usually this is done
        instead via calculation in update_position) """
        self.position = position
        

    @cython.boundscheck(False)
    @cython.wraparound(False)
    #cdef void get_closest_interior_point(self, Py_ssize_t indx, double[:] closest_point) except *:
    cdef void get_closest_interior_point(self, Py_ssize_t indx, np.ndarray[dtype=np.float64_t, ndim=1] closest_point) except *:
        """ Return the point on the boundary closest to the node (i.e. point [1] or 
        point [-2]) and shift it by self._domain_width if it exceeds the periodic b.c. """
        cdef:
            Boundary gb
            double x,y,x0,y0,domain_width
            #np.ndarray[dtype=np.float64_t, ndim=1] closest_point
            #double [:] closest_point = np.empty((2))
            Py_ssize_t bendindx
                    
        gb   = self.boundaries[indx]
        bendindx = 1 if self.ends[indx] == head_end else len(gb.b)-2
        x, y = gb.b[bendindx]
        x0, y0 = self.position
        domain_width = self._domain_width 
        if fabs(x-x0) > domain_width/2.0:
            x = x - domain_width if (x-x0) > 0.0 else x + domain_width
                
        if fabs(y-y0) > domain_width/2.0:
            y = y - domain_width if (y-y0) > 0.0 else y + domain_width

        #closest_point = np.array([x,y], dtype=np.float64)
        closest_point[0] = x
        closest_point[1] = y
        #return closest_point

    #cdef get_triangle(self, double[:] p0, double[:] p1, double[:] p2):
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef get_triangle(self, np.ndarray[dtype=np.float64_t, ndim=1] p0,
        np.ndarray[dtype=np.float64_t, ndim=1] p1, np.ndarray[dtype=np.float64_t, ndim=1] p2):
        """ Returns a tuple of the three end points of the Node in
        counter-clockwise order """
        #cdef:
            #double [:] p0 = np.empty((2))
            #double [:] p1 = np.empty((2))
            #double [:] p2 = np.empty((2))
            #np.ndarray[dtype=np.float64_t, ndim=1] p0 = np.empty((2))
            #np.ndarray[dtype=np.float64_t, ndim=1] p1 = np.empty((2))
            #np.ndarray[dtype=np.float64_t, ndim=1] p2 = np.empty((2))
        self.get_closest_interior_point(0,p0)
        self.get_closest_interior_point(1,p1)
        self.get_closest_interior_point(2,p2)
        return p0,p1,p2


    @cython.boundscheck(False)
    @cython.wraparound(False) # TODO: replace self.bendindx with only non-negative values
    cpdef int update_position( self, bool verbose=False, bool check_angles=False ) except *:
        """ Calculate the new location of the triple-point. """
        cdef:
            Boundary gb, bndry
            double x,y,theta,theta1,theta2,theta3
            double l1, l2, l3
            double alpha, delta, beta, dilation
            np.ndarray[np.float64_t, ndim=2] b
            #double[:,::1] b
            np.ndarray[np.float64_t, ndim=1] b1N = np.empty((2))
            np.ndarray[np.float64_t, ndim=1] b2N = np.empty((2))
            np.ndarray[np.float64_t, ndim=1] b3N = np.empty((2))
            np.ndarray[np.float64_t, ndim=1] v13
            np.ndarray[np.float64_t, ndim=1] v21
            np.ndarray[np.float64_t, ndim=1] v23
            np.ndarray[np.float64_t, ndim=1] C
            #np.ndarray[np.float64_t, ndim=2] rotation
            Py_ssize_t i
            Py_ssize_t bendindx
            
        if self._type == domain_boundary:
            """ Find where the grain boundary 'gb' would intersect the domain boundary if
            it were to arrive normal to the domain boundary.
            self.b needs to be in 'point-by-point' format. """
            gb = self.b
            b = gb.b
            bendindx = 1 if self.bend == head_end else len(b)-2
            x, y = b[bendindx,0], b[bendindx,1]
            theta = atan(fabs(y/x))
            self.position[0] = sign(x)*cos(theta)
            self.position[1] = sign(y)*sin(theta)
        else:
            if 0 == self.fits_inside_triangle():
                for gb in self.boundaries:
                    gb.check_spacing()
                    if gb._to_be_deleted:
                        return 0
                print 'In update_position, adjusting boundaries for node {0} at {1}'.format(self.get_id().split('-')[0], self.position)
                #bndry = self.get_shorty_boundary()
                bndry = self.get_obtuse_boundary()
                #bndry.curve_thy_neighbors(self)
                bndry.retract_from_node(self)
                bndry.extrapolate()

            theta1 = theta2 = theta3 = 2.0*pi/3.0
            self.get_closest_interior_point(0,b1N)
            self.get_closest_interior_point(1,b2N)
            self.get_closest_interior_point(2,b3N)
            #print "  node ID:", self.get_id()
            #print "   triangle:", self.get_triangle()
            #gb = self.boundaries[0]
            #b1N = gb.b[self.endindx[0]]
            #gb = self.boundaries[1]
            #b2N = gb.b[self.endindx[1]]
            #gb = self.boundaries[2]
            #b3N = gb.b[self.endindx[2]]
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
                print "  node ID, location:", self.get_id().split('-')[0], self.position
                #print "   triangle:", self.get_triangle()
                print "  C is:", C
                print "  b1N, b2N, b3N:", b1N, b2N, b3N
                raise ValueError("Bring points together. Your triangle of endpoints is too open to contain the triple point.")
            else:
                #print "   good: b1N, b2N, b3N:", b1N, b2N, b3N
                #print "   good: l2/l1, l3/l1:", l2/l1, l3/l1
                pass
                
            self.position = C
            
    def get_position(self):
        """ Just for testing. """
        return self.position
        
    cdef int fits_inside_triangle(self):
        """ If the triangle formed by the initial endpoints of 
        the three intersecting boundaries is too oblique (i.e. if 
        one of the angles is larger than 120 degrees, then the 
        node point will not fit inside the triangle. """
        cdef:
            double theta_node, l1, l2, l3
            double phi1, phi2, phi3
            np.ndarray[dtype=np.float64_t, ndim=1] p1
            np.ndarray[dtype=np.float64_t, ndim=1] p2
            np.ndarray[dtype=np.float64_t, ndim=1] p3
            np.ndarray[dtype=np.float64_t, ndim=1] v13
            np.ndarray[dtype=np.float64_t, ndim=1] v21
            np.ndarray[dtype=np.float64_t, ndim=1] v23
            int true=1, false=0
        theta_node = 2.0*pi/3.0
        p1 = np.empty((2))
        p2 = np.empty((2))
        p3 = np.empty((2))
        self.get_triangle(p1,p2,p3)
        v13 = p3 - p1 # vector running from pt1 to pt3
        v21 = p1 - p2 # vector running from pt2 to pt1
        v23 = p3 - p2 # vector running from pt2 to pt3
        l1 = norm( v23 )
        l2 = norm( v13 )
        l3 = norm( v21 )
        #l1 = sqrt(

        """ If any of the following three angles is negative, then the node 
        won't fit in the interior of the triangle. """
        phi1 = theta_node - acos(-(v21[0]*v13[0]+v21[1]*v13[1])/l2/l3) # np.arccos( dot(v21,-v13)/l2/l3) )
        phi2 = theta_node - acos( (v21[0]*v23[0] + v21[1]*v23[1])/l1/l3 )
        phi3 = theta_node - acos( (v23[0]*v13[0] + v23[1]*v13[1])/l1/l2 )
        if (phi1 < 0) or (phi2 < 0) or (phi3 < 0):
            return false
        else:
            return true
            

    def get_problematic_boundaries(self):
        """ Return the two Boundary objects that should be brought closer to 
        the node to make self.fits_inside_triangle() return True. """
        theta_node = 2.0*pi/3.0
        p1 = np.zeros((2), dtype=np.float64)
        p2 = np.zeros((2), dtype=np.float64)
        p3 = np.zeros((2), dtype=np.float64)
        self.get_triangle(p1,p2,p3)
        v13 = p3 - p1 # vector running from pt1 to pt3
        v21 = p1 - p2 # vector running from pt2 to pt1
        v23 = p3 - p2 # vector running from pt2 to pt3
        l1 = norm( v23 )
        l2 = norm( v13 )
        l3 = norm( v21 )
        phi1 = theta_node - acos(-(v21[0]*v13[0]+v21[1]*v13[1])/l2/l3) # np.arccos( dot(v21,-v13)/l2/l3) )
        phi2 = theta_node - np.arccos( np.dot(-v21,-v23)/l1/l3 )
        phi3 = theta_node - np.arccos( np.dot(v23,v13)/l1/l2 )
        if (phi1 < 0):
            return self.boundaries[1], self.boundaries[2]
        elif (phi2 < 0):
            return self.boundaries[0], self.boundaries[2]
        elif (phi3 < 0):
            return self.boundaries[0], self.boundaries[1]
        else:
            return False


    cpdef Boundary get_shorty_boundary(self):
        """ Return the shortest Boundary object connected to the current Node.
        (length is calculated as the crow flies from head to tail). """
        cdef Boundary b1,b2,b3
        
        b1 = self.boundaries[0]
        b2 = self.boundaries[1]
        b3 = self.boundaries[2]

        l1 = norm( b1.b[-1] - b1.b[0] )
        l2 = norm( b2.b[-1] - b2.b[0] )
        l3 = norm( b3.b[-1] - b3.b[0] )
        
        if (l1 < l2) and (l1 < l3):
            return self.boundaries[0]
        elif (l2 < l1) and (l2 < l3):
            return self.boundaries[1]
        elif (l3 < l2) and (l3 < l1):
            return self.boundaries[2]
        else:
            raise ValueError("All boundaries have length {0}? WTF?".format(l1))

    cpdef Boundary get_obtuse_boundary(self):
        """ The end points of the three boundaries form a triangle that
        surrounds the Node. Return the Boundary object whose
        end point is the vertex of the triangle with the largest angle.
        This method will be typically only used when this maximum angle
        is actually >120 degrees, so even more than obtuse."""
        cdef Boundary b

        theta_node = 2.0*pi/3.0
        p1 = np.zeros((2), dtype=np.float64)
        p2 = np.zeros((2), dtype=np.float64)
        p3 = np.zeros((2), dtype=np.float64)
        self.get_triangle(p1,p2,p3)
        v13 = p3 - p1 # vector running from pt1 to pt3
        v21 = p1 - p2 # vector running from pt2 to pt1
        v23 = p3 - p2 # vector running from pt2 to pt3
        l1 = norm( v23 )
        l2 = norm( v13 )
        l3 = norm( v21 )

        """ If any of the following three angles is negative, then the node 
        won't fit in the interior of the triangle. """
        phi1 = theta_node - acos(-(v21[0]*v13[0]+v21[1]*v13[1])/l2/l3) # np.arccos( dot(v21,-v13)/l2/l3) )
        phi2 = theta_node - np.arccos( np.dot(-v21,-v23)/l1/l3 ) #TODO: Hand code dot product!!!
        phi3 = theta_node - np.arccos( np.dot(v23,v13)/l1/l2 )
        if (phi1 < 0):
            return self.boundaries[0]
        elif (phi2 < 0):
            return self.boundaries[1]
        elif (phi3 < 0):
            return self.boundaries[2]
        else:
            raise ValueError("All angles are less than 120 degrees, so why are you running this method?")

cdef class ThinFilm(object):
    """ A class for simulating thin-film grain growth using the boundary
        front-tracking method. The physics follows what came out of Carl
        Thompson's lab, but the detailed implementation is an attempt to
        follow Bronsard & Wetton.
    """
    cdef:
        double _t                # current simulation time
        double _k                # time step (fixed for junction iteration)
        double _h                # average resolution of discretization
        double _domain_width     # in case of periodic b.c., width of sim. region
        double _convergence      # convergence level for junction iteration
        list boundaries, nodes
        
    def __init__(self, boundaries=None, nodes=None):
        if boundaries is not None and nodes is not None:
            self.add_boundaries(boundaries)
            self.add_nodes(nodes)
        self._k = 0.0003
        self._h = 1.0/16.0
        self._t = 0.0
        self._domain_width = 1.0
        self._convergence = 1.0e-5

    def set_convergence(self, double level):
        """ sum of changes on all boundaries must be less than this value. """
        self._convergence = level

    def initialize_graph(self, num_grains=2500, num_interior_points_per_boundary=8):
        """ Generate an initial set of Boundary and Node objects
        using a Voronoi diagram from a set of num_grains randomly spaced
        points on a domain with periodic boundary conditions. 
        The width and height of the simulation domain will be set 
        equal to the square root of n (so that one spatial unit of 
        the domain is approximately equal to the average grain 
        diameter in the initial graph)."""
        # magnification factor from Voronoi diagram to our simulation region:
        mag = np.sqrt(num_grains)
        self._domain_width = mag
        segments, nodelist, node_segments = voronoi.periodic_diagram_with_nodes(num_grains)
        boundaries = []
        nodes = []
        for s in segments:
            """ 
            each segment is a dictionary containing:
                points: list of floats, [(x0,y0), (x1,y1)]
                head: int, node id number
                tail: int, node id number """
            
            p0 = np.array(s['points'][0], dtype=np.float64)*mag
            p1 = np.array(s['points'][1], dtype=np.float64)*mag
            n  = num_interior_points_per_boundary
            
            b  = np.zeros((n+2, 2), dtype=np.float64)
            for i in range(n):
                b[i+1,:] = p0 + (i+1.0)/(n+1.0)*(p1-p0)

            # These two spaces in b will be over-written by extrapolated points, 
            # but for now I'll just store the node locations
            b[0,:] = p0
            b[len(b)-1,:] = p1
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
            v1 = np.array(b1b[len(b1b)-2]) - np.array(b1b[1])
            v2 = np.array(b2b[len(b2b)-2]) - np.array(b2b[1])
            v3 = np.array(b3b[len(b3b)-2]) - np.array(b3b[1])
            
            
            """ Flip so that all vectors point away from triple point to 
            enable our cross-product test below """
            if b1end == head_end:
                v1 = -1*v1
            if b2end == head_end:
                v2 = -1*v2

            if b1end == head_end:
                node_location = b1b[0]
            else:
                node_location = b1b[len(b1b)-1]
            # print 'Node location is:', node_location


            """ Order counter-clockwise around node: """
            if v1[0]*v2[1] < v2[0]*v1[1]: # same as np.cross(v1,v2) < 0:
                b_temp = b2
                b_end_temp = b2end
                b2 = b3
                b2end = b3end
                b3 = b_temp
                b3end = b_end_temp
            
            node = Node(node_type=triple_point, period=mag)
            node.set_position(node_location)
            node.add_boundaries(
                dict(b=b1, end=b1end),
                dict(b=b2, end=b2end),
                dict(b=b3, end=b3end))

            nodes.append( node )

        self.add_boundaries(boundaries)
        self.add_nodes(nodes)

    def add_boundaries(self, boundaries):
        """ Add a list of Boundary objects to the ThinFilm object."""
        self.boundaries = boundaries


    def add_nodes(self, nodes):
        """ Add a list of Node objects to the ThinFilm object."""
        self.nodes = nodes
        

    cpdef delete_single_boundary(self, Boundary gb):
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
                        self.nodes[len(self.nodes)-1].add_single_boundary(dict(b=gb2, end=end))
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
                head_prev_end = gb.head.ends[head_index-1]
                if head_prev_end == head_end:
                    head_prev_endindx  = 1
                    head_prev_nodeindx = 0
                else:
                    head_prev_endindx  = len(head_prev_boundary.b)-2
                    head_prev_nodeindx = len(head_prev_boundary.b)-1
                
                subs_indx = head_index+1 if head_index+1 < len(gb.head.boundaries) else 0
                head_subs_boundary = gb.head.boundaries[subs_indx]
                head_subs_end = gb.head.ends[subs_indx]
                if head_subs_end == head_end:
                    head_subs_endindx  = 1
                    head_subs_nodeindx = 0
                else:
                    head_subs_endindx  = len(head_subs_boundary.b)-2
                    head_subs_nodeindx = len(head_subs_boundary.b)-1
                
                tail_index = gb.tail.boundaries.index(gb)
                tail_prev_boundary = gb.tail.boundaries[tail_index-1]
                tail_prev_end = gb.tail.ends[tail_index-1]
                if tail_prev_end == head_end:
                    tail_prev_endindx  = 1
                    tail_prev_nodeindx = 0
                else:
                    tail_prev_endindx  = len(tail_prev_boundary.b)-2
                    tail_prev_nodeindx = len(tail_prev_boundary.b)-1

                subs_indx = tail_index+1 if tail_index+1 < len(gb.tail.boundaries) else 0
                tail_subs_boundary = gb.tail.boundaries[subs_indx]
                tail_subs_end = gb.tail.ends[subs_indx]
                if tail_subs_end == head_end:
                    tail_subs_endindx  = 1
                    tail_subs_nodeindx = 0
                else:
                    tail_subs_endindx  = len(tail_subs_boundary.b)-2
                    tail_subs_nodeindx = len(tail_subs_boundary.b)-1

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
                new_head = p1+(p2-p1)/3.0
                new_tail = p2+(p1-p2)/3.0
                new_node_head = p1+(p2-p1)/6.0
                new_node_tail = p2+(p1-p2)/6.0
                print "new head:", new_head
                print "new tail:", new_tail
                new_gb = Boundary(np.array([new_node_head,new_head,new_tail,new_node_tail]), as_points=True)

                """ Reset the 'extrapolated' points in the surviving boundaries so that when
                the boundaries are added to the node, they can be adjusted to ensure
                that a node forms between the proper end points """
                head_subs_boundary.b[head_subs_nodeindx] = new_node_head
                tail_prev_boundary.b[tail_prev_nodeindx] = new_node_head
                head_prev_boundary.b[head_prev_nodeindx] = new_node_tail
                tail_subs_boundary.b[tail_subs_nodeindx] = new_node_tail
                
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
                        self.nodes[len(self.nodes)-1].add_single_boundary(dict(b=gb2, end=end))
                    else:
                        if verbose: print "  gb2 == gb for gb2={0}".format(gb2)
            return True

    cdef int concatenate_boundaries(self, Boundary a, int aend, Boundary b, int bend) except *:
        """ Stitch boundaries a and b together, connecting aend to bend and
        equalizing the number of points of each boundary so that the number 
        of interior points of the combined boundary will still be a power of 2
        for easier densifying/coarsening later on. """
        cdef:
            np.ndarray[np.float64_t, ndim=2] new_array
            Boundary new_boundary, gb
            Node a_surviving_node, b_surviving_node
            int a_surviving_end, b_surviving_end, indx
        
        print "CONCATENATING!!!"
        while len(a.b) > len(b.b):
            b.densify()
        while len(a.b) < len(b.b):
            a.densify()
        
        if aend == tail_end:
            a_surviving_end = head_end
            if bend == head_end:
                b_surviving_end = tail_end
                new_array = np.concatenate((a.b[:len(a.b)-1], b.b[1:]), axis=0)
            else:
                b_surviving_end = head_end
                new_array = np.concatenate((a.b[:len(a.b)-1], b.b[len(b.b)-1::-1]), axis=0)
        else:
            a_surviving_end = tail_end
            if bend == head_end:
                b_surviving_end = tail_end
                new_array = np.concatenate((a.b[len(a.b)-1::-1], b.b[1:]), axis=0)
            else:                    
                b_surviving_end = head_end
                new_array = np.concatenate((a.b[len(a.b)-1::-1], b.b[len(a.b)-1::-1]), axis=0)

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

    def delete_several_boundaries(self, to_be_deleted):
        """ Parse multiple successive deletion operations
        that occurred within one implicit time-step. """
        
        cdef Boundary bndry, sb, b1, b2, b3, b_cw, survivor 
        cdef Node tp, node1, node2, new_node
            
        for bndry in to_be_deleted:
            if domain_boundary in [bndry.head._type, bndry.tail._type]:
                raise TypeError, "Multiple successive deletion involving domain boundary is not implemented."
        
        """ Possible options for successive deletion of connected boundaries
        that are not connected to the domain boundary:
        4* followed by 5 
            (as is what happens in B&W 7-grain test near time 1.8)
        5 followed by 4*
        """
        if len(to_be_deleted) == 3:
            """ 4* followed by 5. This looks kinda like a three-sided bubble, with 
            one surviving boundary attached to each node.
            First figure out ordering around the bubble.
            Identify a "node set point" to set the extrapolated points of the survivors.
            Then create new Node and attach survivors. This will move the interior points
                of the survivors around the 'node set point' to ensure success of 
                Node.update_position() when it is called for the first time. """
    

            b1, b2, b3 = to_be_deleted
            sb, sb_endindx, sb_end = b1.get_subsequent_boundary(b1.head)
            if sb in [b2,b3]:
                """ b1 runs ccw head->tail around bubble. """
                node1 = b1.head
                node2 = b1.tail
            else:
                node1 = b1.tail
                node2 = b1.head
            
            if b3.has_node(node1):
                """ The three boundaries run b1->b2->b3 ccw around bubble,
                so the boundary clockwise from b1 is b3. """
                b_cw = b3
            else:
                b_cw = b2           
                
            if node1 == b_cw.tail:
                """ b_cw also runs ccw head-> tail """
                ordered_nodes = [node1, node2, b_cw.head]
            else:
                ordered_nodes = [node1, node2, b_cw.tail]
                
            endpoints = np.zeros((3,2), dtype=np.float64)
            for i,tp in enumerate(ordered_nodes):
                survivor, survivor_end = tp.get_sole_survivor()
                indx = 1 if survivor_end == head_end else len(survivor.b)-2
                endpoints[i] = survivor.b[indx]
            new_setpoint = (endpoints[0] + endpoints[1] + endpoints[2])/3.0
            print "new setpoint:", new_setpoint
            print " endpoints:", endpoints
            
            survivors     = []
            survivor_ends = []
            for i,tp in enumerate(ordered_nodes):
                survivor, survivor_end = tp.get_sole_survivor()
                survivors.append(survivor)
                survivor_ends.append(survivor_end)
                indx = 0 if survivor_end == head_end else len(survivor.b)-1
                survivor.b[indx] = new_setpoint
                self.nodes.remove(tp)

            self.boundaries.remove(b1)
            self.boundaries.remove(b2)
            self.boundaries.remove(b3)
            new_node = Node(node_type=triple_point)
            new_node.add_boundaries(
                dict(b=survivors[0], end=survivor_ends[0]),
                dict(b=survivors[1], end=survivor_ends[1]),
                dict(b=survivors[2], end=survivor_ends[2]))
            self.nodes.append(new_node)


        elif len(to_be_deleted) == 4:
            """ 5 followed by 4* """
            pass
        else:
            raise ValueError, "More than 4 boundaries deleted at once? NOT IMPLEMENTED!!!"


    def delete_boundaries(self, list to_be_deleted):
        """ Remove boundaries through surgery. """
        cdef Boundary gb
        for gb in to_be_deleted:
            if gb.will_die_alone():
                if verbose: print "  killing single boundary."
                self.delete_single_boundary(gb)
                to_be_deleted.remove(gb)
                
        if len(to_be_deleted) > 2:
            """This wouldn't happen in explicit stepping with 4RK time condition,
            since all deletions should be combinations of the elementary deletions 
            that B&W discussed. However, when doing implicit time-stepping, you can 
            essentially step through two successive deletion operations in one step."""
            self.delete_several_boundaries(to_be_deleted)
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
        cdef:
            Boundary b
            Py_ssize_t i,j
            
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

    def get_time(self):
        """ Return elapsed simulation time. """
        return self._t

    def set_resolution(self, h):
        """ set resolution parameter h """
        self._h = h


    def test_junction_iteration(self):
        self.initialize_iteration_buffers()
        self.implicit_solve_junction_iteration()
        

    def junction_iteration(self):
        """Use the "junction iteration" method of 
        Bronsard and Wetton (which is basically what
        Frost, Thompson, and Carel used) to do implicit
        time-stepping over a time increment 'timestep' """

        cdef int i, ilimit
        
        i      = 0
        ilimit = 1000
        
        self.initialize_iteration_buffers()
        self.implicit_solve_junction_iteration()
        self.update_nodes()
        self.extrapolate_all_boundaries()
        while self.converged() != 1:
            i += 1
            if i > ilimit:
                raise ValueError("Did not converge within {0} iterations".format(ilimit))

            self.implicit_solve_junction_iteration()
            self.update_nodes()
            self.extrapolate_all_boundaries()

        self._t += self._k
        if verbose: 
            print 'num iterations:', i
        
    def extrapolate_all_boundaries(self):
        """ just for testing. """
        cdef Boundary b
        for b in self.boundaries:
            b.extrapolate()
        
            
    cdef void implicit_solve_junction_iteration(self) except *:
        """ set up and solve the matrix for implicitly solving for the 
        boundary location using B&W's junction iteration model """

        cdef Boundary b
        cdef double lx,ly
        
        for b in self.boundaries:
            b.implicit_step_junction_iteration(self._k)
            lx = b.b[len(b.b)-2,0] - b.b[1,0]
            ly = b.b[len(b.b)-2,1] - b.b[1,1]
            b._length = sqrt(lx*lx + ly*ly)


    cdef int converged(self) except *:
        """ Check if aggregate difference between current and previous (Xi) boundary locations
        is within the limit set by self._convergence """
        cdef:
            double max_change, change
            Boundary b
        max_change = 0.0
        for b in self.boundaries:
            change = b.get_iteration_change()
            if change > max_change:
                max_change = change

        if max_change < self._convergence:
            return 1
        else:
            return 0
        
        
    cpdef initialize_iteration_buffers(self):
        """ Reset the previous-time (Xn) boundary array
        within each Boundary object to equal the 
        current value of Boundary.b. """

        cdef Boundary b
        for b in self.boundaries:
            b._prev_length = b._length
            b.extrapolate()
            b.initialize_iteration_buffer()
        

    cpdef update_grain_boundaries(self):
        """ Cycle through the list of grain boundaries and update
            their locations.
            
            This uses explicit time-stepping, so it's really only
            good for the simple few-boundary tests from Bronsard & Wetton."""
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
    

"""
   ----------------------------------------------
"""
def test_update_node_position(p1,p2,p3):
    """ Send in three points (x0,y0) and see if they would 
    work for a triple-point. """
    cdef:
        Boundary b1, b2, b3 
        Node node
    b1 = Boundary(np.zeros([2,4]))
    b2 = Boundary(np.zeros([2,4]))
    b3 = Boundary(np.zeros([2,4]))
    b1.manual_override(1,0,p1[0])
    b1.manual_override(1,1,p1[1])
    b2.manual_override(1,0,p2[0])
    b2.manual_override(1,1,p2[1])
    b3.manual_override(1,0,p3[0])
    b3.manual_override(1,1,p3[1])
    
    node = Node(node_type=type_triple_point())
    node.add_boundaries(dict(b=b1, end=get_head_end()),
                             dict(b=b2, end=get_head_end()),
                             dict(b=b3, end=get_head_end()))


