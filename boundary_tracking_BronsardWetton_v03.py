from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

"""Try to copy fig 4 from Bronsard & Wetton
    "A Numerical Method for Tracking Curve Networks Moving with Curvature Motion"
    Journal of Computational Physics vol. 120, pp. 66 (1995)
    
    v03: boundary objects!
    """


sigma = np.linspace(0,1,15)
boundary1 = np.array([1-sigma, np.sin(np.pi*sigma)**2/4])
boundary2 = np.array([-1/2*(1-sigma), np.sqrt(3)/2*(1-sigma)])
boundary3 = np.array([-1/2*(1-sigma), -np.sqrt(3)/2*(1-sigma)])

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

""" Tack on elements to beginning and end of array as place-holders for
    the extrapolated points. """
b1 = np.array([ [0]+list(boundary1[0][1:-1])+[0], [0]+list(boundary1[1][1:-1])+[0] ])
b2 = np.array([ [0]+list(boundary2[0][1:-1])+[0], [0]+list(boundary2[1][1:-1])+[0] ])
b3 = np.array([ [0]+list(boundary3[0][1:-1])+[0], [0]+list(boundary3[1][1:-1])+[0] ])
#boundaries = [b1, b2, b3]
boundaries = [boundary1, boundary2, boundary3]

def norm(n):
    """ n is assumed to be a two-element array or list. """
    return np.sqrt(n[0]**2 + n[1]**2)

class Boundary:
    """ Pass in a 2-d array of x-y values of the initial boundary points. By default,
        this array is assumed to be a two-column array, with column0 for x and
        column1 for y. If as_points=True, then the format is already
        converted into a 1-D array of (x,y) values, with one (x,y) value for each point. """
    def __init__(self, b_initial, as_points=False):
        assert type(b_initial) == np.ndarray
        assert not as_points # not yet implemented
        self._as_points = False
        self.b = np.array([ [0]+list(b_initial[0][1:-1])+[0], [0]+list(b_initial[1][1:-1])+[0] ])
        self.convert_to_points()
        self._head_triple_point = [None,None]
        self._tail_triple_point = [None,None]
        

    def convert_to_points(self):
        """ It is easier to do point-by-point vector-style work with the
            x and y points associated with each other, so you
            have just one column of data, with each data point
            being an (x,y) pair. This function converts
            into this point-by-point format."""
        if self._as_points:
            return False
        self.b = np.array(zip(list(self.b[0]), list(self.b[1])))
        self._as_points = True


    def convert_to_columns(self):
        """ Undoes function 'convert_to_points' for easier plotting."""
        if not self._as_points:
            return False
        self.b = np.array(zip(*self.b))
        self._as_points = False


    def get_edgepts(self):
        """ By 'edge', I mean edge of the simulation region, i.e. the domain boundary.
            self.b needs to be in 'point-by-point' format. """
        assert self._as_points
        x, y = self.b[1][0], self.b[1][1]
        theta = np.arctan(np.abs(y/x))
        return np.sign(x)*np.cos(theta), np.sign(y)*np.sin(theta)


    def extrapolate(self):
        """ Extrapolate across the triple points at each end of the boundary.
            boundary needs to be in 'point-by-point' format. """
        assert self._as_points
        self._head_triple_point = self.get_edgepts() # fake triple point, hits domain boundary
        htp, ttp = self._head_triple_point, self._tail_triple_point
        #!!! TODO!!!: CAN I clean this up?
        self.b[0][0], self.b[0][1] = 2*htp[0]-self.b[1][0], 2*htp[1]-self.b[1][1]
        self.b[-1][0], self.b[-1][1] = 2*ttp[0]-self.b[-2][0], 2*ttp[1]-self.b[-2][1]
    

    def get_velocity(self):
        """ Calculate the instantaneous velocity vector for
            each point on the grain boundary.
            b needs to be in 'point-by-point' format.
            This will return an array in point-by-point format
            with n-2 rows, where n is the number of rows
            in the boundary array (which has the 2
            extrapolated extra points.) """
        assert self._as_points
        v = []
        for i in range(1,len(self.b)-1):
            D2 = self.b[i+1] - 2.0*self.b[i] + self.b[i-1]    # numerical second derivative (w/o normalization)
            D1 = (self.b[i+1] - self.b[i-1])/2.0         # numerical first derivative (w/o normalization)
            D1norm2 = D1[0]**2.0 + D1[1]**2.0
            v.append( D2/D1norm2 )             # normalizations cancel since we take ratio
        self.v = np.array(v)


    def update_position(self, k):
        """ Using the velocity vector, update the positions of the interior
            points (i.e. not including the extrapolated points).
            k is the time increment (delta-t), using Bronsard-Wetton notation. """
        self.b[1:-1] += self.v*k


    def densify_boundary(self):
        """ double the number of points """
        #for i in range(1, len(b)-2, 2):
        pass


    def coarsen_boundary(self):
        """ halve the number of points """
        for i in range(2,len(self.b)-2):
            # remove points...
            pass        


    def check_spacing(self):
        """ we need to refine our mesh when it gets too coarse, coarsen it when it gets dense,
            and cut it when it gets to 2 interior points. """

        if len(self.b) < 5:
            """ we're down to two interior points (or fewer? hope that's not possible...) """
            print "do surgery."
            return True
            
        coarsen = densify = False
        for i in range(1,len(self.b)-1):
            D1 = np.abs( (self.b[i+1] - self.b[i-1])/2.0/h )
            if D1 > 4:
                densify = True
            elif D1 < 1/4:
                coarsen = True
        if densify and coarsen: raise ValueError("Can't densify and coarsen at same time...")
        if densify:
            self.densify_boundary()
        if coarsen:
            self.coarsen_boundary()


    def update_triple_point(self, end, position):
        """ Update the location of the triple point specified in 'end'
            (this should be either 'head' or 'tail').
            """
        assert end in ['head', 'tail']
        if end == 'head':
            self._head_triple_point = position
        else:
            self._tail_triple_point = position


class TriplePoint:
    """ A class for managing the triple points where three grain boundaries meet.
        Pass in three dictionaries, each formatted like so:
        b1 = dict( b=Boundary(), end='head' )
        
        where Boundary is a Boundary object and 'end' can be 'head' or 'tail' to
        specify which end of the boundary is connected to the triple point. """
    def __init__(self, b1, b2, b3):
        self.parse_boundaries(b1, b2, b3)
        self.position = [None, None]
        self.update_position()


    def parse_boundaries(self, b1, b2, b3):
        self.b1, self.b1pt, self.b1end = self.get_boundary(b1)
        self.b2, self.b2pt, self.b2end = self.get_boundary(b2)
        self.b3, self.b3pt, self.b3end = self.get_boundary(b3)
    

    def get_boundary(self, bdict):
        """ Parse dictionary, saving boundary and info on proper end point. """
        assert bdict['end'] in ['head', 'tail']
        array_position = 1 if bdict['end'] == 'head' else -2
        return bdict['b'], array_position, bdict['end']


    def update_boundary_triple_points(self):
        """ Update the appropriate triple point values for each of the Boundary objects. """
        self.b1.update_triple_point(self.b1end, self.position)
        self.b2.update_triple_point(self.b2end, self.position)
        self.b3.update_triple_point(self.b3end, self.position)


    def update_position( self, verbose=False ):
        """ Send in the list of boundaries, each of which ends at the center-point,
            and use them to calculate the new center-point. """
        theta1 = theta2 = theta3 = 2*np.pi/3
        b1N = self.b1.b[self.b1pt]
        b2N = self.b2.b[self.b2pt]
        b3N = self.b3.b[self.b3pt]
        v13 = b3N - b1N # vector running from pt1 to pt3
        v21 = b1N - b2N # vector running from pt2 to pt1
        v23 = b3N - b2N # vector running from pt2 to pt3
        l1 = norm( v23 )
        l2 = norm( v13 )
        l3 = norm( v21 )
        alpha = np.arccos(np.dot(-v21, v13)/l2/l3) #np.arccos( (l2**2 + l3**2 - l1**2)/(2*l2*l3) )
        delta = theta2 - alpha
        beta = np.arctan( np.sin(delta)/(l3*np.sin(theta3)/(l2*np.sin(theta1))+np.cos(delta)) )
        dilation = np.sin(np.pi - beta - theta1)/np.sin(theta1)
        if np.cross(v23,v21) > 0:
            # rotate clockwise
            beta *= -1
        rotation = np.matrix([[np.cos(beta), -np.sin(beta)],[np.sin(beta), np.cos(beta)]])
        C = b2N + dilation*np.array(np.dot(rotation, v21))[0]
        if verbose:
            print 'points:', b1N, b2N, b3N
            print 'angles are in degrees:'
            print 'alpha:', alpha*180/np.pi
            print 'delta:', delta*180/np.pi
            print 'beta:', beta*180/np.pi
            print 'l1, l2, l3:', l1, l2, l3
            print 'dilation:', dilation
            print 'new triple point location:', C

        vc1, vc2, vc3 = C-b1N, C-b2N, C-b3N
        t1 = np.arccos(np.dot(vc2,vc1)/norm(vc2)/norm(vc1))*180/np.pi
        t2 = np.arccos(np.dot(vc3,vc2)/norm(vc3)/norm(vc2))*180/np.pi
        t3 = np.arccos(np.dot(vc3,vc1)/norm(vc3)/norm(vc1))*180/np.pi
        if (np.round(t1) != np.round(theta1*180/np.pi)
            or np.round(t2) != np.round(theta2*180/np.pi)
            or np.round(t3) != np.round(theta3*180/np.pi)):
            print "t1, t2, t3:", t1, t2, t3

        self.position = C
        self.update_boundary_triple_points()

def test_update_TriplePoint(p1,p2,p3):
    """ put the three given test points (each a two-point list or tuple) into a list to pass to update_C """
    na = (0,0) # placeholder point, not used
    TriplePoint(np.array([[p1,na],[p2,na],[p3,na]]), verbose=True)
    

""" Done with definitions, start actual simulation.
"""
t = 0.0
k = 0.0003 # time-step
C = [0,0]  # initial value only
h = 1/16   # grid spacing parameter

boundary_objects = []
for b in boundaries:
    boundary_objects.append(Boundary(b))

tps = [TriplePoint( dict(b=boundary_objects[0], end='tail'),
                dict(b=boundary_objects[1], end='tail'),
                dict(b=boundary_objects[2], end='tail') )]

while t <= 1.0: #1.0:
    for tp in tps:
        tp.update_position()

    for b in boundary_objects:
        b.extrapolate()
        b.get_velocity()
        b.update_position(k)

    #for b in boundary_objects:
    #    b.check_spacing()
    #    b.extrapolate()

    t += k


""" Plot the result at t=1.0: """
ax2 = fig.add_subplot(122)
ax2.cla()
markersize=4.0
markers = ['-o','-d','-x']
for b,marker in zip(boundary_objects, markers):
    b.convert_to_columns()
    d = b.b
    ax2.plot( d[0], d[1], marker, ms=markersize )
ax2.plot( domain_boundary[0], domain_boundary[1], '-k' )
ax2.set_aspect('equal')
ax2.set_xlim(-1.1,1.1)
ax2.set_ylim(-1.1,1.1)
ax2.set_xticks([])
ax2.set_yticks([])

plt.show()
fig.canvas.draw()
