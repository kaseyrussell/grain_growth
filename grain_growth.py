from __future__ import division
import numpy as np

""" Implementation of boundary tracking method from Bronsard & Wetton
    "A Numerical Method for Tracking Curve Networks Moving with Curvature Motion"
    Journal of Computational Physics vol. 120, pp. 66 (1995)

    Author: Kasey J. Russell
    Harvard University SEAS
    Laboratory of Evelyn Hu
    Copyright 2012, all rights reserved
    Released as open-source software under
    the GNU public license (GPL). This code
    is not guaranteed to work, not guaranteed
    against anything, use at your own risk, etc.
    """


def norm(n):
    """ n is assumed to be a two-element array or list. """
    return np.sqrt(n[0]**2 + n[1]**2)


class Boundary(object):
    """ Pass in a 2-d array of x-y values of the initial boundary points. By default,
        this array is assumed to be a two-column array, with column0 for x and
        column1 for y. If as_points=True, then the format is already
        converted into a 1-D array of (x,y) values, with one (x,y) value for each point.
        phase_left and phase_right are the phases of grain adjacent to the grain boundary,
        with left and right referring to the perspective looking from head to tail along
        the boundary."""
    def __init__(self, b_initial, phase_left=None, phase_right=None, as_points=False):
        assert type(b_initial) == np.ndarray
        self._as_points = as_points
        if not as_points:
            """ Tack on elements to beginning and end of array as place-holders for
                the extrapolated points. """
            self.b = np.array([ [0]+list(b_initial[0])+[0], [0]+list(b_initial[1])+[0] ])
            self.convert_to_points()
        else:
            """ assume it already has elements at beginning and end for extrapolated points. """
            self.b = b_initial
            
        self.head = None
        self.tail = None
        self._phase_left = phase_left
        self._phase_right = phase_right
        self._to_be_deleted = False
        self._h = 1/16

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
        assert self._as_points
        
        if self.head is None:
            """ The grain boundary loops on itself. """
            if self.tail is not None:
                raise ValueError("Head is {0}, but tail is {1}".format(self.head, self.tail))
            self.b[0] = self.b[-2]
            self.b[-1] = self.b[1]
        
        if type(self.head) == TriplePoint:
            head = self.head.position
        elif type(self.head) == DomainBoundary:
            head = self.head.get_intersection(self)

        if type(self.tail) == TriplePoint:
            tail = self.tail.position
        elif type(self.tail) == DomainBoundary:
            tail = self.tail.get_intersection(self)

        #!!! TODO!!!: CAN I clean this up?
        self.b[0][0], self.b[0][1] = 2*head[0]-self.b[1][0], 2*head[1]-self.b[1][1]
        self.b[-1][0], self.b[-1][1] = 2*tail[0]-self.b[-2][0], 2*tail[1]-self.b[-2][1]


    def get_velocity(self):
        """ Calculate the instantaneous velocity vector for
            each point on the grain boundary.
            b needs to be in 'point-by-point' format.
            This will return an array in point-by-point format
            with n-2 rows, where n is the number of rows
            in the boundary array (which has the 2
            extrapolated extra points.)
            Returns the min velocity (used for 4RK time-stepping adjustment by B&W)."""
        assert self._as_points
        v = []
        min_velocity = None
        for i in range(1,len(self.b)-1):
            D2 = self.b[i+1] - 2.0*self.b[i] + self.b[i-1]    # numerical second derivative (w/o normalization)
            D1 = (self.b[i+1] - self.b[i-1])/2.0         # numerical first derivative (w/o normalization)
            D1norm2 = D1[0]**2.0 + D1[1]**2.0
            v.append( D2/D1norm2 )             # normalizations cancel since we take ratio
            if i == 1:
                min_velocity = D1norm2/4
            elif D1norm2/4 < min_velocity:
                min_velocity = D1norm2/4

        self.v = np.array(v)
        return min_velocity
        

    def get_scheduled_for_deletion(self):
        """ returns whether the gb is slated for deletion. """
        return self._to_be_deleted


    def get_average_velocity(self):
        """ calculate average velocity of points on the gb. """
        print np.mean(self.v)


    def update_position(self, k):
        """ Using the velocity vector, update the positions of the interior
            points (i.e. not including the extrapolated points).
            k is the time increment (delta-t), using Bronsard-Wetton notation. """
        self.b[1:-1] += self.v*k


    def densify(self):
        """ Double the number of interior points for higher resolution
        and redistribute the points evenly along the length of the path."""
        print "densify: current length:", len(self.b)
        
        """Temporarily use the 'extrapolation' spaces in the arrays
        to hold the locations of the triple point or domain boundary. """
        if type(self.head) == TriplePoint:
            self.b[0] = self.head.position
        elif type(self.head) == DomainBoundary:
            self.b[0] = self.head.get_intersection(self)
        else:
            # gb loops on itself, so put the head mid-way between start and end
            self.b[0] = (self.b[-2] - self.b[1])/2

        if type(self.tail) == TriplePoint:
            self.b[-1] = self.tail.position
        elif type(self.tail) == DomainBoundary:
            self.b[-1] = self.tail.get_intersection(self)
        else:
            self.b[-1] = self.b[0]

        old_boundary = self.b[:]
        new_length   = (len(self.b)-2)*2 + 2
        self.b       = np.zeros([new_length,2])

        delta_vectors = np.diff(old_boundary, axis=0)
        delta_scalars = np.array([0]+[norm(d) for d in delta_vectors])
        boundary_length = sum(delta_scalars)
        
        """ We parameterize the original boundary with the parameter s describing
        the fractional distance along the boundary from head to tail. """
        s = np.cumsum(delta_scalars)/boundary_length
        
        """ the array t will hold parameters describing fractional distance along the
        boundary for the new set of interior points. """
        t = np.arange(1,new_length-1)/(new_length-1)

        """ The new points are to be evenly spaced along the boundary """
        for i,ti in enumerate(t):
            n = np.where(s<ti)[0][-1] # ti should always be greater than zero, and s[0]=0
            alpha_n = (ti - s[n])/(s[n+1]-s[n])
            self.b[i+1] = old_boundary[n]*(1-alpha_n) + old_boundary[n+1]*alpha_n

        self.extrapolate()
        
        
    def coarsen(self):
        """ halve the number of interior points, but keep the two interior end points and
            the two exterior extrapolated points. """
        #TODO: should interpolate and make this smarter and keep num interior points
        # to be a multiple of two
        new_length = (len(self.b)-4)/2 + 4
        old_boundary = self.b[:]
        self.b = np.zeros([new_length,2])
        self.b[-2] = old_boundary[-2]
        self.b[1:-2] = old_boundary[1:-2:2]
        self.extrapolate()
        

    def check_spacing(self):
        """ we need to refine our mesh when it gets too coarse, coarsen it when it gets dense,
            and cut it when it gets to 2 interior points. """

        normd1x2 = [norm( (self.b[i+1] - self.b[i-1])/2.0/self._h )**2 for i in range(1,len(self.b)-1)]
        nu = max(normd1x2)
        mu = min(normd1x2)
        
        if len(self.b) < 5 and mu < self._h**2/10:
            """ we're down to two interior points and need to kill the boundary """
            self._to_be_deleted = True
            return False

        if nu > 4:
            self.densify()
        if nu < 1/4 and len(self.b) > 4:
            self.coarsen()


    def set_termination(self, end, termination):
        """ Set the termination object for the specified end of the grain boundary.
            The value of 'end' should be either 'head' or 'tail'.
            The value of termination should be either a TriplePoint object,
            DomainBoundary, or None.
            """
        assert end in ['head', 'tail']
        if end == 'head':
            self.head = termination
        else:
            self.tail = termination


    def get_termination(self, end):
        """ Get the specified triple point (if there is one)
            The value of 'end' should be either 'head' or 'tail'.
            The returned value will be a TriplePoint object,
            DomainBoundary object if the grain boundary terminates at
            the domain boundary, or None if the boundary loops
            onto itself and so has no termination.
            """
        assert end in ['head', 'tail']
        if end == 'head':
            return self.head
        else:
            return self.tail


    def will_die_alone(self):
        """ Are there any to_be_deleted grain boundaries attached
            to the same triple points as this grain boundary? """
        num_also_dying = 0
        
        if type(self.head) == TriplePoint:
            num_also_dying += self.head.get_number_dying_boundaries() - 1

        if type(self.tail) == TriplePoint:
            num_also_dying += self.tail.get_number_dying_boundaries() - 1

        return True if num_also_dying == 0 else False


class TriplePoint(object):
    """ A class for managing the triple points where three grain boundaries meet.
        Pass in an ordered sequence of three dictionaries containing information
        about the boundaries (in anti-clockwise order around the junction),
        each formatted like so:
        b1 = dict( b=Boundary(), end='head' )
        
        where Boundary is a Boundary object and 'end' can be 'head' or 'tail' to
        specify which end of the boundary is connected to the triple point. """
    def __init__(self, b1, b2, b3):
        self.parse_boundaries(b1, b2, b3)
        self.position = [None, None]
        self.update_position()


    def get_number_dying_boundaries(self):
        """ # to be deleted from this node. """
        num_dying = 0
        for gb in self.boundaries:
            if gb.get_scheduled_for_deletion():
                num_dying += 1
        return num_dying
        
        
    def get_sole_survivor(self):
        """ For a node that we know has two gb to be
            deleted, return the one that won't be deleted. """
        for gb in self.boundaries:
            if not gb._to_be_deleted:
                return gb, self.ends[self.boundaries.index(gb)]
                

    def parse_boundaries(self, b1, b2, b3):
        """ split the dictionaries and save similar info for each of 3 boundaries:
        b1: boundary
        b1pt: the index of the point in the grain boundary array
            to use when calculating triple point location (i.e. either 1 or -2)
        b1end: a string (either 'head' or 'tail') specifying the end of the grain boundary
            that is connected to this triple point.
        """
        self.boundaries, self.endindx, self.ends = zip(*[self.parse_boundary(b) for b in [b1,b2,b3]])
        

    def parse_boundary(self, bdict):
        """ Parse dictionary, saving boundary and info on proper end point
        and set_termination on the boundary so that it points to this triple-point. """
        assert bdict['end'] in ['head', 'tail']
        array_position = 1 if bdict['end'] == 'head' else -2
        bdict['b'].set_termination( bdict['end'], self )
        return bdict['b'], array_position, bdict['end']


    def update_position( self, verbose=False ):
        """ Send in the list of boundaries, each of which ends at the center-point,
            and use them to calculate the new center-point. """
        theta1 = theta2 = theta3 = 2*np.pi/3
        b1N, b2N, b3N = [gb.b[endpt] for gb, endpt in zip(self.boundaries, self.endindx)]
        v13 = b3N - b1N # vector running from pt1 to pt3
        v21 = b1N - b2N # vector running from pt2 to pt1
        v23 = b3N - b2N # vector running from pt2 to pt3
        l1 = norm( v23 )
        l2 = norm( v13 )
        l3 = norm( v21 )
        alpha = np.arccos(np.dot(-v21, v13)/l2/l3) # np.arccos( (l2**2 + l3**2 - l1**2)/(2*l2*l3) )
        delta = theta2 - alpha
        beta = np.arctan( np.sin(delta)/(l3*np.sin(theta3)/(l2*np.sin(theta1))+np.cos(delta)) )
        dilation = np.sin(np.pi - beta - theta1)/np.sin(theta1)
        if np.cross(v23,v21) > 0:
            # rotate clockwise
            beta *= -1
        rotation = np.matrix([[np.cos(beta), -np.sin(beta)],[np.sin(beta), np.cos(beta)]])
        C = b2N + dilation*np.array(np.dot(rotation, v21))[0]

        vc1, vc2, vc3 = C-b1N, C-b2N, C-b3N
        t1 = np.arccos(np.dot(vc2,vc1)/norm(vc2)/norm(vc1))*180/np.pi
        t2 = np.arccos(np.dot(vc3,vc2)/norm(vc3)/norm(vc2))*180/np.pi
        t3 = np.arccos(np.dot(vc3,vc1)/norm(vc3)/norm(vc1))*180/np.pi
        if (np.round(t1) != np.round(theta1*180/np.pi)
            or np.round(t2) != np.round(theta2*180/np.pi)
            or np.round(t3) != np.round(theta3*180/np.pi)):
            print "There is some problem with triple point angles."
            print "  t1, t2, t3:", t1, t2, t3
            print "  locations:", b1N, b2N, b3N

        if verbose:
            print 'points:', b1N, b2N, b3N
            print 'angles are in degrees:'
            print 'alpha:', alpha*180/np.pi
            print 'delta:', delta*180/np.pi
            print 'beta:', beta*180/np.pi
            print 'l1, l2, l3:', l1, l2, l3
            print 'dilation:', dilation
            print 'new triple point location:', C
            print "theta1, theta2, theta3:", t1, t2, t3

        if delta < 0:
            raise ValueError("Bring points together. Your triangle of endpoints is too open to contain the triple point.")

        self.position = C
        

class DomainBoundary(object):
    """ A class representing one of the boundary-conditions that can be
    imposed on the end of a grain boundary. The DomainBoundary condition
    restricts the simulation region by imposing Neumann boundary conditions
    on the gb: the gb must intersect the domain boundary normal to the
    domain boundary (i.e. orthogonal to the tangent to the domain boundary).
    This boundary condition is mostly used for testing purposes to re-create
    Bronsard and Wetton's test cases.
    
    The domain boundary is understood to be the unit circle since that is
    what B&W used.
    
    Arguments:
    end -- 'head' or 'tail': the end point to which the DomainBoundary is attached
    """
    def __init__(self, end):
        assert end in ['head', 'tail']
        self.endindx = 1 if end == 'head' else -2
        self.end = end
        

    def get_intersection(self, gb):
        """ Find where the grain boundary 'gb' would intersect the domain boundary if
        it were to arrive normal to the domain boundary.
        self.b needs to be in 'point-by-point' format. """
        assert gb._as_points
        x, y = gb.b[self.endindx][0], gb.b[self.endindx][1]
        theta = np.arctan(np.abs(y/x))
        return np.sign(x)*np.cos(theta), np.sign(y)*np.sin(theta)


class ThinFilm(object):
    """ A class for simulating thin-film grain growth using the boundary
        front-tracking method. The physics follows what came out of Carl
        Thompson's lab, but the detailed implementation is an attempt to
        follow Bronsard & Wetton.
    """
    def __init__(self, boundaries, triple_points):
        self.add_boundaries(boundaries)
        self.add_triple_points(triple_points)
        self._k = 0.0003
        self._h = 1/16
    

    def add_boundaries(self, boundaries):
        """ Add a list of Boundary objects to the ThinFilm object."""
        self.boundaries = boundaries


    def add_triple_points(self, triple_points):
        """ Add a list of TriplePoint objects to the ThinFilm object."""
        self.triple_points = triple_points
        

    def delete_single_boundary(self, gb):
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
        if gb.head is None:
            """ gb loops onto itself, so it can just be removed. """
            self.boundaries.remove(gb)
            return True
        
        if type(gb.head) == DomainBoundary:
            if type(gb.tail) == DomainBoundary:
                """ gb connects two points on domain boundary & can just be removed. """
                self.boundaries.remove(gb)
            else:
                """ tail connects to a triple point. connect the two other gb
                from that triple point to the domain boundary. """
                for gb, end in zip(gb.tail.boundaries, gb.tail.ends):
                    gb.set_termination(end, DomainBoundary(end))
                self.triple_points.remove(gb.tail)
                self.boundaries.remove(gb)
        else:
            if type(gb.tail) == TriplePoint:
                """ gb connects two triple points. 4* situation.
                Make a new grain boundary roughly orthogonal to the one
                we are going to be deleting, then connect it to 
                the living grain boundaries."""

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

                """ Our new triple points will connect the following
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
                a = (head_subs_boundary.b[head_subs_endindx] + tail_prev_boundary.b[tail_prev_endindx])/2
                b = (head_prev_boundary.b[head_prev_endindx] + tail_subs_boundary.b[tail_subs_endindx])/2
                new_head = a+(b-a)/3
                new_tail = b+(a-b)/3
                new_gb = Boundary(np.array([[0,0],new_head,new_tail,[0,0]]), as_points=True)

                """ Now delete the old grain boundary from the list and insert the new one,
                and delete the two old triple points, replacing them with two new ones. """
                self.triple_points.remove(gb.head)
                self.triple_points.remove(gb.tail)
                self.boundaries.remove(gb)
                
                self.boundaries.append(new_gb)
                TP1 = TriplePoint(
                    dict(b=tail_prev_boundary, end=tail_prev_end),
                    dict(b=head_subs_boundary, end=head_subs_end),
                    dict(b=new_gb, end='head'))
                TP2 = TriplePoint(
                    dict(b=head_prev_boundary, end=head_prev_end),
                    dict(b=tail_subs_boundary, end=tail_subs_end),
                    dict(b=new_gb, end='tail'))
                self.triple_points.append(TP1)
                self.triple_points.append(TP2)
            else:
                """ head connects to a triple point. connect the two other gb
                from that triple point to the domain boundary. """
                for gb, end in zip(gb.head.boundaries, gb.head.ends):
                    gb.set_termination(end, DomainBoundary(end))
                self.triple_points.remove(gb.head)
                self.boundaries.remove(gb)
            return True


    def boundaries_share_two_triple_points(self, b1, b2):
        """ Really? You need more of a description? """
        if ((b1.head == b2.head or b1.head == b2.tail) and
            (b1.tail == b2.head or b1.tail == b2.tail)):
            return True
        else:
            return False


    def delete_two_boundaries(self, to_be_deleted):
        """ Remove two boundaries that share at least one triple point. """
        assert len(to_be_deleted) == 2
        d1, d2 = to_be_deleted
        if self.boundaries_share_two_triple_points(*to_be_deleted):
            """ Connect the two living grain boundaries and turn them into
            one longer grain boundary. If the number of points per boundary
            is not the same, B&W increase the density of the less-dense one
            to match the more-dense one to make it easier to keep track of
            the density of points. """
            a, aend = d1.head.get_sole_survivor()
            b, bend = d1.tail.get_sole_survivor()
            while len(a.b) > len(b.b):
                b.densify()
            while len(a.b) < len(b.b):
                a.densify()
            
            if aend == 'tail':
                if bend == 'head':
                    new_array = np.concatenate(a[:-1], b[1:])
                else:
                    new_array = np.concatenate(a[:-1], b[-1::-1])
            else:
                if bend == 'head':
                    new_array = np.concatenate(a[-1::-1], b[1:])
                else:                    
                    new_array = np.concatenate(a[-1::-1], b[-1::-1])
            
            self.boundaries.append( Boundary( new_array, as_points=True ) )
            self.triple_points.remove(d1.head)
            self.triple_points.remove(d1.tail)
        else:
            """ The two are connected to the domain boundary at the other ends, so
            just connect the surviving member of the TriplePoint to the DomainBoundary. """
            assert (type(d1.head) == DomainBoundary) or (type(d1.tail) == DomainBoundary)
            tp = d1.head if type(d1.head) == TriplePoint else d1.tail
            survivor, survivor_end = tp.get_sole_survivor()
            self.triple_points.remove(tp)
            survivor.set_termination( survivor_end, DomainBoundary(survivor_end) )

        self.boundaries.remove(d1)
        self.boundaries.remove(d2)

    def delete_boundaries(self, to_be_deleted):
        """ Remove boundaries through surgery. """
        
        for gb in to_be_deleted:
            if gb.will_die_alone():
                print "killing single boundary."
                self.delete_single_boundary(gb)
                to_be_deleted.remove(gb)
                
        if len(to_be_deleted) > 2:
            """Shit. Delete three boundaries at once? BUT I AM LAZY!!!"""
            raise ValueError("AHHHHHHHHHHHH. Not implemented.")
        elif len(to_be_deleted) == 0:
            return True
        else:
            self.delete_two_boundaries(to_be_deleted)
            

    def get_dying_but_not_dead_neighbors(self, to_be_deleted):
        """ Probably two adjacent grain boundaries would die simultaneously
        in the real world, but in our discretized, time-stepped world, they
        appear to die sequentially. Here we check for neighbors with only
        two interior points that 'almost' meet the requirements for deletion
        and accelerate their demise by killing them pre-emptively. You can think
        of this as the Bush doctrine applied to grain growth. """
        dying = []
        for gb in to_be_deleted:
            for end in [gb.head, gb.tail]:
                if type(end) == TriplePoint:
                    for gb in end.boundaries:
                        if not gb._to_be_deleted and len(gb.b) <= 4:
                            print "Deleting by association!!!"
                            gb._to_be_deleted = True
                            dying.append(gb)
        return dying
                            
        

    def regrid(self):
        """ Check each boundary to ensure that its density of points is
            not too high and not too low. If it only has two interior points
            and is too dense, then it is deleted and the remaining boundaries
            are surgically repaired across the void. """
            
        to_be_deleted = []
        for b in self.boundaries:
            if b.check_spacing() == False:
                to_be_deleted.append(b)
        
        if len(to_be_deleted) > 0:
            dying = self.get_dying_but_not_dead_neighbors(to_be_deleted)
            if len(dying) > 0: to_be_deleted += dying
            print "Num to be deleted:", len(to_be_deleted)
            self.delete_boundaries(to_be_deleted)

        return True


    def calculate_timestep(self, min_velocity):
        """ alter time step according to the overly-restrictive 4RK condition of B&W
            Eq. 15 """
        self._k = min_velocity
        

    def get_timestep(self):
        """ Return the most recent value of time step. """
        return self._k


    def set_timestep(self, k):
        """ Set the time increment to be used for the subsequent iteration. """
        self._k = k


    def set_resolution(self, h):
        """ set resolution parameter h """
        self._h = h


    def update_grain_boundaries(self):
        """ Cycle through the list of grain boundaries and update
            their locations. """
        min_velocity = None
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


    def update_triple_points(self, **kwargs):
        """ Cycle through the list of triple points and update their
            locations using the most recent interior points of the
            grain boundaries. """
        for tp in self.triple_points:
            tp.update_position( kwargs )
    


