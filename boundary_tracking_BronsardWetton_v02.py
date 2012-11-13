from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

"""Try to copy fig 4 from Bronsard & Wetton
    "A Numerical Method for Tracking Curve Networks Moving with Curvature Motion"
    Journal of Computational Physics vol. 120, pp. 66 (1995)
    """

""" This version totally works!!! But it is pretty slow, no optimization implemented...
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
boundaries = [b1, b2, b3]


def norm(n):
    """ n is assumed to be a two-element array or list. """
    return np.sqrt(n[0]**2 + n[1]**2)

def convert_to_points(b):
    """ Incoming boundary is a two-column array, with
    column0 as x-data, column1 as y-data. It is easier
    to do point-by-point vector-style work with the
    x and y points associated with each other, so you
    have just one column of data, with each data point
    being an (x,y) pair. This function converts
    into this point-by-point format."""
    return np.array(zip(list(b[0]), list(b[1])))


def convert_to_columns(b):
    """ Undoes function 'convert_to_points' for easier plotting."""
    return np.array(zip(*b))


def get_edgept( b ):
    """ By 'edge', I mean edge of the simulation region, i.e. the domain boundary.
        b needs to be in 'point-by-point' format. """
    x, y = b[1][0], b[1][1]
    theta = np.arctan(np.abs(y/x))
    return np.sign(x)*np.cos(theta), np.sign(y)*np.sin(theta)


def extrapolate( b, centerpt ):
    """ Extrapolate across the domain boundary or center point.
        b needs to be in 'point-by-point' format. """
    edgept = get_edgept(b)
    b[0][0], b[0][1] = 2*edgept[0]-b[1][0], 2*edgept[1]-b[1][1]
    b[-1][0], b[-1][1] = 2*centerpt[0]-b[-2][0], 2*centerpt[1]-b[-2][1]
    

def get_velocity( b ):
    """ Calculate the instantaneous velocity vector for
        each point on the grain boundary.
        b needs to be in 'point-by-point' format.
        This will return an array in point-by-point format
        with n-2 rows, where n is the number of rows
        in the boundary array (which has the 2
        extrapolated extra points.) """
    v = []
    for i in range(1,len(b)-1):
        D2 = b[i+1] - 2.0*b[i] + b[i-1]
        D1 = (b[i+1] - b[i-1])/2.0
        D1norm2 = D1[0]**2.0 + D1[1]**2.0
        v.append( D2/D1norm2 )
    return np.array(v)


def update_position( b, v, k ):
    """ Using the velocity vector, update the positions of the interior
        points (i.e. not including the extrapolated points).
        k is the time increment (delta-t), using Bronsard-Wetton notation. """
    b[1:-1] += v*k


def update_C( bds, verbose=False ):
    """ Send in the list of boundaries, each of which ends at the center-point,
        and use them to calculate the new center-point. """
    theta1 = theta2 = theta3 = 2*np.pi/3
    b1N = bds[0][-2]
    b2N = bds[1][-2]
    b3N = bds[2][-2]
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
        print 'new C:', C

    vc1, vc2, vc3 = C-b1N, C-b2N, C-b3N
    t1 = np.arccos(np.dot(vc2,vc1)/norm(vc2)/norm(vc1))*180/np.pi
    t2 = np.arccos(np.dot(vc3,vc2)/norm(vc3)/norm(vc2))*180/np.pi
    t3 = np.arccos(np.dot(vc3,vc1)/norm(vc3)/norm(vc1))*180/np.pi
    if (np.round(t1) != np.round(theta1*180/np.pi)
        or np.round(t2) != np.round(theta2*180/np.pi)
        or np.round(t3) != np.round(theta3*180/np.pi)):
        print "t1, t2, t3:", t1, t2, t3
    return C

def test_update_C(p1,p2,p3):
    """ put the three given test points (each a two-point list or tuple) into a list to pass to update_C """
    na = (0,0) # placeholder point, not used
    update_C(np.array([[p1,na],[p2,na],[p3,na]]), verbose=True)
    

""" Done with definitions, start actual simulation.
"""
t=0.0
k = 0.0003 # time-step
C = [0,0] # initial value only

boundaries_by_points = []
for b in boundaries:
    boundaries_by_points.append(convert_to_points(b))

while t <= 1.0: #1.0:
    C = update_C(boundaries_by_points, verbose=False)
    for b in boundaries_by_points:
        extrapolate(b, C)
        v = get_velocity(b)
        update_position(b,v,k)
        extrapolate(b, C)
    t += k

C = update_C(boundaries_by_points, verbose=False) # don't need this again; just testing
  

b1, b2, b3 = boundaries_by_points

""" Plot the result at t=1.0: """
ax2 = fig.add_subplot(122)
ax2.cla()
markersize=4.0
markers = ['-o','-d','-x']
for b,marker in zip(boundaries_by_points, markers):
    d = convert_to_columns(b)
    ax2.plot( d[0], d[1], marker, ms=markersize )
ax2.plot( domain_boundary[0], domain_boundary[1], '-k' )
ax2.set_aspect('equal')
ax2.set_xlim(-1.1,1.1)
ax2.set_ylim(-1.1,1.1)
ax2.set_xticks([])
ax2.set_yticks([])

plt.show()
fig.canvas.draw()
